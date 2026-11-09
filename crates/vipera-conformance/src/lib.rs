//! Shared conformance suites.
//!
//! A component is not "done" because its own tests pass; it is done when the contract suite passes
//! (spec FR-023, SC-008). Each suite is a list of cases plus a runner, so the same contract can be
//! applied to the reference implementation and to every future implementation.
//!
//! Slice 2 ships the GGUF parser suite. The sampler suite now ships too, because
//! the first algorithms exist (`greedy` and `temperature` in `vipera-sampler`);
//! the backend suite is still pending, because the `ForwardPass` contract
//! needs KV fixtures shared with its implementers and those live in
//! `vipera-backend-cpu`, outside this crate.

#![warn(missing_docs)]

use std::fmt;

/// What a case expects the implementation to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expect {
    /// The input must be accepted.
    Accepts,
    /// The input must be rejected, and the message must contain this text.
    Rejects {
        /// Substring the error message must contain, so a rejection for the wrong reason fails.
        containing: &'static str,
    },
}

/// One contract case: bytes in, expected verdict out.
#[derive(Debug, Clone)]
pub struct Case {
    /// Stable case name, used in reports.
    pub name: &'static str,
    /// Input bytes handed to the implementation.
    pub bytes: Vec<u8>,
    /// Expected verdict.
    pub expect: Expect,
}

impl Case {
    /// A case that must be accepted.
    pub fn accepts(name: &'static str, bytes: Vec<u8>) -> Self {
        Self {
            name,
            bytes,
            expect: Expect::Accepts,
        }
    }

    /// A case that must be rejected with a message containing `containing`.
    pub fn rejects(name: &'static str, bytes: Vec<u8>, containing: &'static str) -> Self {
        Self {
            name,
            bytes,
            expect: Expect::Rejects { containing },
        }
    }
}

/// Outcome of one case.
#[derive(Debug, Clone)]
pub struct CaseResult {
    /// Case name.
    pub name: &'static str,
    /// Whether the implementation met the expectation.
    pub passed: bool,
    /// What happened, for failing cases.
    pub detail: String,
}

impl fmt::Display for CaseResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.passed {
            write!(f, "PASS {}", self.name)
        } else {
            write!(f, "FAIL {} — {}", self.name, self.detail)
        }
    }
}

/// Run every case against `check`, which returns `Ok(())` when the input was accepted.
pub fn run(cases: &[Case], check: impl Fn(&[u8]) -> Result<(), String>) -> Vec<CaseResult> {
    cases
        .iter()
        .map(|case| {
            let outcome = check(&case.bytes);
            let (passed, detail) = match (&case.expect, &outcome) {
                (Expect::Accepts, Ok(())) => (true, String::new()),
                (Expect::Accepts, Err(message)) => {
                    (false, format!("expected acceptance, got error: {message}"))
                }
                (Expect::Rejects { containing }, Err(message)) => {
                    if message.contains(containing) {
                        (true, String::new())
                    } else {
                        (
                            false,
                            format!(
                                "rejected for the wrong reason: {message:?} lacks {containing:?}"
                            ),
                        )
                    }
                }
                (Expect::Rejects { containing }, Ok(())) => (
                    false,
                    format!("expected rejection containing {containing:?}, got acceptance"),
                ),
            };
            CaseResult {
                name: case.name,
                passed,
                detail,
            }
        })
        .collect()
}

/// Run a suite and panic with a readable report if any case fails.
///
/// This is what a implementing crate calls from a `#[test]`, so a regression in the contract shows up
/// as a normal test failure.
pub fn assert_conforms(
    contract: &str,
    cases: &[Case],
    check: impl Fn(&[u8]) -> Result<(), String>,
) {
    let results = run(cases, check);
    let failures: Vec<&CaseResult> = results.iter().filter(|result| !result.passed).collect();
    assert!(
        failures.is_empty(),
        "conformance suite \"{contract}\" failed {}/{} cases:\n{}",
        failures.len(),
        results.len(),
        failures
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// Metadata value used when building fixtures.
#[derive(Debug, Clone)]
pub enum MetaValue {
    /// 32-bit unsigned integer.
    U32(u32),
    /// UTF-8 string.
    Str(String),
    /// Array of strings.
    StrArray(Vec<String>),
    /// Array of 32-bit floats, used by vocabulary scores.
    F32Array(Vec<f32>),
    /// Array of 32-bit signed integers, used by token types.
    I32Array(Vec<i32>),
}

/// One tensor to place in a fixture.
#[derive(Debug, Clone)]
pub struct TensorSpec {
    /// Tensor name.
    pub name: String,
    /// Dimensions.
    pub dims: Vec<u64>,
    /// GGML type id.
    pub type_id: u32,
    /// Offset inside the data section.
    pub offset: u64,
}

/// Builds a GGUF file in memory, so fixtures are visible code rather than opaque blobs.
pub struct GgufBuilder {
    /// Format version.
    pub version: u32,
    /// Metadata entries in insertion order.
    pub metadata: Vec<(String, MetaValue)>,
    /// Tensor table entries.
    pub tensors: Vec<TensorSpec>,
    /// Bytes after the aligned tensor table.
    pub data_len: usize,
}

impl GgufBuilder {
    /// A minimal valid v3 file with one F32 tensor.
    pub fn minimal() -> Self {
        Self {
            version: 3,
            metadata: vec![(
                "general.architecture".to_owned(),
                MetaValue::Str("llama".to_owned()),
            )],
            tensors: vec![TensorSpec {
                name: "token_embd.weight".to_owned(),
                dims: vec![4],
                type_id: 0,
                offset: 0,
            }],
            data_len: 16,
        }
    }

    /// Serialize to bytes.
    pub fn build(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());

        for (key, value) in &self.metadata {
            out.extend_from_slice(&encode_string(key));
            match value {
                MetaValue::U32(value) => {
                    out.extend_from_slice(&4u32.to_le_bytes());
                    out.extend_from_slice(&value.to_le_bytes());
                }
                MetaValue::Str(value) => {
                    out.extend_from_slice(&8u32.to_le_bytes());
                    out.extend_from_slice(&encode_string(value));
                }
                MetaValue::StrArray(values) => {
                    out.extend_from_slice(&9u32.to_le_bytes());
                    out.extend_from_slice(&8u32.to_le_bytes());
                    out.extend_from_slice(&(values.len() as u64).to_le_bytes());
                    for value in values {
                        out.extend_from_slice(&encode_string(value));
                    }
                }
                MetaValue::F32Array(values) => {
                    out.extend_from_slice(&9u32.to_le_bytes());
                    out.extend_from_slice(&6u32.to_le_bytes());
                    out.extend_from_slice(&(values.len() as u64).to_le_bytes());
                    for value in values {
                        out.extend_from_slice(&value.to_le_bytes());
                    }
                }
                MetaValue::I32Array(values) => {
                    out.extend_from_slice(&9u32.to_le_bytes());
                    out.extend_from_slice(&5u32.to_le_bytes());
                    out.extend_from_slice(&(values.len() as u64).to_le_bytes());
                    for value in values {
                        out.extend_from_slice(&value.to_le_bytes());
                    }
                }
            }
        }

        for tensor in &self.tensors {
            out.extend_from_slice(&encode_string(&tensor.name));
            out.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
            for dim in &tensor.dims {
                out.extend_from_slice(&dim.to_le_bytes());
            }
            out.extend_from_slice(&tensor.type_id.to_le_bytes());
            out.extend_from_slice(&tensor.offset.to_le_bytes());
        }

        while out.len() % 32 != 0 {
            out.push(0);
        }
        out.extend(std::iter::repeat_n(0u8, self.data_len));
        out
    }
}

fn encode_string(value: &str) -> Vec<u8> {
    let mut out = (value.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(value.as_bytes());
    out
}

/// Contract cases every GGUF v2/v3 parser must pass.
pub fn gguf_cases() -> Vec<Case> {
    let mut cases = Vec::new();

    cases.push(Case::accepts(
        "valid v3 file",
        GgufBuilder::minimal().build(),
    ));

    let mut v2 = GgufBuilder::minimal();
    v2.version = 2;
    cases.push(Case::accepts("valid v2 file", v2.build()));

    let mut wrong_magic = GgufBuilder::minimal().build();
    wrong_magic[0..4].copy_from_slice(b"GGUX");
    cases.push(Case::rejects("wrong magic", wrong_magic, "magic"));

    let mut v1 = GgufBuilder::minimal();
    v1.version = 1;
    cases.push(Case::rejects("version 1 is refused", v1.build(), "v1"));

    let mut future = GgufBuilder::minimal();
    future.version = 9;
    cases.push(Case::rejects(
        "future version is refused",
        future.build(),
        "version 9",
    ));

    let mut truncated = GgufBuilder::minimal().build();
    truncated.truncate(20);
    cases.push(Case::rejects(
        "truncated header",
        truncated,
        "unexpected end of file",
    ));

    let mut overrun_tensor = GgufBuilder::minimal();
    overrun_tensor.tensors[0].dims = vec![1_000_000];
    cases.push(Case::rejects(
        "tensor overruns the file",
        overrun_tensor.build(),
        "ends at byte",
    ));

    let mut unknown_type = GgufBuilder::minimal();
    unknown_type.tensors[0].type_id = 99;
    cases.push(Case::rejects(
        "unknown element type",
        unknown_type.build(),
        "99",
    ));

    let mut rank_zero = GgufBuilder::minimal();
    rank_zero.tensors[0].dims = Vec::new();
    cases.push(Case::rejects("rank zero tensor", rank_zero.build(), "rank"));

    let mut duplicate_key = GgufBuilder::minimal();
    duplicate_key.metadata.push((
        "general.architecture".to_owned(),
        MetaValue::Str("qwen2".to_owned()),
    ));
    cases.push(Case::rejects(
        "duplicate metadata key",
        duplicate_key.build(),
        "more than once",
    ));

    cases
}

/// What a sampler case expects the strategy to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerExpect {
    /// The strategy must return this token id.
    Returns(u32),
    /// The strategy must fail, and the error must contain this text.
    Rejects {
        /// Substring the error must contain.
        containing: &'static str,
    },
}

/// One contract case: logits in, expected token out.
#[derive(Debug, Clone)]
pub struct SamplerCase {
    /// Stable case name, used in reports.
    pub name: &'static str,
    /// One logit per vocabulary entry, in id order.
    pub logits: Vec<f32>,
    /// Expected outcome.
    pub expect: SamplerExpect,
}

impl SamplerCase {
    /// A case that must return `id`.
    pub fn returns(name: &'static str, logits: Vec<f32>, id: u32) -> Self {
        Self {
            name,
            logits,
            expect: SamplerExpect::Returns(id),
        }
    }

    /// A case that must fail with an error containing `containing`.
    pub fn rejects(name: &'static str, logits: Vec<f32>, containing: &'static str) -> Self {
        Self {
            name,
            logits,
            expect: SamplerExpect::Rejects { containing },
        }
    }
}

/// Run sampler cases against `check`, which returns `Ok(id)` when a token was picked.
pub fn run_sampler(
    cases: &[SamplerCase],
    check: impl Fn(&[f32]) -> Result<u32, String>,
) -> Vec<CaseResult> {
    cases
        .iter()
        .map(|case| {
            let outcome = check(&case.logits);
            let (passed, detail) = match (&case.expect, &outcome) {
                (SamplerExpect::Returns(expected), Ok(id)) if *id == *expected => {
                    (true, String::new())
                }
                (SamplerExpect::Returns(expected), Ok(id)) => {
                    (false, format!("expected token {expected}, got {id}"))
                }
                (SamplerExpect::Returns(expected), Err(message)) => (
                    false,
                    format!("expected token {expected}, got error: {message}"),
                ),
                (SamplerExpect::Rejects { containing }, Err(message)) => {
                    if message.contains(containing) {
                        (true, String::new())
                    } else {
                        (
                            false,
                            format!(
                                "rejected for the wrong reason: {message:?} lacks {containing:?}"
                            ),
                        )
                    }
                }
                (SamplerExpect::Rejects { containing }, Ok(id)) => (
                    false,
                    format!("expected rejection containing {containing:?}, got token {id}"),
                ),
            };
            CaseResult {
                name: case.name,
                passed,
                detail,
            }
        })
        .collect()
}

/// Run a sampler suite and panic with a readable report if any case fails.
pub fn assert_sampler_conforms(
    contract: &str,
    cases: &[SamplerCase],
    check: impl Fn(&[f32]) -> Result<u32, String>,
) {
    let results = run_sampler(cases, check);
    let failures: Vec<&CaseResult> = results.iter().filter(|result| !result.passed).collect();
    assert!(
        failures.is_empty(),
        "conformance suite \"{contract}\" failed {}/{} cases:\n{}",
        failures.len(),
        results.len(),
        failures
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// Contract cases every greedy sampler must pass (FR-012, FR-023).
///
/// Every case below is a behavior already pinned by `vipera-sampler`'s own
/// tests (`zero_temperature_is_greedy`, `greedy_breaks_ties_towards_the_lowest_id`,
/// `non_finite_logits_are_skipped`, `empty_or_unusable_logits_are_rejected`);
/// the shared suite copies them so a second algorithm has a contract to
/// meet without reading the first implementation. Temperature sampling is
/// deliberately absent: it is stochastic, and pinning a fixed-seed sequence
/// here would freeze the private `Rng` stream as a contract.
pub fn sampler_cases() -> Vec<SamplerCase> {
    vec![
        SamplerCase::returns("clear argmax", vec![0.1, 5.0, 0.2], 1),
        SamplerCase::returns("tie goes to lowest id", vec![2.0, 2.0, 2.0], 0),
        SamplerCase::returns("single logit", vec![-1.5], 0),
        SamplerCase::returns("negative logits argmax", vec![-5.0, -1.0, -3.0], 1),
        SamplerCase::returns(
            "non-finite logits are skipped",
            vec![f32::NAN, 1.0, f32::INFINITY],
            1,
        ),
        SamplerCase::rejects("empty logits", Vec::new(), "NoFiniteLogits"),
        SamplerCase::rejects(
            "all logits non-finite",
            vec![f32::NAN, f32::NEG_INFINITY],
            "NoFiniteLogits",
        ),
        SamplerCase::rejects(
            "all logits infinite",
            vec![f32::INFINITY, f32::INFINITY],
            "NoFiniteLogits",
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_passing_check_reports_every_case_as_passed() {
        let cases = vec![
            Case::accepts("anything", vec![1]),
            Case::rejects("nothing", vec![2], "nope"),
        ];
        let results = run(&cases, |bytes| match bytes {
            [1] => Ok(()),
            _ => Err("nope".to_owned()),
        });
        assert!(results.iter().all(|result| result.passed));
    }

    #[test]
    fn a_rejection_for_the_wrong_reason_fails() {
        let cases = vec![Case::rejects("needs magic", vec![9], "magic")];
        let results = run(&cases, |_| Err("something else".to_owned()));
        assert!(!results[0].passed);
        assert!(results[0].detail.contains("wrong reason"));
    }

    #[test]
    fn an_acceptance_that_errors_fails() {
        let cases = vec![Case::accepts("must pass", vec![1])];
        let results = run(&cases, |_| Err("boom".to_owned()));
        assert!(!results[0].passed);
        assert!(results[0].detail.contains("expected acceptance"));
    }

    #[test]
    fn fixtures_are_byte_identical_across_runs() {
        assert_eq!(
            GgufBuilder::minimal().build(),
            GgufBuilder::minimal().build()
        );
    }

    #[test]
    fn a_sampler_case_that_returns_the_expected_id_passes() {
        let cases = vec![SamplerCase::returns("only", vec![0.0], 0)];
        let results = run_sampler(&cases, |_| Ok(0));
        assert!(results[0].passed);
    }

    #[test]
    fn a_sampler_case_that_returns_the_wrong_id_fails() {
        let cases = vec![SamplerCase::returns("only", vec![0.0], 0)];
        let results = run_sampler(&cases, |_| Ok(1));
        assert!(!results[0].passed);
        assert!(results[0].detail.contains("expected token 0"));
    }

    #[test]
    fn a_sampler_rejection_for_the_wrong_reason_fails() {
        let cases = vec![SamplerCase::rejects("empty", Vec::new(), "NoFiniteLogits")];
        let results = run_sampler(&cases, |_| Err("something else".to_owned()));
        assert!(!results[0].passed);
        assert!(results[0].detail.contains("wrong reason"));
    }

    #[test]
    fn sampler_cases_accept_a_greedy_stub() {
        // Greedy stub: the argmax, ties to lowest id, skipping non-finite.
        assert_sampler_conforms("sampler-stub", &sampler_cases(), |logits: &[f32]| {
            let mut best: Option<(u32, f32)> = None;
            for (id, logit) in logits.iter().enumerate() {
                let value = *logit;
                if !value.is_finite() {
                    continue;
                }
                let better = match best {
                    None => true,
                    Some((best_id, best_value)) => {
                        value > best_value || (value == best_value && (id as u32) < best_id)
                    }
                };
                if better {
                    best = Some((id as u32, value));
                }
            }
            best.map(|(id, _)| id)
                .ok_or_else(|| "NoFiniteLogits".to_owned())
        });
    }
}
