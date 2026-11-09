//! `vipera` command line.
//!
//! Slice 1 shipped `model-info`; slice 2 added `plan`; slice 3 added `profile`, `tokenize`, `sample`,
//! and `verify`; slice 4 added `generate`, the first command that runs a model end to end. Argument
//! parsing stays manual — a handful of commands do not justify a parser crate (see
//! `specs/001-vipera-rs-engine/research.md`, R3).

use std::io::{Read, Write};

use std::process::ExitCode;
use std::time::Instant;

use vipera_backend_cpu::CpuBackend;
use vipera_core::gguf::{GgufFile, ModelDescriptor};
use vipera_core::hardware::SystemProbe;
use vipera_core::planner::{KvCacheType, ModelShape, PlacementPlan, PlanRequest, plan};
use vipera_errors::ViperaError;
use vipera_runtime::{
    DEFAULT_TOKENS_PER_BLOCK, Engine, EngineConfig, KvCache, KvConfig, Stage, peak_resident_bytes,
};
use vipera_sampler::{Sampler, SamplerConfig};
use vipera_server::serve;
use vipera_tokenizer::{ChatMessage, Tokenizer, chat};
use vipera_traits::{HardwareProbe, HardwareProfile};

const USAGE: &str = "\
vipera — local inference engine

USAGE:
    vipera model-info <model.gguf> [--json]
    vipera plan <model.gguf> [--ctx N] [--kv-type f16|q8_0|q4_0] [--json]
    vipera profile [--json]
    vipera tokenize <model.gguf> [--text TEXT | --text-file PATH | --ids IDS]
                   [--system TEXT] [--user TEXT] [--assistant TEXT] [--no-generation-prompt] [--json]
    vipera sample [--logits CSV | --logits-file PATH] [--model MODEL] [--count N] [--seed N]
                  [--temperature F] [--top-k N] [--top-p F] [--min-p F]
                  [--repetition-penalty F] [--recent IDS] [--json]
    vipera verify <model.gguf> [--ctx N] [--kv-type f16|q8_0|q4_0] [--cpu] [--json]
    vipera generate <model.gguf> --prompt TEXT [--chat] [--system TEXT] [--max-tokens N]
                    [--ctx N] [--kv-type f16|q8_0] [--temperature F] [--top-k N] [--top-p F]
                    [--min-p F] [--repetition-penalty F] [--seed N] [--json]
    vipera serve <model.gguf> [--host ADDR] [--port N] [--api-key KEY] [--model-id NAME]
                 [--ctx N] [--kv-type f16|q8_0]
    vipera --help

COMMANDS:
    model-info    Print the model's architecture, quantization, and tensor summary
    plan          Predict device and host memory for a placement without loading the model
    profile       Print the detected hardware profile and its stable fingerprint
    tokenize      Tokenize text or a rendered chat with the model's own vocabulary
    sample        Sample token ids from logits with a fixed seed and print the result
    verify        Load a model and compare measured memory with the plan's prediction
    generate      Run the model on the CPU and print what it writes next
    serve         Serve OpenAI-compatible chat completions over HTTP, streaming included

EXIT CODES:
    0 success · 1 usage · 2 bad input · 3 infeasible plan · 4 internal
";

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(code) => ExitCode::from(code),
        Err(err) => {
            eprintln!("vipera: {err}");
            ExitCode::from(u8::try_from(err.exit_code()).unwrap_or(4))
        }
    }
}

fn run(args: Vec<String>) -> Result<u8, ViperaError> {
    let Some(command) = args.first() else {
        eprint!("{USAGE}");
        std::process::exit(1);
    };

    match command.as_str() {
        "--help" | "-h" | "help" => {
            emit(USAGE)?;
            Ok(0)
        }
        "model-info" | "inspect" => {
            model_info(&args[1..])?;
            Ok(0)
        }
        "plan" => plan_command(&args[1..]),
        "profile" => profile_command(&args[1..]),
        "tokenize" => tokenize_command(&args[1..]),
        "sample" => sample_command(&args[1..]),
        "verify" => verify_command(&args[1..]),
        "generate" => generate_command(&args[1..]),
        "serve" => serve::command(&args[1..]),
        other => Err(ViperaError::usage(format!("unknown command \"{other}\""))),
    }
}

fn resolve_model_path(path: &str) -> Result<String, ViperaError> {
    let candidate = std::path::Path::new(path);
    if !candidate.is_dir() {
        return Ok(path.to_owned());
    }
    let mut found = Vec::new();
    let entries = std::fs::read_dir(candidate).map_err(|error| {
        ViperaError::internal(format!("cannot list model directory \"{path}\": {error}"))
    })?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            ViperaError::internal(format!("cannot read model directory \"{path}\": {error}"))
        })?;
        let entry_path = entry.path();
        if entry_path.is_file()
            && entry_path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            found.push(entry_path.to_string_lossy().into_owned());
        }
    }
    match found.len() {
        1 => Ok(found.into_iter().next().expect("one gguf")),
        0 => Err(ViperaError::usage(format!(
            "directory \"{path}\" contains no .gguf file; pass a model file directly"
        ))),
        many => Err(ViperaError::usage(format!(
            "directory \"{path}\" contains {many} .gguf files; pass one model file directly"
        ))),
    }
}

fn write_out(writer: &mut dyn std::io::Write, text: &str) -> Result<(), ViperaError> {
    match writer
        .write_all(text.as_bytes())
        .and_then(|()| writer.flush())
    {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
        Err(error) => Err(ViperaError::internal(format!(
            "cannot write to stdout: {error}"
        ))),
    }
}

fn emit(text: &str) -> Result<(), ViperaError> {
    let mut out = std::io::stdout().lock();
    write_out(&mut out, text)
}

fn take_path<'a>(args: &'a [String], command: &str) -> Result<&'a str, ViperaError> {
    args.iter()
        .find(|arg| !arg.starts_with('-'))
        .map(String::as_str)
        .ok_or_else(|| ViperaError::usage(format!("{command} needs a model path")))
}

fn model_info(args: &[String]) -> Result<(), ViperaError> {
    let path = take_path(args, "model-info")?;
    if let Some(unknown) = args
        .iter()
        .find(|arg| arg.starts_with('-') && arg.as_str() != "--json")
    {
        return Err(ViperaError::usage(format!(
            "unknown flag \"{unknown}\" for model-info"
        )));
    }
    let json = args.iter().any(|arg| arg == "--json");

    let bytes = std::fs::read(resolve_model_path(path)?)?;
    let described = GgufFile::parse(&bytes)?.describe();

    emit(
        &(if json {
            render_json(&described)
        } else {
            render_text(&described)
        }),
    )?;
    Ok(())
}

fn plan_command(args: &[String]) -> Result<u8, ViperaError> {
    let mut json = false;
    let mut ctx: Option<u64> = None;
    let mut kv_type = KvCacheType::F16;
    let mut path: Option<&str> = None;

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--json" => json = true,
            "--ctx" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| ViperaError::usage("--ctx needs a token count"))?;
                ctx = Some(value.parse().map_err(|_| {
                    ViperaError::usage(format!("--ctx \"{value}\" is not a number"))
                })?);
                index += 1;
            }
            "--kv-type" => {
                let value = args
                    .get(index + 1)
                    .ok_or_else(|| ViperaError::usage("--kv-type needs a value"))?;
                kv_type = KvCacheType::parse(value)?;
                index += 1;
            }
            flag if flag.starts_with('-') => {
                return Err(ViperaError::usage(format!(
                    "unknown flag \"{flag}\" for plan"
                )));
            }
            other => {
                if path.is_some() {
                    return Err(ViperaError::usage(format!(
                        "unexpected extra argument \"{other}\""
                    )));
                }
                path = Some(other);
            }
        }
        index += 1;
    }

    let path = path.ok_or_else(|| ViperaError::usage("plan needs a model path"))?;
    let bytes = std::fs::read(resolve_model_path(path)?)?;
    let file = GgufFile::parse(&bytes)?;
    let descriptor = file.describe();
    let hardware = SystemProbe::new().profile();
    let request = PlanRequest {
        ctx: ctx.unwrap_or(descriptor.context_limit.unwrap_or(4096)),
        kv_type,
        batch_size: 1,
    };
    let placement = plan(&file, &hardware, &request)?;

    emit(
        &(if json {
            render_plan_json(&placement)
        } else {
            render_plan_text(&placement, &hardware)
        }),
    )?;

    Ok(if placement.feasible { 0 } else { 3 })
}

fn profile_command(args: &[String]) -> Result<u8, ViperaError> {
    let mut json = false;
    for arg in args {
        match arg.as_str() {
            "--json" => json = true,
            flag => {
                return Err(ViperaError::usage(format!(
                    "unknown flag \"{flag}\" for profile"
                )));
            }
        }
    }

    let hardware = SystemProbe::new().profile();
    emit(
        &(if json {
            render_profile_json(&hardware)
        } else {
            render_profile_text(&hardware)
        }),
    )?;
    Ok(0)
}

/// SC-002: predicted peak memory must land within ten percent of measured peak memory.
const MEMORY_TARGET_PERCENT: f64 = 10.0;

/// Predicted versus measured memory for one placement.
struct Verification {
    model_identity: String,
    hardware_fingerprint: String,
    placement: &'static str,
    layers_on_device: u64,
    kv_cache_type: String,
    context_length: u64,
    predicted_device_bytes: u64,
    measured_device_bytes: u64,
    device_deviation_percent: Option<f64>,
    predicted_host_bytes: u64,
    measured_host_bytes: u64,
    host_deviation_percent: Option<f64>,
    weights_bytes: u64,
    kv_cache_bytes: u64,
    kv_blocks: u64,
    verdict: &'static str,
}

fn verify_command(args: &[String]) -> Result<u8, ViperaError> {
    let mut json = false;
    let mut cpu = false;
    let mut ctx: Option<u64> = None;
    let mut kv_type = KvCacheType::F16;
    let mut path: Option<&str> = None;

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--json" => json = true,
            "--cpu" => cpu = true,
            "--ctx" => {
                ctx = Some(parse_number("--ctx", next_value(args, index, "--ctx")?)?);
                index += 1;
            }
            "--kv-type" => {
                kv_type = KvCacheType::parse(next_value(args, index, "--kv-type")?)?;
                index += 1;
            }
            flag if flag.starts_with('-') => {
                return Err(ViperaError::usage(format!(
                    "unknown flag \"{flag}\" for verify"
                )));
            }
            other => {
                if path.is_some() {
                    return Err(ViperaError::usage(format!(
                        "unexpected extra argument \"{other}\""
                    )));
                }
                path = Some(other);
            }
        }
        index += 1;
    }

    let path = path.ok_or_else(|| ViperaError::usage("verify needs a model path"))?;

    // Baseline is taken before anything is loaded, so the delta below is this run's peak.
    let baseline = peak_resident_bytes();
    let bytes = std::fs::read(resolve_model_path(path)?)?;
    let file = GgufFile::parse(&bytes)?;
    let descriptor = file.describe();
    let shape = ModelShape::from_gguf(&file)?;

    let probed = SystemProbe::new().profile();
    let hardware = if cpu {
        HardwareProfile {
            devices: Vec::new(),
            ..probed
        }
    } else {
        probed
    };
    let request = PlanRequest {
        ctx: ctx.unwrap_or(descriptor.context_limit.unwrap_or(4096)),
        kv_type,
        batch_size: 1,
    };
    let placement = plan(&file, &hardware, &request)?;
    if !placement.feasible {
        let reasons: Vec<&str> = placement
            .violations
            .iter()
            .map(|violation| violation.message.as_str())
            .collect();
        return Err(ViperaError::infeasible(reasons.join("; ")));
    }
    if placement.layers_on_device > 0 {
        return Err(ViperaError::unsupported(format!(
            "this build has no device backend, so a placement with {} layers on the device cannot be \
             loaded or measured; re-run with --cpu to verify a CPU placement",
            placement.layers_on_device
        )));
    }

    let bytes_per_token = shape.kv_bytes(1, kv_type)?;
    let kv_budget = shape.kv_bytes(request.ctx, kv_type)?;
    let kv_config = KvConfig::from_bytes(kv_budget, bytes_per_token, DEFAULT_TOKENS_PER_BLOCK)?;
    let kv = KvCache::new(kv_config)?;

    let measured_host_bytes = match (baseline, peak_resident_bytes()) {
        (Some(before), Some(peak)) => peak.saturating_sub(before),
        _ => 0,
    };
    let kv_cache_bytes = kv.config().total_bytes().unwrap_or(0);
    let kv_blocks = kv.config().blocks;
    drop(kv);
    drop(bytes);

    let host_deviation_percent = deviation(placement.predicted_host_bytes, measured_host_bytes);
    let device_deviation_percent = deviation(placement.predicted_device_bytes, 0);
    let measured_host_available = baseline.is_some();
    let verdict = verdict(
        measured_host_available,
        host_deviation_percent,
        device_deviation_percent,
    );

    let record = Verification {
        model_identity: placement.model_identity.clone(),
        hardware_fingerprint: placement.hardware_fingerprint.clone(),
        placement: "cpu",
        layers_on_device: placement.layers_on_device,
        kv_cache_type: placement.kv_cache_type.clone(),
        context_length: placement.context_length,
        predicted_device_bytes: placement.predicted_device_bytes,
        measured_device_bytes: 0,
        device_deviation_percent,
        predicted_host_bytes: placement.predicted_host_bytes,
        measured_host_bytes,
        host_deviation_percent,
        weights_bytes: shape.weights_bytes,
        kv_cache_bytes,
        kv_blocks,
        verdict,
    };

    emit(
        &(if json {
            render_verification_json(&record)
        } else {
            render_verification_text(&record)
        }),
    )?;
    Ok(0)
}

fn deviation(predicted: u64, measured: u64) -> Option<f64> {
    if predicted == 0 {
        return None;
    }
    Some((measured as f64 - predicted as f64) / predicted as f64 * 100.0)
}

fn verdict(measured: bool, host: Option<f64>, device: Option<f64>) -> &'static str {
    if !measured {
        return "not_measured";
    }
    let worst = [host, device]
        .into_iter()
        .flatten()
        .map(f64::abs)
        .fold(0.0f64, f64::max);
    if worst <= MEMORY_TARGET_PERCENT {
        "within_target"
    } else {
        "outside_target"
    }
}

fn render_verification_text(record: &Verification) -> String {
    let mut out = String::new();
    out.push_str(&format!("model_identity: {}\n", record.model_identity));
    out.push_str(&format!(
        "hardware_fingerprint: {}\n",
        record.hardware_fingerprint
    ));
    out.push_str(&format!("placement: {}\n", record.placement));
    out.push_str(&format!("layers_on_device: {}\n", record.layers_on_device));
    out.push_str(&format!("kv_cache_type: {}\n", record.kv_cache_type));
    out.push_str(&format!("context_length: {}\n", record.context_length));
    out.push_str(&format!("weights_bytes: {}\n", record.weights_bytes));
    out.push_str(&format!("kv_cache_bytes: {}\n", record.kv_cache_bytes));
    out.push_str(&format!("kv_blocks: {}\n", record.kv_blocks));
    out.push_str(&format!(
        "predicted_device_bytes: {}\n",
        record.predicted_device_bytes
    ));
    out.push_str(&format!(
        "measured_device_bytes: {}\n",
        record.measured_device_bytes
    ));
    out.push_str(&format!(
        "predicted_host_bytes: {}\n",
        record.predicted_host_bytes
    ));
    out.push_str(&format!(
        "measured_host_bytes: {}\n",
        record.measured_host_bytes
    ));
    out.push_str(&format!(
        "host_deviation_percent: {}\n",
        signed(record.host_deviation_percent)
    ));
    out.push_str(&format!(
        "device_deviation_percent: {}\n",
        signed(record.device_deviation_percent)
    ));
    out.push_str(&format!("verdict: {}\n", record.verdict));
    out
}

fn signed(value: Option<f64>) -> String {
    match value {
        Some(value) => format!("{value:+.2}"),
        None => "not_measured".to_owned(),
    }
}

fn render_verification_json(record: &Verification) -> String {
    let percent = |value: Option<f64>| {
        value
            .map(|value| format!("{value:.2}"))
            .unwrap_or_else(|| "null".to_owned())
    };
    format!(
        "{{\n  \"model_identity\": {},\n  \"hardware_fingerprint\": {},\n  \"placement\": {},\n  \"layers_on_device\": {},\n  \"kv_cache_type\": {},\n  \"context_length\": {},\n  \"weights_bytes\": {},\n  \"kv_cache_bytes\": {},\n  \"kv_blocks\": {},\n  \"predicted_device_bytes\": {},\n  \"measured_device_bytes\": {},\n  \"device_deviation_percent\": {},\n  \"predicted_host_bytes\": {},\n  \"measured_host_bytes\": {},\n  \"host_deviation_percent\": {},\n  \"verdict\": {}\n}}\n",
        quote(&record.model_identity),
        quote(&record.hardware_fingerprint),
        quote(record.placement),
        record.layers_on_device,
        quote(&record.kv_cache_type),
        record.context_length,
        record.weights_bytes,
        record.kv_cache_bytes,
        record.kv_blocks,
        record.predicted_device_bytes,
        record.measured_device_bytes,
        percent(record.device_deviation_percent),
        record.predicted_host_bytes,
        record.measured_host_bytes,
        percent(record.host_deviation_percent),
        quote(record.verdict)
    )
}

/// One generated turn, as the CLI reports it.
struct Generated {
    prompt_tokens: usize,
    tokens: Vec<u32>,
    reused_tokens: usize,
    processed_tokens: u64,
    generated_tokens: usize,
    finish: &'static str,
    text: String,
    tokens_per_second: f64,
    micros: Vec<(&'static str, u64)>,
    log_line: String,
}

fn generate_command(args: &[String]) -> Result<u8, ViperaError> {
    let mut json = false;
    let mut chat = false;
    let mut prompt: Option<String> = None;
    let mut system: Option<String> = None;
    let mut path: Option<&str> = None;
    let mut max_tokens: u32 = 32;
    let mut ctx: Option<u64> = None;
    let mut kv_type = KvCacheType::F16;
    let mut sampler = SamplerConfig {
        temperature: 0.0,
        ..SamplerConfig::default()
    };

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--json" => json = true,
            "--chat" => chat = true,
            "--prompt" => {
                prompt = Some(next_value(args, index, "--prompt")?.to_owned());
                index += 1;
            }
            "--system" => {
                system = Some(next_value(args, index, "--system")?.to_owned());
                index += 1;
            }
            "--max-tokens" => {
                max_tokens =
                    parse_number("--max-tokens", next_value(args, index, "--max-tokens")?)?;
                index += 1;
            }
            "--ctx" => {
                ctx = Some(parse_number("--ctx", next_value(args, index, "--ctx")?)?);
                index += 1;
            }
            "--kv-type" => {
                kv_type = KvCacheType::parse(next_value(args, index, "--kv-type")?)?;
                index += 1;
            }
            "--temperature" => {
                sampler.temperature =
                    parse_number("--temperature", next_value(args, index, "--temperature")?)?;
                index += 1;
            }
            "--top-k" => {
                sampler.top_k = parse_number("--top-k", next_value(args, index, "--top-k")?)?;
                index += 1;
            }
            "--top-p" => {
                sampler.top_p = parse_number("--top-p", next_value(args, index, "--top-p")?)?;
                index += 1;
            }
            "--min-p" => {
                sampler.min_p = parse_number("--min-p", next_value(args, index, "--min-p")?)?;
                index += 1;
            }
            "--repetition-penalty" => {
                sampler.repetition_penalty = parse_number(
                    "--repetition-penalty",
                    next_value(args, index, "--repetition-penalty")?,
                )?;
                index += 1;
            }
            "--seed" => {
                sampler.seed = parse_number("--seed", next_value(args, index, "--seed")?)?;
                index += 1;
            }
            flag if flag.starts_with('-') => {
                return Err(ViperaError::usage(format!(
                    "unknown flag \"{flag}\" for generate"
                )));
            }
            other => {
                if path.is_some() {
                    return Err(ViperaError::usage(format!(
                        "unexpected extra argument \"{other}\""
                    )));
                }
                path = Some(other);
            }
        }
        index += 1;
    }

    let path = path.ok_or_else(|| ViperaError::usage("generate needs a model path"))?;
    let prompt = prompt.ok_or_else(|| ViperaError::usage("generate needs --prompt"))?;
    if max_tokens == 0 {
        return Err(ViperaError::invalid_argument(
            "--max-tokens must be at least 1",
        ));
    }

    // One read serves both the metadata and the weights: the backend keeps the bytes it will read
    // rows from, so nothing is copied twice.
    let bytes = std::fs::read(resolve_model_path(path)?)?;
    let file = GgufFile::parse(&bytes)?;
    let tokenizer = Tokenizer::from_gguf(&file)?;
    let backend = CpuBackend::from_bytes(bytes, &file)?;
    let model = backend.config();

    let bytes_per_token = model.kv_bytes_per_token_for(kv_type).ok_or_else(|| {
        ViperaError::unsupported(format!(
            "this model cannot store its KV cache as {}",
            kv_type.as_str()
        ))
    })?;
    let context_limit = file.describe().context_limit.unwrap_or(2048);
    if let Some(requested) = ctx
        && requested > context_limit
    {
        return Err(ViperaError::unsupported(format!(
            "requested context {requested} exceeds the model's maximum of {context_limit}"
        )));
    }
    let context_length = ctx.unwrap_or(context_limit.min(2048));
    let budget = context_length
        .checked_mul(bytes_per_token)
        .ok_or_else(|| ViperaError::invalid_argument("--ctx is too large for this model"))?;
    let kv = KvCache::new(KvConfig::from_bytes(
        budget,
        bytes_per_token,
        DEFAULT_TOKENS_PER_BLOCK,
    )?)?;

    let mut eos = Vec::new();
    for marker in ["<|im_end|>", "<|endoftext|>", "</s>"] {
        if let Some(id) = tokenizer.token_id(marker)
            && !eos.contains(&id)
        {
            eos.push(id);
        }
    }
    let mut engine = Engine::new(
        kv,
        backend,
        EngineConfig {
            eos,
            sampler,
            recent_window: 128,
            // One request at a time keeps the CLI's tokens exactly what they were before batching.
            max_batch: 1,
        },
    );

    let ids = if chat {
        let mut messages = Vec::new();
        if let Some(system) = system {
            messages.push(ChatMessage::system(system));
        }
        messages.push(ChatMessage::user(prompt));
        tokenizer.encode_chat(&messages, true)?
    } else {
        tokenizer.encode(&prompt)?
    };

    let started = Instant::now();
    let sequence = engine.admit(&ids, max_tokens)?;
    // Text mode prints as tokens arrive; JSON needs the whole turn before it can emit a document.
    let mut seen = Vec::new();
    let mut shown = String::new();
    let mut stdout = std::io::stdout().lock();
    let generation = if json {
        engine.generate(sequence)?
    } else {
        engine.generate_with(sequence, &mut |token| {
            seen.push(token);
            // A token that is not yet a valid string is held back until it is.
            if let Ok(text) = tokenizer.decode(&seen) {
                flush_text(&text, &mut shown, &mut stdout);
            }
        })?
    };
    let elapsed = started.elapsed();
    let text = tokenizer.decode(&generation.tokens)?;
    if !json {
        flush_text(&text, &mut shown, &mut stdout);
    }
    drop(stdout);
    let seconds = elapsed.as_secs_f64();
    let report = Generated {
        prompt_tokens: ids.len(),
        tokens: generation.tokens.clone(),
        reused_tokens: generation.reused_tokens,
        processed_tokens: generation.processed_tokens,
        generated_tokens: generation.tokens.len(),
        finish: generation.finish.as_str(),
        tokens_per_second: if seconds > 0.0 {
            generation.tokens.len() as f64 / seconds
        } else {
            0.0
        },
        micros: Stage::ALL
            .iter()
            .map(|stage| (stage.as_str(), generation.timings.micros(*stage)))
            .collect(),
        log_line: generation.log_line.clone(),
        text,
    };

    if json {
        emit(&render_generate_json(&report))?;
    } else if !report.text.ends_with('\n') {
        emit("\n")?;
    }
    Ok(0)
}

/// Print whatever part of `text` has not been printed, assuming the printed text is its prefix.
///
/// Decoding a growing prefix can only extend it, but a partial multi-byte character can decode as a
/// replacement first; when the prefix no longer matches, nothing is printed and the final flush sends
/// the settled text instead of garbling what is already on screen.
fn flush_text(text: &str, shown: &mut String, out: &mut dyn Write) {
    if text.len() < shown.len() || !text.starts_with(shown.as_str()) {
        return;
    }
    let _ = out.write_all(&text.as_bytes()[shown.len()..]);
    // Flushed per token: a pipe is block-buffered, and a caller watching the output should see it
    // arrive rather than wait for a newline that a completion may never contain.
    let _ = out.flush();
    shown.clear();
    shown.push_str(text);
}

fn render_generate_json(report: &Generated) -> String {
    let stages: Vec<String> = report
        .micros
        .iter()
        .map(|(stage, micros)| format!("\"{stage}\": {micros}"))
        .collect();
    let tokens: Vec<String> = report.tokens.iter().map(u32::to_string).collect();
    format!(
        "{{\n  \"prompt_tokens\": {},\n  \"reused_tokens\": {},\n  \"processed_tokens\": {},\n  \"generated_tokens\": {},\n  \"finish\": {},\n  \"tokens\": [{}],\n  \"text\": {},\n  \"tokens_per_second\": {:.2},\n  \"micros\": {{{}}},\n  \"log\": {}\n}}\n",
        report.prompt_tokens,
        report.reused_tokens,
        report.processed_tokens,
        report.generated_tokens,
        quote(report.finish),
        tokens.join(", "),
        quote(&report.text),
        report.tokens_per_second,
        stages.join(", "),
        quote(&report.log_line)
    )
}

fn next_value<'a>(args: &'a [String], index: usize, flag: &str) -> Result<&'a str, ViperaError> {
    args.get(index + 1)
        .map(String::as_str)
        .ok_or_else(|| ViperaError::usage(format!("{flag} needs a value")))
}

fn parse_number<T: std::str::FromStr>(flag: &str, value: &str) -> Result<T, ViperaError> {
    value
        .parse()
        .map_err(|_| ViperaError::usage(format!("{flag} \"{value}\" is not a number")))
}

fn read_stdin() -> Result<String, ViperaError> {
    let mut buffer = String::new();
    std::io::stdin().read_to_string(&mut buffer)?;
    Ok(buffer)
}

fn tokenize_command(args: &[String]) -> Result<u8, ViperaError> {
    let mut json = false;
    let mut text: Option<&str> = None;
    let mut text_file: Option<&str> = None;
    let mut messages: Vec<ChatMessage> = Vec::new();
    let mut generation_prompt = true;
    let mut ids_text: Option<&str> = None;
    let mut path: Option<&str> = None;

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--json" => json = true,
            "--ids" => {
                ids_text = Some(next_value(args, index, "--ids")?);
                index += 1;
            }
            "--text" => {
                text = Some(next_value(args, index, "--text")?);
                index += 1;
            }
            "--text-file" => {
                text_file = Some(next_value(args, index, "--text-file")?);
                index += 1;
            }
            "--system" => {
                messages.push(ChatMessage::system(next_value(args, index, "--system")?));
                index += 1;
            }
            "--user" => {
                messages.push(ChatMessage::user(next_value(args, index, "--user")?));
                index += 1;
            }
            "--assistant" => {
                messages.push(ChatMessage::assistant(next_value(
                    args,
                    index,
                    "--assistant",
                )?));
                index += 1;
            }
            "--no-generation-prompt" => generation_prompt = false,
            flag if flag.starts_with('-') => {
                return Err(ViperaError::usage(format!(
                    "unknown flag \"{flag}\" for tokenize"
                )));
            }
            other => {
                if path.is_some() {
                    return Err(ViperaError::usage(format!(
                        "unexpected extra argument \"{other}\""
                    )));
                }
                path = Some(other);
            }
        }
        index += 1;
    }

    let path = path.ok_or_else(|| ViperaError::usage("tokenize needs a model path"))?;
    if !messages.is_empty() && (text.is_some() || text_file.is_some()) {
        return Err(ViperaError::usage(
            "--text cannot be combined with --system/--user/--assistant",
        ));
    }
    if ids_text.is_some() && (text.is_some() || text_file.is_some() || !messages.is_empty()) {
        return Err(ViperaError::usage(
            "--ids decodes, so it cannot be combined with text or chat input",
        ));
    }

    let bytes = std::fs::read(resolve_model_path(path)?)?;
    let file = GgufFile::parse(&bytes)?;
    let tokenizer = Tokenizer::from_gguf(&file)?;

    if let Some(raw) = ids_text {
        let ids = parse_ids(raw)?;
        let decoded = tokenizer.decode(&ids)?;
        if json {
            emit(&render_tokenize_json(&ids, &[], None, Some(&decoded)))?;
        } else {
            emit(&format!("tokens: {}\ntext: {decoded}\n", ids.len()))?;
        }
        return Ok(0);
    }

    let (prompt, ids) = if messages.is_empty() {
        let source = if let Some(value) = text {
            value.to_owned()
        } else if let Some(value) = text_file {
            std::fs::read_to_string(value)?
        } else {
            read_stdin()?
        };
        (None, tokenizer.encode(&source)?)
    } else {
        let template = tokenizer.chat_template().ok_or_else(|| {
            ViperaError::unsupported("model does not declare tokenizer.chat_template")
        })?;
        let rendered = chat::render(template, &messages, generation_prompt)?;
        let ids = tokenizer.encode(&rendered)?;
        (Some(rendered), ids)
    };

    let pieces: Vec<String> = ids
        .iter()
        .map(|id| tokenizer.vocab().piece(*id).unwrap_or("").to_owned())
        .collect();

    if json {
        emit(&render_tokenize_json(
            &ids,
            &pieces,
            prompt.as_deref(),
            None,
        ))?;
    } else {
        let mut out = format!("tokens: {}\n", ids.len());
        out.push_str("ids:");
        for id in &ids {
            out.push_str(&format!(" {id}"));
        }
        out.push('\n');
        if let Some(prompt) = &prompt {
            out.push_str(&format!("prompt:\n{prompt}"));
        }
        emit(&out)?;
    }
    Ok(0)
}

fn render_tokenize_json(
    ids: &[u32],
    pieces: &[String],
    prompt: Option<&str>,
    text: Option<&str>,
) -> String {
    let ids: Vec<String> = ids.iter().map(u32::to_string).collect();
    let pieces: Vec<String> = pieces.iter().map(|piece| quote(piece)).collect();
    let prompt = prompt
        .map(|value| format!(",\n  \"prompt\": {}", quote(value)))
        .unwrap_or_default();
    let text = text
        .map(|value| format!(",\n  \"text\": {}", quote(value)))
        .unwrap_or_default();
    let pieces = if pieces.is_empty() {
        String::new()
    } else {
        format!(",\n  \"pieces\": [{}]", pieces.join(", "))
    };
    format!(
        "{{\n  \"count\": {},\n  \"ids\": [{}]{pieces}{prompt}{text}\n}}\n",
        ids.len(),
        ids.join(", ")
    )
}

fn parse_logits(raw: &str) -> Result<Vec<f32>, ViperaError> {
    let mut logits = Vec::new();
    for part in raw.split([',', ' ', '\n', '\t', '\r']) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let value: f32 = part.parse().map_err(|_| {
            ViperaError::invalid_argument(format!("logit \"{part}\" is not a number"))
        })?;
        logits.push(value);
    }
    if logits.is_empty() {
        return Err(ViperaError::invalid_argument(
            "no logits were provided on --logits, --logits-file, or stdin",
        ));
    }
    Ok(logits)
}

fn parse_ids(raw: &str) -> Result<Vec<u32>, ViperaError> {
    let mut ids = Vec::new();
    for part in raw.split([',', ' ']) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        ids.push(part.parse().map_err(|_| {
            ViperaError::invalid_argument(format!("token id \"{part}\" is not a number"))
        })?);
    }
    Ok(ids)
}

fn sample_command(args: &[String]) -> Result<u8, ViperaError> {
    let mut json = false;
    let mut config = SamplerConfig::default();
    let mut count = 1u32;
    let mut logits_text: Option<&str> = None;
    let mut logits_file: Option<&str> = None;
    let mut model: Option<&str> = None;
    let mut recent: Vec<u32> = Vec::new();

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--json" => json = true,
            "--logits" => {
                logits_text = Some(next_value(args, index, "--logits")?);
                index += 1;
            }
            "--logits-file" => {
                logits_file = Some(next_value(args, index, "--logits-file")?);
                index += 1;
            }
            "--model" => {
                model = Some(next_value(args, index, "--model")?);
                index += 1;
            }
            "--count" => {
                count = parse_number("--count", next_value(args, index, "--count")?)?;
                index += 1;
            }
            "--seed" => {
                config.seed = parse_number("--seed", next_value(args, index, "--seed")?)?;
                index += 1;
            }
            "--temperature" => {
                config.temperature =
                    parse_number("--temperature", next_value(args, index, "--temperature")?)?;
                index += 1;
            }
            "--top-k" => {
                config.top_k = parse_number("--top-k", next_value(args, index, "--top-k")?)?;
                index += 1;
            }
            "--top-p" => {
                config.top_p = parse_number("--top-p", next_value(args, index, "--top-p")?)?;
                index += 1;
            }
            "--min-p" => {
                config.min_p = parse_number("--min-p", next_value(args, index, "--min-p")?)?;
                index += 1;
            }
            "--repetition-penalty" => {
                config.repetition_penalty = parse_number(
                    "--repetition-penalty",
                    next_value(args, index, "--repetition-penalty")?,
                )?;
                index += 1;
            }
            "--recent" => {
                recent = parse_ids(next_value(args, index, "--recent")?)?;
                index += 1;
            }
            flag if flag.starts_with('-') => {
                return Err(ViperaError::usage(format!(
                    "unknown flag \"{flag}\" for sample"
                )));
            }
            other => {
                return Err(ViperaError::usage(format!(
                    "unexpected argument \"{other}\" for sample; logits come from --logits, --logits-file, or stdin"
                )));
            }
        }
        index += 1;
    }

    if count == 0 {
        return Err(ViperaError::invalid_argument("--count must be at least 1"));
    }
    let raw = if let Some(value) = logits_text {
        value.to_owned()
    } else if let Some(value) = logits_file {
        std::fs::read_to_string(value)?
    } else {
        read_stdin()?
    };
    let logits = parse_logits(&raw)?;

    let mut sampler = Sampler::new(config)?;
    let mut tokens = Vec::new();
    for _ in 0..count {
        let id = sampler.sample(&logits, &recent)?;
        tokens.push(id);
        recent.push(id);
    }

    let text = match model {
        Some(path) => {
            let bytes = std::fs::read(resolve_model_path(path)?)?;
            let file = GgufFile::parse(&bytes)?;
            Some(Tokenizer::from_gguf(&file)?.decode(&tokens)?)
        }
        None => None,
    };

    if json {
        emit(&render_sample_json(&tokens, config.seed, text.as_deref()))?;
    } else {
        let ids: Vec<String> = tokens.iter().map(u32::to_string).collect();
        let mut out = format!("ids: {}\n", ids.join(" "));
        out.push_str(&format!("seed: {}\n", config.seed));
        if let Some(text) = &text {
            out.push_str(&format!("text: {text}\n"));
        }
        emit(&out)?;
    }
    Ok(0)
}

fn render_sample_json(tokens: &[u32], seed: u64, text: Option<&str>) -> String {
    let tokens: Vec<String> = tokens.iter().map(u32::to_string).collect();
    let text = text
        .map(|value| format!(",\n  \"text\": {}", quote(value)))
        .unwrap_or_default();
    format!(
        "{{\n  \"count\": {},\n  \"tokens\": [{}],\n  \"seed\": {seed}{text}\n}}\n",
        tokens.len(),
        tokens.join(", ")
    )
}

fn render_profile_text(hardware: &HardwareProfile) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "hardware_fingerprint: {}\n",
        hardware.fingerprint()
    ));
    out.push_str(&format!(
        "cpu_cores_physical: {}\n",
        hardware.cpu_cores_physical
    ));
    out.push_str(&format!("ram_total_bytes: {}\n", hardware.ram_total_bytes));
    for device in &hardware.devices {
        out.push_str(&format!(
            "device: {} {} ({} bytes free of {})\n",
            device.kind, device.name, device.memory_free_bytes, device.memory_total_bytes
        ));
        if let Some(driver) = &device.driver_version {
            out.push_str(&format!("device_driver: {driver}\n"));
        }
    }
    out
}

fn render_profile_json(hardware: &HardwareProfile) -> String {
    let devices: Vec<String> = hardware
        .devices
        .iter()
        .map(|device| {
            let driver = match &device.driver_version {
                Some(version) => quote(version),
                None => "null".to_owned(),
            };
            format!(
                "    {{\n      \"kind\": {},\n      \"name\": {},\n      \"memory_total_bytes\": {},\n      \"memory_free_bytes\": {},\n      \"driver_version\": {driver}\n    }}",
                quote(device.kind.as_str()),
                quote(&device.name),
                device.memory_total_bytes,
                device.memory_free_bytes
            )
        })
        .collect();

    format!(
        "{{\n  \"hardware_fingerprint\": {},\n  \"cpu_cores_physical\": {},\n  \"ram_total_bytes\": {},\n  \"devices\": [{}]\n}}\n",
        quote(&hardware.fingerprint()),
        hardware.cpu_cores_physical,
        hardware.ram_total_bytes,
        if devices.is_empty() {
            String::new()
        } else {
            format!("\n{}\n  ", devices.join(",\n"))
        }
    )
}

fn render_text(model: &ModelDescriptor) -> String {
    let mut out = String::new();
    out.push_str(&format!("name: {}\n", model.name));
    out.push_str(&format!("identity: {}\n", model.identity));
    out.push_str(&format!("gguf_version: {}\n", model.gguf_version));
    out.push_str(&format!("tensor_count: {}\n", model.tensor_count));
    out.push_str(&format!("parameter_count: {}\n", model.parameter_count));
    out.push_str(&format!("quantization: {}\n", model.quantization));
    if let Some(value) = model.context_limit {
        out.push_str(&format!("context_limit: {value}\n"));
    }
    if let Some(value) = model.layer_count {
        out.push_str(&format!("layer_count: {value}\n"));
    }
    if let Some(value) = model.vocab_size {
        out.push_str(&format!("vocab_size: {value}\n"));
    }
    out.push_str("tensor_types:\n");
    for (name, count) in &model.tensor_types {
        out.push_str(&format!("  {name}: {count}\n"));
    }
    out
}

fn render_json(model: &ModelDescriptor) -> String {
    let mut out = String::from("{\n");
    out.push_str(&format!("  \"name\": {},\n", quote(&model.name)));
    out.push_str(&format!("  \"identity\": {},\n", quote(&model.identity)));
    out.push_str(&format!("  \"gguf_version\": {},\n", model.gguf_version));
    out.push_str(&format!("  \"tensor_count\": {},\n", model.tensor_count));
    out.push_str(&format!(
        "  \"parameter_count\": {},\n",
        model.parameter_count
    ));
    out.push_str(&format!(
        "  \"quantization\": {},\n",
        quote(&model.quantization)
    ));
    if let Some(value) = model.context_limit {
        out.push_str(&format!("  \"context_limit\": {value},\n"));
    }
    if let Some(value) = model.layer_count {
        out.push_str(&format!("  \"layer_count\": {value},\n"));
    }
    if let Some(value) = model.vocab_size {
        out.push_str(&format!("  \"vocab_size\": {value},\n"));
    }
    let types: Vec<String> = model
        .tensor_types
        .iter()
        .map(|(name, count)| format!("{}: {}", quote(name), count))
        .collect();
    out.push_str(&format!("  \"tensor_types\": {{{}}}\n", types.join(", ")));
    out.push_str("}\n");
    out
}

fn render_plan_text(
    placement: &PlacementPlan,
    hardware: &vipera_traits::HardwareProfile,
) -> String {
    let mut out = String::new();
    out.push_str(&format!("model_identity: {}\n", placement.model_identity));
    out.push_str(&format!(
        "hardware_fingerprint: {}\n",
        placement.hardware_fingerprint
    ));
    out.push_str(&format!(
        "cpu_cores_physical: {}\n",
        hardware.cpu_cores_physical
    ));
    out.push_str(&format!("ram_total_bytes: {}\n", hardware.ram_total_bytes));
    for device in &hardware.devices {
        out.push_str(&format!(
            "device: {} {} ({} bytes free of {})\n",
            device.kind, device.name, device.memory_free_bytes, device.memory_total_bytes
        ));
    }
    out.push_str(&format!(
        "layers_on_device: {}\n",
        placement.layers_on_device
    ));
    out.push_str(&format!("kv_cache_type: {}\n", placement.kv_cache_type));
    out.push_str(&format!("context_length: {}\n", placement.context_length));
    out.push_str(&format!("batch_size: {}\n", placement.batch_size));
    out.push_str(&format!(
        "predicted_device_bytes: {}\n",
        placement.predicted_device_bytes
    ));
    out.push_str(&format!(
        "device_reserve_bytes: {}\n",
        placement.device_reserve_bytes
    ));
    out.push_str(&format!(
        "predicted_host_bytes: {}\n",
        placement.predicted_host_bytes
    ));
    out.push_str(&format!("feasible: {}\n", placement.feasible));
    for violation in &placement.violations {
        out.push_str(&format!(
            "violation: {} ({}) — {}\n",
            violation.constraint, violation.message, violation.required_bytes
        ));
    }
    out
}

fn render_plan_json(placement: &PlacementPlan) -> String {
    let violations: Vec<String> = placement
        .violations
        .iter()
        .map(|violation| {
            format!(
                "{{\n      \"constraint\": {},\n      \"required_bytes\": {},\n      \"available_bytes\": {},\n      \"message\": {}\n    }}",
                quote(violation.constraint),
                violation.required_bytes,
                violation.available_bytes,
                quote(&violation.message)
            )
        })
        .collect();

    format!(
        "{{\n  \"model_identity\": {},\n  \"hardware_fingerprint\": {},\n  \"layers_on_device\": {},\n  \"kv_cache_type\": {},\n  \"context_length\": {},\n  \"batch_size\": {},\n  \"predicted_device_bytes\": {},\n  \"device_reserve_bytes\": {},\n  \"predicted_host_bytes\": {},\n  \"feasible\": {},\n  \"violations\": [{}]\n}}\n",
        quote(&placement.model_identity),
        quote(&placement.hardware_fingerprint),
        placement.layers_on_device,
        quote(&placement.kv_cache_type),
        placement.context_length,
        placement.batch_size,
        placement.predicted_device_bytes,
        placement.device_reserve_bytes,
        placement.predicted_host_bytes,
        placement.feasible,
        if violations.is_empty() {
            String::new()
        } else {
            format!("\n    {}\n  ", violations.join(",\n    "))
        }
    )
}

fn quote(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            c if (c as u32) < 0x20 => escaped.push_str(&format!("\\u{:04x}", c as u32)),
            c => escaped.push(c),
        }
    }
    escaped.push('"');
    escaped
}

#[cfg(test)]
mod tests {
    use super::{
        Verification, deviation, parse_ids, parse_logits, quote, render_profile_json,
        render_profile_text, render_sample_json, render_tokenize_json, render_verification_json,
        render_verification_text, verdict,
    };
    use vipera_traits::{Device, DeviceKind, HardwareProfile};

    fn machine() -> HardwareProfile {
        HardwareProfile {
            cpu_cores_physical: 8,
            ram_total_bytes: 16 * 1024 * 1024 * 1024,
            devices: vec![Device {
                kind: DeviceKind::Cuda,
                name: "NVIDIA GeForce RTX 3060".to_owned(),
                memory_total_bytes: 12 * 1024 * 1024 * 1024,
                memory_free_bytes: 11 * 1024 * 1024 * 1024,
                driver_version: Some("560.35".to_owned()),
            }],
        }
    }

    #[test]
    fn profile_json_exposes_the_frozen_keys() {
        let json = render_profile_json(&machine());
        for key in [
            "hardware_fingerprint",
            "cpu_cores_physical",
            "ram_total_bytes",
            "devices",
            "kind",
            "name",
            "memory_total_bytes",
            "memory_free_bytes",
            "driver_version",
        ] {
            assert!(json.contains(key), "missing key {key} in {json}");
        }
        assert!(json.contains("hw_"));
        assert!(json.contains("\"cuda\""));
    }

    #[test]
    fn profile_json_omits_no_device_and_reports_a_null_driver() {
        let mut hardware = machine();
        hardware.devices[0].driver_version = None;
        assert!(render_profile_json(&hardware).contains("\"driver_version\": null"));

        hardware.devices.clear();
        assert!(render_profile_json(&hardware).contains("\"devices\": []"));
    }

    #[test]
    fn profile_text_lists_the_fingerprint_and_the_device() {
        let text = render_profile_text(&machine());
        assert!(text.starts_with("hardware_fingerprint: hw_"));
        assert!(text.contains("device: cuda NVIDIA GeForce RTX 3060"));
        assert!(text.contains("device_driver: 560.35"));
    }

    #[test]
    fn directory_with_one_gguf_resolves_to_that_file() {
        let dir = std::env::temp_dir().join("vipera-dir-test-one");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.gguf"), b"fake").unwrap();
        let resolved = super::resolve_model_path(dir.to_str().unwrap()).unwrap();
        assert_eq!(resolved, dir.join("model.gguf").to_str().unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn directory_without_gguf_is_an_error() {
        let dir = std::env::temp_dir().join("vipera-dir-test-empty");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(super::resolve_model_path(dir.to_str().unwrap()).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn directory_with_two_gguf_is_an_error() {
        let dir = std::env::temp_dir().join("vipera-dir-test-two");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.gguf"), b"a").unwrap();
        std::fs::write(dir.join("b.gguf"), b"b").unwrap();
        assert!(super::resolve_model_path(dir.to_str().unwrap()).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_file_path_passes_through_unchanged() {
        assert_eq!(
            super::resolve_model_path("some/model.gguf").unwrap(),
            "some/model.gguf"
        );
    }

    struct BrokenPipe;
    impl std::io::Write for BrokenPipe {
        fn write(&mut self, _buf: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "pipe closed",
            ))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn broken_pipe_is_not_an_error() {
        // Writing to a closed pipe must not surface as an internal error (or panic).
        let mut writer = BrokenPipe;
        assert!(super::write_out(&mut writer, "hello").is_ok());
    }

    fn verification() -> Verification {
        Verification {
            model_identity: "fnv1a64:1".to_owned(),
            hardware_fingerprint: "hw_1".to_owned(),
            placement: "cpu",
            layers_on_device: 0,
            kv_cache_type: "f16".to_owned(),
            context_length: 2048,
            predicted_device_bytes: 0,
            measured_device_bytes: 0,
            device_deviation_percent: None,
            predicted_host_bytes: 1_200_000_000,
            measured_host_bytes: 1_212_000_000,
            host_deviation_percent: Some(1.0),
            weights_bytes: 1_100_000_000,
            kv_cache_bytes: 56_000_000,
            kv_blocks: 120,
            verdict: "within_target",
        }
    }

    #[test]
    fn deviation_is_signed_and_undefined_without_a_prediction() {
        assert_eq!(deviation(0, 100), None);
        assert_eq!(deviation(1000, 1100), Some(10.0));
        assert_eq!(deviation(1000, 900), Some(-10.0));
    }

    #[test]
    fn the_verdict_uses_the_ten_percent_target() {
        assert_eq!(verdict(true, Some(9.99), None), "within_target");
        assert_eq!(verdict(true, Some(10.01), None), "outside_target");
        assert_eq!(verdict(true, None, Some(-25.0)), "outside_target");
        assert_eq!(verdict(false, None, None), "not_measured");
    }

    #[test]
    fn generate_json_carries_the_text_and_the_stage_breakdown() {
        let report = super::Generated {
            prompt_tokens: 5,
            tokens: vec![220, 16, 20066, 335],
            reused_tokens: 0,
            processed_tokens: 8,
            generated_tokens: 4,
            finish: "length",
            text: " 4\n".to_owned(),
            tokens_per_second: 0.38,
            micros: vec![("compute", 10_570_470), ("sample", 43_903)],
            log_line: "{\"event\":\"generation\"}".to_owned(),
        };
        let json = super::render_generate_json(&report);
        for key in [
            "prompt_tokens",
            "reused_tokens",
            "processed_tokens",
            "generated_tokens",
            "finish",
            "text",
            "tokens_per_second",
            "micros",
            "log",
        ] {
            assert!(json.contains(key), "missing {key} in {json}");
        }
        assert!(json.contains("\"text\": \" 4\\n\""));
        assert!(json.contains("\"tokens\": [220, 16, 20066, 335]"), "{json}");
        assert!(json.contains("\"tokens_per_second\": 0.38"));
        assert!(json.contains("\"compute\": 10570470"));
    }

    #[test]
    fn verification_json_carries_the_frozen_keys_and_null_percentages() {
        let json = render_verification_json(&verification());
        for key in [
            "model_identity",
            "hardware_fingerprint",
            "placement",
            "layers_on_device",
            "kv_cache_type",
            "context_length",
            "predicted_device_bytes",
            "measured_device_bytes",
            "device_deviation_percent",
            "predicted_host_bytes",
            "measured_host_bytes",
            "host_deviation_percent",
            "verdict",
        ] {
            assert!(json.contains(key), "missing {key} in {json}");
        }
        assert!(json.contains("\"device_deviation_percent\": null"));
        assert!(json.contains("\"host_deviation_percent\": 1.00"));
        assert!(json.contains("\"verdict\": \"within_target\""));
    }

    #[test]
    fn verification_text_is_key_value_lines() {
        let text = render_verification_text(&verification());
        assert!(text.starts_with("model_identity: fnv1a64:1"));
        assert!(text.contains("host_deviation_percent: +1.00"));
        assert!(text.contains("device_deviation_percent: not_measured"));
        assert!(text.contains("verdict: within_target"));
    }

    #[test]
    fn escapes_json_specials() {
        assert_eq!(quote("a\"b\\c\nd"), "\"a\\\"b\\\\c\\nd\"");
    }

    #[test]
    fn escapes_control_characters() {
        assert_eq!(quote("\u{1}"), "\"\\u0001\"");
    }

    #[test]
    fn logits_accept_commas_spaces_and_newlines() {
        assert_eq!(
            parse_logits("1, 2\n3\t-0.5").unwrap(),
            vec![1.0, 2.0, 3.0, -0.5]
        );
        assert!(parse_logits("1, nope").is_err());
        assert!(parse_logits("  ").is_err());
    }

    #[test]
    fn token_ids_are_parsed_and_junk_is_rejected() {
        assert_eq!(parse_ids("1, 2 3").unwrap(), vec![1, 2, 3]);
        assert!(parse_ids("1, x").is_err());
    }

    #[test]
    fn tokenize_json_has_count_ids_pieces_and_an_optional_prompt() {
        let json = render_tokenize_json(&[1, 2], &["a".to_owned(), "b".to_owned()], None, None);
        assert!(json.contains("\"count\": 2"));
        assert!(json.contains("\"ids\": [1, 2]"));
        assert!(json.contains("\"pieces\": [\"a\", \"b\"]"));
        assert!(!json.contains("prompt"));

        let with_prompt =
            render_tokenize_json(&[1], &["a".to_owned()], Some("<|im_start|>user\nhi"), None);
        assert!(with_prompt.contains("\"prompt\": \"<|im_start|>user\\nhi\""));
    }

    #[test]
    fn tokenize_json_in_decode_mode_carries_text_and_no_pieces() {
        let json = render_tokenize_json(&[1, 2], &[], None, Some("hi"));
        assert!(json.contains("\"text\": \"hi\""));
        assert!(!json.contains("pieces"));
    }

    #[test]
    fn sample_json_reports_the_seed_and_an_optional_text() {
        let json = render_sample_json(&[3, 4], 7, None);
        assert!(json.contains("\"count\": 2"));
        assert!(json.contains("\"tokens\": [3, 4]"));
        assert!(json.contains("\"seed\": 7"));
        assert!(!json.contains("text"));

        let with_text = render_sample_json(&[3], 7, Some("hi"));
        assert!(with_text.contains("\"text\": \"hi\""));
    }
}
