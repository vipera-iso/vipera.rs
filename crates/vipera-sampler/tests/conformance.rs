//! The shipped `greedy` strategy must pass the shared sampler contract, not just its own tests.

use vipera_conformance::{assert_sampler_conforms, sampler_cases};
use vipera_sampler::{Greedy, Rng};
use vipera_traits::{NoPenalty, SampleRequest, Sampler as Strategy};

#[test]
fn greedy_meets_the_sampler_conformance_contract() {
    assert_sampler_conforms("sampler-greedy", &sampler_cases(), |logits: &[f32]| {
        Greedy
            .draw(
                &SampleRequest {
                    logits,
                    penalty: &NoPenalty,
                    temperature: 0.0,
                    top_k: 0,
                    top_p: 1.0,
                    min_p: 0.0,
                },
                &mut Rng::from_seed(0),
            )
            .map_err(|err| format!("{err:?}"))
    });
}
