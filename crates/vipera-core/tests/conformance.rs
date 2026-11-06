//! The reference parser must pass the shared GGUF contract, not just its own tests.

use vipera_conformance::{assert_conforms, gguf_cases};
use vipera_core::gguf::GgufFile;

#[test]
fn gguf_parser_meets_the_conformance_contract() {
    assert_conforms("gguf-v2-v3", &gguf_cases(), |bytes| {
        GgufFile::parse(bytes)
            .map(|_| ())
            .map_err(|err| err.to_string())
    });
}
