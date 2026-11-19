//! The HTTP surface from `contracts/http-chat.md`, as a library.
//!
//! [`serve::command`] parses the flags `vipera serve` accepts and runs the server; [`json`] is the
//! small parser and writer behind the request and response bodies. The workspace kept its dependency
//! list empty on purpose (`specs/001-vipera-rs-engine/research.md`, R3); the one exception is
//! `rustls`, taken for `--tls-cert`/`--tls-key`, because a TLS stack is not something to hand-write
//! and the alternative was telling every operator to run a proxy.
//!
//! The binary stays thin — it owns argument parsing for the other commands and calls
//! [`serve::command`] for this one.

#![warn(missing_docs)]

#[cfg(feature = "acme")]
pub mod acme;
pub mod handler;
pub mod http;
pub mod json;
pub mod serve;
pub mod tls;

pub use serve::command;
