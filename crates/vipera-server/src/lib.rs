//! The HTTP surface from `contracts/http-chat.md`, as a library.
//!
//! [`serve::command`] parses the flags `vipera serve` accepts and runs the server; [`json`] is the
//! small parser and writer behind the request and response bodies. Neither pulls a dependency: the
//! workspace keeps its list empty on purpose (`specs/001-vipera-rs-engine/research.md`, R3).
//!
//! The binary stays thin — it owns argument parsing for the other commands and calls
//! [`serve::command`] for this one.

pub mod json;
pub mod serve;

pub use serve::command;
