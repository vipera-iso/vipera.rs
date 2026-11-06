//! Core model handling: GGUF reading, hardware probing, and placement planning.
//!
//! Nothing in this crate loads weights or talks to a GPU, so every function here is testable on any
//! machine.

#![warn(missing_docs)]

pub mod gguf;
pub mod hardware;
pub mod planner;

pub use gguf::{GgufFile, Header, Metadata, ModelDescriptor, TensorInfo};
pub use hardware::SystemProbe;
pub use planner::{
    KvCacheType, ModelShape, PlacementPlan, PlanRequest, Violation, plan, plan_shape,
};
