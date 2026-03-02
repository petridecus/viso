//! Background mesh generation pipeline.
//!
//! Converts scene data into GPU-ready byte buffers on a background
//! thread. The main thread only does GPU uploads and render passes.

mod mesh_concat;
mod mesh_gen;
pub(crate) mod prepared;
pub mod processor;

pub use prepared::{PreparedScene, SceneRequest};
pub use processor::SceneProcessor;
