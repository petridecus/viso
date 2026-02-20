//! Post-processing effect passes.
//!
//! Provides screen-space ambient occlusion (SSAO), bloom, depth fog,
//! tone-mapping composite, and FXAA anti-aliasing.

pub mod bloom;
pub mod composite;
pub mod fxaa;
pub(crate) mod post_process;
pub mod ssao;
