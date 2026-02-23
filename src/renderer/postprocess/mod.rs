//! Post-processing effect passes.
//!
//! Provides screen-space ambient occlusion (SSAO), bloom, depth fog,
//! tone-mapping composite, and FXAA anti-aliasing.

/// Bloom extraction and multi-level Gaussian blur.
pub mod bloom;
/// Final composite pass with SSAO, outlines, fog, and tone-mapping.
pub mod composite;
/// FXAA screen-space anti-aliasing pass.
pub mod fxaa;
pub(crate) mod post_process;
/// Uniform interface for fullscreen post-processing passes.
pub(crate) mod screen_pass;
/// Screen-space ambient occlusion (SSAO) renderer.
pub mod ssao;
