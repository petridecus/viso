//! Post-processing effect passes.
//!
//! Provides screen-space ambient occlusion (SSAO), bloom, depth fog,
//! tone-mapping composite, and FXAA anti-aliasing.

/// Bloom extraction and multi-level Gaussian blur.
pub(crate) mod bloom;
/// Final composite pass with SSAO, outlines, fog, and tone-mapping.
pub(crate) mod composite;
/// FXAA screen-space anti-aliasing pass.
pub(crate) mod fxaa;
pub(crate) mod post_process;
/// Uniform interface for fullscreen post-processing passes.
pub(crate) mod screen_pass;
/// Screen-space ambient occlusion (SSAO) renderer.
pub(crate) mod ssao;

pub(crate) use bloom::BloomPass;
pub(crate) use composite::{CompositeInputs, CompositePass};
pub(crate) use fxaa::FxaaPass;
pub(crate) use post_process::PostProcessStack;
pub(crate) use screen_pass::ScreenPass;
pub(crate) use ssao::SsaoRenderer;
