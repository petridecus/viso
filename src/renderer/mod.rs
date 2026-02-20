//! Rendering subsystems for molecular visualization.
//!
//! Contains molecular renderers (tubes, ribbons, sidechains, ball-and-stick,
//! nucleic acids) and post-processing effects (SSAO, bloom, FXAA).

pub mod molecular;
pub(crate) mod pipeline_util;
pub mod postprocess;
