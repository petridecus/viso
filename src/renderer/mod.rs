//! Rendering subsystems for molecular visualization.
//!
//! Contains molecular renderers (tubes, ribbons, sidechains, ball-and-stick,
//! nucleic acids) and post-processing effects (SSAO, bloom, FXAA).

/// Bind groups shared across all molecular draw calls.
pub mod draw_context;
/// Molecular geometry assemblers (backbone, sidechain, ball-and-stick, etc.).
pub mod geometry;
/// Reusable impostor-pass primitives and instance types.
pub mod impostor;
/// Shared indexed-mesh draw-pass abstraction.
pub(crate) mod mesh;
/// GPU-based object picking and selection management.
pub mod picking;
pub(crate) mod pipeline_util;
/// Post-processing effects (SSAO, bloom, FXAA).
pub mod postprocess;

/// Bind group layouts shared by all molecular geometry pipelines.
///
/// Every molecular renderer (backbone, sidechain, ball-and-stick, band, pull,
/// nucleic acid) binds the same camera, lighting, and selection groups.
pub struct PipelineLayouts {
    pub camera: wgpu::BindGroupLayout,
    pub lighting: wgpu::BindGroupLayout,
    pub selection: wgpu::BindGroupLayout,
}
