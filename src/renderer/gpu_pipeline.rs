//! All GPU infrastructure grouped together.

use glam::Vec3;

use crate::gpu::lighting::Lighting;
use crate::gpu::{RenderContext, ShaderComposer};
use crate::renderer::picking::PickingSystem;
use crate::renderer::pipeline::SceneProcessor;
use crate::renderer::postprocess::PostProcessStack;
use crate::renderer::Renderers;

/// All GPU infrastructure grouped together: device/queue, renderers,
/// picking, background mesh processor, post-processing, lighting, and
/// per-frame cursor/culling state.
pub(crate) struct GpuPipeline {
    /// Core wgpu device, queue, and surface.
    pub context: RenderContext,
    /// All geometry renderers (backbone, sidechain, band, pull,
    /// ball-and-stick, nucleic acid).
    pub renderers: Renderers,
    /// GPU picking, selection, and per-residue color buffers.
    pub pick: PickingSystem,
    /// Background thread for off-main-thread mesh generation.
    pub scene_processor: SceneProcessor,
    /// Post-processing pass stack (SSAO, bloom, composite, FXAA).
    pub post_process: PostProcessStack,
    /// GPU lighting uniform and bind group.
    pub lighting: Lighting,
    /// Current cursor position in physical pixels (set by the viewer /
    /// input processor each frame for GPU picking).
    pub cursor_pos: (f32, f32),
    /// Camera eye position at the last frustum-culling update.
    pub last_cull_camera_eye: Vec3,
    /// Retained so compiled shader modules stay alive for the engine lifetime.
    #[allow(dead_code)]
    pub(crate) shader_composer: ShaderComposer,
}
