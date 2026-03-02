//! High-level GPU pipeline initialization.
//!
//! Wires together all GPU subsystems (shaders, camera, lighting, renderers,
//! picking, post-processing) from a scene and render coords.

use foldit_conv::render::RenderCoords;

use crate::camera::controller::CameraController;
use crate::error::VisoError;
use crate::gpu::lighting::Lighting;
use crate::gpu::residue_color::ResidueColorBuffer;
use crate::gpu::{RenderContext, ShaderComposer};
use crate::renderer::picking::{PickingSystem, SelectionBuffer};
use crate::renderer::postprocess::PostProcessStack;
use crate::renderer::{PipelineLayouts, Renderers};
use crate::scene::Scene;

/// Intermediate state holding all initialized GPU subsystems.
///
/// Produced by [`init_gpu_pipeline`] and consumed by
/// [`VisoEngine::assemble`](super::VisoEngine::assemble) to build
/// the final engine struct.
pub(crate) struct GpuBootstrap {
    /// WGSL shader composer with `#import` support.
    pub shader_composer: ShaderComposer,
    /// Orbital camera controller.
    pub camera_controller: CameraController,
    /// GPU lighting uniform and IBL.
    pub lighting: Lighting,
    /// All geometry renderers (backbone, sidechain, band, pull, BnS, NA).
    pub renderers: Renderers,
    /// GPU picking, selection, and per-residue color buffers.
    pub pick: PickingSystem,
    /// Post-processing stack (SSAO, bloom, fog, FXAA).
    pub post_process: PostProcessStack,
    /// The scene that was used to initialize the pipeline.
    pub scene: Scene,
}

/// Initialize all shared GPU subsystems from a scene and render coords.
///
/// This is the common pipeline setup for both empty and loaded constructors.
pub(super) fn init_gpu_pipeline(
    context: &RenderContext,
    scene: Scene,
    render_coords: &RenderCoords,
) -> Result<GpuBootstrap, VisoError> {
    let mut shader_composer = ShaderComposer::new()?;
    let mut camera_controller = CameraController::new(context);
    let lighting = Lighting::new(context);

    let n = render_coords.residue_count().max(1);
    let selection = SelectionBuffer::new(&context.device, n);
    let residue_colors = ResidueColorBuffer::new(&context.device, n);
    let layouts = PipelineLayouts {
        camera: camera_controller.layout.clone(),
        lighting: lighting.layout.clone(),
        selection: selection.layout.clone(),
        color: residue_colors.layout.clone(),
    };
    let renderers = Renderers::new(
        context,
        &layouts,
        render_coords,
        &scene,
        &mut shader_composer,
    )?;
    let pick = PickingSystem::new(
        context,
        &camera_controller.layout,
        selection,
        residue_colors,
        &mut shader_composer,
    )?;
    let post_process = PostProcessStack::new(context, &mut shader_composer)?;
    camera_controller.fit_to_positions(&[]);

    Ok(GpuBootstrap {
        shader_composer,
        camera_controller,
        lighting,
        renderers,
        pick,
        post_process,
        scene,
    })
}
