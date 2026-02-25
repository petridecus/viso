use glam::Vec3;

use super::renderers::Renderers;
use crate::gpu::render_context::RenderContext;
use crate::gpu::residue_color::ResidueColorBuffer;
use crate::gpu::shader_composer::ShaderComposer;
use crate::renderer::picking::state::PickingState;
use crate::renderer::picking::{PickMap, PickTarget, Picking, SelectionBuffer};

/// GPU picking, selection, and per-residue color buffers grouped together.
pub(crate) struct PickingSystem {
    pub picking: Picking,
    pub groups: PickingState,
    pub selection: SelectionBuffer,
    pub residue_colors: ResidueColorBuffer,
    pub pick_map: Option<PickMap>,
    pub hovered_target: PickTarget,
}

impl PickingSystem {
    /// Create the picking system from pre-built selection and residue-color
    /// buffers.
    pub fn new(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        selection: SelectionBuffer,
        residue_colors: ResidueColorBuffer,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, crate::error::VisoError> {
        let picking = Picking::new(context, camera_layout, shader_composer)?;
        Ok(Self {
            picking,
            groups: PickingState::new(),
            selection,
            residue_colors,
            pick_map: None,
            hovered_target: PickTarget::None,
        })
    }

    /// Compute initial per-residue colors and build picking bind groups.
    pub fn init_colors_and_groups(
        &mut self,
        context: &RenderContext,
        backbone_chains: &[Vec<Vec3>],
        renderers: &Renderers,
    ) {
        let total_residues =
            backbone_chains.iter().map(|c| c.len() / 3).sum::<usize>();
        let initial =
            super::initial_chain_colors(backbone_chains, total_residues);
        self.residue_colors
            .set_colors_immediate(&context.queue, &initial);

        self.groups.rebuild_capsule(
            &self.picking,
            &context.device,
            &renderers.sidechain,
        );
        self.groups.rebuild_bns_bond(
            &self.picking,
            &context.device,
            &renderers.ball_and_stick,
        );
        self.groups.rebuild_bns_sphere(
            &self.picking,
            &context.device,
            &renderers.ball_and_stick,
        );
    }
}
