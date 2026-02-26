use foldit_conv::secondary_structure::SSType;
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

    /// Poll the GPU for a completed picking readback and resolve the raw ID
    /// to a typed [`PickTarget`].
    ///
    /// Non-blocking: returns immediately if no readback data is ready yet.
    /// When data is available, the internal `hovered_target` is updated.
    pub fn poll_and_resolve(&mut self, device: &wgpu::Device) {
        if let Some(raw_id) = self.picking.complete_readback(device) {
            self.hovered_target = self
                .pick_map
                .as_ref()
                .map_or(PickTarget::None, |pm| pm.resolve(raw_id));
        }
    }

    /// Clear the residue selection. Returns `true` if the selection was
    /// non-empty (i.e. it actually changed).
    pub fn clear_selection(&mut self) -> bool {
        if self.picking.selected_residues.is_empty() {
            false
        } else {
            self.picking.selected_residues.clear();
            true
        }
    }

    /// Select all residues in the same secondary-structure segment as
    /// `residue_idx`. If `extend` is true the new residues are added to the
    /// existing selection; otherwise the selection is replaced.
    ///
    /// `ss_types` is the per-residue secondary-structure classification
    /// (indexed by flat residue index).
    ///
    /// Returns `true` if the selection changed.
    pub fn select_segment(
        &mut self,
        residue_idx: i32,
        ss_types: &[SSType],
        extend: bool,
    ) -> bool {
        if residue_idx < 0 || (residue_idx as usize) >= ss_types.len() {
            return false;
        }

        let idx = residue_idx as usize;
        let target_ss = ss_types[idx];

        // Walk backwards to the start of this SS segment
        let mut start = idx;
        while start > 0 && ss_types[start - 1] == target_ss {
            start -= 1;
        }

        // Walk forwards to the end of this SS segment
        let mut end = idx;
        while end + 1 < ss_types.len() && ss_types[end + 1] == target_ss {
            end += 1;
        }

        if !extend {
            self.picking.selected_residues.clear();
        }

        for i in start..=end {
            let residue = i as i32;
            if !self.picking.selected_residues.contains(&residue) {
                self.picking.selected_residues.push(residue);
            }
        }

        true
    }

    /// Select all residues in the same chain as `residue_idx`. Chains are
    /// described by `backbone_chains` (each entry is a chain's backbone
    /// points, with 3 points per residue).
    ///
    /// If `extend` is true the new residues are added to the existing
    /// selection; otherwise the selection is replaced.
    ///
    /// Returns `true` if the selection changed.
    pub fn select_chain(
        &mut self,
        residue_idx: i32,
        backbone_chains: &[Vec<Vec3>],
        extend: bool,
    ) -> bool {
        if residue_idx < 0 {
            return false;
        }
        let target = residue_idx as usize;

        let mut global_start = 0usize;
        let chain_range = backbone_chains.iter().find_map(|chain| {
            let chain_residues = chain.len() / 3;
            let global_end = global_start + chain_residues;
            let result = (target >= global_start && target < global_end)
                .then_some(global_start..global_end);
            global_start = global_end;
            result
        });

        let Some(range) = chain_range else {
            return false;
        };

        if !extend {
            self.picking.selected_residues.clear();
        }
        for i in range {
            let residue = i as i32;
            if !self.picking.selected_residues.contains(&residue) {
                self.picking.selected_residues.push(residue);
            }
        }
        true
    }

    /// Build the picking geometry descriptor from current renderer state.
    pub fn build_geometry<'a>(
        &'a self,
        renderers: &'a Renderers,
        show_sidechains: bool,
    ) -> crate::renderer::picking::PickingGeometry<'a> {
        crate::renderer::picking::PickingGeometry {
            backbone_vertex_buffer: renderers.backbone.vertex_buffer(),
            backbone_tube_index_buffer: renderers.backbone.tube_index_buffer(),
            backbone_tube_index_count: renderers.backbone.tube_index_count(),
            backbone_ribbon_index_buffer: renderers
                .backbone
                .ribbon_index_buffer(),
            backbone_ribbon_index_count: renderers
                .backbone
                .ribbon_index_count(),
            capsule_bind_group: self.groups.capsule.as_ref(),
            capsule_count: if show_sidechains {
                renderers.sidechain.instance_count()
            } else {
                0
            },
            bns_capsule_bind_group: self.groups.bns_bond.as_ref(),
            bns_capsule_count: renderers.ball_and_stick.bond_count(),
            bns_sphere_bind_group: self.groups.bns_sphere.as_ref(),
            bns_sphere_count: renderers.ball_and_stick.sphere_count(),
        }
    }

    /// Read-only access to the currently selected residue indices.
    pub fn selected_residues(&self) -> &[i32] {
        &self.picking.selected_residues
    }

    /// Upload the current selection state to the GPU selection buffer.
    pub fn update_selection_buffer(&self, queue: &wgpu::Queue) {
        self.selection
            .update(queue, &self.picking.selected_residues);
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
        let initial = super::construction::initial_chain_colors(
            backbone_chains,
            total_residues,
        );
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
