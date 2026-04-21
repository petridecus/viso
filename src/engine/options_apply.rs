//! Options application: push runtime config to GPU subsystems.

use super::VisoEngine;
use crate::options::score_color;

impl VisoEngine {
    /// Push lighting options to the GPU uniform.
    pub(super) fn apply_lighting(&mut self) {
        self.gpu.apply_lighting(&self.options.lighting);
    }

    /// Push post-processing options to the composite pass.
    pub(super) fn apply_post_processing(&mut self) {
        self.gpu.apply_post_processing(&self.options);
    }

    /// Push camera options to the controller.
    pub(super) fn apply_camera(&mut self) {
        self.camera_controller
            .apply_camera_options(&self.options.camera);
    }

    /// Push debug options to the camera uniform.
    pub(super) fn apply_debug(&mut self) {
        self.camera_controller
            .apply_debug_options(&self.options.debug, &self.gpu.context.queue);
    }

    /// Recompute backbone per-residue colors from current options and
    /// push them to the GPU color buffer.
    pub(super) fn recompute_backbone_colors(&mut self) {
        let chains = self.gpu.renderers.backbone.cached_chains().to_vec();
        let per_entity_scores: Vec<Option<&[f64]>> = self
            .entities
            .entities()
            .iter()
            .map(|e| e.per_residue_scores.as_deref())
            .collect();
        let entity_chain_counts: Vec<usize> = self
            .entities
            .entities()
            .iter()
            .filter(|e| e.visible)
            .filter_map(|e| {
                e.entity.as_protein().and_then(|p| {
                    let segs = p.to_interleaved_segments();
                    if segs.is_empty() {
                        None
                    } else {
                        Some(segs.len())
                    }
                })
            })
            .collect();
        let new_colors = score_color::compute_per_residue_colors_styled(
            &chains,
            &self.topology.ss_types,
            &per_entity_scores,
            &self.options.display.backbone_color_scheme,
            &self.options.display.backbone_palette(),
            Some(&entity_chain_counts),
        );
        self.gpu.set_target_colors(&new_colors);
        self.topology.per_residue_colors = Some(new_colors);
    }

    /// Trigger a full background remesh so ball-and-stick geometry
    /// reflects current visibility / drawing-mode flags.
    ///
    /// Bumps every entity's `mesh_version` so the background mesh cache
    /// regenerates from scratch, then dispatches a `FullRebuild`. The
    /// previous synchronous-upload path relied on the legacy
    /// `EntityStore` which the render path no longer reads.
    pub(crate) fn refresh_ball_and_stick(&mut self) {
        for state in self.entity_state.values_mut() {
            state.mesh_version = state.mesh_version.wrapping_add(1);
        }
        self.entities.force_dirty();
        self.sync_scene_to_renderers(std::collections::HashMap::new());
    }
}
