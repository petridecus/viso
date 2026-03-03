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
        let co = &self.options.camera;
        self.camera_controller.camera.fovy = co.fovy;
        self.camera_controller.camera.znear = co.znear;
        self.camera_controller.camera.zfar = co.zfar;
        self.camera_controller.rotate_speed = co.rotate_speed * 0.02;
        self.camera_controller.pan_speed = co.pan_speed * 0.2;
        self.camera_controller.zoom_speed = co.zoom_speed * 0.5;
    }

    /// Push debug options to the camera uniform.
    pub(super) fn apply_debug(&mut self) {
        self.camera_controller.uniform.debug_mode =
            u32::from(self.options.debug.show_normals);
        self.camera_controller.update_gpu(&self.gpu.context.queue);
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
        let new_colors = score_color::compute_per_residue_colors(
            &chains,
            &self.topology.ss_types,
            &per_entity_scores,
            &self.options.display.backbone_color_mode,
        );
        self.gpu.set_target_colors(&new_colors);
        self.topology.per_residue_colors = Some(new_colors);
    }

    /// Refresh ball-and-stick renderer with current visibility flags.
    pub(crate) fn refresh_ball_and_stick(&mut self) {
        let entities =
            self.entities.ligand_entities().map(|se| se.entity.clone());
        self.gpu.refresh_ball_and_stick(
            entities,
            &self.options.display,
            Some(&self.options.colors),
        );
    }
}
