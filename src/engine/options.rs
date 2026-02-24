//! Options methods for ProteinRenderEngine

use super::ProteinRenderEngine;
use crate::options::Options;

impl ProteinRenderEngine {
    /// Replace options and apply all changes to subsystems.
    pub fn set_options(&mut self, new: Options) {
        self.options = new;
        self.apply_options();
    }

    /// Push current option values to all subsystems (lighting, camera,
    /// composite, etc.).
    pub fn apply_options(&mut self) {
        self.apply_lighting();
        self.apply_post_processing();
        self.apply_camera();
        self.apply_debug();
        self.refresh_ball_and_stick();

        // Regenerate backbone mesh with updated geometry options (on
        // background thread to avoid blocking the main thread / exceeding
        // the 256 MB buffer limit with synchronous allocation).
        let camera_eye = self.camera_controller.camera.eye;
        self.submit_per_chain_lod_remesh(camera_eye);

        // Recompute backbone colors (handles backbone_color_mode changes)
        let chains = self.backbone_renderer.cached_chains().to_vec();
        let new_colors = self.compute_per_residue_colors(&chains);
        self.residue_color_buffer.set_target_colors(&new_colors);
        self.sc.cached_per_residue_colors = Some(new_colors);
    }

    /// Push lighting options to the GPU uniform.
    fn apply_lighting(&mut self) {
        let lo = &self.options.lighting;
        self.lighting.uniform.light1_intensity = lo.light1_intensity;
        self.lighting.uniform.light2_intensity = lo.light2_intensity;
        self.lighting.uniform.ambient = lo.ambient;
        self.lighting.uniform.specular_intensity = lo.specular_intensity;
        self.lighting.uniform.shininess = lo.shininess;
        self.lighting.uniform.rim_power = lo.rim_power;
        self.lighting.uniform.rim_intensity = lo.rim_intensity;
        self.lighting.uniform.rim_directionality = lo.rim_directionality;
        self.lighting.uniform.rim_color = lo.rim_color;
        self.lighting.uniform.ibl_strength = lo.ibl_strength;
        self.lighting.uniform.roughness = lo.roughness;
        self.lighting.uniform.metalness = lo.metalness;
        self.lighting.update_gpu(&self.context.queue);
    }

    /// Push post-processing options to the composite pass.
    fn apply_post_processing(&mut self) {
        self.post_process
            .apply_options(&self.options, &self.context.queue);
    }

    /// Push camera options to the controller.
    fn apply_camera(&mut self) {
        let co = &self.options.camera;
        self.camera_controller.camera.fovy = co.fovy;
        self.camera_controller.camera.znear = co.znear;
        self.camera_controller.camera.zfar = co.zfar;
        self.camera_controller.rotate_speed = co.rotate_speed * 0.02;
        self.camera_controller.pan_speed = co.pan_speed * 0.2;
        self.camera_controller.zoom_speed = co.zoom_speed * 0.5;
    }

    /// Push debug options to the camera uniform.
    fn apply_debug(&mut self) {
        self.camera_controller.uniform.debug_mode =
            u32::from(self.options.debug.show_normals);
        self.camera_controller.update_gpu(&self.context.queue);
    }

    /// Load a named view preset from the presets directory.
    /// Returns true on success.
    pub fn load_preset(
        &mut self,
        name: &str,
        presets_dir: &std::path::Path,
    ) -> bool {
        let path = presets_dir.join(format!("{name}.toml"));
        match Options::load(&path) {
            Ok(opts) => {
                log::info!("Loaded view preset '{name}'");
                self.set_options(opts);
                self.active_preset = Some(name.to_owned());
                true
            }
            Err(e) => {
                log::error!("Failed to load view preset '{name}': {e}");
                false
            }
        }
    }

    /// Save the current options as a named view preset.
    /// Returns true on success.
    pub fn save_preset(
        &mut self,
        name: &str,
        presets_dir: &std::path::Path,
    ) -> bool {
        let path = presets_dir.join(format!("{name}.toml"));
        match self.options.save(&path) {
            Ok(()) => {
                log::info!("Saved view preset '{name}'");
                self.active_preset = Some(name.to_owned());
                true
            }
            Err(e) => {
                log::error!("Failed to save view preset '{name}': {e}");
                false
            }
        }
    }

    /// Toggle water visibility
    pub fn toggle_waters(&mut self) {
        self.options.display.show_waters = !self.options.display.show_waters;
        self.refresh_ball_and_stick();
    }

    /// Toggle ion visibility
    pub fn toggle_ions(&mut self) {
        self.options.display.show_ions = !self.options.display.show_ions;
        self.refresh_ball_and_stick();
    }

    /// Toggle solvent visibility
    pub fn toggle_solvent(&mut self) {
        self.options.display.show_solvent = !self.options.display.show_solvent;
        self.refresh_ball_and_stick();
    }

    /// Cycle lipid display mode (coarse → ball_and_stick → coarse)
    pub fn toggle_lipids(&mut self) {
        self.options.display.lipid_mode =
            if self.options.display.lipid_ball_and_stick() {
                crate::options::LipidMode::Coarse
            } else {
                crate::options::LipidMode::BallAndStick
            };
        self.refresh_ball_and_stick();
    }
}
