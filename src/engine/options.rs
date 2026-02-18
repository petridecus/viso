//! Options methods for ProteinRenderEngine

use super::ProteinRenderEngine;
use crate::util::options::Options;

impl ProteinRenderEngine {
    /// Get a reference to the current options.
    pub fn options(&self) -> &Options {
        &self.options
    }

    /// Replace options and apply all changes to subsystems.
    pub fn set_options(&mut self, new: Options) {
        self.options = new;
        self.apply_options();
    }

    /// Push current option values to all subsystems (lighting, camera,
    /// composite, etc.).
    pub fn apply_options(&mut self) {
        // Lighting
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

        // Post-processing (outline/AO params; fog is dynamic per-frame)
        self.post_process
            .apply_options(&self.options, &self.context.queue);

        // Camera
        let co = &self.options.camera;
        self.camera_controller.camera.fovy = co.fovy;
        self.camera_controller.camera.znear = co.znear;
        self.camera_controller.camera.zfar = co.zfar;
        self.camera_controller.rotate_speed = co.rotate_speed * 0.02; // scale to match internal units
        self.camera_controller.pan_speed = co.pan_speed * 0.2;
        self.camera_controller.zoom_speed = co.zoom_speed * 0.5;

        // Display: ball-and-stick visibility
        self.refresh_ball_and_stick();
    }

    /// Apply a single view option by key/value from the frontend.
    /// Returns true if the option was recognized and applied.
    pub fn apply_view_option(
        &mut self,
        key: &str,
        value: &serde_json::Value,
    ) -> bool {
        match key {
            "show_sidechains" => {
                if let Some(v) = value.as_bool() {
                    self.options.display.show_sidechains = v;
                    true
                } else {
                    false
                }
            }
            "show_waters" => {
                if let Some(v) = value.as_bool() {
                    self.options.display.show_waters = v;
                    self.refresh_ball_and_stick();
                    true
                } else {
                    false
                }
            }
            "show_ions" => {
                if let Some(v) = value.as_bool() {
                    self.options.display.show_ions = v;
                    self.refresh_ball_and_stick();
                    true
                } else {
                    false
                }
            }
            "show_solvent" => {
                if let Some(v) = value.as_bool() {
                    self.options.display.show_solvent = v;
                    self.refresh_ball_and_stick();
                    true
                } else {
                    false
                }
            }
            "backbone_color_mode" => {
                if let Some(mode_str) = value.as_str() {
                    use crate::util::options::BackboneColorMode;
                    let mode = match mode_str {
                        "score" => BackboneColorMode::Score,
                        "score_relative" => BackboneColorMode::ScoreRelative,
                        "secondary_structure" => {
                            BackboneColorMode::SecondaryStructure
                        }
                        "chain" => BackboneColorMode::Chain,
                        _ => return false,
                    };
                    self.options.display.backbone_color_mode = mode;
                    // Compute new colors and transition smoothly via GPU buffer
                    let chains = self.tube_renderer.cached_chains().to_vec();
                    let new_colors = self.compute_per_residue_colors(&chains);
                    self.residue_color_buffer.set_target_colors(&new_colors);
                    self.sc.cached_per_residue_colors = Some(new_colors);
                    true
                } else {
                    false
                }
            }
            // --- Lighting ---
            "lighting.light1_intensity" => {
                self.set_f32_option(value, |s, v| {
                    s.options.lighting.light1_intensity = v;
                    s.apply_options();
                })
            }
            "lighting.light2_intensity" => {
                self.set_f32_option(value, |s, v| {
                    s.options.lighting.light2_intensity = v;
                    s.apply_options();
                })
            }
            "lighting.ambient" => self.set_f32_option(value, |s, v| {
                s.options.lighting.ambient = v;
                s.apply_options();
            }),
            "lighting.specular_intensity" => {
                self.set_f32_option(value, |s, v| {
                    s.options.lighting.specular_intensity = v;
                    s.apply_options();
                })
            }
            "lighting.shininess" => self.set_f32_option(value, |s, v| {
                s.options.lighting.shininess = v;
                s.apply_options();
            }),
            "lighting.rim_power" => self.set_f32_option(value, |s, v| {
                s.options.lighting.rim_power = v;
                s.apply_options();
            }),
            "lighting.rim_intensity" => self.set_f32_option(value, |s, v| {
                s.options.lighting.rim_intensity = v;
                s.apply_options();
            }),
            "lighting.ibl_strength" => self.set_f32_option(value, |s, v| {
                s.options.lighting.ibl_strength = v;
                s.apply_options();
            }),
            // --- Post-processing ---
            "post_processing.outline_thickness" => {
                self.set_f32_option(value, |s, v| {
                    s.options.post_processing.outline_thickness = v;
                    s.apply_options();
                })
            }
            "post_processing.outline_strength" => {
                self.set_f32_option(value, |s, v| {
                    s.options.post_processing.outline_strength = v;
                    s.apply_options();
                })
            }
            "post_processing.ao_strength" => {
                self.set_f32_option(value, |s, v| {
                    s.options.post_processing.ao_strength = v;
                    s.apply_options();
                })
            }
            "post_processing.fog_start" => {
                self.set_f32_option(value, |s, v| {
                    s.options.post_processing.fog_start = v;
                    s.apply_options();
                })
            }
            "post_processing.fog_density" => {
                self.set_f32_option(value, |s, v| {
                    s.options.post_processing.fog_density = v;
                    s.apply_options();
                })
            }
            "post_processing.ao_radius" => {
                self.set_f32_option(value, |s, v| {
                    s.options.post_processing.ao_radius = v;
                    s.apply_options();
                })
            }
            "post_processing.ao_bias" => self.set_f32_option(value, |s, v| {
                s.options.post_processing.ao_bias = v;
                s.apply_options();
            }),
            "post_processing.ao_power" => self.set_f32_option(value, |s, v| {
                s.options.post_processing.ao_power = v;
                s.apply_options();
            }),
            "post_processing.exposure" => self.set_f32_option(value, |s, v| {
                s.options.post_processing.exposure = v;
                s.apply_options();
            }),
            "post_processing.normal_outline_strength" => {
                self.set_f32_option(value, |s, v| {
                    s.options.post_processing.normal_outline_strength = v;
                    s.apply_options();
                })
            }
            "post_processing.bloom_intensity" => {
                self.set_f32_option(value, |s, v| {
                    s.options.post_processing.bloom_intensity = v;
                    s.apply_options();
                })
            }
            "post_processing.bloom_threshold" => {
                self.set_f32_option(value, |s, v| {
                    s.options.post_processing.bloom_threshold = v;
                    s.apply_options();
                })
            }
            "lighting.roughness" => self.set_f32_option(value, |s, v| {
                s.options.lighting.roughness = v;
                s.apply_options();
            }),
            "lighting.metalness" => self.set_f32_option(value, |s, v| {
                s.options.lighting.metalness = v;
                s.apply_options();
            }),
            // --- Camera ---
            "camera.fovy" => self.set_f32_option(value, |s, v| {
                s.options.camera.fovy = v;
                s.apply_options();
            }),
            "camera.rotate_speed" => self.set_f32_option(value, |s, v| {
                s.options.camera.rotate_speed = v;
                s.apply_options();
            }),
            "camera.pan_speed" => self.set_f32_option(value, |s, v| {
                s.options.camera.pan_speed = v;
                s.apply_options();
            }),
            "camera.zoom_speed" => self.set_f32_option(value, |s, v| {
                s.options.camera.zoom_speed = v;
                s.apply_options();
            }),
            _ => {
                log::debug!("Unhandled view option: {}", key);
                false
            }
        }
    }

    /// Helper: extract an f64/f32 from a JSON value and apply a mutation.
    fn set_f32_option(
        &mut self,
        value: &serde_json::Value,
        apply: impl FnOnce(&mut Self, f32),
    ) -> bool {
        if let Some(v) = value.as_f64() {
            apply(self, v as f32);
            true
        } else {
            false
        }
    }

    /// Load a named view preset from the presets directory.
    /// Returns true on success.
    pub fn load_preset(
        &mut self,
        name: &str,
        presets_dir: &std::path::Path,
    ) -> bool {
        let path = presets_dir.join(format!("{}.toml", name));
        match Options::load(&path) {
            Ok(opts) => {
                log::info!("Loaded view preset '{}'", name);
                self.set_options(opts);
                self.active_preset = Some(name.to_string());
                true
            }
            Err(e) => {
                log::error!("Failed to load view preset '{}': {}", name, e);
                false
            }
        }
    }

    /// Save the current options as a named view preset.
    /// Returns true on success.
    pub fn save_preset(
        &self,
        name: &str,
        presets_dir: &std::path::Path,
    ) -> bool {
        let path = presets_dir.join(format!("{}.toml", name));
        match self.options.save(&path) {
            Ok(()) => {
                log::info!("Saved view preset '{}'", name);
                true
            }
            Err(e) => {
                log::error!("Failed to save view preset '{}': {}", name, e);
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
                crate::util::options::LipidMode::Coarse
            } else {
                crate::util::options::LipidMode::BallAndStick
            };
        self.refresh_ball_and_stick();
    }
}
