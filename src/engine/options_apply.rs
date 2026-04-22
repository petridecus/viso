//! Engine-side methods that propagate option / preset changes through
//! the GPU, scene, and overlay sub-structs.
//!
//! Every method here takes `&mut self` on [`VisoEngine`] and touches
//! multiple sibling sub-structs (`gpu`, `scene`, `annotations`,
//! `camera_controller`) — the code can't live in `options/` because
//! that would invert the dependency direction (options is a leaf
//! module, engine depends on it).

use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use molex::entity::molecule::id::EntityId;
use molex::MoleculeType;

use super::VisoEngine;
use crate::options::VisoOptions;

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

    /// Replace options and apply only the sections that changed.
    pub fn set_options(&mut self, new: VisoOptions) {
        let old = &self.options;
        let lighting_changed = old.lighting != new.lighting;
        let post_changed = old.post_processing != new.post_processing;
        let camera_changed = old.camera != new.camera;
        let debug_changed = old.debug != new.debug;
        let present_mode_changed =
            old.display.present_mode != new.display.present_mode;
        let display_changed = {
            let mut old_d = old.display.clone();
            old_d.present_mode = new.display.present_mode;
            old_d != new.display
        };
        let geometry_changed = old.geometry != new.geometry;
        let colors_changed = old.colors != new.colors;
        let waters_changed = old.display.show_waters != new.display.show_waters;
        let ions_changed = old.display.show_ions != new.display.show_ions;
        let solvent_changed =
            old.display.show_solvent != new.display.show_solvent;
        let surface_changed = old.display.surface_kind
            != new.display.surface_kind
            || old.display.surface_opacity != new.display.surface_opacity
            || old.display.show_cavities != new.display.show_cavities;

        let helix_sheet_changed = old.display.helix_style
            != new.display.helix_style
            || old.display.sheet_style != new.display.sheet_style;

        let drawing_mode_changed =
            old.display.drawing_mode != new.display.drawing_mode;

        self.options = new;

        if waters_changed {
            self.set_type_visibility(
                MoleculeType::Water,
                self.options.display.show_waters,
            );
        }
        if ions_changed {
            self.set_type_visibility(
                MoleculeType::Ion,
                self.options.display.show_ions,
            );
        }
        if solvent_changed {
            self.set_type_visibility(
                MoleculeType::Solvent,
                self.options.display.show_solvent,
            );
        }

        if lighting_changed {
            self.apply_lighting();
        }
        if post_changed {
            self.apply_post_processing();
        }
        if camera_changed {
            self.apply_camera();
        }
        if debug_changed {
            self.apply_debug();
        }
        if present_mode_changed {
            self.gpu
                .context
                .set_present_mode(self.options.display.present_mode.to_wgpu());
        }
        if drawing_mode_changed {
            self.reresolve_drawing_modes();
        }
        if display_changed || geometry_changed || colors_changed {
            self.invalidate_all_mesh_versions();
        }
        if display_changed {
            self.refresh_ball_and_stick();
            self.recompute_backbone_colors();
        }
        if geometry_changed || helix_sheet_changed {
            let camera_eye = self.camera_controller.camera.eye;
            self.submit_per_chain_lod_remesh(camera_eye);
        }
        if colors_changed {
            self.refresh_ball_and_stick();
        }
        if display_changed || colors_changed {
            log::debug!("set_options: display/colors changed, triggering sync");
            self.sync_scene_to_renderers(HashMap::new());
        }
        if surface_changed {
            self.regenerate_entity_surfaces();
        }
    }

    /// Force-refresh all subsystems from current options (escape hatch).
    pub fn apply_options(&mut self) {
        self.gpu
            .context
            .set_present_mode(self.options.display.present_mode.to_wgpu());
        self.apply_lighting();
        self.apply_post_processing();
        self.apply_camera();
        self.apply_debug();
        self.refresh_ball_and_stick();

        let camera_eye = self.camera_controller.camera.eye;
        self.submit_per_chain_lod_remesh(camera_eye);
        self.recompute_backbone_colors();
    }

    /// Load a named view preset from the presets directory.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_preset(&mut self, name: &str, presets_dir: &Path) -> bool {
        let path = presets_dir.join(format!("{name}.toml"));
        match VisoOptions::load(&path) {
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
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save_preset(&mut self, name: &str, presets_dir: &Path) -> bool {
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

    /// Set the GPU render scale (supersampling factor).
    pub fn set_render_scale(&mut self, scale: u32) {
        self.gpu.context.render_scale = scale;
    }

    /// Update render scale from a DPI scale factor and resize if needed.
    pub fn set_surface_scale(&mut self, scale: f64) {
        if self.gpu.context.set_surface_scale(scale) {
            let w = self.gpu.context.config.width;
            let h = self.gpu.context.config.height;
            self.gpu.resize(w, h);
        }
    }

    /// Bump every entity's `mesh_version` so the next sync regenerates
    /// all meshes.
    fn invalidate_all_mesh_versions(&mut self) {
        let ids: Vec<EntityId> =
            self.scene.entity_state.keys().copied().collect();
        for id in ids {
            let v = self.scene.bump_mesh_version();
            if let Some(state) = self.scene.entity_state.get_mut(&id) {
                state.mesh_version = v;
            }
        }
    }

    /// Re-resolve every entity's `drawing_mode` against the current
    /// global option + per-entity appearance overrides. Needed when the
    /// global `display.drawing_mode` changes between Assembly syncs —
    /// `sync_from_assembly` only runs on generation bumps, so without
    /// this the per-entity `state.drawing_mode` stays stale.
    fn reresolve_drawing_modes(&mut self) {
        let pairs: Vec<(EntityId, MoleculeType)> = self
            .scene
            .entity_state
            .iter()
            .map(|(id, state)| (*id, state.topology.molecule_type))
            .collect();
        for (id, mol_type) in pairs {
            let resolved = self.resolved_drawing_mode(id, mol_type);
            if let Some(state) = self.scene.entity_state.get_mut(&id) {
                state.drawing_mode = resolved;
            }
        }
    }
}
