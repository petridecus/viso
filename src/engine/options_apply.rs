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

/// Fine-grained diff of what changed between two [`VisoOptions`] values.
///
/// Each field corresponds to one dispatch step in
/// [`VisoEngine::apply_options_change`]. Adding a field here forces a
/// matching dispatch arm — the struct initializer in [`Self::diff`] and
/// the OR-chain in [`Self::any`] both break if fields drift out of sync.
///
/// The struct is intentionally a flat bag of bools: each bit maps to
/// one independent dispatch arm, so a state-machine rewrite would lose
/// the 1-to-1 mapping that makes diff/apply enforceable.
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct OptionsChange {
    /// Lighting uniform (`options.lighting`) changed.
    pub(crate) lighting: bool,
    /// Post-processing stack (`options.post_processing`) changed.
    pub(crate) post_processing: bool,
    /// Camera options (`options.camera`) changed.
    pub(crate) camera: bool,
    /// Debug overlay options (`options.debug`) changed.
    pub(crate) debug: bool,
    /// Surface present mode changed (requires swapchain reconfigure).
    pub(crate) present_mode: bool,
    /// Display section changed (ignoring `present_mode`, which has its
    /// own flag).
    pub(crate) display: bool,
    /// Geometry tuning section changed.
    pub(crate) geometry: bool,
    /// Colors section changed.
    pub(crate) colors: bool,
    /// Water visibility toggled.
    pub(crate) waters: bool,
    /// Ion visibility toggled.
    pub(crate) ions: bool,
    /// Solvent visibility toggled.
    pub(crate) solvent: bool,
    /// Surface kind / opacity / cavities changed (requires surface
    /// regeneration).
    pub(crate) surface: bool,
    /// Helix or sheet style changed (requires backbone remesh).
    pub(crate) helix_sheet: bool,
    /// Global drawing mode changed (requires per-entity re-resolution).
    pub(crate) drawing_mode: bool,
}

impl OptionsChange {
    /// Compute the fine-grained diff between two options.
    pub(crate) fn diff(old: &VisoOptions, new: &VisoOptions) -> Self {
        let display = {
            let mut old_d = old.display.clone();
            old_d.present_mode = new.display.present_mode;
            old_d != new.display
        };
        Self {
            lighting: old.lighting != new.lighting,
            post_processing: old.post_processing != new.post_processing,
            camera: old.camera != new.camera,
            debug: old.debug != new.debug,
            present_mode: old.display.present_mode != new.display.present_mode,
            display,
            geometry: old.geometry != new.geometry,
            colors: old.colors != new.colors,
            waters: old.display.show_waters != new.display.show_waters,
            ions: old.display.show_ions != new.display.show_ions,
            solvent: old.display.show_solvent != new.display.show_solvent,
            surface: old.display.surface_kind != new.display.surface_kind
                || old.display.surface_opacity != new.display.surface_opacity
                || old.display.show_cavities != new.display.show_cavities,
            helix_sheet: old.display.helix_style != new.display.helix_style
                || old.display.sheet_style != new.display.sheet_style,
            drawing_mode: old.display.drawing_mode != new.display.drawing_mode,
        }
    }

    /// True if any part of the options changed.
    pub(crate) fn any(&self) -> bool {
        self.lighting
            || self.post_processing
            || self.camera
            || self.debug
            || self.present_mode
            || self.display
            || self.geometry
            || self.colors
            || self.waters
            || self.ions
            || self.solvent
            || self.surface
            || self.helix_sheet
            || self.drawing_mode
    }
}

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
        let change = OptionsChange::diff(&self.options, &new);
        if !change.any() {
            return;
        }
        self.options = new;
        self.apply_options_change(&change);
    }

    /// Dispatch one action per field of [`OptionsChange`]. Adding a
    /// field to [`OptionsChange`] forces a matching arm here — the
    /// compiler enforces diff/apply stay in sync via the struct
    /// initializer in [`OptionsChange::diff`] and the `any()` OR-chain.
    fn apply_options_change(&mut self, change: &OptionsChange) {
        if change.waters {
            self.set_type_visibility(
                MoleculeType::Water,
                self.options.display.show_waters,
            );
        }
        if change.ions {
            self.set_type_visibility(
                MoleculeType::Ion,
                self.options.display.show_ions,
            );
        }
        if change.solvent {
            self.set_type_visibility(
                MoleculeType::Solvent,
                self.options.display.show_solvent,
            );
        }
        if change.lighting {
            self.apply_lighting();
        }
        if change.post_processing {
            self.apply_post_processing();
        }
        if change.camera {
            self.apply_camera();
        }
        if change.debug {
            self.apply_debug();
        }
        if change.present_mode {
            self.gpu
                .context
                .set_present_mode(self.options.display.present_mode.to_wgpu());
        }
        if change.drawing_mode {
            self.reresolve_drawing_modes();
        }
        if change.display || change.geometry || change.colors {
            self.invalidate_all_mesh_versions();
        }
        if change.display {
            self.refresh_ball_and_stick();
            self.recompute_backbone_colors();
        }
        if change.geometry || change.helix_sheet {
            let camera_eye = self.camera_controller.camera.eye;
            self.submit_per_chain_lod_remesh(camera_eye);
        }
        if change.colors {
            self.refresh_ball_and_stick();
        }
        if change.display || change.colors {
            log::debug!("set_options: display/colors changed, triggering sync");
            self.sync_scene_to_renderers(HashMap::new());
        }
        if change.surface {
            super::surface_regen::regenerate_surfaces(
                &self.scene,
                &self.annotations,
                &self.density,
                &self.options,
                &self.surface_regen,
            );
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
