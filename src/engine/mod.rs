mod bootstrap;
/// The engine's complete interactive vocabulary.
pub mod command;
mod constraint;
mod density;
pub(crate) mod density_store;
mod entity;
pub(crate) mod entity_store;
mod options_apply;
/// Scene types: Focus, topology, and visual state.
pub mod scene;
/// Entity data types, bond topology, and scene aggregation functions.
pub(crate) mod scene_data;
pub(crate) mod surface;
mod sync;
pub(crate) mod trajectory;

use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

pub(crate) use bootstrap::FrameTiming;
use density_store::DensityStore;
use entity_store::EntityStore;
use rustc_hash::FxHashMap;
use scene::{Focus, SceneTopology, VisualState};
use surface::EntitySurface;
use web_time::Instant;

use crate::animation::AnimationState;
use crate::camera::controller::CameraController;
use crate::options::VisoOptions;
use crate::renderer::GpuPipeline;

/// Stored constraint specifications (bands + pull), resolved to world-space
/// each frame.
pub(crate) struct ConstraintSpecs {
    /// Band constraint specs.
    pub band_specs: Vec<command::BandInfo>,
    /// Pull constraint spec.
    pub pull_spec: Option<command::PullInfo>,
}

/// The core rendering engine for protein visualization.
///
/// Manages the full GPU pipeline: molecular renderers (tube, ribbon,
/// ball-and-stick, sidechain capsules, nucleic acid, bands, pulls),
/// post-processing (SSAO, bloom, depth fog, FXAA), and GPU picking.
///
/// # Construction
///
/// Use [`VisoEngine::new`] for a default molecule or
/// [`VisoEngine::new_with_path`] to load a specific `.cif`/`.pdb`
/// file.
///
/// # Frame loop
///
/// Each frame, call [`render`](Self::render) to draw and present. Call
/// [`resize`](Self::resize) when the window size changes. Input is forwarded
/// via [`execute`](Self::execute).
///
/// # Scene management
///
/// Load structures with [`load_entities`](Self::load_entities), update
/// coordinates with [`update_backbone`](Self::update_backbone) or
/// [`update_entity_coords`](Self::update_entity_coords), and sync changes
/// to renderers with
/// [`sync_scene_to_renderers`](Self::sync_scene_to_renderers).
///
/// # Animation
///
/// Structural changes are animated via the internal `StructureAnimator`.
/// Animation transitions are managed internally by the engine pipeline.
pub struct VisoEngine {
    /// All GPU infrastructure (device, renderers, picking, post-process,
    /// lighting, cursor, culling state).
    pub(crate) gpu: GpuPipeline,
    /// Orbital camera controller.
    pub camera_controller: CameraController,
    /// Derived topology (SS types, residue ranges, sidechain topology).
    pub(crate) topology: SceneTopology,
    /// Animation output buffer (interpolated positions).
    pub(crate) visual: VisualState,
    /// Consolidated entity ownership (source + scene entities + behaviors).
    pub(crate) entities: EntityStore,
    /// Stored band/pull constraint specs.
    pub(crate) constraints: ConstraintSpecs,
    /// Structural animation, trajectory, and pending transitions.
    pub(crate) animation: AnimationState,
    /// Runtime display, lighting, color, and geometry options.
    pub(crate) options: VisoOptions,
    /// Currently applied options preset name, if any.
    pub(crate) active_preset: Option<String>,
    /// Per-frame timing and FPS tracking.
    pub(crate) frame_timing: FrameTiming,
    /// Loaded electron density maps.
    pub(crate) density: DensityStore,
    /// Per-entity molecular surfaces (internal mesh generation params).
    pub(crate) entity_surfaces: FxHashMap<u32, EntitySurface>,
}

// ── Frame loop ──

impl VisoEngine {
    /// Per-frame updates: animation ticks, uniform uploads, frustum culling.
    fn pre_render(&mut self) {
        self.apply_pending_animation();
        self.tick_animation();

        // Camera uniform (hover state + time from GPU picking)
        self.camera_controller.uniform.hovered_residue =
            self.gpu.pick.hovered_target.as_residue_i32();
        self.camera_controller.uniform.time =
            self.frame_timing.elapsed_secs();
        self.camera_controller.update_gpu(&self.gpu.context.queue);

        // Depth-buffer fog from camera distance
        let fog_start = self.camera_controller.distance();
        let fog_density =
            2.0 / self.camera_controller.bounding_radius().max(10.0);
        self.gpu.post_process.update_fog(
            &self.gpu.context.queue,
            fog_start,
            fog_density,
        );

        self.check_and_submit_lod();
        self.gpu
            .pick
            .update_selection_buffer(&self.gpu.context.queue);
        let _color_transitioning =
            self.gpu.pick.residue_colors.update(&self.gpu.context.queue);
        self.gpu.update_headlamp(&self.camera_controller.camera);
        self.update_frustum_culling();

        // Resolve band/pull specs to world-space each frame (auto-tracks
        // animated atoms)
        if !self.constraints.band_specs.is_empty()
            || self.constraints.pull_spec.is_some()
        {
            self.resolve_and_render_constraints();
        }

        // Poll for pending density mesh data from background thread
        let _ = self.gpu.apply_pending_density_mesh();
    }

    /// Tick animation (both trajectory and structural), submitting any
    /// interpolated frame to the background thread.
    ///
    /// Trajectory frames are fed through `animate_entity()` with
    /// `Transition::snap()`, so both paths converge through the
    /// animator's update loop.
    fn tick_animation(&mut self) {
        let now = Instant::now();
        self.animation
            .advance_trajectory(now, &self.topology.entity_residue_ranges);
        let Some(frame) = self.animation.tick(now) else {
            return;
        };
        self.visual.update(
            frame.backbone_chains.clone(),
            frame.sidechain_positions.clone().unwrap_or_else(|| {
                self.topology.sidechain_topology.target_positions.clone()
            }),
            frame.backbone_sidechain_bonds.clone().unwrap_or_else(|| {
                self.topology
                    .sidechain_topology
                    .target_backbone_bonds
                    .clone()
            }),
        );
        if self.visual.is_dirty() {
            self.submit_animation_frame_from(&frame);
        }
    }

    /// Core render — geometry, post-process, picking — targeting the given
    /// view. Returns the encoder so the caller can submit it.
    fn render_to_view(
        &mut self,
        view: &wgpu::TextureView,
    ) -> wgpu::CommandEncoder {
        self.gpu.render_to_view(
            view,
            &self.camera_controller,
            self.options.display.show_sidechains,
        )
    }

    /// Execute one frame: update animations, run the geometry pass,
    /// post-process, and present.
    ///
    /// # Errors
    ///
    /// Returns [`wgpu::SurfaceError`] if the swapchain frame cannot be
    /// acquired.
    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // Check if we should render based on FPS limit
        if !self.frame_timing.should_render() {
            return Ok(());
        }

        self.pre_render();

        let frame = self.gpu.context.get_next_frame()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let encoder = self.render_to_view(&view);
        self.gpu.context.submit(encoder);

        // Start async GPU picking readback (non-blocking)
        self.gpu.pick.picking.start_readback();

        // Try to complete any pending readback from previous frame
        // (non-blocking poll)
        self.gpu.pick.poll_and_resolve(&self.gpu.context.device);

        frame.present();

        // Update frame timing
        self.frame_timing.end_frame();

        Ok(())
    }

    /// Render the scene to the given texture view (for embedding in
    /// dioxus/etc). The caller owns the texture — no surface present happens.
    pub fn render_to_texture(&mut self, view: &wgpu::TextureView) {
        self.pre_render();
        let encoder = self.render_to_view(view);
        self.gpu.context.submit(encoder);
        self.frame_timing.end_frame();
    }

    /// Resize all GPU surfaces and the camera projection to match the new
    /// window size.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.gpu.resize(width, height);
            self.camera_controller.resize(width, height);
        }
    }
}

// ── Command dispatch ──

impl VisoEngine {
    /// Execute a single [`command::VisoCommand`].
    ///
    /// This is **the** centralized entry point for all interactive behavior.
    /// Keyboard bindings, mouse gestures, GUI buttons, and programmatic
    /// callers all go through here.
    ///
    /// Returns `true` if the residue selection changed (relevant for
    /// selection commands only).
    pub fn execute(&mut self, cmd: command::VisoCommand) -> bool {
        use command::VisoCommand;
        match cmd {
            // Camera
            VisoCommand::RecenterCamera => self.execute_no_selection(|e| {
                e.fit_camera_to_focus();
            }),
            VisoCommand::ToggleAutoRotate => self.execute_no_selection(|e| {
                let _ = e.camera_controller.toggle_auto_rotate();
            }),
            VisoCommand::RotateCamera { delta } => {
                self.execute_no_selection(|e| e.camera_controller.rotate(delta))
            }
            VisoCommand::PanCamera { delta } => {
                self.execute_no_selection(|e| e.camera_controller.pan(delta))
            }
            VisoCommand::Zoom { delta } => {
                self.execute_no_selection(|e| e.camera_controller.zoom(delta))
            }
            // Focus
            VisoCommand::CycleFocus => self.execute_no_selection(|e| {
                let _ = e.entities.cycle_focus();
                e.fit_camera_to_focus();
            }),
            VisoCommand::ResetFocus => self.execute_no_selection(|e| {
                e.entities.set_focus(Focus::Session);
                e.fit_camera_to_focus();
            }),
            // Playback
            VisoCommand::ToggleTrajectory => {
                self.execute_no_selection(|e| e.animation.toggle_trajectory())
            }
            // Selection
            VisoCommand::ClearSelection => self.gpu.pick.clear_selection(),
            VisoCommand::SelectResidue { index, extend } => {
                self.gpu.pick.picking.handle_click(index, extend)
            }
            VisoCommand::SelectSegment { index, extend } => self
                .gpu
                .pick
                .select_segment(index, &self.topology.ss_types, extend),
            VisoCommand::SelectChain { index, extend } => {
                let chains = self.gpu.renderers.backbone.cached_chains();
                self.gpu.pick.select_chain(index, chains, extend)
            }
            // Entity management
            VisoCommand::FocusEntity { id } => self.execute_no_selection(|e| {
                e.entities.set_focus(Focus::Entity(id));
                e.fit_camera_to_focus();
            }),
            VisoCommand::ToggleEntityVisibility { id } => {
                let vis = self.entities.entity(id).is_some_and(|e| !e.visible);
                self.set_entity_visible(id, vis);
                false
            }
            VisoCommand::RemoveEntity { id } => {
                self.execute_no_selection(|e| e.remove_entity(id))
            }
            // Display toggles — toggle entity-level visibility for
            // all entities of the corresponding molecule type, and keep
            // the display option in sync (renderer safety net).
            VisoCommand::ToggleIons => {
                self.options.display.show_ions =
                    !self.options.display.show_ions;
                self.entities.set_type_visible(
                    molex::MoleculeType::Ion,
                    self.options.display.show_ions,
                );
                self.sync_scene_to_renderers(HashMap::new());
                false
            }
            VisoCommand::ToggleWaters => {
                self.options.display.show_waters =
                    !self.options.display.show_waters;
                self.entities.set_type_visible(
                    molex::MoleculeType::Water,
                    self.options.display.show_waters,
                );
                self.sync_scene_to_renderers(HashMap::new());
                false
            }
            VisoCommand::ToggleSolvent => {
                self.options.display.show_solvent =
                    !self.options.display.show_solvent;
                self.entities.set_type_visible(
                    molex::MoleculeType::Solvent,
                    self.options.display.show_solvent,
                );
                self.sync_scene_to_renderers(HashMap::new());
                false
            }
            VisoCommand::CycleLipidMode => {
                self.options.display.lipid_mode =
                    if self.options.display.lipid_ball_and_stick() {
                        crate::options::LipidMode::Coarse
                    } else {
                        crate::options::LipidMode::BallAndStick
                    };
                self.refresh_ball_and_stick();
                false
            }
        }
    }

    /// Run a non-selection command, always returning `false`.
    fn execute_no_selection(&mut self, f: impl FnOnce(&mut Self)) -> bool {
        f(self);
        false
    }
}

// ── Public API: lifecycle ──

impl VisoEngine {
    /// Advance camera animation and apply any pending scene from the
    /// background processor.
    ///
    /// Call once per frame before [`render`](Self::render):
    /// ```ignore
    /// engine.update(dt);
    /// engine.render()?;
    /// ```
    pub fn update(&mut self, dt: f32) {
        let _ = self.camera_controller.update_animation(dt);
        self.apply_pending_scene();
    }

    /// Stop the background scene processor thread.
    pub fn shutdown(&mut self) {
        self.gpu.shutdown();
    }
}

// ── Public API: camera ──

impl VisoEngine {
    /// Fit camera to the currently focused element.
    pub fn fit_camera_to_focus(&mut self) {
        match *self.entities.focus() {
            Focus::Session => {
                if let Some((centroid, radius)) =
                    self.entities.bounding_sphere()
                {
                    self.camera_controller
                        .fit_to_sphere_animated(centroid, radius);
                }
            }
            Focus::Entity(eid) => {
                if let Some(se) = self.entities.entity(eid) {
                    self.camera_controller.fit_to_sphere_animated(
                        se.cached_centroid,
                        se.cached_bounding_radius,
                    );
                }
            }
        }
    }
}

// ── Public API: queries ──

impl VisoEngine {
    /// The pick target currently under the cursor (resolved from the
    /// previous frame's GPU picking pass).
    pub fn hovered_target(&self) -> crate::renderer::picking::PickTarget {
        self.gpu.pick.hovered_target
    }

    /// The currently focused entity ID, or `None` when focus is session-wide.
    #[must_use]
    pub fn focused_entity(&self) -> Option<u32> {
        match *self.entities.focus() {
            Focus::Entity(id) => Some(id),
            Focus::Session => None,
        }
    }

    /// Current smoothed frames-per-second.
    #[must_use]
    pub fn fps(&self) -> f32 {
        self.frame_timing.fps()
    }

    /// Currently selected residue indices.
    #[must_use]
    pub fn selected_residues(&self) -> &[i32] {
        self.gpu.pick.selected_residues()
    }

    /// Read-only access to the current options.
    #[must_use]
    pub fn options(&self) -> &VisoOptions {
        &self.options
    }

    /// Name of the currently active preset, if any.
    #[must_use]
    pub fn active_preset(&self) -> Option<&str> {
        self.active_preset.as_deref()
    }

    /// Whether a trajectory is loaded.
    #[must_use]
    pub fn has_trajectory(&self) -> bool {
        self.animation.trajectory_player.is_some()
    }

    /// Current focus state.
    #[must_use]
    pub fn focus(&self) -> Focus {
        *self.entities.focus()
    }

    /// Set the focus state directly.
    pub fn set_focus(&mut self, focus: Focus) {
        self.entities.set_focus(focus);
    }

    /// Resolve an atom position from structural references.
    ///
    /// Uses interpolated visual positions during animation so lookups
    /// track animated atoms. Returns `None` if the residue/atom is
    /// out of range.
    #[must_use]
    pub fn resolve_atom_position(
        &self,
        residue: u32,
        atom_name: &str,
    ) -> Option<glam::Vec3> {
        let scene = constraint::ScenePositions {
            visual: &self.visual,
            topology: &self.topology,
            cached_chains: self.gpu.renderers.backbone.cached_chains(),
        };
        constraint::resolve_atom_ref_pub(
            &scene,
            &command::AtomRef {
                residue,
                atom_name: atom_name.to_owned(),
            },
        )
    }

    /// Number of entities currently in the scene.
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.entities.entity_count()
    }

    /// Current viewport dimensions in physical pixels.
    #[must_use]
    pub fn viewport_size(&self) -> glam::UVec2 {
        glam::UVec2::new(
            self.gpu.context.config.width,
            self.gpu.context.config.height,
        )
    }

    /// Update the cursor position for GPU picking.
    ///
    /// Call each frame with the current mouse position in physical pixels.
    pub fn set_cursor_pos(&mut self, x: f32, y: f32) {
        self.gpu.cursor_pos = (x, y);
    }
}

// ── Public API: options ──

impl VisoEngine {
    /// Replace options and apply only the sections that changed.
    pub fn set_options(&mut self, new: VisoOptions) {
        let old = &self.options;
        let lighting_changed = old.lighting != new.lighting;
        let post_changed = old.post_processing != new.post_processing;
        let camera_changed = old.camera != new.camera;
        let debug_changed = old.debug != new.debug;
        let present_mode_changed =
            old.display.present_mode != new.display.present_mode;
        // Compare display options excluding present_mode (which only
        // affects surface configuration, not scene geometry).
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
            || old.display.surface_opacity != new.display.surface_opacity;

        let helix_sheet_changed = old.display.helix_style
            != new.display.helix_style
            || old.display.sheet_style != new.display.sheet_style;

        self.options = new;

        // Sync entity-level visibility with ambient type display toggles.
        if waters_changed {
            self.entities.set_type_visible(
                molex::MoleculeType::Water,
                self.options.display.show_waters,
            );
        }
        if ions_changed {
            self.entities.set_type_visible(
                molex::MoleculeType::Ion,
                self.options.display.show_ions,
            );
        }
        if solvent_changed {
            self.entities.set_type_visible(
                molex::MoleculeType::Solvent,
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
        if display_changed || geometry_changed || colors_changed {
            self.entities.force_dirty();
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

        // Regenerate backbone mesh with updated geometry options (on
        // background thread to avoid blocking the main thread / exceeding
        // the 256 MB buffer limit with synchronous allocation).
        let camera_eye = self.camera_controller.camera.eye;
        self.submit_per_chain_lod_remesh(camera_eye);

        // Recompute backbone colors (handles backbone_color_mode changes)
        self.recompute_backbone_colors();
    }

    /// Load a named view preset from the presets directory.
    /// Returns true on success.
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
    /// Returns true on success.
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
    ///
    /// Low-DPI displays (scale < 2.0) get 2x SSAA; HiDPI (>= 2.0) renders
    /// at native resolution. Triggers a full render target resize when the
    /// scale changes.
    pub fn set_surface_scale(&mut self, scale: f64) {
        if self.gpu.context.set_surface_scale(scale) {
            let w = self.gpu.context.config.width;
            let h = self.gpu.context.config.height;
            self.gpu.resize(w, h);
        }
    }
}

#[cfg(test)]
pub(crate) mod test_fixtures;
