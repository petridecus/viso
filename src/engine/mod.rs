pub(crate) mod annotations;
pub(crate) mod assembly_consumer;
mod bootstrap;
/// The engine's complete interactive vocabulary.
pub mod command;
mod constraint;
mod culling;
mod density;
pub(crate) mod density_store;
pub(crate) mod entity_view;
/// Focus state for tab cycling.
pub mod focus;
mod options_apply;
pub(crate) mod positions;
pub(crate) mod scene;
pub(crate) mod scene_state;
pub(crate) mod surface;
mod sync;
pub(crate) mod trajectory;

use std::collections::HashMap;

use annotations::EntityAnnotations;
pub(crate) use bootstrap::FrameTiming;
use density_store::DensityStore;
use focus::Focus;
use molex::entity::molecule::id::EntityId;
use molex::{Assembly, MoleculeEntity, MoleculeType};
use scene::Scene;
use web_time::Instant;

use crate::animation::AnimationState;
use crate::camera;
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
/// `VisoEngine` is a read-only consumer of structural state: the host
/// application owns an [`molex::Assembly`] and publishes snapshots
/// through a triple buffer. In standalone deployments
/// ([`crate::VisoApp`]) the app plays the host role.
///
/// The engine is never mutated directly for structural changes —
/// all such mutations route through [`crate::VisoApp`]. Viso-side
/// annotations (appearance overrides, behavior overrides, camera state)
/// are mutated through engine methods directly.
pub struct VisoEngine {
    // ── GPU + camera ──────────────────────────────────────────────
    /// All GPU infrastructure (device, renderers, picking, post-process,
    /// lighting, cursor, culling state).
    pub(crate) gpu: GpuPipeline,
    /// Orbital camera controller.
    pub camera_controller: CameraController,

    // ── Runtime state ─────────────────────────────────────────────
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

    // ── Assembly ingest + derived per-entity state ────────────────
    /// Triple-buffer reader, latest snapshot, generation tracker,
    /// plus the per-entity render-ready derived state rebuilt on
    /// every sync (`SceneRenderState`, `EntityView`s, positions).
    /// Also owns the monotonic `mesh_version` dispenser. See
    /// [`Scene`].
    pub(crate) scene: Scene,

    // ── User-authored per-entity annotations ──────────────────────
    /// Per-entity opinions that ride alongside the Assembly: focus,
    /// visibility, behaviors, appearance overrides, scores, SS
    /// overrides, surfaces. All maps keyed on [`EntityId`] so lookups
    /// are O(1). See [`EntityAnnotations`].
    pub(crate) annotations: EntityAnnotations,
}

// ── Frame loop ──

impl VisoEngine {
    /// Per-frame updates: animation ticks, uniform uploads, frustum
    /// culling.
    fn pre_render(&mut self) {
        self.apply_pending_animation();
        self.tick_animation();

        self.camera_controller.uniform.hovered_residue =
            self.gpu.pick.hovered_target.as_residue_i32();
        self.camera_controller.uniform.time = self.frame_timing.elapsed_secs();
        self.camera_controller.update_gpu(&self.gpu.context.queue);

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

        if !self.constraints.band_specs.is_empty()
            || self.constraints.pull_spec.is_some()
        {
            self.resolve_and_render_constraints();
        }

        let _ = self.gpu.apply_pending_density_mesh();
    }

    /// Tick animation (both trajectory and structural), submitting any
    /// interpolated frame to the background thread.
    fn tick_animation(&mut self) {
        let now = Instant::now();
        let trajectory_frame = self.animation.advance_trajectory(now);
        if let Some(frame) = trajectory_frame {
            self.apply_trajectory_frame(&frame);
            self.submit_animation_frame();
        }
        if self.animation.tick(now, &mut self.scene.positions) {
            self.submit_animation_frame();
        }
    }

    /// Core render — geometry, post-process, picking — targeting the
    /// given view. Returns the encoder so the caller can submit it.
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

        self.gpu.pick.picking.start_readback();
        self.gpu.pick.poll_and_resolve(&self.gpu.context.device);

        frame.present();
        self.frame_timing.end_frame();

        Ok(())
    }

    /// Render the scene to the given texture view (for embedding in
    /// dioxus/etc). The caller owns the texture — no surface present
    /// happens.
    pub fn render_to_texture(&mut self, view: &wgpu::TextureView) {
        self.pre_render();
        let encoder = self.render_to_view(view);
        self.gpu.context.submit(encoder);
        self.frame_timing.end_frame();
    }

    /// Resize all GPU surfaces and the camera projection to match the
    /// new window size.
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
    /// Returns `true` if the residue selection changed.
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
                let _ = e.cycle_focus();
                e.fit_camera_to_focus();
            }),
            VisoCommand::ResetFocus => self.execute_no_selection(|e| {
                e.annotations.focus = Focus::Session;
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
            VisoCommand::SelectSegment { index, extend } => {
                let ss = self.concatenated_cartoon_ss();
                self.gpu.pick.select_segment(index, &ss, extend)
            }
            VisoCommand::SelectChain { index, extend } => {
                let chains = self.gpu.renderers.backbone.cached_chains();
                self.gpu.pick.select_chain(index, chains, extend)
            }
            // Entity management
            VisoCommand::FocusEntity { id } => self.execute_no_selection(|e| {
                if let Some(eid) = e.entity_id(id) {
                    e.annotations.focus = Focus::Entity(eid);
                    e.fit_camera_to_focus();
                }
            }),
            VisoCommand::ToggleEntityVisibility { id } => {
                let currently_visible = self.is_entity_visible(id);
                self.set_entity_visible_internal(id, !currently_visible);
                self.sync_scene_to_renderers(HashMap::new());
                false
            }
            VisoCommand::RemoveEntity { .. } => {
                // Entity removal must go through VisoApp (it mutates the
                // Assembly). Ignored when dispatched as a bare command.
                false
            }
            // Display toggles — toggle engine-side visibility for all
            // entities of a type and keep the display option in sync.
            VisoCommand::ToggleIons => {
                self.options.display.show_ions =
                    !self.options.display.show_ions;
                self.set_type_visibility_internal(
                    MoleculeType::Ion,
                    self.options.display.show_ions,
                );
                self.sync_scene_to_renderers(HashMap::new());
                false
            }
            VisoCommand::ToggleWaters => {
                self.options.display.show_waters =
                    !self.options.display.show_waters;
                self.set_type_visibility_internal(
                    MoleculeType::Water,
                    self.options.display.show_waters,
                );
                self.sync_scene_to_renderers(HashMap::new());
                false
            }
            VisoCommand::ToggleSolvent => {
                self.options.display.show_solvent =
                    !self.options.display.show_solvent;
                self.set_type_visibility_internal(
                    MoleculeType::Solvent,
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

    fn execute_no_selection(&mut self, f: impl FnOnce(&mut Self)) -> bool {
        f(self);
        false
    }
}

// ── Public API: lifecycle ──

impl VisoEngine {
    /// Advance camera animation and apply any pending scene from the
    /// background processor.
    pub fn update(&mut self, dt: f32) {
        let _ = self.camera_controller.update_animation(dt);
        self.poll_assembly();
        self.apply_pending_scene();
    }

    /// Poll the consumer for a new Assembly snapshot and, if one is
    /// ready, rederive viso-side state.
    fn poll_assembly(&mut self) {
        let Some(assembly) = self.scene.consumer.latest() else {
            return;
        };
        if assembly.generation() == self.scene.last_seen_generation {
            return;
        }
        self.sync_from_assembly(&assembly);
        self.scene.current = assembly;
        self.scene.last_seen_generation = self.scene.current.generation();
    }

    /// Stop the background scene processor thread.
    pub fn shutdown(&mut self) {
        self.gpu.shutdown();
    }
}

// ── Camera + focus (thin dispatchers; logic lives in camera::fit +
// annotations) ──

impl VisoEngine {
    /// Fit the camera to the currently focused element (session-wide
    /// bounding sphere, or the focused entity's bounding sphere).
    pub fn fit_camera_to_focus(&mut self) {
        match self.annotations.focus {
            Focus::Session => self.fit_session_camera(),
            Focus::Entity(eid) => {
                if let Some(entity) =
                    self.scene.current.entities().iter().find(|e| e.id() == eid)
                {
                    camera::fit::fit_to_entity(
                        &mut self.camera_controller,
                        entity,
                    );
                }
            }
        }
    }

    /// Fit the camera to the combined bounding sphere of every visible
    /// entity.
    pub(crate) fn fit_session_camera(&mut self) {
        let visible: Vec<&MoleculeEntity> = self
            .scene
            .current
            .entities()
            .iter()
            .filter(|e| self.is_entity_visible(e.id().raw()))
            .collect();
        camera::fit::fit_to_entities(&mut self.camera_controller, visible);
    }

    /// Advance focus to the next visible, focusable entity. Wraps to
    /// session after the last. Returns the new focus.
    fn cycle_focus(&mut self) -> Focus {
        let focusable: Vec<EntityId> = self
            .scene
            .current
            .entities()
            .iter()
            .filter(|e| {
                self.is_entity_visible(e.id().raw()) && e.is_focusable()
            })
            .map(MoleculeEntity::id)
            .collect();
        self.annotations.cycle_focus(&focusable)
    }
}

// ── Lifecycle + EntityId translation ──

impl VisoEngine {
    /// Reset all scene-local state (animation, scene ingest, derived
    /// per-entity views, annotations). Called when replacing or
    /// clearing the scene.
    ///
    /// Also resets `last_seen_generation` to `u64::MAX` so that the
    /// next Assembly snapshot triggers a sync unconditionally — the
    /// app-side replace path rebuilds the assembly in one shot via
    /// `Assembly::new(...)`, which starts at generation 0 and would
    /// otherwise collide with a previously-observed
    /// `last_seen_generation` of 0.
    pub(crate) fn reset_scene_local_state(&mut self) {
        self.animation = AnimationState::new();
        self.scene.reset_local_state();
        self.annotations.reset();
        self.regenerate_entity_surfaces();
    }

    /// Look up the opaque [`EntityId`] for a raw `u32` id. Returns
    /// `None` if no entity with that raw id exists. The boundary
    /// translator: callers arriving from a wire format (IPC, TOML,
    /// CLI) translate *once* here and then pass [`EntityId`] to the
    /// per-entity engine methods.
    #[must_use]
    pub fn entity_id(&self, raw: u32) -> Option<EntityId> {
        self.scene.entity_id(raw)
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
    pub fn focused_entity(&self) -> Option<EntityId> {
        match self.annotations.focus {
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
        self.annotations.focus
    }

    /// Set the focus state directly.
    pub fn set_focus(&mut self, focus: Focus) {
        self.annotations.focus = focus;
    }

    /// Resolve an atom position from structural references, using
    /// interpolated visual positions during animation.
    #[must_use]
    pub fn resolve_atom_position(
        &self,
        residue: u32,
        atom_name: &str,
    ) -> Option<glam::Vec3> {
        constraint::resolve_atom_ref_pub(
            self,
            &command::AtomRef {
                residue,
                atom_name: atom_name.to_owned(),
            },
        )
    }

    /// Number of entities currently in the scene.
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.scene.current.entities().len()
    }

    /// Read-only access to the last `Assembly` snapshot applied to
    /// viso-side state.
    #[must_use]
    pub fn assembly(&self) -> &Assembly {
        self.scene.current.as_ref()
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
    pub fn set_cursor_pos(&mut self, x: f32, y: f32) {
        self.gpu.cursor_pos = (x, y);
    }
}
