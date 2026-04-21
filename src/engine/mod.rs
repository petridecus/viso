pub(crate) mod assembly_consumer;
mod bootstrap;
/// The engine's complete interactive vocabulary.
pub mod command;
mod constraint;
mod density;
pub(crate) mod density_store;
pub(crate) mod entity_view;
/// Focus state for tab cycling.
pub mod focus;
mod options_apply;
pub(crate) mod positions;
pub(crate) mod scene_state;
pub(crate) mod surface;
mod sync;
pub(crate) mod trajectory;

use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::sync::Arc;

use assembly_consumer::AssemblyConsumer;
pub(crate) use bootstrap::FrameTiming;
use density_store::DensityStore;
use focus::Focus;
use molex::entity::molecule::id::EntityId;
use molex::{Assembly, MoleculeEntity, MoleculeType, SSType};
use rustc_hash::FxHashMap;
use surface::EntitySurface;
use entity_view::EntityView;
use positions::EntityPositions;
use scene_state::SceneRenderState;
use web_time::Instant;

use crate::animation::transition::Transition;
use crate::animation::AnimationState;
use crate::camera::controller::CameraController;
use crate::options::{DrawingMode, EntityAppearance, VisoOptions};
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
/// overlays (appearance overrides, behavior overrides, camera state)
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
    /// Per-entity molecular surfaces.
    pub(crate) entity_surfaces: FxHashMap<u32, EntitySurface>,

    // ── Assembly ingest ───────────────────────────────────────────
    /// Triple-buffer reader for the host-owned [`Assembly`].
    pub(crate) assembly_consumer: AssemblyConsumer,
    /// Last `Assembly` snapshot applied by [`Self::sync_from_assembly`].
    /// Held for read-only queries (entity metadata for the UI panel,
    /// atom positions for the picking pipeline). Always up to date
    /// with `entity_state` / `positions`.
    pub(crate) current_assembly: Arc<Assembly>,
    /// Cross-entity rendering data (disulfide + H-bond endpoints)
    /// rederived on every sync.
    pub(crate) scene_state: Arc<SceneRenderState>,
    /// Per-entity rendering state (topology + drawing mode + SS
    /// override + mesh version) rederived on every sync.
    pub(crate) entity_state: FxHashMap<EntityId, EntityView>,
    /// Per-entity animator write surface; renderer reads back each
    /// frame. Seeded from the assembly's reference positions when new
    /// entities appear, animated locally thereafter.
    pub(crate) positions: EntityPositions,
    /// Generation of the most recently consumed `Assembly` snapshot.
    /// Initialized to `u64::MAX` so the first snapshot (generation 0)
    /// always triggers a sync.
    pub(crate) last_seen_generation: u64,
    /// Monotonic mesh-version dispenser. Survives `reset_scene_local_state`
    /// so a Vacant insert never collides with a stale worker `MeshCache`
    /// entry for the same `EntityId` (fresh per-file allocators reuse low
    /// ids across `replace_scene`).
    pub(crate) next_mesh_version: u64,

    // ── Viso-side per-entity overlays ─────────────────────────────
    /// Current focus state (session-wide or a specific entity).
    pub(crate) focus: Focus,
    /// Per-entity visibility (`true` = drawn). Missing = visible.
    pub(crate) entity_visibility: FxHashMap<u32, bool>,
    /// Per-entity animation behavior overrides.
    pub(crate) entity_behaviors: FxHashMap<u32, Transition>,
    /// Per-entity appearance overrides (None-fields inherit global).
    pub(crate) appearance_overrides: FxHashMap<u32, EntityAppearance>,
    /// Per-entity scores (for color-by-score visualization).
    pub(crate) entity_scores: FxHashMap<u32, Vec<f64>>,
    /// Per-entity SS overrides (from puzzle annotations).
    pub(crate) entity_ss_overrides: FxHashMap<u32, Vec<SSType>>,
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
        if self.animation.tick(now, &mut self.positions) {
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
                e.focus = Focus::Session;
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
                e.focus = Focus::Entity(id);
                e.fit_camera_to_focus();
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
        let Some(assembly) = self.assembly_consumer.latest() else {
            return;
        };
        if assembly.generation() == self.last_seen_generation {
            return;
        }
        self.sync_from_assembly(&assembly);
        self.current_assembly = assembly;
        self.last_seen_generation = self.current_assembly.generation();
    }

    /// Stop the background scene processor thread.
    pub fn shutdown(&mut self) {
        self.gpu.shutdown();
    }
}

// ── Camera + focus ──

impl VisoEngine {
    /// Fit camera to the currently focused element.
    pub fn fit_camera_to_focus(&mut self) {
        match self.focus {
            Focus::Session => {
                self.fit_session_camera();
            }
            Focus::Entity(eid) => {
                if let Some((centroid, radius)) = self.entity_bounds(eid) {
                    self.camera_controller
                        .fit_to_sphere_animated(centroid, radius);
                }
            }
        }
    }

    /// Fit the camera to the combined session bounding sphere.
    pub(crate) fn fit_session_camera(&mut self) {
        if let Some((centroid, radius)) = self.session_bounds() {
            self.camera_controller
                .fit_to_sphere_animated(centroid, radius);
        }
    }

    /// Cycle focus: Session → first focusable entity → … → Session.
    fn cycle_focus(&mut self) -> Focus {
        let focusable: Vec<u32> = self
            .current_assembly
            .entities()
            .iter()
            .filter(|e| self.is_entity_visible(e.id().raw()) && e.is_focusable())
            .map(|e| e.id().raw())
            .collect();

        self.focus = match self.focus {
            Focus::Session => focusable
                .first()
                .map_or(Focus::Session, |&id| Focus::Entity(id)),
            Focus::Entity(current_id) => {
                let idx = focusable.iter().position(|&id| id == current_id);
                match idx {
                    Some(i) if i + 1 < focusable.len() => {
                        Focus::Entity(focusable[i + 1])
                    }
                    _ => Focus::Session,
                }
            }
        };
        self.focus
    }

    /// Bounding sphere of a single entity from its Assembly positions.
    fn entity_bounds(&self, raw_id: u32) -> Option<(glam::Vec3, f32)> {
        let entity = self
            .current_assembly
            .entities()
            .iter()
            .find(|e| e.id().raw() == raw_id)?;
        bounding_sphere_of(entity)
    }

    /// Combined bounding sphere across all visible entities.
    pub(crate) fn session_bounds(&self) -> Option<(glam::Vec3, f32)> {
        let visible: Vec<&MoleculeEntity> = self
            .current_assembly
            .entities()
            .iter()
            .filter(|e| self.is_entity_visible(e.id().raw()))
            .collect();
        if visible.is_empty() {
            return None;
        }

        let mut total_weight = 0.0f32;
        let mut weighted_sum = glam::Vec3::ZERO;
        let mut radii: Vec<(glam::Vec3, f32)> = Vec::with_capacity(visible.len());
        for entity in &visible {
            let Some((centroid, radius)) = bounding_sphere_of(entity) else {
                continue;
            };
            let w = entity.atom_count() as f32;
            if w > 0.0 {
                weighted_sum += centroid * w;
                total_weight += w;
                radii.push((centroid, radius));
            }
        }
        if total_weight == 0.0 {
            return None;
        }
        let centroid = weighted_sum / total_weight;
        let radius = radii
            .iter()
            .map(|(c, r)| (*c - centroid).length() + r)
            .fold(0.0f32, f32::max);
        Some((centroid, radius))
    }
}

/// Compute (centroid, radius) for a molecule entity's atoms.
pub(crate) fn bounding_sphere_of(
    entity: &MoleculeEntity,
) -> Option<(glam::Vec3, f32)> {
    let atoms = entity.atom_set();
    if atoms.is_empty() {
        return None;
    }
    let n = atoms.len() as f32;
    let centroid =
        atoms.iter().fold(glam::Vec3::ZERO, |acc, a| acc + a.position) / n;
    let radius = atoms
        .iter()
        .map(|a| (a.position - centroid).length())
        .fold(0.0f32, f32::max);
    Some((centroid, radius))
}

// ── Viso-side overlays (pub(crate) helpers for VisoApp) ──

impl VisoEngine {
    /// Whether an entity is currently visible. Absent entries in the
    /// visibility overlay default to visible.
    pub(crate) fn is_entity_visible(&self, id: u32) -> bool {
        self.entity_visibility.get(&id).copied().unwrap_or(true)
    }

    /// Record the visibility of a single entity.
    pub(crate) fn set_entity_visible_internal(
        &mut self,
        id: u32,
        visible: bool,
    ) {
        let _ = self.entity_visibility.insert(id, visible);
        if let Some(eid) = self.entity_id(id) {
            if let Some(state) = self.entity_state.get_mut(&eid) {
                state.mesh_version = state.mesh_version.wrapping_add(1);
            }
        }
    }

    /// Set visibility for every entity of a given molecule type.
    pub(crate) fn set_type_visibility_internal(
        &mut self,
        mol_type: MoleculeType,
        visible: bool,
    ) {
        let ids: Vec<u32> = self
            .current_assembly
            .entities()
            .iter()
            .filter(|e| e.molecule_type() == mol_type)
            .map(|e| e.id().raw())
            .collect();
        for id in ids {
            self.set_entity_visible_internal(id, visible);
        }
    }

    /// Set per-entity scores (`None` clears).
    pub(crate) fn set_per_residue_scores_internal(
        &mut self,
        id: u32,
        scores: Option<Vec<f64>>,
    ) {
        match scores {
            Some(s) => {
                let _ = self.entity_scores.insert(id, s);
            }
            None => {
                let _ = self.entity_scores.remove(&id);
            }
        }
        if let Some(eid) = self.entity_id(id) {
            if let Some(state) = self.entity_state.get_mut(&eid) {
                state.mesh_version = state.mesh_version.wrapping_add(1);
            }
        }
    }

    /// Set an SS override for an entity.
    pub(crate) fn set_ss_override_internal(
        &mut self,
        id: u32,
        ss: Vec<SSType>,
    ) {
        let _ = self.entity_ss_overrides.insert(id, ss);
        if let Some(eid) = self.entity_id(id) {
            if let Some(state) = self.entity_state.get_mut(&eid) {
                state.mesh_version = state.mesh_version.wrapping_add(1);
            }
        }
    }

    /// Clear an entity's per-entity animation behavior override.
    pub(crate) fn clear_entity_behavior_internal(&mut self, id: u32) {
        let _ = self.entity_behaviors.remove(&id);
    }

    /// Reset all scene-local state (animation, positions, entity_state,
    /// surfaces). Called when replacing or clearing the scene.
    ///
    /// Also resets `last_seen_generation` to `u64::MAX` so that the next
    /// Assembly snapshot triggers a sync unconditionally — the app-side
    /// replace path rebuilds the assembly in one shot via
    /// `Assembly::new(...)`, which starts at generation 0 and would
    /// otherwise collide with a previously-observed `last_seen_generation`
    /// of 0.
    pub(crate) fn reset_scene_local_state(&mut self) {
        self.animation = AnimationState::new();
        self.entity_state.clear();
        self.positions = EntityPositions::new();
        self.entity_visibility.clear();
        self.entity_behaviors.clear();
        self.appearance_overrides.clear();
        self.entity_scores.clear();
        self.entity_ss_overrides.clear();
        self.focus = Focus::Session;
        self.entity_surfaces.clear();
        self.last_seen_generation = u64::MAX;
        self.regenerate_entity_surfaces();
    }

    /// Look up the opaque `EntityId` for a raw ID. Returns `None` if
    /// no entity with that raw ID exists.
    pub(crate) fn entity_id(&self, raw: u32) -> Option<EntityId> {
        self.current_assembly
            .entities()
            .iter()
            .map(MoleculeEntity::id)
            .find(|eid| eid.raw() == raw)
    }

    /// Read access to the per-entity behavior override for `id`.
    pub(crate) fn entity_behavior(&self, id: u32) -> Option<&Transition> {
        self.entity_behaviors.get(&id)
    }
}

// ── Public API: overlays (engine-side; not Assembly-related) ──

impl VisoEngine {
    /// Set the animation behavior for a specific entity.
    pub fn set_entity_behavior(
        &mut self,
        entity_id: u32,
        transition: Transition,
    ) {
        let _ = self.entity_behaviors.insert(entity_id, transition);
    }

    /// Clear a per-entity behavior override.
    pub fn clear_entity_behavior(&mut self, entity_id: u32) {
        self.clear_entity_behavior_internal(entity_id);
    }

    /// Set per-entity appearance overrides.
    pub fn set_entity_appearance(
        &mut self,
        entity_id: u32,
        overrides: EntityAppearance,
    ) {
        let _ = self.appearance_overrides.insert(entity_id, overrides);
        self.apply_appearance_change(entity_id);
        self.sync_scene_to_renderers(HashMap::new());
    }

    /// Clear a per-entity appearance override.
    pub fn clear_entity_appearance(&mut self, entity_id: u32) {
        let _ = self.appearance_overrides.remove(&entity_id);
        self.apply_appearance_change(entity_id);
        self.sync_scene_to_renderers(HashMap::new());
    }

    /// Bump mesh version and re-resolve drawing mode for an entity
    /// whose appearance override just changed. Internal helper that
    /// sidesteps the borrow-check pitfall of reading `self` while
    /// holding a `&mut` into `entity_state`.
    fn apply_appearance_change(&mut self, entity_id: u32) {
        let Some(eid) = self.entity_id(entity_id) else {
            return;
        };
        let mol_type = self
            .entity_state
            .get(&eid)
            .map(|s| s.topology.molecule_type);
        let Some(mol_type) = mol_type else {
            return;
        };
        let drawing_mode = self.resolved_drawing_mode(entity_id, mol_type);
        if let Some(state) = self.entity_state.get_mut(&eid) {
            state.mesh_version = state.mesh_version.wrapping_add(1);
            state.drawing_mode = drawing_mode;
        }
    }

    /// Look up a per-entity appearance override.
    #[must_use]
    pub fn entity_appearance(
        &self,
        entity_id: u32,
    ) -> Option<&EntityAppearance> {
        self.appearance_overrides.get(&entity_id)
    }

    /// Resolve the drawing mode for an entity: per-entity override wins,
    /// else global (with Cartoon falling back to type-default).
    pub(crate) fn resolved_drawing_mode(
        &self,
        entity_id: u32,
        mol_type: MoleculeType,
    ) -> DrawingMode {
        self.appearance_overrides
            .get(&entity_id)
            .and_then(|ovr| ovr.drawing_mode)
            .unwrap_or_else(|| {
                if self.options.display.drawing_mode == DrawingMode::Cartoon {
                    DrawingMode::default_for(mol_type)
                } else {
                    self.options.display.drawing_mode
                }
            })
    }

    /// Replace the current set of constraint bands.
    pub fn update_bands(&mut self, bands: Vec<command::BandInfo>) {
        self.constraints.band_specs = bands;
        self.resolve_and_render_constraints();
    }

    /// Set or clear the active pull constraint.
    pub fn update_pull(&mut self, pull: Option<command::PullInfo>) {
        self.constraints.pull_spec = pull;
        self.resolve_and_render_constraints();
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
        match self.focus {
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
        self.focus
    }

    /// Set the focus state directly.
    pub fn set_focus(&mut self, focus: Focus) {
        self.focus = focus;
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
        self.current_assembly.entities().len()
    }

    /// Read-only access to the last `Assembly` snapshot applied to
    /// viso-side state.
    #[must_use]
    pub fn assembly(&self) -> &Assembly {
        &self.current_assembly
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
            self.set_type_visibility_internal(
                MoleculeType::Water,
                self.options.display.show_waters,
            );
        }
        if ions_changed {
            self.set_type_visibility_internal(
                MoleculeType::Ion,
                self.options.display.show_ions,
            );
        }
        if solvent_changed {
            self.set_type_visibility_internal(
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
            log::debug!(
                "set_options: display/colors changed, triggering sync"
            );
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
    /// all meshes. Pulls fresh versions from the engine's monotonic
    /// dispenser so a version never collides with a stale worker
    /// `MeshCache` entry for the same `EntityId`.
    fn invalidate_all_mesh_versions(&mut self) {
        let ids: Vec<EntityId> = self.entity_state.keys().copied().collect();
        for id in ids {
            let v = self.next_mesh_version;
            self.next_mesh_version = self.next_mesh_version.wrapping_add(1);
            if let Some(state) = self.entity_state.get_mut(&id) {
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
            .entity_state
            .iter()
            .map(|(id, state)| (*id, state.topology.molecule_type))
            .collect();
        for (id, mol_type) in pairs {
            let resolved = self.resolved_drawing_mode(id.raw(), mol_type);
            if let Some(state) = self.entity_state.get_mut(&id) {
                state.drawing_mode = resolved;
            }
        }
    }
}
