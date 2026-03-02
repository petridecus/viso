mod bootstrap;
/// The engine's complete interactive vocabulary.
pub mod command;
mod gpu_init;
pub(crate) mod trajectory;

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use foldit_conv::secondary_structure::SSType;
use foldit_conv::types::entity::MoleculeEntity;
use glam::{Mat4, Vec3};

use self::command::{BandInfo, PullInfo};
use self::gpu_init::GpuBootstrap;
use self::trajectory::TrajectoryPlayer;
use crate::animation::transition::Transition;
use crate::animation::{
    AnimationFrame, EntitySidechainData, StructureAnimator,
};
use crate::camera::controller::CameraController;
use crate::error::VisoError;
use crate::gpu::lighting::Lighting;
use crate::gpu::{RenderContext, ShaderComposer};
use crate::options::{score_color, VisoOptions};
use crate::renderer::draw_context::DrawBindGroups;
use crate::renderer::geometry::{
    BackboneUpdateData, PreparedBackboneData, PreparedBallAndStickData,
    SidechainView,
};
use crate::renderer::picking::PickingSystem;
use crate::renderer::pipeline::{PreparedScene, SceneProcessor, SceneRequest};
use crate::renderer::postprocess::PostProcessStack;
use crate::renderer::{GeometryPassInput, Renderers};
use crate::scene::{Focus, Scene, SceneEntity};

/// Target FPS limit
const TARGET_FPS: u32 = 300;

// ---------------------------------------------------------------------------
// FrameTiming
// ---------------------------------------------------------------------------

/// Frame timing with FPS calculation and optional frame limiting.
pub(crate) struct FrameTiming {
    target_fps: u32,
    min_frame_duration: Duration,
    last_frame: Instant,
    smoothed_fps: f32,
    smoothing: f32,
}

impl FrameTiming {
    /// Create a new frame timer with the given FPS target (0 = unlimited).
    fn new(target_fps: u32) -> Self {
        let min_frame_duration = if target_fps > 0 {
            Duration::from_secs_f64(1.0 / f64::from(target_fps))
        } else {
            Duration::ZERO
        };
        Self {
            target_fps,
            min_frame_duration,
            last_frame: Instant::now(),
            smoothed_fps: 60.0,
            smoothing: 0.05,
        }
    }

    /// Returns true if enough time has passed to render the next frame.
    fn should_render(&self) -> bool {
        if self.target_fps == 0 {
            return true;
        }
        self.last_frame.elapsed() >= self.min_frame_duration
    }

    /// Update timing after rendering a frame.
    fn end_frame(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_frame);
        self.last_frame = now;

        let frame_time = elapsed.as_secs_f32();
        if frame_time > 0.0 {
            let instant_fps = 1.0 / frame_time;
            self.smoothed_fps = self
                .smoothed_fps
                .mul_add(1.0 - self.smoothing, instant_fps * self.smoothing);
        }
    }

    /// Current smoothed FPS.
    pub fn fps(&self) -> f32 {
        self.smoothed_fps
    }
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
    /// Core wgpu device, queue, and surface.
    pub context: RenderContext,
    _shader_composer: ShaderComposer,

    /// Post-processing pass stack (SSAO, bloom, composite, FXAA).
    pub(crate) post_process: PostProcessStack,
    /// Current cursor position in physical pixels (set by the viewer /
    /// input processor each frame for GPU picking).
    pub(crate) cursor_pos: (f32, f32),
    /// Camera eye position at the last frustum-culling update.
    pub(crate) last_cull_camera_eye: Vec3,

    /// Orbital camera controller.
    pub camera_controller: CameraController,
    /// GPU lighting uniform and bind group.
    pub lighting: Lighting,
    /// Scene graph holding all entities.
    pub(crate) scene: Scene,
    /// Canonical entity data (source of truth). Scene is derived from this.
    pub(crate) entities: Vec<MoleculeEntity>,
    /// Per-entity animation behavior overrides (default = smooth).
    pub(crate) entity_behaviors: HashMap<u32, Transition>,
    /// Transitions pending from the last sync (avoids round-trip through
    /// the background thread). Empty when no sync is pending.
    pending_transitions: HashMap<u32, Transition>,
    /// Background thread for off-main-thread mesh generation.
    pub(crate) scene_processor: SceneProcessor,
    /// Structural animation driver.
    pub(crate) animator: StructureAnimator,
    /// Runtime display, lighting, color, and geometry options.
    pub(crate) options: VisoOptions,
    /// Currently applied options preset name, if any.
    pub(crate) active_preset: Option<String>,
    /// Per-frame timing and FPS tracking.
    pub(crate) frame_timing: FrameTiming,
    /// Multi-frame trajectory player, if loaded.
    pub trajectory_player: Option<TrajectoryPlayer>,

    /// All geometry renderers (backbone, sidechain, band, pull,
    /// ball-and-stick, nucleic acid).
    pub(crate) renderers: Renderers,
    /// GPU picking, selection, and per-residue color buffers.
    pub(crate) pick: PickingSystem,
}

// ── Construction ──

impl VisoEngine {
    /// Engine with a default molecule path.
    ///
    /// # Errors
    ///
    /// Returns [`VisoError`] if GPU initialization
    /// or structure loading fails.
    pub async fn new(
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        size: (u32, u32),
        scale_factor: f64,
    ) -> Result<Self, VisoError> {
        Self::new_with_path(
            window,
            size,
            scale_factor,
            "assets/models/4pnk.cif",
        )
        .await
    }

    /// Engine with a specified molecule path.
    ///
    /// # Errors
    ///
    /// Returns [`VisoError`] if GPU initialization
    /// or structure loading fails.
    pub async fn new_with_path(
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        size: (u32, u32),
        scale_factor: f64,
        cif_path: &str,
    ) -> Result<Self, VisoError> {
        let mut context = RenderContext::new(window, size).await?;

        // 2x supersampling on standard-DPI displays to compensate for low pixel
        // density
        if scale_factor < 2.0 {
            context.render_scale = 2;
        }

        Self::init_with_context(context, cif_path)
    }

    /// Engine from a pre-built [`RenderContext`] (for embedding in dioxus,
    /// headless rendering, etc.).
    ///
    /// Use [`RenderContext::from_device`] to create a surface-less context
    /// from an externally-owned `wgpu::Device` and `wgpu::Queue`.
    ///
    /// # Errors
    ///
    /// Returns [`VisoError`] if structure loading
    /// fails.
    pub fn new_from_context(
        mut context: RenderContext,
        scale_factor: f64,
        cif_path: &str,
    ) -> Result<Self, VisoError> {
        if scale_factor < 2.0 {
            context.render_scale = 2;
        }

        Self::init_with_context(context, cif_path)
    }

    /// Engine with an empty scene (no entities loaded).
    ///
    /// Initializes all GPU resources but starts with no visible geometry.
    /// Entities can be loaded later via [`load_entities`](Self::load_entities)
    /// or [`sync_scene_to_renderers`](Self::sync_scene_to_renderers).
    ///
    /// # Errors
    ///
    /// Returns [`VisoError`] if GPU pipeline initialization fails.
    pub fn new_empty(context: RenderContext) -> Result<Self, VisoError> {
        Self::init_empty(context)
    }

    /// Shared empty-init logic.
    fn init_empty(context: RenderContext) -> Result<Self, VisoError> {
        let render_coords = bootstrap::empty_render_coords();
        let scene = Scene::new();
        let bootstrap =
            gpu_init::init_gpu_pipeline(&context, scene, &render_coords)?;
        let options = VisoOptions::default();
        Self::assemble(context, options, bootstrap, Vec::new())
    }

    /// Shared construction logic for both windowed and headless modes.
    fn init_with_context(
        context: RenderContext,
        cif_path: &str,
    ) -> Result<Self, VisoError> {
        let (scene, render_coords) = bootstrap::load_scene_from_file(cif_path)?;
        let options = VisoOptions::default();

        let mut bootstrap =
            gpu_init::init_gpu_pipeline(&context, scene, &render_coords)?;
        bootstrap.renderers.init_ball_and_stick_entities(
            &context,
            &bootstrap.scene,
            &options,
        );
        let initial_colors = bootstrap::initial_chain_colors(
            &render_coords.backbone_chains,
            render_coords
                .backbone_chains
                .iter()
                .map(|c| c.len() / 3)
                .sum(),
        );
        bootstrap.pick.init_colors_and_groups(
            &context,
            &initial_colors,
            &bootstrap.renderers,
        );
        let positions = bootstrap::collect_all_positions(
            &render_coords,
            &bootstrap.scene,
            &options,
        );
        bootstrap.camera_controller.fit_to_positions(&positions);

        let entities: Vec<MoleculeEntity> = bootstrap
            .scene
            .entities()
            .iter()
            .map(|se| se.entity.clone())
            .collect();
        Self::assemble(context, options, bootstrap, entities)
    }

    /// Build the final `VisoEngine` from initialized GPU subsystems.
    fn assemble(
        context: RenderContext,
        options: VisoOptions,
        bootstrap: GpuBootstrap,
        entities: Vec<MoleculeEntity>,
    ) -> Result<Self, VisoError> {
        Ok(Self {
            context,
            _shader_composer: bootstrap.shader_composer,
            post_process: bootstrap.post_process,
            cursor_pos: (0.0, 0.0),
            last_cull_camera_eye: Vec3::ZERO,
            camera_controller: bootstrap.camera_controller,
            lighting: bootstrap.lighting,
            scene: bootstrap.scene,
            entities,
            entity_behaviors: HashMap::new(),
            pending_transitions: HashMap::new(),
            scene_processor: SceneProcessor::new()
                .map_err(VisoError::ThreadSpawn)?,
            animator: StructureAnimator::new(),
            options,
            active_preset: None,
            frame_timing: FrameTiming::new(TARGET_FPS),
            trajectory_player: None,
            renderers: bootstrap.renderers,
            pick: bootstrap.pick,
        })
    }
}

// ── Frame loop ──

impl VisoEngine {
    /// Per-frame updates: animation ticks, uniform uploads, frustum culling.
    fn pre_render(&mut self) {
        self.apply_pending_animation();
        self.tick_animation();

        // Camera uniform (hover state from GPU picking)
        self.camera_controller.uniform.hovered_residue =
            self.pick.hovered_target.as_residue_i32();
        self.camera_controller.update_gpu(&self.context.queue);

        // Depth-buffer fog from camera distance
        let fog_start = self.camera_controller.distance();
        let fog_density =
            2.0 / self.camera_controller.bounding_radius().max(10.0);
        self.post_process.update_fog(
            &self.context.queue,
            fog_start,
            fog_density,
        );

        self.check_and_submit_lod();
        self.pick.update_selection_buffer(&self.context.queue);
        let _color_transitioning =
            self.pick.residue_colors.update(&self.context.queue);
        self.lighting
            .update_headlamp_from_camera(&self.camera_controller.camera);
        self.lighting.update_gpu(&self.context.queue);
        self.update_frustum_culling();
    }

    /// Tick animation (both trajectory and structural), submitting any
    /// interpolated frame to the background thread.
    ///
    /// Trajectory frames are fed through `animate_entity()` with
    /// `Transition::snap()`, so both paths converge through the
    /// animator's update loop.
    fn tick_animation(&mut self) {
        let now = Instant::now();

        // If a trajectory is active, feed its frame through per-entity
        // animation so it converges with the standard path.
        self.advance_trajectory(now);

        if !self.animator.update(now) {
            return;
        }

        let frame = self.animator.get_frame();
        self.scene.update_visual_state(
            frame.backbone_chains.clone(),
            frame.sidechain_positions.clone().unwrap_or_else(|| {
                self.scene.target_sidechain_positions.clone()
            }),
            frame.backbone_sidechain_bonds.clone().unwrap_or_else(|| {
                self.scene.target_backbone_sidechain_bonds.clone()
            }),
        );

        // Only submit if the previous animation frame has been consumed
        // by the renderer — avoids flooding the background thread.
        if self.scene.is_position_dirty() {
            self.submit_animation_frame_from(&frame);
        }
    }

    /// Core render — geometry, post-process, picking — targeting the given
    /// view. Returns the encoder so the caller can submit it.
    fn render_to_view(
        &mut self,
        view: &wgpu::TextureView,
    ) -> wgpu::CommandEncoder {
        let mut encoder = self.context.create_encoder();

        // Geometry pass
        let input = GeometryPassInput {
            color: self.post_process.color_view(),
            normal: &self.post_process.normal_view,
            depth: &self.post_process.depth_view,
            show_sidechains: self.options.display.show_sidechains,
        };
        let bind_groups = DrawBindGroups {
            camera: &self.camera_controller.bind_group,
            lighting: &self.lighting.bind_group,
            selection: &self.pick.selection.bind_group,
            color: Some(&self.pick.residue_colors.bind_group),
        };
        let frustum = self.camera_controller.frustum();
        self.renderers.encode_geometry_pass(
            &mut encoder,
            &input,
            &bind_groups,
            &frustum,
        );

        // Post-processing: SSAO → bloom → composite → FXAA
        let camera = &self.camera_controller.camera;
        self.post_process.render(
            &mut encoder,
            &self.context.queue,
            &crate::renderer::postprocess::post_process::PostProcessCamera {
                proj: camera.build_projection(),
                view_matrix: Mat4::look_at_rh(
                    camera.eye,
                    camera.target,
                    camera.up,
                ),
                znear: camera.znear,
                zfar: camera.zfar,
            },
            view.clone(),
        );

        // GPU Picking pass
        let picking_geometry = self.pick.build_geometry(
            &self.renderers,
            self.options.display.show_sidechains,
        );
        self.pick.picking.render(
            &mut encoder,
            &self.camera_controller.bind_group,
            &picking_geometry,
            (self.cursor_pos.0 as u32, self.cursor_pos.1 as u32),
        );

        encoder
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

        let frame = self.context.get_next_frame()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let encoder = self.render_to_view(&view);
        self.context.submit(encoder);

        // Start async GPU picking readback (non-blocking)
        self.pick.picking.start_readback();

        // Try to complete any pending readback from previous frame
        // (non-blocking poll)
        self.pick.poll_and_resolve(&self.context.device);

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
        self.context.submit(encoder);
        self.frame_timing.end_frame();
    }

    /// Resize all GPU surfaces and the camera projection to match the new
    /// window size.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.context.resize(width, height);
            self.camera_controller.resize(width, height);
            self.post_process.resize(&self.context);
            self.pick
                .picking
                .resize(&self.context.device, width, height);
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
            // ── Camera ──
            VisoCommand::RecenterCamera => {
                self.fit_camera_to_focus();
                false
            }
            VisoCommand::ToggleAutoRotate => {
                let _ = self.camera_controller.toggle_auto_rotate();
                false
            }
            VisoCommand::RotateCamera { delta } => {
                self.camera_controller.rotate(delta);
                false
            }
            VisoCommand::PanCamera { delta } => {
                self.camera_controller.pan(delta);
                false
            }
            VisoCommand::Zoom { delta } => {
                self.camera_controller.zoom(delta);
                false
            }

            // ── Focus ──
            VisoCommand::CycleFocus => {
                let _ = self.scene.cycle_focus();
                self.fit_camera_to_focus();
                false
            }
            VisoCommand::ResetFocus => {
                self.scene.set_focus(Focus::Session);
                self.fit_camera_to_focus();
                false
            }

            // ── Playback ──
            VisoCommand::ToggleTrajectory => {
                if self.trajectory_player.is_some() {
                    self.toggle_trajectory();
                }
                false
            }

            // ── Selection ──
            VisoCommand::ClearSelection => self.pick.clear_selection(),
            VisoCommand::SelectResidue { index, extend } => {
                self.pick.picking.handle_click(index, extend)
            }
            VisoCommand::SelectSegment { index, extend } => self
                .pick
                .select_segment(index, &self.scene.ss_types, extend),
            VisoCommand::SelectChain { index, extend } => {
                let chains = self.renderers.backbone.cached_chains();
                self.pick.select_chain(index, chains, extend)
            }

            // ── Constraint visualization ──
            VisoCommand::UpdateBands { bands } => {
                self.update_bands(&bands);
                false
            }
            VisoCommand::UpdatePull { pull } => {
                self.update_pull(pull.as_ref());
                false
            }
        }
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
        self.scene_processor.shutdown();
    }
}

// ── Public API: camera ──

impl VisoEngine {
    /// Fit camera to the currently focused element.
    pub fn fit_camera_to_focus(&mut self) {
        match *self.scene.focus() {
            Focus::Session => {
                let positions = self.scene.all_positions();
                if !positions.is_empty() {
                    self.camera_controller
                        .fit_to_positions_animated(&positions);
                }
            }
            Focus::Entity(eid) => {
                if let Some(se) = self.scene.entity(eid) {
                    let positions = se.entity.positions();
                    if !positions.is_empty() {
                        self.camera_controller
                            .fit_to_positions_animated(&positions);
                    }
                }
            }
        }
    }
}

// ── Public API: entity behavior ──

impl VisoEngine {
    /// Set the animation behavior for a specific entity.
    ///
    /// This behavior will be used when the entity is next updated.
    /// Overrides the default smooth transition for the given entity.
    pub fn set_entity_behavior(
        &mut self,
        entity_id: u32,
        transition: Transition,
    ) {
        let _ = self.entity_behaviors.insert(entity_id, transition);
    }

    /// Clear a per-entity behavior override, reverting to default (smooth).
    pub fn clear_entity_behavior(&mut self, entity_id: u32) {
        let _ = self.entity_behaviors.remove(&entity_id);
    }
}

// ── Public API: queries ──

impl VisoEngine {
    /// The pick target currently under the cursor (resolved from the
    /// previous frame's GPU picking pass).
    pub fn hovered_target(&self) -> crate::renderer::picking::PickTarget {
        self.pick.hovered_target
    }

    /// Name of the currently active options preset, if any.
    pub fn active_preset(&self) -> Option<&str> {
        self.active_preset.as_deref()
    }

    /// GPU buffer sizes across all renderers.
    ///
    /// Each entry is `(label, used_bytes, allocated_bytes)`.
    pub fn gpu_buffer_stats(&self) -> Vec<(&str, usize, usize)> {
        let mut stats = Vec::new();
        stats.extend(self.renderers.buffer_info());
        stats.extend(self.pick.selection.buffer_info());
        stats.extend(self.pick.residue_colors.buffer_info());
        stats
    }

    /// Unproject screen coordinates to a world-space point on a plane at
    /// the depth of `reference_point`.
    ///
    /// Uses the current screen size internally — callers do not need to
    /// pass width/height.
    pub fn unproject(
        &self,
        screen_x: f32,
        screen_y: f32,
        reference_point: Vec3,
    ) -> Vec3 {
        let w = self.context.config.width;
        let h = self.context.config.height;
        self.camera_controller.screen_to_world_at_depth(
            glam::Vec2::new(screen_x, screen_y),
            glam::UVec2::new(w, h),
            reference_point,
        )
    }
}

// ── Public API: position queries ──

impl VisoEngine {
    /// Get the CA position of a residue by index.
    /// Returns None if the residue index is out of bounds.
    pub fn get_residue_ca_position(&self, residue_idx: usize) -> Option<Vec3> {
        // First try animator (has interpolated positions during animation)
        if let Some(pos) = self.animator.get_ca_position(residue_idx) {
            return Some(pos);
        }

        // Fall back to backbone_renderer's cached chains
        foldit_conv::ops::transform::get_ca_position_from_chains(
            self.renderers.backbone.cached_chains(),
            residue_idx,
        )
    }

    /// Get the current visual backbone chains (interpolated during animation).
    pub fn get_current_backbone_chains(&self) -> Vec<Vec<Vec3>> {
        if self.scene.visual_backbone_chains.is_empty() {
            self.renderers.backbone.cached_chains().to_vec()
        } else {
            self.scene.visual_backbone_chains.clone()
        }
    }

    /// Get the current visual sidechain positions (interpolated during
    /// animation).
    pub fn get_current_sidechain_positions(&self) -> Vec<Vec3> {
        if self.scene.visual_sidechain_positions.is_empty() {
            self.scene.target_sidechain_positions.clone()
        } else {
            self.scene.visual_sidechain_positions.clone()
        }
    }

    /// Get the current visual CA positions for all residues (interpolated
    /// during animation).
    pub fn get_current_ca_positions(&self) -> Vec<Vec3> {
        let chains = self.get_current_backbone_chains();
        foldit_conv::ops::transform::extract_ca_from_chains(&chains)
    }

    /// Get a single interpolated CA position by residue index.
    pub fn get_current_ca_position(&self, residue_idx: usize) -> Option<Vec3> {
        if let Some(pos) = self.animator.get_ca_position(residue_idx) {
            return Some(pos);
        }
        foldit_conv::ops::transform::get_ca_position_from_chains(
            self.renderers.backbone.cached_chains(),
            residue_idx,
        )
    }

    /// Get the interpolated position of the closest atom to a reference point
    /// for a given residue.
    pub fn get_closest_atom_for_residue(
        &self,
        residue_idx: usize,
        reference_point: Vec3,
    ) -> Option<Vec3> {
        let backbone_chains = self.get_current_backbone_chains();
        let sidechain_positions = self.get_current_sidechain_positions();

        foldit_conv::ops::transform::get_closest_atom_for_residue(
            &backbone_chains,
            &sidechain_positions,
            &self.scene.sidechain_residue_indices,
            residue_idx,
            reference_point,
        )
    }

    /// Get the closest atom position and name for a given residue relative to
    /// a reference point. Returns both backbone and sidechain atoms.
    pub fn get_closest_atom_with_name(
        &self,
        residue_idx: usize,
        reference_point: Vec3,
    ) -> Option<(Vec3, String)> {
        let backbone_chains = self.get_current_backbone_chains();
        let sidechain_positions = self.get_current_sidechain_positions();

        foldit_conv::ops::transform::get_closest_atom_with_name(
            &backbone_chains,
            &sidechain_positions,
            &self.scene.sidechain_residue_indices,
            &self.scene.sidechain_atom_names,
            residue_idx,
            reference_point,
        )
    }

    /// Get the interpolated position of a specific atom by residue index and
    /// atom name.
    pub fn get_atom_position_by_name(
        &self,
        residue_idx: usize,
        atom_name: &str,
    ) -> Option<Vec3> {
        // Check backbone atoms first (N, CA, C)
        if atom_name == "N" || atom_name == "CA" || atom_name == "C" {
            let backbone_chains = self.get_current_backbone_chains();
            let offset = match atom_name {
                "N" => 0,
                "CA" => 1,
                "C" => 2,
                _ => return None,
            };

            let mut current_idx = 0;
            for chain in &backbone_chains {
                let residues_in_chain = chain.len() / 3;
                if residue_idx < current_idx + residues_in_chain {
                    let local_idx = residue_idx - current_idx;
                    let atom_idx = local_idx * 3 + offset;
                    return chain.get(atom_idx).copied();
                }
                current_idx += residues_in_chain;
            }
            return None;
        }

        // Check sidechain atoms
        let sidechain_positions = self.get_current_sidechain_positions();
        for (i, (res_idx, name)) in self
            .scene
            .sidechain_residue_indices
            .iter()
            .zip(self.scene.sidechain_atom_names.iter())
            .enumerate()
        {
            if *res_idx as usize == residue_idx && name == atom_name {
                return sidechain_positions.get(i).copied();
            }
        }

        None
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
        let display_changed = old.display != new.display;
        let color_mode_changed =
            old.display.backbone_color_mode != new.display.backbone_color_mode;
        let geometry_changed = old.geometry != new.geometry;
        let colors_changed = old.colors != new.colors;

        self.options = new;

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
        if display_changed || geometry_changed || colors_changed {
            self.scene.force_dirty();
        }
        if display_changed {
            self.refresh_ball_and_stick();
            if color_mode_changed {
                self.recompute_backbone_colors();
            }
        }
        if geometry_changed {
            let camera_eye = self.camera_controller.camera.eye;
            self.submit_per_chain_lod_remesh(camera_eye);
        }
        if colors_changed {
            self.refresh_ball_and_stick();
        }
    }

    /// Force-refresh all subsystems from current options (escape hatch).
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
        self.recompute_backbone_colors();
    }

    /// Load a named view preset from the presets directory.
    /// Returns true on success.
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
        self.context.render_scale = scale;
    }
}

// ── Public API: scene data ──

impl VisoEngine {
    /// Load entities into the scene. Optionally fits camera.
    /// Returns the assigned entity IDs.
    pub fn load_entities(
        &mut self,
        entities: Vec<MoleculeEntity>,
        fit_camera: bool,
    ) -> Vec<u32> {
        // Store canonical copy on the engine (source of truth)
        self.entities.clone_from(&entities);

        let ids = self.scene.add_entities(entities);
        if fit_camera {
            // Sync immediately so entity data is available for camera fit
            let snap_transitions: HashMap<u32, Transition> =
                ids.iter().map(|&id| (id, Transition::snap())).collect();
            self.sync_scene_to_renderers(snap_transitions);
            let positions = self.scene.all_positions();
            if !positions.is_empty() {
                self.camera_controller.fit_to_positions(&positions);
            }
        }
        ids
    }

    /// Update backbone with new chains (regenerates the backbone mesh)
    /// Use this for designed backbones from ML models like RFDiffusion3
    pub fn update_backbone(&mut self, backbone_chains: &[Vec<Vec3>]) {
        self.renderers.backbone.update(
            &self.context,
            &BackboneUpdateData {
                protein_chains: backbone_chains,
                na_chains: &[],
                ss_types: None,
                geometry: &self.options.geometry,
            },
        );
    }

    /// Set SS override (from puzzle.toml annotation). Updates cached types
    /// and forces backbone renderer regeneration.
    pub fn set_ss_override(&mut self, ss_types: &[SSType]) {
        self.scene.ss_types = ss_types.to_vec();
        self.renderers
            .backbone
            .set_ss_override(Some(ss_types.to_vec()));
        let camera_eye = self.camera_controller.camera.eye;
        self.submit_per_chain_lod_remesh(camera_eye);
    }

    /// Update the band visualization.
    /// Call this when bands are added, removed, or modified.
    pub fn update_bands(&mut self, bands: &[BandInfo]) {
        self.renderers.band.update(
            &self.context.device,
            &self.context.queue,
            bands,
            Some(&self.options.colors),
        );
    }

    /// Update the pull visualization (only one pull at a time).
    /// Pass None to clear the pull visualization.
    pub fn update_pull(&mut self, pull: Option<&PullInfo>) {
        self.renderers.pull.update(
            &self.context.device,
            &self.context.queue,
            pull,
        );
    }
}

// ── Public API: entity updates ──

impl VisoEngine {
    /// Update protein coords for a specific entity.
    ///
    /// Updates the engine's source-of-truth entities first, then derives
    /// Scene from them. Uses the entity's per-entity behavior override if
    /// set, otherwise falls back to the provided transition.
    pub fn update_entity_coords(
        &mut self,
        id: u32,
        coords: foldit_conv::types::coords::Coords,
        transition: Transition,
    ) {
        // 1. Update source-of-truth on the engine
        if let Some(entity) =
            self.entities.iter_mut().find(|e| e.entity_id == id)
        {
            let mut entities = vec![entity.clone()];
            foldit_conv::types::assembly::update_protein_entities(
                &mut entities,
                coords.clone(),
            );
            if let Some(updated) = entities.into_iter().next() {
                *entity = updated;
            }
        }

        // 2. Update Scene (derived copy)
        self.scene.update_entity_protein_coords(id, coords);

        // 3. Look up per-entity behavior override
        let effective_transition = self
            .entity_behaviors
            .get(&id)
            .cloned()
            .unwrap_or(transition);

        // 4. Sync with per-entity transition
        let mut entity_transitions = HashMap::new();
        let _ = entity_transitions.insert(id, effective_transition);
        self.sync_scene_to_renderers(entity_transitions);
    }

    /// Replace one or more entities with new `MoleculeEntity` data.
    ///
    /// Each entity is matched by `entity_id`. The engine's source-of-truth
    /// and Scene are both updated, then a targeted sync is triggered for all
    /// changed entities. Per-entity behavior overrides are used when set,
    /// otherwise `default_transition` is applied.
    pub fn update_entities(
        &mut self,
        updated: Vec<MoleculeEntity>,
        default_transition: &Transition,
    ) {
        let mut entity_transitions = HashMap::new();

        for new_entity in updated {
            let id = new_entity.entity_id;

            // Update engine source-of-truth
            if let Some(slot) =
                self.entities.iter_mut().find(|e| e.entity_id == id)
            {
                *slot = new_entity.clone();
            }

            // Update Scene (derived copy)
            self.scene.replace_entity(new_entity);

            // Resolve per-entity behavior override
            let transition = self
                .entity_behaviors
                .get(&id)
                .cloned()
                .unwrap_or_else(|| default_transition.clone());
            let _ = entity_transitions.insert(id, transition);
        }

        if !entity_transitions.is_empty() {
            self.sync_scene_to_renderers(entity_transitions);
        }
    }
}

// ── Public API: trajectory ──

impl VisoEngine {
    /// Load a DCD trajectory file and begin playback.
    pub fn load_trajectory(&mut self, path: &Path) {
        use foldit_conv::adapters::dcd::dcd_file_to_frames;
        use foldit_conv::ops::transform::protein_only;

        use self::trajectory::build_backbone_atom_indices;

        let (header, frames) = match dcd_file_to_frames(path) {
            Ok(r) => r,
            Err(e) => {
                log::error!("Failed to load DCD trajectory: {e}");
                return;
            }
        };

        // Get protein coords from the first visible entity to build backbone
        // mapping
        let protein_coords = self
            .scene
            .entities()
            .iter()
            .filter(|e| e.visible)
            .find_map(SceneEntity::protein_coords);

        let protein_coords = if let Some(c) = protein_coords {
            protein_only(&c)
        } else {
            log::error!("No protein structure loaded — cannot play trajectory");
            return;
        };

        // Validate atom count
        if (header.num_atoms as usize) < protein_coords.num_atoms {
            log::error!(
                "DCD atom count ({}) is less than protein atom count ({})",
                header.num_atoms,
                protein_coords.num_atoms,
            );
            return;
        }

        // Build backbone atom index mapping
        let backbone_indices = build_backbone_atom_indices(&protein_coords);

        // Get current backbone chains for topology
        let backbone_chains =
            foldit_conv::ops::transform::extract_backbone_chains(
                &protein_coords,
            );

        let num_atoms = header.num_atoms as usize;
        let num_frames = frames.len();
        let duration_secs = num_frames as f64 / 30.0;

        let player = TrajectoryPlayer::new(
            frames,
            num_atoms,
            &backbone_chains,
            backbone_indices,
        );
        self.trajectory_player = Some(player);

        log::info!(
            "Trajectory loaded: {num_frames} frames, {num_atoms} atoms, \
             ~{duration_secs:.1}s at 30fps",
        );
    }

    /// Toggle trajectory playback (play/pause). No-op if no trajectory loaded.
    pub fn toggle_trajectory(&mut self) {
        if let Some(ref mut player) = self.trajectory_player {
            player.toggle_playback();
            let state = if player.is_playing() {
                "playing"
            } else {
                "paused"
            };
            log::info!(
                "Trajectory {state} (frame {}/{})",
                player.current_frame(),
                player.total_frames()
            );
        }
    }
}

// ── Internals: options application ──

impl VisoEngine {
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

    /// Recompute backbone per-residue colors from current options and
    /// push them to the GPU color buffer.
    fn recompute_backbone_colors(&mut self) {
        let chains = self.renderers.backbone.cached_chains().to_vec();
        let per_entity_scores: Vec<Option<&[f64]>> = self
            .scene
            .entities()
            .iter()
            .map(|e| e.per_residue_scores.as_deref())
            .collect();
        let new_colors = score_color::compute_per_residue_colors(
            &chains,
            &self.scene.ss_types,
            &per_entity_scores,
            &self.options.display.backbone_color_mode,
        );
        self.pick.residue_colors.set_target_colors(&new_colors);
        self.scene.per_residue_colors = Some(new_colors);
    }

    /// Refresh ball-and-stick renderer with current visibility flags.
    pub(crate) fn refresh_ball_and_stick(&mut self) {
        // Collect all ligand entities (not protein, not nucleic acid)
        let entities: Vec<MoleculeEntity> = self
            .scene
            .ligand_entities()
            .iter()
            .map(|se| se.entity.clone())
            .collect();
        self.renderers.ball_and_stick.update_from_entities(
            &self.context,
            &entities,
            &self.options.display,
            Some(&self.options.colors),
        );
        // Recreate picking bind groups
        self.pick.groups.rebuild_bns_bond(
            &self.pick.picking,
            &self.context.device,
            &self.renderers.ball_and_stick,
        );
        self.pick.groups.rebuild_bns_sphere(
            &self.pick.picking,
            &self.context.device,
            &self.renderers.ball_and_stick,
        );
    }
}

// ── Internals: scene sync ──

impl VisoEngine {
    /// Compute metadata from entities and store on Scene before background
    /// submission. Returns the per-entity data and entity transitions.
    fn prepare_scene_metadata(
        &mut self,
        entity_transitions: HashMap<u32, Transition>,
    ) -> (Vec<crate::scene::PerEntityData>, HashMap<u32, Transition>) {
        let mut entities = self.scene.per_entity_data();

        // Compute entity residue ranges on main thread
        let ranges = crate::scene::compute_entity_residue_ranges(&entities);
        self.scene.set_entity_residue_ranges(ranges.clone());

        // Compute concatenated sidechain topology on main thread
        let sidechain =
            crate::scene::concatenate_sidechain_atoms(&entities, &ranges);
        self.scene.update_sidechain_topology(&sidechain);

        // Compute concatenated SS types on main thread
        self.scene.ss_types =
            crate::scene::concatenate_ss_types(&entities, &ranges);

        // Concatenate backbone and NA chains on main thread
        let backbone_chains: Vec<Vec<Vec3>> = entities
            .iter()
            .flat_map(|e| e.backbone_chains.iter().cloned())
            .collect();
        let na_chains: Vec<Vec<Vec3>> = entities
            .iter()
            .flat_map(|e| e.nucleic_acid_chains.iter().cloned())
            .collect();
        // Store on Scene for use by apply_pending_scene / animation
        self.scene
            .visual_backbone_chains
            .clone_from(&backbone_chains);
        self.scene.na_chains = na_chains;

        // Compute per-residue colors on main thread and distribute to
        // entities for vertex coloring (avoids background round-trip)
        let per_entity_scores: Vec<Option<&[f64]>> = self
            .scene
            .entities()
            .iter()
            .map(|e| e.per_residue_scores.as_deref())
            .collect();
        let colors = score_color::compute_per_residue_colors(
            &backbone_chains,
            &self.scene.ss_types,
            &per_entity_scores,
            &self.options.display.backbone_color_mode,
        );
        for (e, range) in entities.iter_mut().zip(&ranges) {
            let start = range.start as usize;
            let end = range.end() as usize;
            e.per_residue_colors = colors.get(start..end).map(<[_]>::to_vec);
        }
        self.scene.per_residue_colors = Some(colors);

        (entities, entity_transitions)
    }

    /// Sync scene data to renderers with per-entity transitions.
    ///
    /// Entities in the map animate with their transition; entities not in
    /// the map snap. Pass an empty map for a non-animated sync.
    pub fn sync_scene_to_renderers(
        &mut self,
        entity_transitions: HashMap<u32, Transition>,
    ) {
        if !self.scene.is_dirty() && entity_transitions.is_empty() {
            return;
        }

        let (entities, transitions) =
            self.prepare_scene_metadata(entity_transitions);
        self.pending_transitions = transitions;
        self.scene.mark_rendered();

        self.scene_processor.submit(SceneRequest::FullRebuild {
            entities,
            display: self.options.display.clone(),
            colors: self.options.colors.clone(),
            geometry: self.options.geometry.clone(),
        });
    }

    /// Upload prepared scene geometry to GPU renderers.
    fn upload_prepared_to_gpu(
        &mut self,
        prepared: &PreparedScene,
        animating: bool,
        suppress_sidechains: bool,
    ) {
        let ss_types = if self.scene.ss_types.is_empty() {
            None
        } else {
            Some(self.scene.ss_types.clone())
        };
        let backbone_chains = self.scene.visual_backbone_chains.clone();
        let na_chains = self.scene.na_chains.clone();

        if animating {
            self.renderers.backbone.update_metadata(
                backbone_chains,
                na_chains,
                ss_types,
            );
        } else {
            self.renderers.backbone.apply_prepared(
                &self.context.device,
                &self.context.queue,
                PreparedBackboneData {
                    vertices: &prepared.backbone.vertices,
                    tube_indices: &prepared.backbone.tube_indices,
                    ribbon_indices: &prepared.backbone.ribbon_indices,
                    tube_index_count: prepared.backbone.tube_index_count,
                    ribbon_index_count: prepared.backbone.ribbon_index_count,
                    sheet_offsets: prepared.backbone.sheet_offsets.clone(),
                    chain_ranges: prepared.backbone.chain_ranges.clone(),
                    cached_chains: backbone_chains,
                    cached_na_chains: na_chains,
                    ss_override: ss_types,
                },
            );
            if !suppress_sidechains {
                let _ = self.renderers.sidechain.apply_prepared(
                    &self.context.device,
                    &self.context.queue,
                    &prepared.sidechain_instances,
                    prepared.sidechain_instance_count,
                );
            }
        }
        self.upload_non_backbone(prepared);
    }

    /// Upload BnS, NA, and pick data (shared by animating and non-animating).
    fn upload_non_backbone(&mut self, prepared: &PreparedScene) {
        self.renderers.ball_and_stick.apply_prepared(
            &self.context.device,
            &self.context.queue,
            &PreparedBallAndStickData {
                sphere_bytes: &prepared.bns.sphere_instances,
                sphere_count: prepared.bns.sphere_count,
                capsule_bytes: &prepared.bns.capsule_instances,
                capsule_count: prepared.bns.capsule_count,
            },
        );
        self.renderers.nucleic_acid.apply_prepared(
            &self.context.device,
            &self.context.queue,
            &prepared.na,
        );
        self.pick.pick_map = Some(prepared.pick_map.clone());
        self.pick.groups.rebuild_all(
            &self.pick.picking,
            &self.context.device,
            &self.renderers.sidechain,
            &self.renderers.ball_and_stick,
        );
    }

    /// Apply any pending scene data from the background SceneProcessor.
    ///
    /// Called every frame from the main loop. If the background thread has
    /// finished generating geometry, this uploads it to the GPU (<1ms) and
    /// sets up animation.
    pub fn apply_pending_scene(&mut self) {
        let Some(prepared) = self.scene_processor.try_recv_scene() else {
            return;
        };

        let entity_transitions = std::mem::take(&mut self.pending_transitions);
        let animating = !entity_transitions.is_empty();

        if animating {
            self.setup_per_entity_animation(&entity_transitions);
            let frame = self.animator.get_frame();
            self.submit_animation_frame_from(&frame);
        } else {
            self.snap_from_prepared();
        }

        let suppress_sidechains = entity_transitions
            .values()
            .any(|t| t.suppress_initial_sidechains);
        self.upload_prepared_to_gpu(&prepared, animating, suppress_sidechains);
    }

    /// Snap update from prepared scene data (no animation).
    ///
    /// All metadata (backbone chains, sidechain topology, SS types, colors)
    /// is already on Scene from `prepare_scene_metadata`. This just sets
    /// visual state and ensures GPU buffer capacity.
    fn snap_from_prepared(&mut self) {
        // Write full at-rest visual state to Scene (backbone chains
        // already set in prepare_scene_metadata; sidechain topology too)
        self.scene.update_visual_state(
            self.scene.visual_backbone_chains.clone(),
            self.scene.target_sidechain_positions.clone(),
            self.scene.target_backbone_sidechain_bonds.clone(),
        );

        let total_residues =
            crate::renderer::geometry::sheet_adjust::backbone_residue_count(
                &self.scene.visual_backbone_chains,
            );
        self.pick
            .selection
            .ensure_capacity(&self.context.device, total_residues);
        self.pick
            .residue_colors
            .ensure_capacity(&self.context.device, total_residues);

        // Colors already computed in prepare_scene_metadata
        if let Some(ref colors) = self.scene.per_residue_colors {
            self.pick
                .residue_colors
                .set_colors_immediate(&self.context.queue, colors);
        }
    }

    /// Apply any pending animation frame from the background thread.
    pub(crate) fn apply_pending_animation(&mut self) {
        let Some(prepared) = self.scene_processor.try_recv_animation() else {
            return;
        };

        self.renderers.backbone.apply_mesh(
            &self.context.device,
            &self.context.queue,
            prepared.backbone,
        );

        if let Some(ref instances) = prepared.sidechain_instances {
            let reallocated = self.renderers.sidechain.apply_prepared(
                &self.context.device,
                &self.context.queue,
                instances,
                prepared.sidechain_instance_count,
            );
            if reallocated {
                self.pick.groups.rebuild_capsule(
                    &self.pick.picking,
                    &self.context.device,
                    &self.renderers.sidechain,
                );
            }
        }

        self.scene.mark_position_rendered();
    }
}

// ── Internals: animation setup ──

impl VisoEngine {
    /// Set up per-entity animation from prepared scene data.
    ///
    /// For each entity with a transition, dispatches to
    /// `animator.animate_entity()` so each entity gets its own runner.
    /// Entities without transitions are not animated.
    fn setup_per_entity_animation(
        &mut self,
        entity_transitions: &HashMap<u32, Transition>,
    ) {
        let new_backbone = self.scene.visual_backbone_chains.clone();

        // Read sidechain data from Scene (already computed on main thread
        // in prepare_scene_metadata)
        let ca_positions =
            foldit_conv::render::backbone::ca_positions_from_chains(
                &new_backbone,
            );
        let sidechain_positions = self.scene.target_sidechain_positions.clone();
        let sidechain_residue_indices =
            self.scene.sidechain_residue_indices.clone();
        let sidechain_backbone_bonds =
            self.scene.target_backbone_sidechain_bonds.clone();

        // Set global sidechain residue indices on animator (once per scene
        // update) so compute_interpolated_bonds() can resolve CB → residue.
        self.animator
            .set_sidechain_residue_indices(sidechain_residue_indices.clone());

        // Dispatch per-entity animation with sidechain data
        for range in &self.scene.entity_residue_ranges.clone() {
            let transition = entity_transitions
                .get(&range.entity_id)
                .cloned()
                .unwrap_or_default();

            let positions = crate::scene::extract_entity_sidechain(
                &sidechain_positions,
                &sidechain_residue_indices,
                &ca_positions,
                range,
                entity_transitions.get(&range.entity_id),
            );

            let backbone_bonds = crate::scene::extract_entity_backbone_bonds(
                &sidechain_backbone_bonds,
                &sidechain_residue_indices,
                range,
            );

            self.animator.animate_entity(
                range,
                &new_backbone,
                &transition,
                EntitySidechainData {
                    positions,
                    backbone_bonds,
                },
            );
        }

        // Ensure selection/color buffers have capacity and update colors
        let total_residues =
            crate::renderer::geometry::sheet_adjust::backbone_residue_count(
                &new_backbone,
            );
        self.pick
            .selection
            .ensure_capacity(&self.context.device, total_residues);
        self.pick
            .residue_colors
            .ensure_capacity(&self.context.device, total_residues);

        // Animate colors to new target (already computed in
        // prepare_scene_metadata)
        if let Some(ref colors) = self.scene.per_residue_colors {
            self.pick.residue_colors.set_target_colors(colors);
        }
    }

    /// Submit an animation frame to the background thread for mesh
    /// generation, using a unified [`AnimationFrame`] from the animator.
    pub(crate) fn submit_animation_frame_from(&self, frame: &AnimationFrame) {
        let has_sc = !self.scene.target_sidechain_positions.is_empty()
            && frame.sidechains_visible;

        let sidechains = if has_sc {
            let positions = frame
                .sidechain_positions
                .as_deref()
                .unwrap_or(&self.scene.target_sidechain_positions);
            let bonds = frame
                .backbone_sidechain_bonds
                .as_deref()
                .unwrap_or(&self.scene.target_backbone_sidechain_bonds);
            Some(self.scene.to_interpolated_sidechain_atoms(positions, bonds))
        } else {
            None
        };

        self.scene_processor.submit(SceneRequest::AnimationFrame {
            backbone_chains: frame.backbone_chains.clone(),
            na_chains: None,
            sidechains,
            ss_types: None,
            per_residue_colors: None,
            geometry: self.options.geometry.clone(),
            per_chain_lod: None,
        });
    }

    /// Feed the current trajectory frame (if any) through per-entity
    /// animation with `Transition::snap()`.
    fn advance_trajectory(&mut self, now: Instant) {
        let Some(ref mut player) = self.trajectory_player else {
            return;
        };
        let Some(backbone_chains) = player.tick(now) else {
            return;
        };

        let snap = Transition::snap();
        for range in &self.scene.entity_residue_ranges {
            self.animator.animate_entity(
                range,
                &backbone_chains,
                &snap,
                EntitySidechainData {
                    positions: None,
                    backbone_bonds: Vec::new(),
                },
            );
        }
    }
}

// ── Internals: frustum + LOD ──

impl VisoEngine {
    /// Update sidechain instances with frustum culling when camera moves
    /// significantly. This filters out sidechains behind the camera to
    /// reduce draw calls.
    pub(crate) fn update_frustum_culling(&mut self) {
        // Skip if no sidechain data
        if self.scene.target_sidechain_positions.is_empty() {
            return;
        }

        // Only update culling when camera moves more than 5 units.
        // Exception: always update during animation so sidechain positions
        // reflect the interpolated state.
        if !self.should_update_culling() {
            return;
        }

        let camera_eye = self.camera_controller.camera.eye;
        self.last_cull_camera_eye = camera_eye;

        let frustum = self.camera_controller.frustum();
        // Read visual state from Scene (populated by tick_animation or
        // snap_from_prepared).
        let positions = if self.scene.visual_sidechain_positions.is_empty() {
            &self.scene.target_sidechain_positions
        } else {
            &self.scene.visual_sidechain_positions
        };
        let bs_bonds = if self.scene.visual_backbone_sidechain_bonds.is_empty()
        {
            self.scene.target_backbone_sidechain_bonds.clone()
        } else {
            self.scene.visual_backbone_sidechain_bonds.clone()
        };

        // Translate sidechains onto sheet surface and apply frustum culling
        let offset_map = self.sheet_offset_map();
        let raw_view = SidechainView {
            positions,
            bonds: &self.scene.sidechain_bonds,
            backbone_bonds: &bs_bonds,
            hydrophobicity: &self.scene.sidechain_hydrophobicity,
            residue_indices: &self.scene.sidechain_residue_indices,
        };
        let adjusted =
            crate::renderer::geometry::sheet_adjust::sheet_adjusted_view(
                &raw_view,
                &offset_map,
            );

        self.renderers.sidechain.update_with_frustum(
            &self.context.device,
            &self.context.queue,
            &adjusted.as_view(),
            Some(&frustum),
        );

        // Recreate picking bind group since buffer may have changed
        self.pick.groups.rebuild_capsule(
            &self.pick.picking,
            &self.context.device,
            &self.renderers.sidechain,
        );
    }

    /// Whether frustum culling should be recalculated this frame.
    ///
    /// Returns `true` when the camera has moved more than the threshold or
    /// an animation with sidechain data is active (positions are
    /// interpolated and need continuous updates).
    fn should_update_culling(&self) -> bool {
        const CULL_UPDATE_THRESHOLD: f32 = 5.0;

        let animating = self.animator.is_animating()
            && !self.scene.target_sidechain_positions.is_empty();
        if animating {
            return true;
        }

        let camera_eye = self.camera_controller.camera.eye;
        let camera_delta = (camera_eye - self.last_cull_camera_eye).length();
        camera_delta >= CULL_UPDATE_THRESHOLD
    }

    /// Check per-chain LOD tiers and submit a background remesh if any
    /// chain's tier has changed.
    pub(crate) fn check_and_submit_lod(&mut self) {
        let camera_eye = self.camera_controller.camera.eye;
        let per_chain_tiers: Vec<u8> = self
            .renderers
            .backbone
            .chain_ranges()
            .iter()
            .map(|r| {
                crate::options::select_chain_lod_tier(
                    r.bounding_center,
                    camera_eye,
                )
            })
            .collect();
        if per_chain_tiers != self.renderers.backbone.cached_lod_tiers() {
            self.renderers
                .backbone
                .set_cached_lod_tiers(per_chain_tiers);
            self.submit_per_chain_lod_remesh(camera_eye);
        }
    }

    /// Submit a backbone-only remesh with per-chain LOD to the background
    /// thread. Each chain gets its own `(spr, csv)` based on its distance
    /// from the camera. No sidechains — they don't change with LOD.
    pub(crate) fn submit_per_chain_lod_remesh(&self, camera_eye: Vec3) {
        use crate::options::{lod_scaled, select_chain_lod_tier};

        // Use clamped geometry as the base for LOD scaling
        let total_residues =
            crate::renderer::geometry::sheet_adjust::backbone_residue_count(
                self.renderers.backbone.cached_chains(),
            ) + self
                .renderers
                .backbone
                .cached_na_chains()
                .iter()
                .map(Vec::len)
                .sum::<usize>();
        let base_geo =
            self.options.geometry.clamped_for_residues(total_residues);
        let max_spr = base_geo.segments_per_residue;
        let max_csv = base_geo.cross_section_verts;

        let per_chain_lod: Vec<(usize, usize)> = self
            .renderers
            .backbone
            .chain_ranges()
            .iter()
            .map(|r| {
                let tier = select_chain_lod_tier(r.bounding_center, camera_eye);
                lod_scaled(max_spr, max_csv, tier)
            })
            .collect();

        self.scene_processor.submit(SceneRequest::AnimationFrame {
            backbone_chains: self.renderers.backbone.cached_chains().to_vec(),
            na_chains: None,
            sidechains: None,
            ss_types: None,
            per_residue_colors: None,
            geometry: base_geo,
            per_chain_lod: Some(per_chain_lod),
        });
    }

    /// Build a map of sheet residue offsets (residue_idx -> offset vector).
    pub(crate) fn sheet_offset_map(&self) -> HashMap<u32, Vec3> {
        self.renderers
            .backbone
            .sheet_offsets()
            .iter()
            .copied()
            .collect()
    }
}
