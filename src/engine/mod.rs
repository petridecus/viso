mod bootstrap;
/// The engine's complete interactive vocabulary.
pub mod command;
mod entity;
pub(crate) mod entity_store;
mod options_apply;
pub(crate) mod scene;
/// Entity data types, bond topology, and scene aggregation functions.
pub(crate) mod scene_data;
mod sync;
pub(crate) mod trajectory;

use std::path::Path;
use std::time::{Duration, Instant};

use entity_store::EntityStore;
use foldit_conv::render::RenderCoords;
use glam::{Mat4, Vec3};
use scene::{Focus, SceneTopology, VisualState};
use scene_data::SceneEntity;

use crate::animation::AnimationState;
use crate::camera::controller::CameraController;
use crate::error::VisoError;
use crate::gpu::lighting::Lighting;
use crate::gpu::residue_color::ResidueColorBuffer;
use crate::gpu::{RenderContext, ShaderComposer};
use crate::options::VisoOptions;
use crate::renderer::draw_context::DrawBindGroups;
use crate::renderer::picking::{PickingSystem, SelectionBuffer};
use crate::renderer::pipeline::SceneProcessor;
use crate::renderer::postprocess::PostProcessStack;
use crate::renderer::{GeometryPassInput, PipelineLayouts, Renderers};

/// Target FPS limit
const TARGET_FPS: u32 = 300;

// ---------------------------------------------------------------------------
// GpuPipeline
// ---------------------------------------------------------------------------

/// All GPU infrastructure grouped together: device/queue, renderers,
/// picking, background mesh processor, and post-processing.
pub(crate) struct GpuPipeline {
    /// Core wgpu device, queue, and surface.
    pub context: RenderContext,
    /// All geometry renderers (backbone, sidechain, band, pull,
    /// ball-and-stick, nucleic acid).
    pub renderers: Renderers,
    /// GPU picking, selection, and per-residue color buffers.
    pub pick: PickingSystem,
    /// Background thread for off-main-thread mesh generation.
    pub scene_processor: SceneProcessor,
    /// Post-processing pass stack (SSAO, bloom, composite, FXAA).
    pub post_process: PostProcessStack,
    /// Retained so compiled shader modules stay alive for the engine lifetime.
    #[allow(dead_code)]
    shader_composer: ShaderComposer,
}

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

// ---------------------------------------------------------------------------
// GpuBootstrap — intermediate state for GPU pipeline initialization
// ---------------------------------------------------------------------------

/// Intermediate state holding all initialized GPU subsystems.
///
/// Produced by [`init_gpu_pipeline`] and consumed by
/// [`VisoEngine::assemble`] to build the final engine struct.
struct GpuBootstrap {
    shader_composer: ShaderComposer,
    camera_controller: CameraController,
    lighting: Lighting,
    renderers: Renderers,
    pick: PickingSystem,
    post_process: PostProcessStack,
    entities: EntityStore,
}

/// Initialize all shared GPU subsystems from entity data and render coords.
///
/// This is the common pipeline setup for both empty and loaded constructors.
fn init_gpu_pipeline(
    context: &RenderContext,
    entities: EntityStore,
    render_coords: &RenderCoords,
) -> Result<GpuBootstrap, VisoError> {
    let mut shader_composer = ShaderComposer::new()?;
    let mut camera_controller = CameraController::new(context);
    let lighting = Lighting::new(context);

    let n = render_coords.residue_count().max(1);
    let selection = SelectionBuffer::new(&context.device, n);
    let residue_colors = ResidueColorBuffer::new(&context.device, n);
    let layouts = PipelineLayouts {
        camera: camera_controller.layout.clone(),
        lighting: lighting.layout.clone(),
        selection: selection.layout.clone(),
        color: residue_colors.layout.clone(),
    };
    let renderers = Renderers::new(
        context,
        &layouts,
        render_coords,
        &entities,
        &mut shader_composer,
    )?;
    let pick = PickingSystem::new(
        context,
        &camera_controller.layout,
        selection,
        residue_colors,
        &mut shader_composer,
    )?;
    let post_process = PostProcessStack::new(context, &mut shader_composer)?;
    camera_controller.fit_to_sphere(Vec3::ZERO, 0.0);

    Ok(GpuBootstrap {
        shader_composer,
        camera_controller,
        lighting,
        renderers,
        pick,
        post_process,
        entities,
    })
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
    /// All GPU infrastructure (device, renderers, picking, post-process).
    pub(crate) gpu: GpuPipeline,

    /// Current cursor position in physical pixels (set by the viewer /
    /// input processor each frame for GPU picking).
    pub(crate) cursor_pos: (f32, f32),
    /// Camera eye position at the last frustum-culling update.
    pub(crate) last_cull_camera_eye: Vec3,

    /// Orbital camera controller.
    pub camera_controller: CameraController,
    /// GPU lighting uniform and bind group.
    pub lighting: Lighting,
    /// Derived topology (SS types, residue ranges, sidechain topology).
    pub(crate) topology: SceneTopology,
    /// Animation output buffer (interpolated positions).
    pub(crate) visual: VisualState,
    /// Consolidated entity ownership (source + scene entities + behaviors).
    pub(crate) entities: EntityStore,
    /// Stored band constraint specs (resolved to world-space each frame).
    pub(crate) band_specs: Vec<command::BandInfo>,
    /// Stored pull constraint spec (resolved to world-space each frame).
    pub(crate) pull_spec: Option<command::PullInfo>,
    /// Structural animation, trajectory, and pending transitions.
    pub(crate) animation: AnimationState,
    /// Runtime display, lighting, color, and geometry options.
    pub(crate) options: VisoOptions,
    /// Currently applied options preset name, if any.
    pub(crate) active_preset: Option<String>,
    /// Per-frame timing and FPS tracking.
    pub(crate) frame_timing: FrameTiming,
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
        let entities = EntityStore::new();
        let bootstrap = init_gpu_pipeline(&context, entities, &render_coords)?;
        let options = VisoOptions::default();
        Self::assemble(context, options, bootstrap)
    }

    /// Shared construction logic for both windowed and headless modes.
    fn init_with_context(
        context: RenderContext,
        cif_path: &str,
    ) -> Result<Self, VisoError> {
        let (entities, render_coords) =
            bootstrap::load_scene_from_file(cif_path)?;
        let options = VisoOptions::default();

        let mut bootstrap =
            init_gpu_pipeline(&context, entities, &render_coords)?;
        bootstrap.renderers.init_ball_and_stick_entities(
            &context,
            &bootstrap.entities,
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
        if let Some((centroid, radius)) = bootstrap.entities.bounding_sphere() {
            bootstrap.camera_controller.fit_to_sphere(centroid, radius);
        }

        Self::assemble(context, options, bootstrap)
    }

    /// Build the final `VisoEngine` from initialized GPU subsystems.
    fn assemble(
        context: RenderContext,
        options: VisoOptions,
        bootstrap: GpuBootstrap,
    ) -> Result<Self, VisoError> {
        Ok(Self {
            gpu: GpuPipeline {
                context,
                renderers: bootstrap.renderers,
                pick: bootstrap.pick,
                scene_processor: SceneProcessor::new()
                    .map_err(VisoError::ThreadSpawn)?,
                post_process: bootstrap.post_process,
                shader_composer: bootstrap.shader_composer,
            },
            cursor_pos: (0.0, 0.0),
            last_cull_camera_eye: Vec3::ZERO,
            camera_controller: bootstrap.camera_controller,
            lighting: bootstrap.lighting,
            topology: SceneTopology::new(),
            visual: VisualState::new(),
            entities: bootstrap.entities,
            band_specs: Vec::new(),
            pull_spec: None,
            animation: AnimationState::new(),
            options,
            active_preset: None,
            frame_timing: FrameTiming::new(TARGET_FPS),
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
            self.gpu.pick.hovered_target.as_residue_i32();
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
        self.lighting
            .update_headlamp_from_camera(&self.camera_controller.camera);
        self.lighting.update_gpu(&self.gpu.context.queue);
        self.update_frustum_culling();

        // Resolve band/pull specs to world-space each frame (auto-tracks
        // animated atoms)
        if !self.band_specs.is_empty() || self.pull_spec.is_some() {
            self.resolve_and_render_constraints();
        }
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
        let mut encoder = self.gpu.context.create_encoder();

        // Geometry pass
        let input = GeometryPassInput {
            color: self.gpu.post_process.color_view(),
            normal: &self.gpu.post_process.normal_view,
            depth: &self.gpu.post_process.depth_view,
            show_sidechains: self.options.display.show_sidechains,
        };
        let bind_groups = DrawBindGroups {
            camera: &self.camera_controller.bind_group,
            lighting: &self.lighting.bind_group,
            selection: &self.gpu.pick.selection.bind_group,
            color: Some(&self.gpu.pick.residue_colors.bind_group),
        };
        let frustum = self.camera_controller.frustum();
        self.gpu.renderers.encode_geometry_pass(
            &mut encoder,
            &input,
            &bind_groups,
            &frustum,
        );

        // Post-processing: SSAO → bloom → composite → FXAA
        let camera = &self.camera_controller.camera;
        self.gpu.post_process.render(
            &mut encoder,
            &self.gpu.context.queue,
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
        let picking_geometry = self.gpu.pick.build_geometry(
            &self.gpu.renderers,
            self.options.display.show_sidechains,
        );
        self.gpu.pick.picking.render(
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
            self.gpu.context.resize(width, height);
            self.camera_controller.resize(width, height);
            self.gpu.post_process.resize(&self.gpu.context);
            self.gpu.pick.picking.resize(
                &self.gpu.context.device,
                width,
                height,
            );
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
                let _ = self.entities.cycle_focus();
                self.fit_camera_to_focus();
                false
            }
            VisoCommand::ResetFocus => {
                self.entities.set_focus(Focus::Session);
                self.fit_camera_to_focus();
                false
            }

            // ── Playback ──
            VisoCommand::ToggleTrajectory => {
                self.animation.toggle_trajectory();
                false
            }

            // ── Selection ──
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
        self.gpu.scene_processor.shutdown();
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
            self.entities.force_dirty();
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
        self.gpu.context.render_scale = scale;
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
            .entities
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
        self.animation.load_trajectory(
            frames,
            num_atoms,
            &backbone_chains,
            backbone_indices,
        );
    }
}
