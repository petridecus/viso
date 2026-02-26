mod accessors;
mod animation;
/// The engine's complete interactive vocabulary.
pub mod command;
mod construction;
mod options;
pub(crate) mod picking_system;
mod queries;
pub(crate) mod renderers;
mod scene_management;
mod scene_sync;

use std::collections::HashMap;

use foldit_conv::types::entity::MoleculeEntity;
use glam::{Mat4, Vec3};

use self::picking_system::PickingSystem;
use self::renderers::Renderers;
use crate::animation::animator::StructureAnimator;
use crate::animation::sidechain_state::{SidechainAnimData, SidechainCache};
use crate::animation::transition::Transition;
use crate::camera::controller::CameraController;
use crate::error::VisoError;
use crate::gpu::lighting::Lighting;
use crate::gpu::render_context::RenderContext;
use crate::gpu::shader_composer::ShaderComposer;
use crate::options::VisoOptions;
use crate::renderer::draw_context::DrawBindGroups;
use crate::renderer::postprocess::post_process::PostProcessStack;
use crate::scene::processor::SceneProcessor;
use crate::scene::{EntityResidueRange, Focus, Scene};
use crate::util::frame_timing::FrameTiming;
use crate::util::trajectory::TrajectoryPlayer;

/// Target FPS limit
const TARGET_FPS: u32 = 300;

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
/// [`update_entity_coords`](Self::update_entity_coords), and sync changes to
/// renderers with [`sync_scene_to_renderers`](Self::sync_scene_to_renderers).
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
    cursor_pos: (f32, f32),
    /// Sidechain animation start/target pairs.
    pub(crate) sc_anim: SidechainAnimData,
    /// Cached scene-derived sidechain data.
    pub(crate) sc_cache: SidechainCache,
    /// Camera eye position at the last frustum-culling update.
    pub(crate) last_cull_camera_eye: Vec3,

    /// Orbital camera controller.
    pub camera_controller: CameraController,
    /// GPU lighting uniform and bind group.
    pub lighting: Lighting,
    /// Scene graph holding all entities.
    scene: Scene,
    /// Canonical entity data (source of truth). Scene is derived from this.
    pub(crate) entities: Vec<MoleculeEntity>,
    /// Per-entity residue ranges in the flat array (maps entity_id to
    /// start/count).
    pub(crate) entity_ranges: Vec<EntityResidueRange>,
    /// Per-entity animation behavior overrides (default = smooth).
    pub(crate) entity_behaviors: HashMap<u32, Transition>,
    /// Background thread for off-main-thread mesh generation.
    pub(crate) scene_processor: SceneProcessor,
    /// Structural animation driver.
    pub(crate) animator: StructureAnimator,
    /// Runtime display, lighting, color, and geometry options.
    options: VisoOptions,
    /// Currently applied options preset name, if any.
    active_preset: Option<String>,
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

// =============================================================================
// Core
// =============================================================================

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
        let render_coords = construction::empty_render_coords();
        let scene = Scene::new();
        let bootstrap =
            construction::init_gpu_pipeline(&context, scene, &render_coords)?;
        let options = VisoOptions::default();
        Self::assemble(context, options, bootstrap, Vec::new())
    }

    /// Shared construction logic for both windowed and headless modes.
    fn init_with_context(
        context: RenderContext,
        cif_path: &str,
    ) -> Result<Self, VisoError> {
        let (scene, render_coords) =
            construction::load_scene_from_file(cif_path)?;
        let options = VisoOptions::default();

        let mut bootstrap =
            construction::init_gpu_pipeline(&context, scene, &render_coords)?;
        bootstrap.renderers.init_ball_and_stick_entities(
            &context,
            &bootstrap.scene,
            &options,
        );
        bootstrap.pick.init_colors_and_groups(
            &context,
            &render_coords.backbone_chains,
            &bootstrap.renderers,
        );
        let positions = construction::collect_all_positions(
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
        bootstrap: construction::GpuBootstrap,
        entities: Vec<MoleculeEntity>,
    ) -> Result<Self, VisoError> {
        Ok(Self {
            context,
            _shader_composer: bootstrap.shader_composer,
            post_process: bootstrap.post_process,
            cursor_pos: (0.0, 0.0),
            sc_anim: SidechainAnimData::new(),
            sc_cache: SidechainCache::new(),
            last_cull_camera_eye: Vec3::ZERO,
            camera_controller: bootstrap.camera_controller,
            lighting: bootstrap.lighting,
            scene: bootstrap.scene,
            entities,
            entity_ranges: Vec::new(),
            entity_behaviors: HashMap::new(),
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

    /// Core render — geometry, post-process, picking — targeting the given
    /// view. Returns the encoder so the caller can submit it.
    fn render_to_view(
        &mut self,
        view: &wgpu::TextureView,
    ) -> wgpu::CommandEncoder {
        let mut encoder = self.context.create_encoder();

        // Geometry pass
        let input = renderers::GeometryPassInput {
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

// =============================================================================
// Command dispatch
// =============================================================================

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
                .select_segment(index, &self.sc_cache.ss_types, extend),
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
