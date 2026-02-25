mod accessors;
mod animation;
mod input;
mod options;
pub(crate) mod picking_system;
mod queries;
pub(crate) mod renderers;
mod scene_management;
mod scene_sync;

use std::time::Instant;

use foldit_conv::adapters::pdb::structure_file_to_coords;
use foldit_conv::render::RenderCoords;
use foldit_conv::types::entity::split_into_entities;
use glam::{Mat4, Vec3};

use self::picking_system::PickingSystem;
use self::renderers::Renderers;
use crate::animation::animator::StructureAnimator;
use crate::animation::sidechain_state::SidechainAnimationState;
use crate::camera::controller::CameraController;
use crate::error::VisoError;
use crate::gpu::render_context::RenderContext;
use crate::gpu::residue_color::ResidueColorBuffer;
use crate::gpu::shader_composer::ShaderComposer;
use crate::input::InputState;
use crate::options::Options;
use crate::renderer::draw_context::DrawBindGroups;
use crate::renderer::geometry::ball_and_stick::BallAndStickRenderer;
use crate::renderer::picking::{PickTarget, SelectionBuffer};
use crate::renderer::postprocess::post_process::PostProcessStack;
use crate::renderer::PipelineLayouts;
use crate::scene::processor::SceneProcessor;
use crate::scene::{Focus, Scene, SceneEntity};
use crate::util::bond_topology::{get_residue_bonds, is_hydrophobic};
use crate::util::frame_timing::FrameTiming;
use crate::util::lighting::Lighting;
use crate::util::trajectory::TrajectoryPlayer;

/// Load a structure file and split into entities, returning a populated Scene
/// and the derived protein `RenderCoords`.
fn load_scene_from_file(
    cif_path: &str,
) -> Result<(Scene, RenderCoords), VisoError> {
    let coords = structure_file_to_coords(std::path::Path::new(cif_path))
        .map_err(|e| VisoError::StructureLoad(e.to_string()))?;

    let entities = split_into_entities(&coords);

    for e in &entities {
        log::debug!(
            "  entity {} — {:?}: {} atoms",
            e.entity_id,
            e.molecule_type,
            e.coords.num_atoms
        );
    }

    let mut scene = Scene::new();
    let entity_ids = scene.add_entities(entities);

    let render_coords = extract_render_coords(&scene, &entity_ids);
    Ok((scene, render_coords))
}

/// Derive protein `RenderCoords` from a populated scene.
fn extract_render_coords(scene: &Scene, entity_ids: &[u32]) -> RenderCoords {
    let protein_entity_id = entity_ids
        .iter()
        .find(|&&id| scene.entity(id).is_some_and(SceneEntity::is_protein));

    if let Some(protein_coords) = protein_entity_id
        .and_then(|&id| scene.entity(id).and_then(SceneEntity::protein_coords))
    {
        log::debug!("protein_coords: {} atoms", protein_coords.num_atoms);
        let protein_coords =
            foldit_conv::ops::transform::protein_only(&protein_coords);
        log::debug!("after protein_only: {} atoms", protein_coords.num_atoms);
        let rc = RenderCoords::from_coords_with_topology(
            &protein_coords,
            is_hydrophobic,
            |name| get_residue_bonds(name).map(<[(&str, &str)]>::to_vec),
        );
        log::debug!(
            "render_coords: {} backbone chains, {} residues",
            rc.backbone_chains.len(),
            rc.backbone_chains
                .iter()
                .map(|c| c.len() / 3)
                .sum::<usize>()
        );
        rc
    } else {
        log::debug!("no protein coords found");
        let empty = foldit_conv::types::coords::Coords {
            num_atoms: 0,
            atoms: Vec::new(),
            chain_ids: Vec::new(),
            res_names: Vec::new(),
            res_nums: Vec::new(),
            atom_names: Vec::new(),
            elements: Vec::new(),
        };
        RenderCoords::from_coords_with_topology(
            &empty,
            is_hydrophobic,
            |name| get_residue_bonds(name).map(<[(&str, &str)]>::to_vec),
        )
    }
}

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
/// Use [`ProteinRenderEngine::new`] for a default molecule or
/// [`ProteinRenderEngine::new_with_path`] to load a specific `.cif`/`.pdb`
/// file.
///
/// # Frame loop
///
/// Each frame, call [`render`](Self::render) to draw and present. Call
/// [`resize`](Self::resize) when the window size changes. Input is forwarded
/// via [`handle_input`](Self::handle_input).
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
/// Structural changes are animated via the `StructureAnimator`.
/// Control the animation style per-update by passing a
/// [`Transition`](crate::animation::transition::Transition).
pub struct ProteinRenderEngine {
    /// Core wgpu device, queue, and surface.
    pub context: RenderContext,
    _shader_composer: ShaderComposer,

    /// Post-processing pass stack (SSAO, bloom, composite, FXAA).
    pub(crate) post_process: PostProcessStack,
    /// Mouse and keyboard input state.
    pub input: InputState,
    /// Cached sidechain animation and SS-type state.
    pub(crate) sc: SidechainAnimationState,

    /// Orbital camera controller.
    pub camera_controller: CameraController,
    /// GPU lighting uniform and bind group.
    pub lighting: Lighting,
    /// Scene graph holding all entities.
    scene: Scene,
    /// Background thread for off-main-thread mesh generation.
    pub(crate) scene_processor: SceneProcessor,
    /// Structural animation driver.
    pub(crate) animator: StructureAnimator,
    /// Runtime display, lighting, color, and geometry options.
    options: Options,
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

impl ProteinRenderEngine {
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

    /// Shared construction logic for both windowed and headless modes.
    fn init_with_context(
        context: RenderContext,
        cif_path: &str,
    ) -> Result<Self, VisoError> {
        let mut shader_composer = ShaderComposer::new()?;
        let mut camera_controller = CameraController::new(&context);
        let lighting = Lighting::new(&context);
        let options = Options::default();

        let (scene, render_coords) = load_scene_from_file(cif_path)?;
        let n = render_coords.residue_count().max(1);
        let selection = SelectionBuffer::new(&context.device, n);
        let residue_colors = ResidueColorBuffer::new(&context.device, n);
        let layouts = PipelineLayouts {
            camera: camera_controller.layout.clone(),
            lighting: lighting.layout.clone(),
            selection: selection.layout.clone(),
            color: residue_colors.layout.clone(),
        };
        let mut renderers = Renderers::new(
            &context,
            &layouts,
            &render_coords,
            &scene,
            &mut shader_composer,
        )?;
        renderers.init_ball_and_stick_entities(&context, &scene, &options);
        let mut pick = PickingSystem::new(
            &context,
            &camera_controller.layout,
            selection,
            residue_colors,
            &mut shader_composer,
        )?;
        pick.init_colors_and_groups(
            &context,
            &render_coords.backbone_chains,
            &renderers,
        );
        let post_process =
            PostProcessStack::new(&context, &mut shader_composer)?;
        let positions = collect_all_positions(&render_coords, &scene, &options);
        camera_controller.fit_to_positions(&positions);
        Ok(Self {
            context,
            _shader_composer: shader_composer,
            post_process,
            input: InputState::new(),
            sc: SidechainAnimationState::new(),
            camera_controller,
            lighting,
            scene,
            scene_processor: SceneProcessor::new()
                .map_err(VisoError::ThreadSpawn)?,
            animator: StructureAnimator::new(),
            options,
            active_preset: None,
            frame_timing: FrameTiming::new(TARGET_FPS),
            trajectory_player: None,
            renderers,
            pick,
        })
    }

    /// Per-frame updates: animation ticks, uniform uploads, frustum culling.
    fn pre_render(&mut self) {
        // Apply any pending animation frame from the background thread
        // (non-blocking)
        self.apply_pending_animation();

        // Trajectory playback — submit frames to background thread
        // (non-blocking)
        if let Some(ref mut player) = self.trajectory_player {
            if let Some(backbone_chains) = player.tick(Instant::now()) {
                self.submit_animation_frame_with_backbone(
                    backbone_chains,
                    false,
                );
            }
        } else {
            // Standard animator path
            let animating = self.animator.update(Instant::now());

            // If animation is active, submit interpolated positions to
            // background thread
            if animating {
                self.submit_animation_frame();
            }
        }

        // Update hover state in camera uniform (from GPU picking)
        self.camera_controller.uniform.hovered_residue =
            self.pick.hovered_target.as_residue_i32();
        self.camera_controller.update_gpu(&self.context.queue);

        // Compute fog params from camera state each frame (depth-buffer fog,
        // always in sync) fog_start = focus point distance → front half
        // stays crisp fog_density scaled so back of protein reaches
        // ~87% fog (exp(-2) ≈ 0.13)
        let distance = self.camera_controller.distance();
        let bounding_radius = self.camera_controller.bounding_radius();
        let fog_start = distance;
        let fog_density = 2.0 / bounding_radius.max(10.0);
        self.post_process.update_fog(
            &self.context.queue,
            fog_start,
            fog_density,
        );

        // Per-chain LOD — submit background remesh when any chain's tier
        // changes
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

        // Update selection buffer (from GPU picking)
        self.pick
            .selection
            .update(&self.context.queue, &self.pick.picking.selected_residues);

        // Update per-residue color buffer (transition interpolation)
        let _color_transitioning =
            self.pick.residue_colors.update(&self.context.queue);

        // Update lighting to follow camera (headlamp mode)
        // Use camera.up (set by quaternion) for consistent basis vectors
        let camera = &self.camera_controller.camera;
        let forward = (camera.target - camera.eye).normalize();
        let right = camera.up.cross(forward).normalize(); // right = up × forward
        let up = forward.cross(right); // recalculate up to ensure orthonormal
        self.lighting.update_headlamp(right, up, forward);
        self.lighting.update_gpu(&self.context.queue);

        // Frustum culling for sidechains - update when camera moves
        // significantly
        self.update_frustum_culling();
    }

    /// Encode the main geometry render pass.
    fn encode_geometry_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main render pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: self.post_process.color_view(),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.post_process.normal_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
            ],
            depth_stencil_attachment: Some(
                wgpu::RenderPassDepthStencilAttachment {
                    view: &self.post_process.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                },
            ),
            ..Default::default()
        });

        let bind_groups = DrawBindGroups {
            camera: &self.camera_controller.bind_group,
            lighting: &self.lighting.bind_group,
            selection: &self.pick.selection.bind_group,
            color: Some(&self.pick.residue_colors.bind_group),
        };

        let frustum = self.camera_controller.frustum();
        self.renderers
            .backbone
            .draw_culled(&mut rp, &bind_groups, &frustum);

        if self.options.display.show_sidechains {
            self.renderers.sidechain.draw(&mut rp, &bind_groups);
        }

        self.renderers.ball_and_stick.draw(&mut rp, &bind_groups);
        self.renderers.nucleic_acid.draw(&mut rp, &bind_groups);
        self.renderers.band.draw(&mut rp, &bind_groups);
        self.renderers.pull.draw(&mut rp, &bind_groups);
    }

    /// Build the picking geometry descriptor from current renderer state.
    fn build_picking_geometry(
        &self,
    ) -> crate::renderer::picking::PickingGeometry<'_> {
        crate::renderer::picking::PickingGeometry {
            backbone_vertex_buffer: self.renderers.backbone.vertex_buffer(),
            backbone_tube_index_buffer: self
                .renderers
                .backbone
                .tube_index_buffer(),
            backbone_tube_index_count: self
                .renderers
                .backbone
                .tube_index_count(),
            backbone_ribbon_index_buffer: self
                .renderers
                .backbone
                .ribbon_index_buffer(),
            backbone_ribbon_index_count: self
                .renderers
                .backbone
                .ribbon_index_count(),
            capsule_bind_group: self.pick.groups.capsule.as_ref(),
            capsule_count: if self.options.display.show_sidechains {
                self.renderers.sidechain.instance_count()
            } else {
                0
            },
            bns_capsule_bind_group: self.pick.groups.bns_bond.as_ref(),
            bns_capsule_count: self.renderers.ball_and_stick.bond_count(),
            bns_sphere_bind_group: self.pick.groups.bns_sphere.as_ref(),
            bns_sphere_count: self.renderers.ball_and_stick.sphere_count(),
        }
    }

    /// Core render — geometry, post-process, picking — targeting the given
    /// view. Returns the encoder so the caller can submit it.
    fn render_to_view(
        &mut self,
        view: &wgpu::TextureView,
    ) -> wgpu::CommandEncoder {
        let mut encoder = self.context.create_encoder();

        self.encode_geometry_pass(&mut encoder);

        // Post-processing: SSAO → bloom → composite → FXAA
        let proj = self.camera_controller.camera.build_projection();
        let view_matrix = Mat4::look_at_rh(
            self.camera_controller.camera.eye,
            self.camera_controller.camera.target,
            self.camera_controller.camera.up,
        );
        self.post_process.render(
            &mut encoder,
            &self.context.queue,
            &crate::renderer::postprocess::post_process::PostProcessCamera {
                proj,
                view_matrix,
                znear: self.camera_controller.camera.znear,
                zfar: self.camera_controller.camera.zfar,
            },
            view.clone(),
        );

        // GPU Picking pass
        let picking_geometry = self.build_picking_geometry();
        self.pick.picking.render(
            &mut encoder,
            &self.camera_controller.bind_group,
            &picking_geometry,
            (self.input.mouse_pos.0 as u32, self.input.mouse_pos.1 as u32),
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
        if let Some(raw_id) =
            self.pick.picking.complete_readback(&self.context.device)
        {
            self.pick.hovered_target = self
                .pick
                .pick_map
                .as_ref()
                .map_or(PickTarget::None, |pm| pm.resolve(raw_id));
        }

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

/// Compute initial per-residue colors from chain hue ramp.
fn initial_chain_colors(
    backbone_chains: &[Vec<Vec3>],
    total_residues: usize,
) -> Vec<[f32; 3]> {
    if backbone_chains.is_empty() {
        return vec![[0.5, 0.5, 0.5]; total_residues.max(1)];
    }
    let num_chains = backbone_chains.len();
    let mut colors = Vec::with_capacity(total_residues);
    for (chain_idx, chain) in backbone_chains.iter().enumerate() {
        let t = if num_chains > 1 {
            chain_idx as f32 / (num_chains - 1) as f32
        } else {
            0.0
        };
        let color = ProteinRenderEngine::chain_color(t);
        let n_residues = chain.len() / 3;
        colors.extend(std::iter::repeat_n(color, n_residues));
    }
    colors
}

/// Collect all atom positions for initial camera fit (protein + ligands + NA).
fn collect_all_positions(
    render_coords: &RenderCoords,
    scene: &Scene,
    options: &Options,
) -> Vec<Vec3> {
    let mut positions = render_coords.all_positions.clone();
    let non_protein: Vec<foldit_conv::types::entity::MoleculeEntity> = scene
        .entities()
        .iter()
        .filter(|se| !se.is_protein())
        .map(|se| se.entity.clone())
        .collect();
    positions.extend(BallAndStickRenderer::collect_positions(
        &non_protein,
        &options.display,
    ));
    for chain in scene
        .nucleic_acid_entities()
        .iter()
        .flat_map(|se| se.entity.extract_p_atom_chains())
    {
        positions.extend(&chain);
    }
    positions
}

// =============================================================================
// Camera
// =============================================================================

impl ProteinRenderEngine {
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
