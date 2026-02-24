mod accessors;
mod animation;
mod input;
mod options;
mod queries;
mod scene_management;
mod scene_sync;

use std::time::Instant;

use foldit_conv::coords::{
    split_into_entities, structure_file_to_coords, MoleculeEntity,
    MoleculeType, RenderCoords,
};
use glam::{Mat4, Vec3};

use crate::{
    animation::{
        animator::StructureAnimator, sidechain_state::SidechainAnimationState,
    },
    camera::controller::CameraController,
    gpu::{
        render_context::RenderContext, residue_color::ResidueColorBuffer,
        shader_composer::ShaderComposer,
    },
    input::InputState,
    options::Options,
    picking::{picking_state::PickingState, Picking, SelectionBuffer},
    renderer::{
        molecular::{
            backbone::BackboneRenderer,
            ball_and_stick::BallAndStickRenderer,
            band::BandRenderer,
            capsule_sidechain::{CapsuleSidechainRenderer, SidechainData},
            draw_context::DrawBindGroups,
            nucleic_acid::NucleicAcidRenderer,
            pull::PullRenderer,
        },
        postprocess::post_process::PostProcessStack,
    },
    scene::{processor::SceneProcessor, Focus, Scene},
    util::{
        bond_topology::{get_residue_bonds, is_hydrophobic},
        frame_timing::FrameTiming,
        lighting::Lighting,
        trajectory::TrajectoryPlayer,
    },
};

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
/// via [`handle_mouse_move`](Self::handle_mouse_move),
/// [`handle_mouse_button`](Self::handle_mouse_button), and
/// [`handle_input`](Self::handle_input).
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
/// Structural changes are animated via the [`StructureAnimator`].
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
    /// GPU picking bind groups for capsule geometry.
    pub(crate) picking_groups: PickingState,
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

    /// Unified backbone renderer (protein + nucleic acid).
    pub backbone_renderer: BackboneRenderer,
    /// Capsule-based sidechain renderer.
    pub sidechain_renderer: CapsuleSidechainRenderer,
    /// Constraint band renderer.
    pub band_renderer: BandRenderer,
    /// Interactive pull arrow renderer.
    pub pull_renderer: PullRenderer,
    /// Ball-and-stick renderer for small molecules.
    pub ball_and_stick_renderer: BallAndStickRenderer,
    /// Nucleic acid backbone renderer.
    pub nucleic_acid_renderer: NucleicAcidRenderer,

    /// GPU picking pass (offscreen R32Uint render + readback).
    pub picking: Picking,
    /// Per-residue selection bit-array on GPU.
    pub(crate) selection_buffer: SelectionBuffer,
    /// Per-residue color override buffer on GPU.
    pub(crate) residue_color_buffer: ResidueColorBuffer,

    /// Last cursor position for computing deltas in
    /// [`handle_input`](Self::handle_input).
    last_cursor_pos: Option<(f32, f32)>,

    /// DPI scale factor for mapping mouse coordinates (logical/CSS pixels)
    /// to the picking texture (physical pixels). Default 1.0.
    dpi_scale: f32,
}

// =============================================================================
// Core
// =============================================================================

impl ProteinRenderEngine {
    /// Engine with a default molecule path.
    pub async fn new(
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        size: (u32, u32),
        scale_factor: f64,
    ) -> Result<Self, crate::error::VisoError> {
        Self::new_with_path(
            window,
            size,
            scale_factor,
            "assets/models/4pnk.cif",
        )
        .await
    }

    /// Engine with a specified molecule path.
    pub async fn new_with_path(
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        size: (u32, u32),
        scale_factor: f64,
        cif_path: &str,
    ) -> Result<Self, crate::error::VisoError> {
        let mut context = RenderContext::new(window, size).await?;

        // 2x supersampling on standard-DPI displays to compensate for low pixel
        // density
        if scale_factor < 2.0 {
            context.render_scale = 2;
        }

        Self::init_with_context(context, scale_factor, cif_path)
    }

    /// Engine from a pre-built [`RenderContext`] (for embedding in dioxus,
    /// headless rendering, etc.).
    ///
    /// Use [`RenderContext::from_device`] to create a surface-less context
    /// from an externally-owned `wgpu::Device` and `wgpu::Queue`.
    pub fn new_from_context(
        mut context: RenderContext,
        scale_factor: f64,
        cif_path: &str,
    ) -> Result<Self, crate::error::VisoError> {
        if scale_factor < 2.0 {
            context.render_scale = 2;
        }

        Self::init_with_context(context, scale_factor, cif_path)
    }

    /// Shared construction logic for both windowed and headless modes.
    fn init_with_context(
        context: RenderContext,
        scale_factor: f64,
        cif_path: &str,
    ) -> Result<Self, crate::error::VisoError> {
        let mut shader_composer = ShaderComposer::new();

        let mut camera_controller = CameraController::new(&context);
        let lighting = Lighting::new(&context);
        // Load coords from structure file (PDB or mmCIF, detected by extension)
        let coords = structure_file_to_coords(std::path::Path::new(cif_path))
            .map_err(|e| {
            crate::error::VisoError::StructureLoad(e.to_string())
        })?;

        let entities = split_into_entities(&coords);

        // Log entity breakdown
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

        // Extract protein coords for rendering (may be absent for
        // nucleic-acid-only structures)
        let protein_entity_id = entity_ids.iter().find(|&&id| {
            scene.entity(id).is_some_and(|e| {
                e.entity.molecule_type == MoleculeType::Protein
            })
        });
        let render_coords = if let Some(protein_coords) = protein_entity_id
            .and_then(|&id| {
                scene
                    .entity(id)
                    .and_then(super::scene::SceneEntity::protein_coords)
            }) {
            log::debug!("protein_coords: {} atoms", protein_coords.num_atoms);
            let protein_coords =
                foldit_conv::coords::protein_only(&protein_coords);
            log::debug!(
                "after protein_only: {} atoms",
                protein_coords.num_atoms
            );
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
            let empty = foldit_conv::coords::Coords {
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
        };

        // Count total residues for selection buffer sizing
        let total_residues = render_coords.residue_count();

        // Create selection buffer (shared by all renderers)
        let selection_buffer =
            SelectionBuffer::new(&context.device, total_residues.max(1));

        // Create per-residue color buffer (shared by all renderers)
        let mut residue_color_buffer =
            ResidueColorBuffer::new(&context.device, total_residues.max(1));

        // Extract NA chains early so BackboneRenderer can handle both
        let na_chains: Vec<Vec<Vec3>> = scene
            .entities()
            .iter()
            .filter(|se| {
                matches!(
                    se.entity.molecule_type,
                    MoleculeType::DNA | MoleculeType::RNA
                )
            })
            .flat_map(|se| se.entity.extract_p_atom_chains())
            .collect();

        let options = Options::default();

        let backbone_renderer = BackboneRenderer::new(
            &context,
            &camera_controller.layout,
            &lighting.layout,
            &selection_buffer.layout,
            &residue_color_buffer.layout,
            &render_coords.backbone_chains,
            &na_chains,
            &mut shader_composer,
        );

        // Get sidechain data from RenderCoords
        let sidechain_positions = render_coords.sidechain_positions();
        let sidechain_hydrophobicity = render_coords.sidechain_hydrophobicity();
        let sidechain_residue_indices =
            render_coords.sidechain_residue_indices();

        let sidechain_renderer = CapsuleSidechainRenderer::new(
            &context,
            &camera_controller.layout,
            &lighting.layout,
            &selection_buffer.layout,
            &SidechainData {
                positions: &sidechain_positions,
                bonds: &render_coords.sidechain_bonds,
                backbone_bonds: &render_coords.backbone_sidechain_bonds,
                hydrophobicity: &sidechain_hydrophobicity,
                residue_indices: &sidechain_residue_indices,
            },
            &mut shader_composer,
        );

        // Create band renderer (starts empty)
        let band_renderer = BandRenderer::new(
            &context,
            &camera_controller.layout,
            &lighting.layout,
            &selection_buffer.layout,
            &mut shader_composer,
        );

        // Create pull renderer (starts empty, only one pull at a time)
        let pull_renderer = PullRenderer::new(
            &context,
            &camera_controller.layout,
            &lighting.layout,
            &selection_buffer.layout,
            &mut shader_composer,
        );

        // Create the full post-processing stack (depth/normal textures, SSAO,
        // bloom, composite, FXAA)
        let post_process =
            PostProcessStack::new(&context, &mut shader_composer);

        // Create frame timing with 300 FPS limit
        let frame_timing = FrameTiming::new(TARGET_FPS);

        // Create GPU-based picking
        let picking = Picking::new(
            &context,
            &camera_controller.layout,
            &mut shader_composer,
        );

        // Create initial picking bind groups
        let mut picking_groups = PickingState::new();
        picking_groups.rebuild_capsule(
            &picking,
            &context.device,
            &sidechain_renderer,
        );

        // Create ball-and-stick renderer for non-protein entities
        let mut ball_and_stick_renderer = BallAndStickRenderer::new(
            &context,
            &camera_controller.layout,
            &lighting.layout,
            &selection_buffer.layout,
            &mut shader_composer,
        );

        // Compute initial per-residue colors so the first frame isn't gray
        {
            let backbone_chains = &render_coords.backbone_chains;
            let num_chains = backbone_chains.len();
            let initial_colors: Vec<[f32; 3]> = if num_chains == 0 {
                vec![[0.5, 0.5, 0.5]; total_residues.max(1)]
            } else {
                let mut colors = Vec::with_capacity(total_residues);
                for (chain_idx, chain) in backbone_chains.iter().enumerate() {
                    let t = if num_chains > 1 {
                        chain_idx as f32 / (num_chains - 1) as f32
                    } else {
                        0.0
                    };
                    let color = Self::chain_color(t);
                    let n_residues = chain.len() / 3;
                    for _ in 0..n_residues {
                        colors.push(color);
                    }
                }
                colors
            };
            residue_color_buffer
                .set_colors_immediate(&context.queue, &initial_colors);
        }

        // Collect non-protein entities for ball-and-stick
        let non_protein_refs: Vec<MoleculeEntity> = scene
            .entities()
            .iter()
            .filter(|se| se.entity.molecule_type != MoleculeType::Protein)
            .map(|se| se.entity.clone())
            .collect();
        ball_and_stick_renderer.update_from_entities(
            &context.device,
            &context.queue,
            &non_protein_refs,
            &options.display,
            Some(&options.colors),
        );

        // Create picking bind group for ball-and-stick
        picking_groups.rebuild_bns(
            &picking,
            &context.device,
            &ball_and_stick_renderer,
        );

        // Create nucleic acid renderer for base rings + stem tubes
        // (backbone is now handled by BackboneRenderer)
        let na_rings: Vec<foldit_conv::coords::entity::NucleotideRing> = scene
            .entities()
            .iter()
            .filter(|se| {
                matches!(
                    se.entity.molecule_type,
                    MoleculeType::DNA | MoleculeType::RNA
                )
            })
            .flat_map(|se| se.entity.extract_base_rings())
            .collect();
        let nucleic_acid_renderer = NucleicAcidRenderer::new(
            &context,
            &camera_controller.layout,
            &lighting.layout,
            &selection_buffer.layout,
            &na_chains,
            &na_rings,
            &mut shader_composer,
        );

        // Fit camera to all atom positions (protein + non-protein + nucleic
        // acid P-atoms)
        let mut all_fit_positions = render_coords.all_positions.clone();
        all_fit_positions.extend(BallAndStickRenderer::collect_positions(
            &non_protein_refs,
            &options.display,
        ));
        for chain in &na_chains {
            all_fit_positions.extend(chain);
        }
        camera_controller.fit_to_positions(&all_fit_positions);

        // Create structure animator
        let animator = StructureAnimator::new();

        // Create background scene processor
        let scene_processor = SceneProcessor::new()
            .map_err(crate::error::VisoError::ThreadSpawn)?;

        Ok(Self {
            context,
            _shader_composer: shader_composer,
            post_process,
            input: InputState::new(),
            picking_groups,
            sc: SidechainAnimationState::new(),
            camera_controller,
            lighting,
            scene,
            scene_processor,
            animator,
            options,
            active_preset: None,
            frame_timing,
            trajectory_player: None,
            backbone_renderer,
            sidechain_renderer,
            band_renderer,
            pull_renderer,
            ball_and_stick_renderer,
            nucleic_acid_renderer,
            picking,
            selection_buffer,
            residue_color_buffer,
            last_cursor_pos: None,
            dpi_scale: scale_factor as f32,
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
            self.picking.hovered_residue;
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
            .backbone_renderer
            .chain_ranges()
            .iter()
            .map(|r| {
                crate::options::select_chain_lod_tier(
                    r.bounding_center,
                    camera_eye,
                )
            })
            .collect();
        if per_chain_tiers != self.backbone_renderer.cached_lod_tiers() {
            self.backbone_renderer.set_cached_lod_tiers(per_chain_tiers);
            self.submit_per_chain_lod_remesh(camera_eye);
        }

        // Update selection buffer (from GPU picking)
        self.selection_buffer
            .update(&self.context.queue, &self.picking.selected_residues);

        // Update per-residue color buffer (transition interpolation)
        let _color_transitioning =
            self.residue_color_buffer.update(&self.context.queue);

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

    /// Core render — geometry, post-process, picking — targeting the given
    /// view. Returns the encoder so the caller can submit it.
    fn render_to_view(
        &mut self,
        view: &wgpu::TextureView,
    ) -> wgpu::CommandEncoder {
        let mut encoder = self.context.create_encoder();

        // Geometry pass — render to intermediate color/normal textures at
        // render_scale resolution.
        {
            let mut rp =
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("main render pass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: self.post_process.color_view(),
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &self.post_process.normal_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.0,
                                    g: 0.0,
                                    b: 0.0,
                                    a: 0.0,
                                }),
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

            // Render order: backbone (tube or ribbon) -> sidechains (all
            // opaque)
            let bind_groups = DrawBindGroups {
                camera: &self.camera_controller.bind_group,
                lighting: &self.lighting.bind_group,
                selection: &self.selection_buffer.bind_group,
                color: Some(&self.residue_color_buffer.bind_group),
            };

            // Backbone: unified renderer (tube + ribbon passes) with frustum
            // culling
            let frustum = self.camera_controller.frustum();
            self.backbone_renderer
                .draw_culled(&mut rp, &bind_groups, &frustum);

            if self.options.display.show_sidechains {
                self.sidechain_renderer.draw(&mut rp, &bind_groups);
            }

            self.ball_and_stick_renderer.draw(&mut rp, &bind_groups);
            self.nucleic_acid_renderer.draw(&mut rp, &bind_groups);
            self.band_renderer.draw(&mut rp, &bind_groups);
            self.pull_renderer.draw(&mut rp, &bind_groups);
        }

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

        // GPU Picking pass — render residue IDs to picking buffer
        // BackboneRenderer shares one vertex buffer for both tube and ribbon
        // index buffers; the picking pass draws both.
        self.picking.render(
            &mut encoder,
            &self.camera_controller.bind_group,
            &crate::picking::PickingGeometry {
                backbone_vertex_buffer: self.backbone_renderer.vertex_buffer(),
                backbone_tube_index_buffer: self
                    .backbone_renderer
                    .tube_index_buffer(),
                backbone_tube_index_count: self
                    .backbone_renderer
                    .tube_index_count(),
                backbone_ribbon_index_buffer: self
                    .backbone_renderer
                    .ribbon_index_buffer(),
                backbone_ribbon_index_count: self
                    .backbone_renderer
                    .ribbon_index_count(),
                capsule_bind_group: self
                    .picking_groups
                    .capsule_picking_bind_group
                    .as_ref(),
                capsule_count: if self.options.display.show_sidechains {
                    self.sidechain_renderer.instance_count()
                } else {
                    0
                },
                bns_capsule_bind_group: self
                    .picking_groups
                    .bns_picking_bind_group
                    .as_ref(),
                bns_capsule_count: self.ball_and_stick_renderer.picking_count(),
            },
            self.input.mouse_pos.0 as u32,
            self.input.mouse_pos.1 as u32,
        );

        encoder
    }

    /// Execute one frame: update animations, run the geometry pass,
    /// post-process, and present.
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
        self.picking.start_readback();

        // Try to complete any pending readback from previous frame
        // (non-blocking poll)
        let _ = self.picking.complete_readback(&self.context.device);

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
            self.picking.resize(&self.context.device, width, height);
        }
    }
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
