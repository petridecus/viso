mod animation;
mod input;
mod options;
mod queries;
mod scene_management;
mod scene_sync;

use crate::animation::StructureAnimator;
use crate::renderer::molecular::ball_and_stick::BallAndStickRenderer;
use crate::renderer::molecular::band::BandRenderer;
use crate::renderer::molecular::nucleic_acid::NucleicAcidRenderer;
use crate::camera::controller::CameraController;
use crate::camera::input::InputHandler;
use crate::renderer::molecular::capsule_sidechain::{CapsuleSidechainRenderer, SidechainData};
use crate::renderer::molecular::draw_context::DrawBindGroups;
use crate::util::frame_timing::FrameTiming;
use crate::camera::input_state::InputState;
use crate::picking::picking_state::PickingState;
use crate::renderer::postprocess::post_process::PostProcessStack;
use crate::animation::sidechain_state::SidechainAnimationState;
use crate::util::options::Options;
use crate::gpu::residue_color::ResidueColorBuffer;
use crate::picking::{Picking, SelectionBuffer};
use crate::util::lighting::Lighting;
use crate::renderer::molecular::pull::PullRenderer;

use crate::gpu::render_context::RenderContext;
use crate::renderer::molecular::ribbon::RibbonRenderer;
use crate::scene::processor::SceneProcessor;
use crate::gpu::shader_composer::ShaderComposer;
use crate::util::trajectory::TrajectoryPlayer;
use foldit_conv::secondary_structure::SSType;
use crate::renderer::molecular::tube::TubeRenderer;

use crate::util::bond_topology::{get_residue_bonds, is_hydrophobic};
use crate::scene::{Focus, Scene};
use foldit_conv::coords::{
    structure_file_to_coords, split_into_entities,
    MoleculeEntity, MoleculeType, RenderCoords,
};
use glam::{Mat4, Vec3};
use std::collections::HashSet;
use std::time::Instant;

/// Target FPS limit
const TARGET_FPS: u32 = 300;

pub struct ProteinRenderEngine {
    // GPU context
    pub(crate) context: RenderContext,
    pub(crate) _shader_composer: ShaderComposer,

    // Composed sub-structs
    pub(crate) post_process: PostProcessStack,
    pub(crate) input: InputState,
    pub(crate) picking_groups: PickingState,
    pub(crate) sc: SidechainAnimationState,

    // Existing composed types
    pub(crate) camera_controller: CameraController,
    pub(crate) lighting: Lighting,
    pub(crate) scene: Scene,
    pub(crate) scene_processor: SceneProcessor,
    pub(crate) animator: StructureAnimator,
    pub(crate) options: Options,
    pub(crate) active_preset: Option<String>,
    pub(crate) frame_timing: FrameTiming,
    pub(crate) _input_handler: InputHandler,
    pub(crate) trajectory_player: Option<TrajectoryPlayer>,

    // Molecular renderers
    pub(crate) tube_renderer: TubeRenderer,
    pub(crate) ribbon_renderer: RibbonRenderer,
    pub(crate) sidechain_renderer: CapsuleSidechainRenderer,
    pub(crate) band_renderer: BandRenderer,
    pub(crate) pull_renderer: PullRenderer,
    pub(crate) ball_and_stick_renderer: BallAndStickRenderer,
    pub(crate) nucleic_acid_renderer: NucleicAcidRenderer,

    // Shared GPU state
    pub(crate) picking: Picking,
    pub(crate) selection_buffer: SelectionBuffer,
    pub(crate) residue_color_buffer: ResidueColorBuffer,
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
    ) -> Self {
        Self::new_with_path(window, size, scale_factor, "assets/models/4pnk.cif").await
    }

    /// Engine with a specified molecule path.
    pub async fn new_with_path(
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        size: (u32, u32),
        scale_factor: f64,
        cif_path: &str,
    ) -> Self {
        let mut context = RenderContext::new(window, size).await;

        // 2x supersampling on standard-DPI displays to compensate for low pixel density
        if scale_factor < 2.0 {
            context.render_scale = 2;
        }

        let mut shader_composer = ShaderComposer::new();

        let mut camera_controller = CameraController::new(&context);
        let lighting = Lighting::new(&context);
        let input_handler = InputHandler::new();

        // Load coords from structure file (PDB or mmCIF, detected by extension)
        let coords = structure_file_to_coords(std::path::Path::new(cif_path))
            .expect("Failed to load coords from structure file");

        let entities = split_into_entities(&coords);

        // Log entity breakdown
        for e in &entities {
            log::debug!("  entity {} — {:?}: {} atoms", e.entity_id, e.molecule_type, e.coords.num_atoms);
        }

        let group_name = std::path::Path::new(cif_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(cif_path);
        let mut scene = Scene::new();
        let group_id = scene.add_group(entities, group_name);

        // Extract protein coords for rendering (may be absent for nucleic-acid-only structures)
        let render_coords = if let Some(protein_coords) = scene.group(group_id)
            .and_then(|g| g.protein_coords())
        {
            log::debug!("protein_coords: {} atoms", protein_coords.num_atoms);
            let protein_coords = foldit_conv::coords::protein_only(&protein_coords);
            log::debug!("after protein_only: {} atoms", protein_coords.num_atoms);
            let rc = RenderCoords::from_coords_with_topology(
                &protein_coords,
                is_hydrophobic,
                |name| get_residue_bonds(name).map(|b| b.to_vec()),
            );
            log::debug!("render_coords: {} backbone chains, {} residues",
                rc.backbone_chains.len(),
                rc.backbone_chains.iter().map(|c| c.len() / 3).sum::<usize>());
            rc
        } else {
            log::debug!("no protein coords found for group");
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
                |name| get_residue_bonds(name).map(|b| b.to_vec()),
            )
        };

        // Count total residues for selection buffer sizing
        let total_residues = render_coords.residue_count();

        // Create selection buffer (shared by all renderers)
        let selection_buffer = SelectionBuffer::new(&context.device, total_residues.max(1));

        // Create per-residue color buffer (shared by all renderers)
        let mut residue_color_buffer = ResidueColorBuffer::new(&context.device, total_residues.max(1));

        let mut tube_renderer = TubeRenderer::new(
            &context,
            &camera_controller.layout,
            &lighting.layout,
            &selection_buffer.layout,
            &residue_color_buffer.layout,
            &render_coords.backbone_chains,
            &mut shader_composer,
        );

        // Create ribbon renderer for secondary structure visualization
        // Use the Foldit-style renderer if we have full backbone residue data (N, CA, C, O)
        let ribbon_renderer = if !render_coords.backbone_residue_chains.is_empty() {
            RibbonRenderer::new_from_residues(
                &context,
                &camera_controller.layout,
                &lighting.layout,
                &selection_buffer.layout,
                &residue_color_buffer.layout,
                &render_coords.backbone_residue_chains,
                &mut shader_composer,
            )
        } else {
            // Fallback to legacy renderer if only backbone_chains available
            RibbonRenderer::new(
                &context,
                &camera_controller.layout,
                &lighting.layout,
                &selection_buffer.layout,
                &residue_color_buffer.layout,
                &render_coords.backbone_chains,
                &mut shader_composer,
            )
        };

        // Get sidechain data from RenderCoords
        let sidechain_positions = render_coords.sidechain_positions();
        let sidechain_hydrophobicity = render_coords.sidechain_hydrophobicity();
        let sidechain_residue_indices = render_coords.sidechain_residue_indices();

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

        // Create the full post-processing stack (depth/normal textures, SSAO, bloom, composite, FXAA)
        let post_process = PostProcessStack::new(&context, &mut shader_composer);

        // Create frame timing with 300 FPS limit
        let frame_timing = FrameTiming::new(TARGET_FPS);

        // Create GPU-based picking
        let picking = Picking::new(&context, &camera_controller.layout, &mut shader_composer);

        // Create initial picking bind groups
        let mut picking_groups = PickingState::new();
        picking_groups.rebuild_capsule(&picking, &context.device, &sidechain_renderer);

        // Create ball-and-stick renderer for non-protein entities
        let mut ball_and_stick_renderer = BallAndStickRenderer::new(
            &context,
            &camera_controller.layout,
            &lighting.layout,
            &selection_buffer.layout,
            &mut shader_composer,
        );
        let options = Options::default();

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
            residue_color_buffer.set_colors_immediate(&context.queue, &initial_colors);
        }

        // Collect non-protein entities from the group for ball-and-stick
        let non_protein_entities: Vec<&MoleculeEntity> = scene.group(group_id)
            .map(|g| g.entities().iter()
                .filter(|e| e.molecule_type != MoleculeType::Protein)
                .collect())
            .unwrap_or_default();
        let non_protein_refs: Vec<MoleculeEntity> = non_protein_entities.into_iter().cloned().collect();
        ball_and_stick_renderer.update_from_entities(
            &context.device,
            &context.queue,
            &non_protein_refs,
            &options.display,
            Some(&options.colors),
        );

        // Create picking bind group for ball-and-stick
        picking_groups.rebuild_bns(&picking, &context.device, &ball_and_stick_renderer);

        // Create nucleic acid renderer for DNA/RNA backbone ribbons + base rings
        let na_entities: Vec<&MoleculeEntity> = scene.group(group_id)
            .map(|g| g.entities().iter()
                .filter(|e| matches!(e.molecule_type, MoleculeType::DNA | MoleculeType::RNA))
                .collect())
            .unwrap_or_default();
        let na_chains: Vec<Vec<Vec3>> = na_entities.iter()
            .flat_map(|e| e.extract_p_atom_chains())
            .collect();
        let na_rings: Vec<foldit_conv::coords::entity::NucleotideRing> = na_entities.iter()
            .flat_map(|e| e.extract_base_rings())
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

        // Fit camera to all atom positions (protein + non-protein + nucleic acid P-atoms)
        let mut all_fit_positions = render_coords.all_positions.clone();
        all_fit_positions.extend(BallAndStickRenderer::collect_positions(&non_protein_refs, &options.display));
        for chain in &na_chains {
            all_fit_positions.extend(chain);
        }
        camera_controller.fit_to_positions(&all_fit_positions);

        // Create structure animator
        let animator = StructureAnimator::new();

        // Create background scene processor
        let scene_processor = SceneProcessor::new();

        // Tube renderer only renders coils; ribbons handle helices/sheets
        {
            let mut coil_only = HashSet::new();
            coil_only.insert(SSType::Coil);
            tube_renderer.set_ss_filter(Some(coil_only));
            tube_renderer.regenerate(&context.device, &context.queue);
        }

        Self {
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
            _input_handler: input_handler,
            trajectory_player: None,
            tube_renderer,
            ribbon_renderer,
            sidechain_renderer,
            band_renderer,
            pull_renderer,
            ball_and_stick_renderer,
            nucleic_acid_renderer,
            picking,
            selection_buffer,
            residue_color_buffer,
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // Check if we should render based on FPS limit
        if !self.frame_timing.should_render() {
            return Ok(());
        }

        // Apply any pending animation frame from the background thread (non-blocking)
        self.apply_pending_animation();

        // Trajectory playback — submit frames to background thread (non-blocking)
        if let Some(ref mut player) = self.trajectory_player {
            if let Some(backbone_chains) = player.tick(Instant::now()) {
                self.submit_animation_frame_with_backbone(backbone_chains, false);
            }
        } else {
            // Standard animator path
            let animating = self.animator.update(Instant::now());

            // If animation is active, submit interpolated positions to background thread
            if animating {
                self.submit_animation_frame();
            }
        }

        // Update hover state in camera uniform (from GPU picking)
        self.camera_controller.uniform.hovered_residue = self.picking.hovered_residue;
        self.camera_controller.update_gpu(&self.context.queue);

        // Compute fog params from camera state each frame (depth-buffer fog, always in sync)
        // fog_start = focus point distance → front half stays crisp
        // fog_density scaled so back of protein reaches ~87% fog (exp(-2) ≈ 0.13)
        let distance = self.camera_controller.distance();
        let bounding_radius = self.camera_controller.bounding_radius();
        let fog_start = distance;
        let fog_density = 2.0 / bounding_radius.max(10.0);
        self.post_process.update_fog(&self.context.queue, fog_start, fog_density);

        // Update selection buffer (from GPU picking)
        self.selection_buffer.update(&self.context.queue, &self.picking.selected_residues);

        // Update per-residue color buffer (transition interpolation)
        let _color_transitioning = self.residue_color_buffer.update(&self.context.queue);

        // Update lighting to follow camera (headlamp mode)
        // Use camera.up (set by quaternion) for consistent basis vectors
        let camera = &self.camera_controller.camera;
        let forward = (camera.target - camera.eye).normalize();
        let right = camera.up.cross(forward).normalize();  // right = up × forward
        let up = forward.cross(right);  // recalculate up to ensure orthonormal
        self.lighting.update_headlamp(right, up, forward);
        self.lighting.update_gpu(&self.context.queue);

        // Frustum culling for sidechains - update when camera moves significantly
        self.update_frustum_culling();

        let frame = self.context.get_next_frame()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.context.create_encoder();

        // Geometry pass — render to intermediate color/normal textures at render_scale resolution.
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.post_process.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            // Render order: backbone (tube or ribbon) -> sidechains (all opaque)
            let bind_groups = DrawBindGroups {
                camera: &self.camera_controller.bind_group,
                lighting: &self.lighting.bind_group,
                selection: &self.selection_buffer.bind_group,
                color: Some(&self.residue_color_buffer.bind_group),
            };

            // Backbone: tubes for coils, ribbons for helices/sheets
            self.tube_renderer.draw(&mut rp, &bind_groups);
            self.ribbon_renderer.draw(&mut rp, &bind_groups);

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
            proj,
            view_matrix,
            self.camera_controller.camera.znear,
            self.camera_controller.camera.zfar,
            &view,
        );

        // GPU Picking pass - render residue IDs to picking buffer (includes ribbons)
        let (ribbon_vb, ribbon_ib, ribbon_count) = (
            Some(self.ribbon_renderer.vertex_buffer()),
            Some(self.ribbon_renderer.index_buffer()),
            self.ribbon_renderer.index_count,
        );

        self.picking.render(
            &mut encoder,
            &self.camera_controller.bind_group,
            self.tube_renderer.vertex_buffer(),
            self.tube_renderer.index_buffer(),
            self.tube_renderer.index_count,
            ribbon_vb,
            ribbon_ib,
            ribbon_count,
            self.picking_groups.capsule_picking_bind_group.as_ref(),
            if self.options.display.show_sidechains { self.sidechain_renderer.instance_count } else { 0 },
            self.picking_groups.bns_picking_bind_group.as_ref(),
            self.ball_and_stick_renderer.picking_count(),
            self.input.mouse_pos.0 as u32,
            self.input.mouse_pos.1 as u32,
        );

        self.context.submit(encoder);

        // Start async GPU picking readback (non-blocking)
        self.picking.start_readback();

        // Try to complete any pending readback from previous frame (non-blocking poll)
        self.picking.complete_readback(&self.context.device);

        frame.present();

        // Update frame timing
        self.frame_timing.end_frame();

        Ok(())
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.context.resize(width, height);
            self.camera_controller.resize(width, height);
            self.post_process.resize(&self.context);
            self.picking.resize(&self.context.device, width, height);
        }
    }

    pub fn set_scale_factor(&mut self, scale: f64) {
        let new_render_scale: u32 = if scale < 2.0 { 2 } else { 1 };
        self.context.render_scale = new_render_scale;
    }

    /// Shut down the background scene processor thread.
    pub fn shutdown_scene_processor(&mut self) {
        self.scene_processor.shutdown();
    }
}

// =============================================================================
// Camera
// =============================================================================

impl ProteinRenderEngine {
    /// Fit the camera to show all provided positions (instant)
    pub fn fit_camera_to_positions(&mut self, positions: &[Vec3]) {
        if !positions.is_empty() {
            self.camera_controller.fit_to_positions(positions);
        }
    }

    /// Fit the camera to show all provided positions (animated)
    pub fn fit_camera_to_positions_animated(&mut self, positions: &[Vec3]) {
        if !positions.is_empty() {
            self.camera_controller.fit_to_positions_animated(positions);
        }
    }

    /// Update camera animation. Call this every frame.
    /// Returns true if animation is still in progress.
    pub fn update_camera_animation(&mut self, dt: f32) -> bool {
        self.camera_controller.update_animation(dt)
    }

    /// Convert screen delta (pixels) to world-space offset.
    /// Useful for drag operations like pulling.
    pub fn screen_delta_to_world(&self, delta_x: f32, delta_y: f32) -> Vec3 {
        self.camera_controller.screen_delta_to_world(delta_x, delta_y)
    }

    /// Unproject screen coordinates to a world-space point at the depth of a reference point.
    /// Useful for pull operations where the target should be on a plane at the residue's depth.
    pub fn screen_to_world_at_depth(
        &self,
        screen_x: f32,
        screen_y: f32,
        world_point: Vec3,
    ) -> Vec3 {
        self.camera_controller.screen_to_world_at_depth(
            screen_x,
            screen_y,
            self.context.config.width,
            self.context.config.height,
            world_point,
        )
    }

    /// Fit camera to the currently focused element.
    pub fn fit_camera_to_focus(&mut self) {
        match *self.scene.focus() {
            Focus::Session => {
                let positions = self.scene.all_positions();
                if !positions.is_empty() {
                    self.camera_controller.fit_to_positions_animated(&positions);
                }
            }
            Focus::Group(id) => {
                if let Some(group) = self.scene.group(id) {
                    let positions: Vec<Vec3> = group.entities().iter()
                        .flat_map(|e: &MoleculeEntity| e.positions())
                        .collect();
                    if !positions.is_empty() {
                        self.camera_controller.fit_to_positions_animated(&positions);
                    }
                }
            }
            Focus::Entity(eid) => {
                for g in self.scene.iter() {
                    for e in g.entities() {
                        if e.entity_id == eid {
                            let positions = e.positions();
                            if !positions.is_empty() {
                                self.camera_controller.fit_to_positions_animated(&positions);
                            }
                            return;
                        }
                    }
                }
            }
        }
    }
}
