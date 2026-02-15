use crate::animation::{AnimationAction, StructureAnimator};
use crate::renderer::molecular::ball_and_stick::BallAndStickRenderer;
use crate::renderer::molecular::band::{BandRenderer, BandRenderInfo};
use crate::renderer::postprocess::bloom::BloomPass;
use crate::renderer::molecular::nucleic_acid::NucleicAcidRenderer;
use crate::camera::controller::CameraController;
use crate::camera::input::InputHandler;
use crate::renderer::molecular::capsule_sidechain::CapsuleSidechainRenderer;
use crate::renderer::postprocess::composite::CompositePass;
use crate::renderer::postprocess::fxaa::FxaaPass;
use crate::engine::frame_timing::FrameTiming;
use crate::util::options::Options;
use crate::picking::{Picking, SelectionBuffer};
use crate::util::lighting::Lighting;
use crate::renderer::molecular::pull::{PullRenderer, PullRenderInfo};

use crate::engine::render_context::RenderContext;
use crate::renderer::molecular::ribbon::RibbonRenderer;
use crate::engine::scene_processor::{AnimationSidechainData, PreparedScene, SceneProcessor, SceneRequest};
use crate::engine::shader_composer::ShaderComposer;
use crate::util::trajectory::TrajectoryPlayer;
use foldit_conv::secondary_structure::SSType;
use crate::renderer::postprocess::ssao::SsaoRenderer;
use crate::renderer::molecular::tube::TubeRenderer;

use crate::util::bond_topology::{get_residue_bonds, is_hydrophobic};
use crate::engine::scene::{
    CombinedCoordsResult, EntityGroup, Focus, GroupId, Scene,
};
use foldit_conv::coords::{
    get_ca_position_from_chains, structure_file_to_coords, split_into_entities,
    MoleculeEntity, MoleculeType, RenderCoords,
};
use glam::{Mat4, Vec2, Vec3};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;
use winit::event::MouseButton;
use winit::keyboard::ModifiersState;

/// Target FPS limit
const TARGET_FPS: u32 = 300;

pub struct ProteinRenderEngine {
    pub context: RenderContext,
    pub camera_controller: CameraController,
    pub lighting: Lighting,
    pub sidechain_renderer: CapsuleSidechainRenderer,
    pub band_renderer: BandRenderer,
    pub pull_renderer: PullRenderer,
    pub tube_renderer: TubeRenderer,
    pub ribbon_renderer: RibbonRenderer,
    pub frame_timing: FrameTiming,
    pub input_handler: InputHandler,
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    pub normal_texture: wgpu::Texture,
    pub normal_view: wgpu::TextureView,
    pub ssao_renderer: SsaoRenderer,
    pub bloom_pass: BloomPass,
    pub composite_pass: CompositePass,
    pub fxaa_pass: FxaaPass,
    pub picking: Picking,
    pub selection_buffer: SelectionBuffer,
    /// Bind group for capsule picking (needs to be recreated when capsule buffer changes)
    capsule_picking_bind_group: Option<wgpu::BindGroup>,
    /// Current mouse position for picking
    mouse_pos: (f32, f32),
    /// Residue that was under cursor at mouse down (-1 = background, used for drag vs click logic)
    mouse_down_residue: i32,
    /// Whether we're in a drag operation (mouse moved significantly after mouse down)
    is_dragging: bool,
    /// Last click time for double-click detection
    last_click_time: Instant,
    /// Last clicked residue for double-click detection
    last_click_residue: i32,
    /// Cached secondary structure types per residue (for double-click segment selection)
    cached_ss_types: Vec<SSType>,
    /// Cached per-residue colors (derived from scores by scene processor, reused for animation)
    cached_per_residue_colors: Option<Vec<[f32; 3]>>,
    /// Structure animator for smooth transitions
    pub animator: StructureAnimator,
    /// Start sidechain positions (for animation interpolation)
    start_sidechain_positions: Vec<Vec3>,
    /// Target sidechain positions (animation end state)
    target_sidechain_positions: Vec<Vec3>,
    /// Start backbone-sidechain CA positions (for animation interpolation)
    start_backbone_sidechain_bonds: Vec<(Vec3, u32)>,
    /// Target backbone-sidechain bonds (animation end state)
    target_backbone_sidechain_bonds: Vec<(Vec3, u32)>,
    /// Sidechain bond topology (doesn't change during animation)
    cached_sidechain_bonds: Vec<(u32, u32)>,
    cached_sidechain_hydrophobicity: Vec<bool>,
    cached_sidechain_residue_indices: Vec<u32>,
    /// Sidechain atom names (for looking up atoms by name during animation)
    cached_sidechain_atom_names: Vec<String>,
    /// Last camera eye position for frustum culling change detection
    last_cull_camera_eye: Vec3,
    /// Authoritative scene (all entity groups).
    pub scene: Scene,
    /// Ball-and-stick renderer for ligands, ions, and waters
    pub ball_and_stick_renderer: BallAndStickRenderer,
    /// Nucleic acid backbone ribbon renderer
    pub nucleic_acid_renderer: NucleicAcidRenderer,
    /// Centralized rendering/display options (replaces show_waters, show_ions, etc.)
    pub options: Options,
    /// Currently loaded view preset name (if any)
    pub active_preset: Option<String>,
    /// Bind group for ball-and-stick picking (degenerate capsules)
    bns_picking_bind_group: Option<wgpu::BindGroup>,
    /// Trajectory playback (DCD file)
    trajectory_player: Option<TrajectoryPlayer>,
    /// Background scene processor for non-blocking geometry generation
    scene_processor: SceneProcessor,
    /// Shader composer for naga_oil-based shader composition
    pub shader_composer: ShaderComposer,
}

impl ProteinRenderEngine {
    /// Create a new engine with a default molecule path
    pub async fn new(
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        size: (u32, u32),
        scale_factor: f64,
    ) -> Self {
        Self::new_with_path(window, size, scale_factor, "assets/models/4pnk.cif").await
    }

    /// Create a new engine with a specified molecule path
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
            eprintln!("  entity {} — {:?}: {} atoms", e.entity_id, e.molecule_type, e.coords.num_atoms);
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
            eprintln!("[engine::new] protein_coords: {} atoms", protein_coords.num_atoms);
            let protein_coords = foldit_conv::coords::protein_only(&protein_coords);
            eprintln!("[engine::new] after protein_only: {} atoms", protein_coords.num_atoms);
            let rc = RenderCoords::from_coords_with_topology(
                &protein_coords,
                is_hydrophobic,
                |name| get_residue_bonds(name).map(|b| b.to_vec()),
            );
            eprintln!("[engine::new] render_coords: {} backbone chains, {} residues",
                rc.backbone_chains.len(),
                rc.backbone_chains.iter().map(|c| c.len() / 3).sum::<usize>());
            rc
        } else {
            eprintln!("[engine::new] no protein coords found for group");
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

        let mut tube_renderer = TubeRenderer::new(
            &context,
            &camera_controller.layout,
            &lighting.layout,
            &selection_buffer.layout,
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
            &sidechain_positions,
            &render_coords.sidechain_bonds,
            &render_coords.backbone_sidechain_bonds,
            &sidechain_hydrophobicity,
            &sidechain_residue_indices,
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

        // Create shared depth texture (bindable for SSAO)
        let (depth_texture, depth_view) = Self::create_depth_texture(&context);

        // Create normal G-buffer (world-space normals for SSAO)
        let (normal_texture, normal_view) = Self::create_normal_texture(&context);

        // Create SSAO renderer
        let ssao_renderer = SsaoRenderer::new(&context, &depth_view, &normal_view, &mut shader_composer);

        // Create bloom pass (initially with placeholder input; rebind after composite creates color texture)
        let mut bloom_pass = BloomPass::new(&context, &normal_view, &mut shader_composer); // placeholder input

        // Create composite pass (applies SSAO and outlines to final image)
        let mut composite_pass = CompositePass::new(&context, ssao_renderer.get_ssao_view(), &depth_view, &normal_view, bloom_pass.get_output_view(), &mut shader_composer);

        // Now rebind bloom to read from composite's HDR color texture
        bloom_pass.rebind_input(&context, composite_pass.get_color_view());
        // Set gamma based on whether the swapchain surface format is sRGB
        // If sRGB, the hardware does gamma correction, so gamma = 1.0
        // If non-sRGB (linear), we apply gamma = 1/2.2 in the shader
        composite_pass.params.gamma = if context.config.format.is_srgb() { 1.0 } else { 1.0 / 2.2 };

        // Create FXAA post-process pass (smooths remaining edges after composite)
        let fxaa_pass = FxaaPass::new(&context, &mut shader_composer);

        // Create frame timing with 300 FPS limit
        let frame_timing = FrameTiming::new(TARGET_FPS);

        // Create GPU-based picking
        let picking = Picking::new(&context, &camera_controller.layout, &mut shader_composer);

        // Create initial capsule picking bind group
        let capsule_picking_bind_group = Some(picking.create_capsule_bind_group(
            &context.device,
            sidechain_renderer.capsule_buffer(),
        ));

        // Create ball-and-stick renderer for non-protein entities
        let mut ball_and_stick_renderer = BallAndStickRenderer::new(
            &context,
            &camera_controller.layout,
            &lighting.layout,
            &selection_buffer.layout,
            &mut shader_composer,
        );
        let options = Options::default();
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

        // Create picking bind group for ball-and-stick (degenerate capsules)
        let bns_picking_bind_group = if ball_and_stick_renderer.picking_count() > 0 {
            Some(picking.create_capsule_bind_group(
                &context.device,
                ball_and_stick_renderer.picking_buffer(),
            ))
        } else {
            None
        };

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

        // Mark scene as rendered (initial state is synced)
        scene.mark_rendered();

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
            camera_controller,
            lighting,
            tube_renderer,
            ribbon_renderer,
            sidechain_renderer,
            band_renderer,
            pull_renderer,
            frame_timing,
            input_handler,
            depth_texture,
            depth_view,
            normal_texture,
            normal_view,
            ssao_renderer,
            bloom_pass,
            composite_pass,
            fxaa_pass,
            picking,
            selection_buffer,
            capsule_picking_bind_group,
            mouse_pos: (0.0, 0.0),
            mouse_down_residue: -1,
            is_dragging: false,
            last_click_time: Instant::now(),
            last_click_residue: -1,
            cached_ss_types: Vec::new(),
            cached_per_residue_colors: None,
            animator,
            start_sidechain_positions: Vec::new(),
            target_sidechain_positions: Vec::new(),
            start_backbone_sidechain_bonds: Vec::new(),
            target_backbone_sidechain_bonds: Vec::new(),
            cached_sidechain_bonds: Vec::new(),
            cached_sidechain_hydrophobicity: Vec::new(),
            cached_sidechain_residue_indices: Vec::new(),
            cached_sidechain_atom_names: Vec::new(),
            last_cull_camera_eye: Vec3::ZERO,
            scene,
            ball_and_stick_renderer,
            nucleic_acid_renderer,
            options,
            active_preset: None,
            bns_picking_bind_group,
            trajectory_player: None,
            scene_processor,
            shader_composer,
        }
    }

    fn create_depth_texture(context: &RenderContext) -> (wgpu::Texture, wgpu::TextureView) {
        let size = wgpu::Extent3d {
            width: context.render_width(),
            height: context.render_height(),
            depth_or_array_layers: 1,
        };

        let texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            // Add TEXTURE_BINDING so SSAO can sample from it
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_normal_texture(context: &RenderContext) -> (wgpu::Texture, wgpu::TextureView) {
        let size = wgpu::Extent3d {
            width: context.render_width(),
            height: context.render_height(),
            depth_or_array_layers: 1,
        };

        let texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Normal G-Buffer"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
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
        self.composite_pass.update_fog(&self.context.queue, fog_start, fog_density);

        // Update selection buffer (from GPU picking)
        self.selection_buffer.update(&self.context.queue, &self.picking.selected_residues);

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
                        view: &self.composite_pass.color_view,
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
                        view: &self.normal_view,
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
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            // Render order: backbone (tube or ribbon) -> sidechains (all opaque)
            // Backbone: tubes for coils, ribbons for helices/sheets
            self.tube_renderer.draw(
                &mut rp,
                &self.camera_controller.bind_group,
                &self.lighting.bind_group,
                &self.selection_buffer.bind_group,
            );
            self.ribbon_renderer.draw(
                &mut rp,
                &self.camera_controller.bind_group,
                &self.lighting.bind_group,
                &self.selection_buffer.bind_group,
            );

            if self.options.display.show_sidechains {
                self.sidechain_renderer.draw(
                    &mut rp,
                    &self.camera_controller.bind_group,
                    &self.lighting.bind_group,
                    &self.selection_buffer.bind_group,
                );
            }

            self.ball_and_stick_renderer.draw(
                &mut rp,
                &self.camera_controller.bind_group,
                &self.lighting.bind_group,
                &self.selection_buffer.bind_group,
            );

            self.nucleic_acid_renderer.draw(
                &mut rp,
                &self.camera_controller.bind_group,
                &self.lighting.bind_group,
                &self.selection_buffer.bind_group,
            );

            self.band_renderer.draw(
                &mut rp,
                &self.camera_controller.bind_group,
                &self.lighting.bind_group,
                &self.selection_buffer.bind_group,
            );

            self.pull_renderer.draw(
                &mut rp,
                &self.camera_controller.bind_group,
                &self.lighting.bind_group,
                &self.selection_buffer.bind_group,
            );
        }

        // SSAO pass (compute ambient occlusion from depth buffer)
        let proj = self.camera_controller.camera.build_projection();
        let view_matrix = Mat4::look_at_rh(
            self.camera_controller.camera.eye,
            self.camera_controller.camera.target,
            self.camera_controller.camera.up,
        );
        self.ssao_renderer.update_matrices(
            &self.context.queue,
            proj,
            view_matrix,
            self.camera_controller.camera.znear,
            self.camera_controller.camera.zfar,
        );
        self.ssao_renderer.render_ssao(&mut encoder);

        // Bloom pass — extract bright pixels and blur for glow effect
        self.bloom_pass.render(&mut encoder);

        // Composite pass — apply SSAO + bloom + outlines, output to FXAA input texture
        self.composite_pass.render(&mut encoder, self.fxaa_pass.get_input_view());

        // FXAA pass — screen-space anti-aliasing, output to swapchain
        self.fxaa_pass.render(&mut encoder, &view);

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
            self.capsule_picking_bind_group.as_ref(),
            if self.options.display.show_sidechains { self.sidechain_renderer.instance_count } else { 0 },
            self.bns_picking_bind_group.as_ref(),
            self.ball_and_stick_renderer.picking_count(),
            self.mouse_pos.0 as u32,
            self.mouse_pos.1 as u32,
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
            let (depth_texture, depth_view) = Self::create_depth_texture(&self.context);
            self.depth_texture = depth_texture;
            self.depth_view = depth_view;
            let (normal_texture, normal_view) = Self::create_normal_texture(&self.context);
            self.normal_texture = normal_texture;
            self.normal_view = normal_view;
            self.ssao_renderer.resize(&self.context, &self.depth_view, &self.normal_view);
            self.bloom_pass.resize(&self.context, &self.normal_view); // placeholder, rebind below
            self.composite_pass.resize(&self.context, self.ssao_renderer.get_ssao_view(), &self.depth_view, &self.normal_view, self.bloom_pass.get_output_view());
            self.bloom_pass.rebind_input(&self.context, self.composite_pass.get_color_view());
            self.fxaa_pass.resize(&self.context);
            self.picking.resize(&self.context.device, width, height);
        }
    }

    pub fn set_scale_factor(&mut self, scale: f64) {
        let new_render_scale: u32 = if scale < 2.0 { 2 } else { 1 };
        self.context.render_scale = new_render_scale;
    }

    pub fn handle_mouse_move(&mut self, delta_x: f32, delta_y: f32) {
        // Only allow rotation/pan if mouse down was on background (not on a residue)
        if self.camera_controller.mouse_pressed && self.mouse_down_residue < 0 {
            let delta = Vec2::new(delta_x, delta_y);
            // Mark that we're dragging (moved after mouse down)
            if delta.length_squared() > 1.0 {
                self.is_dragging = true;
            }
            if self.camera_controller.shift_pressed {
                self.camera_controller.pan(delta);
            } else {
                self.camera_controller.rotate(delta);
            }
        }
    }

    pub fn handle_mouse_wheel(&mut self, delta_y: f32) {
        self.camera_controller.zoom(delta_y);
    }

    /// Handle mouse button press/release
    /// On press: record what residue (if any) is under cursor
    /// On release: handled by handle_mouse_up
    pub fn handle_mouse_button(&mut self, button: MouseButton, pressed: bool) {
        if button == MouseButton::Left {
            if pressed {
                // Mouse down - record what's under cursor
                self.mouse_down_residue = self.picking.hovered_residue;
                self.is_dragging = false;
            }
            self.camera_controller.mouse_pressed = pressed;
        }
    }

    /// Handle mouse button release for selection
    /// Returns true if selection changed
    pub fn handle_mouse_up(&mut self) -> bool {
        use std::time::Duration;

        const DOUBLE_CLICK_THRESHOLD: Duration = Duration::from_millis(400);

        let mouse_up_residue = self.picking.hovered_residue;
        let mouse_down_residue = self.mouse_down_residue;
        let now = Instant::now();

        // Reset state
        self.mouse_down_residue = -1;
        let was_dragging = self.is_dragging;
        self.is_dragging = false;

        // If we were dragging (mouse moved significantly), don't do selection
        if was_dragging {
            self.last_click_time = now;
            self.last_click_residue = -1;
            return false;
        }

        // Selection only happens when:
        // 1. Mouse down was on a residue AND mouse up is on the SAME residue
        // 2. Mouse down AND up are both on background (clear selection)
        if mouse_down_residue >= 0 && mouse_down_residue == mouse_up_residue {
            // Check for double-click on the same residue
            let is_double_click = now.duration_since(self.last_click_time) < DOUBLE_CLICK_THRESHOLD
                && self.last_click_residue == mouse_up_residue;

            // Update click tracking
            self.last_click_time = now;
            self.last_click_residue = mouse_up_residue;

            let shift_held = self.camera_controller.shift_pressed;

            if is_double_click {
                // Double-click: select entire secondary structure segment
                // With shift: add to existing selection
                self.select_ss_segment(mouse_up_residue, shift_held)
            } else {
                // Single click: select just this residue
                self.picking.handle_click(shift_held)
            }
        } else if mouse_down_residue < 0 && mouse_up_residue < 0 {
            // Clicked on background - clear selection (only if not dragging)
            self.last_click_time = now;
            self.last_click_residue = -1;
            if !self.picking.selected_residues.is_empty() {
                self.picking.selected_residues.clear();
                true
            } else {
                false
            }
        } else {
            // Mouse down and up on different things - no action
            self.last_click_time = now;
            self.last_click_residue = -1;
            false
        }
    }

    /// Select all residues in the same secondary structure segment as the given residue
    /// If shift_held is true, adds to existing selection; otherwise replaces selection
    fn select_ss_segment(&mut self, residue_idx: i32, shift_held: bool) -> bool {
        if residue_idx < 0 || (residue_idx as usize) >= self.cached_ss_types.len() {
            return false;
        }

        let idx = residue_idx as usize;
        let target_ss = self.cached_ss_types[idx];

        // Find the start of this SS segment (walk backwards)
        let mut start = idx;
        while start > 0 && self.cached_ss_types[start - 1] == target_ss {
            start -= 1;
        }

        // Find the end of this SS segment (walk forwards)
        let mut end = idx;
        while end + 1 < self.cached_ss_types.len() && self.cached_ss_types[end + 1] == target_ss {
            end += 1;
        }

        // If shift is NOT held, clear existing selection first
        if !shift_held {
            self.picking.selected_residues.clear();
        }

        // Add all residues in this segment to selection (avoid duplicates)
        for i in start..=end {
            let residue = i as i32;
            if !self.picking.selected_residues.contains(&residue) {
                self.picking.selected_residues.push(residue);
            }
        }

        true
    }

    /// Handle mouse position update for hover detection
    /// GPU picking uses this position in the next render pass
    pub fn handle_mouse_position(&mut self, x: f32, y: f32) {
        self.mouse_pos = (x, y);
    }

    /// Handle click for residue selection (deprecated - use handle_mouse_up instead)
    /// Returns true if selection changed
    pub fn handle_click(&mut self, _x: f32, _y: f32) -> bool {
        // This is now handled by handle_mouse_up
        false
    }

    /// Get currently hovered residue index (-1 if none)
    pub fn hovered_residue(&self) -> i32 {
        self.picking.hovered_residue
    }

    /// Get currently selected residue indices
    pub fn selected_residues(&self) -> &[i32] {
        &self.picking.selected_residues
    }

    pub fn update_modifiers(&mut self, modifiers: ModifiersState) {
        self.camera_controller.shift_pressed = modifiers.shift_key();
    }

    /// Set shift state directly (from frontend IPC, no winit dependency needed).
    pub fn set_shift_pressed(&mut self, shift: bool) {
        self.camera_controller.shift_pressed = shift;
    }

    /// Update protein atom positions for animation
    /// Handles dynamic buffer resizing if new positions exceed current capacity
    pub fn update_positions(&mut self, _positions: &[Vec3], _hydrophobicity: &[bool]) {
        todo!("need a new update_positions method for capsule sidechains");
    }

    /// Update backbone with new chains (regenerates the tube mesh)
    /// Use this for designed backbones from ML models like RFDiffusion3
    pub fn update_backbone(&mut self, backbone_chains: &[Vec<Vec3>]) {
        self.tube_renderer
            .update_chains(&self.context.device, backbone_chains);
        self.ribbon_renderer.update(
            &self.context.device,
            &self.context.queue,
            backbone_chains,
            None, // use cached ss_override
        );
    }

    /// Update sidechain instances with frustum culling when camera moves significantly.
    /// This filters out sidechains behind the camera to reduce draw calls.
    fn update_frustum_culling(&mut self) {
        // Skip if no sidechain data
        if self.target_sidechain_positions.is_empty() {
            return;
        }

        let camera_eye = self.camera_controller.camera.eye;
        let camera_delta = (camera_eye - self.last_cull_camera_eye).length();

        // Only update culling when camera moves more than 5 units
        // This prevents expensive updates on minor camera movements
        const CULL_UPDATE_THRESHOLD: f32 = 5.0;
        if camera_delta < CULL_UPDATE_THRESHOLD {
            return;
        }

        self.last_cull_camera_eye = camera_eye;

        // Get current frustum
        let frustum = self.camera_controller.frustum();

        // Get current sidechain positions (may be interpolated during animation)
        let positions = if self.animator.is_animating() && self.animator.has_sidechain_data() {
            self.animator.get_sidechain_positions()
        } else {
            self.target_sidechain_positions.clone()
        };

        // Get current backbone-sidechain bonds (may be interpolated)
        let bs_bonds = if self.animator.is_animating() {
            // Interpolate CA positions
            self.target_backbone_sidechain_bonds
                .iter()
                .map(|(target_ca, cb_idx)| {
                    let res_idx = self
                        .cached_sidechain_residue_indices
                        .get(*cb_idx as usize)
                        .copied()
                        .unwrap_or(0) as usize;
                    let ca_pos = self
                        .animator
                        .get_ca_position(res_idx)
                        .unwrap_or(*target_ca);
                    (ca_pos, *cb_idx)
                })
                .collect::<Vec<_>>()
        } else {
            self.target_backbone_sidechain_bonds.clone()
        };

        // Translate entire sidechains onto sheet surface
        let offset_map = self.sheet_offset_map();
        let res_indices = self.cached_sidechain_residue_indices.clone();
        let adjusted_positions = Self::adjust_sidechains_for_sheet(
            &positions, &res_indices, &offset_map,
        );
        let adjusted_bonds = Self::adjust_bonds_for_sheet(
            &bs_bonds, &res_indices, &offset_map,
        );

        // Update sidechains with frustum culling
        self.sidechain_renderer.update_with_frustum(
            &self.context.device,
            &self.context.queue,
            &adjusted_positions,
            &self.cached_sidechain_bonds,
            &adjusted_bonds,
            &self.cached_sidechain_hydrophobicity,
            &self.cached_sidechain_residue_indices,
            Some(&frustum),
        );

        // Recreate picking bind group since buffer may have changed
        self.capsule_picking_bind_group = Some(self.picking.create_capsule_bind_group(
            &self.context.device,
            self.sidechain_renderer.capsule_buffer(),
        ));
    }

    /// Get a reference to the current options.
    pub fn options(&self) -> &Options {
        &self.options
    }

    /// Replace options and apply all changes to subsystems.
    pub fn set_options(&mut self, new: Options) {
        self.options = new;
        self.apply_options();
    }

    /// Push current option values to all subsystems (lighting, camera, composite, etc.).
    pub fn apply_options(&mut self) {
        // Lighting
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

        // Post-processing (outline/AO params; fog is dynamic per-frame)
        let pp = &self.options.post_processing;
        self.composite_pass.params.outline_thickness = pp.outline_thickness;
        self.composite_pass.params.outline_strength = pp.outline_strength;
        self.composite_pass.params.ao_strength = pp.ao_strength;
        self.composite_pass.params.normal_outline_strength = pp.normal_outline_strength;
        self.composite_pass.params.exposure = pp.exposure;
        self.composite_pass.params.bloom_intensity = pp.bloom_intensity;
        self.composite_pass.flush_params(&self.context.queue);

        // Bloom tuning
        self.bloom_pass.threshold = pp.bloom_threshold;
        self.bloom_pass.intensity = pp.bloom_intensity;
        self.bloom_pass.update_params(&self.context.queue);

        // SSAO tuning
        self.ssao_renderer.radius = pp.ao_radius;
        self.ssao_renderer.bias = pp.ao_bias;
        self.ssao_renderer.power = pp.ao_power;

        // Camera
        let co = &self.options.camera;
        self.camera_controller.camera.fovy = co.fovy;
        self.camera_controller.camera.znear = co.znear;
        self.camera_controller.camera.zfar = co.zfar;
        self.camera_controller.rotate_speed = co.rotate_speed * 0.02; // scale to match internal units
        self.camera_controller.pan_speed = co.pan_speed * 0.2;
        self.camera_controller.zoom_speed = co.zoom_speed * 0.5;

        // Display: ball-and-stick visibility
        self.refresh_ball_and_stick();
    }

    /// Apply a single view option by key/value from the frontend.
    /// Returns true if the option was recognized and applied.
    pub fn apply_view_option(&mut self, key: &str, value: &serde_json::Value) -> bool {
        match key {
            "show_sidechains" => {
                if let Some(v) = value.as_bool() {
                    self.options.display.show_sidechains = v;
                    true
                } else {
                    false
                }
            }
            "show_waters" => {
                if let Some(v) = value.as_bool() {
                    self.options.display.show_waters = v;
                    self.refresh_ball_and_stick();
                    true
                } else {
                    false
                }
            }
            "show_ions" => {
                if let Some(v) = value.as_bool() {
                    self.options.display.show_ions = v;
                    self.refresh_ball_and_stick();
                    true
                } else {
                    false
                }
            }
            "show_solvent" => {
                if let Some(v) = value.as_bool() {
                    self.options.display.show_solvent = v;
                    self.refresh_ball_and_stick();
                    true
                } else {
                    false
                }
            }
            "backbone_color_mode" => {
                if let Some(mode_str) = value.as_str() {
                    self.options.display.backbone_color_mode = mode_str.to_string();
                    // When switching back to secondary structure, clear cached colors
                    if mode_str == "secondary_structure" {
                        self.cached_per_residue_colors = None;
                    }
                    // Force dirty so FullRebuild is sent even though scene data hasn't changed
                    self.scene.force_dirty();
                    self.sync_scene_to_renderers(None);
                    true
                } else {
                    false
                }
            }
            // --- Lighting ---
            "lighting.light1_intensity" => self.set_f32_option(value, |s, v| {
                s.options.lighting.light1_intensity = v;
                s.apply_options();
            }),
            "lighting.light2_intensity" => self.set_f32_option(value, |s, v| {
                s.options.lighting.light2_intensity = v;
                s.apply_options();
            }),
            "lighting.ambient" => self.set_f32_option(value, |s, v| {
                s.options.lighting.ambient = v;
                s.apply_options();
            }),
            "lighting.specular_intensity" => self.set_f32_option(value, |s, v| {
                s.options.lighting.specular_intensity = v;
                s.apply_options();
            }),
            "lighting.shininess" => self.set_f32_option(value, |s, v| {
                s.options.lighting.shininess = v;
                s.apply_options();
            }),
            "lighting.rim_power" => self.set_f32_option(value, |s, v| {
                s.options.lighting.rim_power = v;
                s.apply_options();
            }),
            "lighting.rim_intensity" => self.set_f32_option(value, |s, v| {
                s.options.lighting.rim_intensity = v;
                s.apply_options();
            }),
            "lighting.ibl_strength" => self.set_f32_option(value, |s, v| {
                s.options.lighting.ibl_strength = v;
                s.apply_options();
            }),
            // --- Post-processing ---
            "post_processing.outline_thickness" => self.set_f32_option(value, |s, v| {
                s.options.post_processing.outline_thickness = v;
                s.apply_options();
            }),
            "post_processing.outline_strength" => self.set_f32_option(value, |s, v| {
                s.options.post_processing.outline_strength = v;
                s.apply_options();
            }),
            "post_processing.ao_strength" => self.set_f32_option(value, |s, v| {
                s.options.post_processing.ao_strength = v;
                s.apply_options();
            }),
            "post_processing.fog_start" => self.set_f32_option(value, |s, v| {
                s.options.post_processing.fog_start = v;
                s.apply_options();
            }),
            "post_processing.fog_density" => self.set_f32_option(value, |s, v| {
                s.options.post_processing.fog_density = v;
                s.apply_options();
            }),
            "post_processing.ao_radius" => self.set_f32_option(value, |s, v| {
                s.options.post_processing.ao_radius = v;
                s.apply_options();
            }),
            "post_processing.ao_bias" => self.set_f32_option(value, |s, v| {
                s.options.post_processing.ao_bias = v;
                s.apply_options();
            }),
            "post_processing.ao_power" => self.set_f32_option(value, |s, v| {
                s.options.post_processing.ao_power = v;
                s.apply_options();
            }),
            "post_processing.exposure" => self.set_f32_option(value, |s, v| {
                s.options.post_processing.exposure = v;
                s.apply_options();
            }),
            "post_processing.normal_outline_strength" => self.set_f32_option(value, |s, v| {
                s.options.post_processing.normal_outline_strength = v;
                s.apply_options();
            }),
            "post_processing.bloom_intensity" => self.set_f32_option(value, |s, v| {
                s.options.post_processing.bloom_intensity = v;
                s.apply_options();
            }),
            "post_processing.bloom_threshold" => self.set_f32_option(value, |s, v| {
                s.options.post_processing.bloom_threshold = v;
                s.apply_options();
            }),
            "lighting.roughness" => self.set_f32_option(value, |s, v| {
                s.options.lighting.roughness = v;
                s.apply_options();
            }),
            "lighting.metalness" => self.set_f32_option(value, |s, v| {
                s.options.lighting.metalness = v;
                s.apply_options();
            }),
            // --- Camera ---
            "camera.fovy" => self.set_f32_option(value, |s, v| {
                s.options.camera.fovy = v;
                s.apply_options();
            }),
            "camera.rotate_speed" => self.set_f32_option(value, |s, v| {
                s.options.camera.rotate_speed = v;
                s.apply_options();
            }),
            "camera.pan_speed" => self.set_f32_option(value, |s, v| {
                s.options.camera.pan_speed = v;
                s.apply_options();
            }),
            "camera.zoom_speed" => self.set_f32_option(value, |s, v| {
                s.options.camera.zoom_speed = v;
                s.apply_options();
            }),
            _ => {
                log::debug!("Unhandled view option: {}", key);
                false
            }
        }
    }

    /// Helper: extract an f64/f32 from a JSON value and apply a mutation.
    fn set_f32_option(
        &mut self,
        value: &serde_json::Value,
        apply: impl FnOnce(&mut Self, f32),
    ) -> bool {
        if let Some(v) = value.as_f64() {
            apply(self, v as f32);
            true
        } else {
            false
        }
    }

    /// Load a named view preset from the presets directory.
    /// Returns true on success.
    pub fn load_preset(&mut self, name: &str, presets_dir: &std::path::Path) -> bool {
        let path = presets_dir.join(format!("{}.toml", name));
        match Options::load(&path) {
            Ok(opts) => {
                log::info!("Loaded view preset '{}'", name);
                self.set_options(opts);
                self.active_preset = Some(name.to_string());
                true
            }
            Err(e) => {
                log::error!("Failed to load view preset '{}': {}", name, e);
                false
            }
        }
    }

    /// Save the current options as a named view preset.
    /// Returns true on success.
    pub fn save_preset(&self, name: &str, presets_dir: &std::path::Path) -> bool {
        let path = presets_dir.join(format!("{}.toml", name));
        match self.options.save(&path) {
            Ok(()) => {
                log::info!("Saved view preset '{}'", name);
                true
            }
            Err(e) => {
                log::error!("Failed to save view preset '{}': {}", name, e);
                false
            }
        }
    }

    /// Toggle water visibility
    pub fn toggle_waters(&mut self) {
        self.options.display.show_waters = !self.options.display.show_waters;
        self.refresh_ball_and_stick();
    }

    /// Toggle ion visibility
    pub fn toggle_ions(&mut self) {
        self.options.display.show_ions = !self.options.display.show_ions;
        self.refresh_ball_and_stick();
    }

    /// Toggle solvent visibility
    pub fn toggle_solvent(&mut self) {
        self.options.display.show_solvent = !self.options.display.show_solvent;
        self.refresh_ball_and_stick();
    }

    /// Cycle lipid display mode (coarse → ball_and_stick → coarse)
    pub fn toggle_lipids(&mut self) {
        self.options.display.lipid_mode = if self.options.display.lipid_ball_and_stick() {
            "coarse".to_string()
        } else {
            "ball_and_stick".to_string()
        };
        self.refresh_ball_and_stick();
    }

    /// Refresh ball-and-stick renderer with current visibility flags.
    fn refresh_ball_and_stick(&mut self) {
        // Collect all non-protein entities from visible groups
        let entities: Vec<MoleculeEntity> = self.scene.iter()
            .filter(|g| g.visible)
            .flat_map(|g| g.entities().iter())
            .filter(|e| e.molecule_type != MoleculeType::Protein
                && !matches!(e.molecule_type, MoleculeType::DNA | MoleculeType::RNA))
            .cloned()
            .collect();
        self.ball_and_stick_renderer.update_from_entities(
            &self.context.device,
            &self.context.queue,
            &entities,
            &self.options.display,
            Some(&self.options.colors),
        );
        // Recreate picking bind group
        self.bns_picking_bind_group = if self.ball_and_stick_renderer.picking_count() > 0 {
            Some(self.picking.create_capsule_bind_group(
                &self.context.device,
                self.ball_and_stick_renderer.picking_buffer(),
            ))
        } else {
            None
        };
    }

    /// Load a DCD trajectory file and begin playback.
    pub fn load_trajectory(&mut self, path: &Path) {
        use foldit_conv::coords::{dcd_file_to_frames, protein_only};
        use crate::util::trajectory::build_backbone_atom_indices;

        let (header, frames) = match dcd_file_to_frames(path) {
            Ok(r) => r,
            Err(e) => {
                log::error!("Failed to load DCD trajectory: {e}");
                return;
            }
        };

        // Get protein coords from the first visible group to build backbone mapping
        let protein_coords = self.scene.iter()
            .filter(|g| g.visible)
            .find_map(|g| g.protein_coords());

        let protein_coords = match protein_coords {
            Some(c) => protein_only(&c),
            None => {
                log::error!("No protein structure loaded — cannot play trajectory");
                return;
            }
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
        let backbone_chains = foldit_conv::coords::extract_backbone_chains(&protein_coords);

        let num_atoms = header.num_atoms as usize;
        let num_frames = frames.len();
        let duration_secs = num_frames as f64 / 30.0;

        let player = TrajectoryPlayer::new(frames, num_atoms, &backbone_chains, backbone_indices);
        self.trajectory_player = Some(player);

        log::info!(
            "Trajectory loaded: {} frames, {} atoms, ~{:.1}s at 30fps",
            num_frames,
            num_atoms,
            duration_secs,
        );
    }

    /// Toggle trajectory playback (play/pause). No-op if no trajectory loaded.
    pub fn toggle_trajectory(&mut self) {
        if let Some(ref mut player) = self.trajectory_player {
            player.toggle_playback();
            let state = if player.is_playing() { "playing" } else { "paused" };
            log::info!("Trajectory {state} (frame {}/{})", player.current_frame(), player.total_frames());
        }
    }

    /// Whether a trajectory is loaded.
    pub fn has_trajectory(&self) -> bool {
        self.trajectory_player.is_some()
    }

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

    /// Get the GPU queue for buffer updates
    pub fn queue(&self) -> &wgpu::Queue {
        &self.context.queue
    }

    /// Update all renderers from aggregated scene data
    ///
    /// This is the main integration point for the Scene-based rendering model.
    /// Call this whenever structures are added, removed, or modified in the scene.
    pub fn update_from_aggregated(
        &mut self,
        backbone_chains: &[Vec<Vec3>],
        sidechain_positions: &[Vec3],
        sidechain_hydrophobicity: &[bool],
        sidechain_residue_indices: &[u32],
        sidechain_atom_names: &[String],
        sidechain_bonds: &[(u32, u32)],
        backbone_sidechain_bonds: &[(Vec3, u32)], // (CA position, CB index)
        all_positions: &[Vec3],
        fit_camera: bool,
        ss_types: Option<&[SSType]>,
    ) {
        // Calculate total residues from backbone chains (3 atoms per residue: N, CA, C)
        let total_residues: usize = backbone_chains.iter().map(|c| c.len() / 3).sum();

        // Ensure selection buffer has capacity for all residues (including new structures)
        self.selection_buffer.ensure_capacity(&self.context.device, total_residues);

        // Update backbone tubes
        self.tube_renderer.update(
            &self.context.device,
            &self.context.queue,
            backbone_chains,
            ss_types,
        );

        // Update ribbon renderer
        self.ribbon_renderer.update(
            &self.context.device,
            &self.context.queue,
            backbone_chains,
            ss_types,
        );

        // Translate sidechains onto sheet surface (whole sidechain, not just CA-CB bond)
        let offset_map = self.sheet_offset_map();
        let adjusted_positions = Self::adjust_sidechains_for_sheet(
            sidechain_positions, sidechain_residue_indices, &offset_map,
        );
        let adjusted_bonds = Self::adjust_bonds_for_sheet(
            backbone_sidechain_bonds, sidechain_residue_indices, &offset_map,
        );

        self.sidechain_renderer.update(
            &self.context.device,
            &self.context.queue,
            &adjusted_positions,
            sidechain_bonds,
            &adjusted_bonds,
            sidechain_hydrophobicity,
            sidechain_residue_indices,
        );

        // Update capsule picking bind group (buffer may have been reallocated)
        self.capsule_picking_bind_group = Some(self.picking.create_capsule_bind_group(
            &self.context.device,
            self.sidechain_renderer.capsule_buffer(),
        ));

        // Cache secondary structure types for double-click segment selection
        self.cached_ss_types = if let Some(ss) = ss_types {
            ss.to_vec()
        } else {
            self.compute_ss_types(backbone_chains)
        };

        // Cache atom names for lookup by name (used for band tracking during animation)
        self.cached_sidechain_atom_names = sidechain_atom_names.to_vec();

        // Fit camera if requested and we have positions
        if fit_camera && !all_positions.is_empty() {
            self.camera_controller.fit_to_positions(all_positions);
        }
    }

    /// Set SS override (from puzzle.toml annotation). Updates cached types
    /// and forces tube/ribbon renderer regeneration.
    pub fn set_ss_override(&mut self, ss_types: &[SSType]) {
        self.cached_ss_types = ss_types.to_vec();
        self.tube_renderer.set_ss_override(Some(ss_types.to_vec()));
        self.tube_renderer.regenerate(&self.context.device, &self.context.queue);
        self.ribbon_renderer.set_ss_override(Some(ss_types.to_vec()));
        self.ribbon_renderer.regenerate(&self.context.device, &self.context.queue);
    }

    /// Compute secondary structure types for all residues across all chains
    fn compute_ss_types(&self, backbone_chains: &[Vec<Vec3>]) -> Vec<SSType> {
        use foldit_conv::secondary_structure::auto::detect as detect_secondary_structure;

        let mut all_ss_types = Vec::new();

        for chain in backbone_chains {
            // Extract CA positions (every 3rd atom starting at index 1: N, CA, C pattern)
            let ca_positions: Vec<Vec3> = chain
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 3 == 1)
                .map(|(_, &pos)| pos)
                .collect();

            let ss_types = detect_secondary_structure(&ca_positions);
            all_ss_types.extend(ss_types);
        }

        all_ss_types
    }

    /// Build a map of sheet residue offsets (residue_idx -> offset vector).
    fn sheet_offset_map(&self) -> HashMap<u32, Vec3> {
        self.ribbon_renderer.sheet_offsets().iter().copied().collect()
    }

    /// Adjust backbone-sidechain bond CA positions by sheet flattening offsets.
    fn adjust_bonds_for_sheet(
        bonds: &[(Vec3, u32)],
        sidechain_residue_indices: &[u32],
        offset_map: &HashMap<u32, Vec3>,
    ) -> Vec<(Vec3, u32)> {
        if offset_map.is_empty() { return bonds.to_vec(); }
        bonds.iter().map(|(ca_pos, cb_idx)| {
            let res_idx = sidechain_residue_indices
                .get(*cb_idx as usize)
                .copied()
                .unwrap_or(u32::MAX);
            if let Some(&offset) = offset_map.get(&res_idx) {
                (*ca_pos + offset, *cb_idx)
            } else {
                (*ca_pos, *cb_idx)
            }
        }).collect()
    }

    /// Translate all sidechain atom positions by sheet flattening offsets.
    /// Moves entire sidechains onto the sheet surface, not just the CA-CB bond.
    fn adjust_sidechains_for_sheet(
        positions: &[Vec3],
        sidechain_residue_indices: &[u32],
        offset_map: &HashMap<u32, Vec3>,
    ) -> Vec<Vec3> {
        if offset_map.is_empty() { return positions.to_vec(); }
        positions.iter().enumerate().map(|(i, &pos)| {
            let res_idx = sidechain_residue_indices
                .get(i)
                .copied()
                .unwrap_or(u32::MAX);
            if let Some(&offset) = offset_map.get(&res_idx) {
                pos + offset
            } else {
                pos
            }
        }).collect()
    }

    // =========================================================================
    // Animation Methods (delegate to StructureAnimator)
    // =========================================================================

    /// Animate backbone to new pose with specified action.
    pub fn animate_to_pose(&mut self, new_backbone: &[Vec<Vec3>], action: AnimationAction) {
        self.animator.set_target(new_backbone, action);

        // If animator has visual state, update renderers
        if self.animator.residue_count() > 0 {
            let visual_backbone = self.animator.get_backbone();
            self.update_backbone(&visual_backbone);
        }
    }

    /// Animate to new pose with sidechain data.
    ///
    /// Uses AnimationAction::Wiggle by default for backwards compatibility.
    pub fn animate_to_full_pose(
        &mut self,
        new_backbone: &[Vec<Vec3>],
        sidechain_positions: &[Vec3],
        sidechain_bonds: &[(u32, u32)],
        sidechain_hydrophobicity: &[bool],
        sidechain_residue_indices: &[u32],
        sidechain_atom_names: &[String],
        backbone_sidechain_bonds: &[(Vec3, u32)],
    ) {
        self.animate_to_full_pose_with_action(
            new_backbone,
            sidechain_positions,
            sidechain_bonds,
            sidechain_hydrophobicity,
            sidechain_residue_indices,
            sidechain_atom_names,
            backbone_sidechain_bonds,
            AnimationAction::Wiggle,
        );
    }

    /// Animate to new pose with sidechain data and explicit action.
    pub fn animate_to_full_pose_with_action(
        &mut self,
        new_backbone: &[Vec<Vec3>],
        sidechain_positions: &[Vec3],
        sidechain_bonds: &[(u32, u32)],
        sidechain_hydrophobicity: &[bool],
        sidechain_residue_indices: &[u32],
        sidechain_atom_names: &[String],
        backbone_sidechain_bonds: &[(Vec3, u32)],
        action: AnimationAction,
    ) {
        // Capture current VISUAL positions as start (for smooth preemption)
        // If animation is in progress, use interpolated positions, not old targets
        if self.target_sidechain_positions.len() == sidechain_positions.len() {
            if self.animator.is_animating() && self.animator.has_sidechain_data() {
                // Animation in progress - sync to current visual state (like backbone does)
                self.start_sidechain_positions = self.animator.get_sidechain_positions();
                // Also interpolate backbone-sidechain bonds
                let ctx = self.animator.interpolation_context();
                self.start_backbone_sidechain_bonds = self
                    .start_backbone_sidechain_bonds
                    .iter()
                    .zip(self.target_backbone_sidechain_bonds.iter())
                    .map(|((start_pos, idx), (target_pos, _))| {
                        let pos = *start_pos + (*target_pos - *start_pos) * ctx.eased_t;
                        (pos, *idx)
                    })
                    .collect();
            } else {
                // No animation - use previous target as new start
                self.start_sidechain_positions = self.target_sidechain_positions.clone();
                self.start_backbone_sidechain_bonds = self.target_backbone_sidechain_bonds.clone();
            }
        } else {
            // Size changed - snap to new positions
            self.start_sidechain_positions = sidechain_positions.to_vec();
            self.start_backbone_sidechain_bonds = backbone_sidechain_bonds.to_vec();
        }

        // Set new targets and cached data
        self.target_sidechain_positions = sidechain_positions.to_vec();
        self.target_backbone_sidechain_bonds = backbone_sidechain_bonds.to_vec();
        self.cached_sidechain_bonds = sidechain_bonds.to_vec();
        self.cached_sidechain_hydrophobicity = sidechain_hydrophobicity.to_vec();
        self.cached_sidechain_residue_indices = sidechain_residue_indices.to_vec();
        self.cached_sidechain_atom_names = sidechain_atom_names.to_vec();

        // Extract CA positions from backbone for sidechain collapse animation
        // CA is the second atom (index 1) in each group of 3 (N, CA, C) per residue
        let ca_positions: Vec<Vec3> = new_backbone
            .iter()
            .flat_map(|chain| chain.chunks(3).filter_map(|chunk| chunk.get(1).copied()))
            .collect();

        // Pass sidechain data to animator FIRST (before set_target)
        // This allows set_target to detect sidechain changes and force animation
        // even when backbone is unchanged (for Shake/MPNN animations)
        self.animator.set_sidechain_target_with_action(
            sidechain_positions,
            sidechain_residue_indices,
            &ca_positions,
            Some(action),
        );

        // Set backbone target (this starts the animation, checking sidechain changes)
        self.animator.set_target(new_backbone, action);

        // Update renderers with START visual state (animation will interpolate from here)
        if self.animator.residue_count() > 0 {
            let visual_backbone = self.animator.get_backbone();
            self.update_backbone(&visual_backbone);
        }

        // Update sidechain renderer with start positions (adjusted for sheet surface)
        let offset_map = self.sheet_offset_map();
        let adjusted_positions = Self::adjust_sidechains_for_sheet(
            &self.start_sidechain_positions, sidechain_residue_indices, &offset_map,
        );
        let adjusted_bonds = Self::adjust_bonds_for_sheet(
            &self.start_backbone_sidechain_bonds, sidechain_residue_indices, &offset_map,
        );
        self.sidechain_renderer.update(
            &self.context.device,
            &self.context.queue,
            &adjusted_positions,
            sidechain_bonds,
            &adjusted_bonds,
            sidechain_hydrophobicity,
            sidechain_residue_indices,
        );
    }

    /// Skip all animations to final state.
    pub fn skip_animations(&mut self) {
        self.animator.skip();
    }

    /// Cancel all animations.
    pub fn cancel_animations(&mut self) {
        self.animator.cancel();
    }

    /// Check if animations are active.
    #[inline]
    pub fn is_animating(&self) -> bool {
        self.animator.is_animating()
    }

    /// Set animation enabled/disabled.
    pub fn set_animation_enabled(&mut self, enabled: bool) {
        self.animator.set_enabled(enabled);
    }

    // =========================================================================
    // Band Methods
    // =========================================================================

    /// Update the band visualization.
    /// Call this when bands are added, removed, or modified.
    pub fn update_bands(&mut self, bands: &[BandRenderInfo]) {
        self.band_renderer.update(
            &self.context.device,
            &self.context.queue,
            bands,
            Some(&self.options.colors),
        );
    }

    /// Clear all band visualizations.
    pub fn clear_bands(&mut self) {
        self.band_renderer.clear();
    }

    // =========================================================================
    // Pull Renderer Methods
    // =========================================================================

    /// Update the pull visualization (only one pull at a time).
    /// Pass None to clear the pull visualization.
    pub fn update_pull(&mut self, pull: Option<&PullRenderInfo>) {
        self.pull_renderer.update(
            &self.context.device,
            &self.context.queue,
            pull,
        );
    }

    /// Clear the pull visualization.
    pub fn clear_pull(&mut self) {
        self.pull_renderer.clear();
    }

    // =========================================================================
    // Pull Action Methods (camera/position helpers)
    // =========================================================================

    /// Get the CA position of a residue by index.
    /// Returns None if the residue index is out of bounds.
    pub fn get_residue_ca_position(&self, residue_idx: usize) -> Option<Vec3> {
        // First try animator (has interpolated positions during animation)
        if let Some(pos) = self.animator.get_ca_position(residue_idx) {
            return Some(pos);
        }

        // Fall back to tube_renderer's cached backbone chains
        // Uses foldit-conv's get_ca_position_from_chains which extracts CA (index 1 in N, CA, C triplet)
        get_ca_position_from_chains(self.tube_renderer.cached_chains(), residue_idx)
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

    /// Get current screen dimensions.
    pub fn screen_size(&self) -> (u32, u32) {
        (self.context.config.width, self.context.config.height)
    }

    // =========================================================================
    // Interpolated Position Methods (for bands/constraints during animation)
    // =========================================================================

    /// Get the current visual backbone chains (interpolated during animation).
    /// Use this for constraint visualizations that need to follow the animated backbone.
    pub fn get_current_backbone_chains(&self) -> Vec<Vec<Vec3>> {
        if self.animator.is_animating() {
            self.animator.get_backbone()
        } else {
            // Return cached chains from tube renderer when not animating
            self.tube_renderer.cached_chains().to_vec()
        }
    }

    /// Get the current visual sidechain positions (interpolated during animation).
    /// Use this for constraint visualizations that need to follow the animated sidechains.
    pub fn get_current_sidechain_positions(&self) -> Vec<Vec3> {
        if self.animator.is_animating() && self.animator.has_sidechain_data() {
            self.animator.get_sidechain_positions()
        } else {
            self.target_sidechain_positions.clone()
        }
    }

    /// Get the current visual CA positions for all residues (interpolated during animation).
    /// This is useful for band endpoints that need to track residue positions.
    pub fn get_current_ca_positions(&self) -> Vec<Vec3> {
        let chains = self.get_current_backbone_chains();
        foldit_conv::coords::extract_ca_from_chains(&chains)
    }

    /// Get a single interpolated CA position by residue index.
    /// Returns None if out of bounds.
    pub fn get_current_ca_position(&self, residue_idx: usize) -> Option<Vec3> {
        // Animator already handles interpolation in get_ca_position
        if let Some(pos) = self.animator.get_ca_position(residue_idx) {
            return Some(pos);
        }

        // Fall back to tube_renderer's cached chains
        get_ca_position_from_chains(self.tube_renderer.cached_chains(), residue_idx)
    }

    /// Check if structure animation is currently in progress.
    /// Useful for determining if band positions should be updated.
    #[inline]
    pub fn needs_band_update(&self) -> bool {
        self.animator.is_animating()
    }

    /// Get the current animation progress (0.0 to 1.0).
    /// Returns 1.0 if no animation is active.
    pub fn animation_progress(&self) -> f32 {
        self.animator.progress()
    }

    /// Get the interpolated position of the closest atom to a reference point for a given residue.
    /// This is useful for bands that need to track a specific atom (not just CA) during animation.
    /// The reference_point is typically the original atom position when the band was created.
    pub fn get_closest_atom_for_residue(
        &self,
        residue_idx: usize,
        reference_point: Vec3,
    ) -> Option<Vec3> {
        let backbone_chains = self.get_current_backbone_chains();
        let sidechain_positions = self.get_current_sidechain_positions();

        foldit_conv::coords::get_closest_atom_for_residue(
            &backbone_chains,
            &sidechain_positions,
            &self.cached_sidechain_residue_indices,
            residue_idx,
            reference_point,
        )
    }

    /// Get the interpolated position of a specific atom by residue index and atom name.
    /// This is the preferred method for band endpoint tracking as it reliably identifies
    /// the same atom across animation frames.
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
        for (i, (res_idx, name)) in self.cached_sidechain_residue_indices
            .iter()
            .zip(self.cached_sidechain_atom_names.iter())
            .enumerate()
        {
            if *res_idx as usize == residue_idx && name == atom_name {
                return sidechain_positions.get(i).copied();
            }
        }

        None
    }

    // =========================================================================
    // Scene API — Group Management (delegates to self.scene)
    // =========================================================================

    /// Load entities into a new group. Optionally fits camera.
    pub fn load_entities(
        &mut self,
        entities: Vec<MoleculeEntity>,
        name: &str,
        fit_camera: bool,
    ) -> GroupId {
        let id = self.scene.add_group(entities, name);
        if fit_camera {
            // Sync immediately so aggregated data is available for camera fit
            self.sync_scene_to_renderers(Some(AnimationAction::Load));
            let positions = self.scene.all_positions();
            if !positions.is_empty() {
                self.camera_controller.fit_to_positions(&positions);
            }
        }
        id
    }

    /// Remove a group by ID.
    pub fn remove_group(&mut self, id: GroupId) -> Option<EntityGroup> {
        self.scene.remove_group(id)
    }

    /// Set group visibility.
    pub fn set_group_visible(&mut self, id: GroupId, visible: bool) {
        self.scene.set_visible(id, visible);
    }

    /// Clear all groups.
    pub fn clear_scene(&mut self) {
        self.scene.clear();
    }

    /// Read access to a group.
    pub fn group(&self, id: GroupId) -> Option<&EntityGroup> {
        self.scene.group(id)
    }

    /// Write access to a group (invalidates scene cache).
    pub fn group_mut(&mut self, id: GroupId) -> Option<&mut EntityGroup> {
        self.scene.group_mut(id)
    }

    /// Ordered group IDs.
    pub fn group_ids(&self) -> Vec<GroupId> {
        self.scene.group_ids()
    }

    /// Number of groups.
    pub fn group_count(&self) -> usize {
        self.scene.group_count()
    }

    // ── Focus / tab cycling ──

    /// Cycle focus: Session → Group1 → ... → GroupN → focusable entities → Session.
    pub fn cycle_focus(&mut self) -> Focus {
        self.scene.cycle_focus()
    }

    /// Current focus.
    pub fn focus(&self) -> &Focus {
        self.scene.focus()
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

    // ── Backend support ──

    /// Combined coords for Rosetta.
    pub fn combined_coords_for_backend(&self) -> Option<CombinedCoordsResult> {
        self.scene.combined_coords_for_backend()
    }

    /// Visible group IDs and their residue counts (for Rosetta topology check).
    pub fn visible_residue_counts(&self) -> Vec<(GroupId, usize)> {
        self.scene.visible_residue_counts()
    }

    // ── Updates from backends (with animation) ──

    /// Sync scene to renderers with the given animation action.
    ///
    /// **Non-blocking**: clones the aggregated data and sends it to the background
    /// SceneProcessor thread for CPU geometry generation. Call `apply_pending_scene()`
    /// each frame to pick up the results.
    pub fn sync_scene_to_renderers(&mut self, action: Option<AnimationAction>) {
        if !self.scene.is_dirty() && action.is_none() {
            return;
        }

        let groups = self.scene.per_group_data();
        let agg = self.scene.aggregated(); // Arc::clone (~1ns), no deep copy
        self.scene.mark_rendered();

        self.scene_processor.submit(SceneRequest::FullRebuild {
            groups,
            aggregated: agg,
            action,
            display: self.options.display.clone(),
            colors: self.options.colors.clone(),
        });
    }

    /// Apply any pending scene data from the background SceneProcessor.
    ///
    /// Called every frame from the main loop. If the background thread has finished
    /// generating geometry, this uploads it to the GPU (<1ms) and sets up animation.
    pub fn apply_pending_scene(&mut self) {
        let prepared = match self.scene_processor.try_recv_scene() {
            Some(p) => p,
            None => return,
        };

        // Triple buffer automatically returns only the latest result,
        // so no drain loop needed — stale intermediates are skipped.

        // Animation target setup or snap update (fast: array copies + animator)
        if let Some(action) = prepared.action {
            self.setup_animation_targets_from_prepared(&prepared, action);
        } else {
            self.snap_from_prepared(&prepared);
        }

        // GPU uploads only — each is <0.2ms
        self.tube_renderer.apply_prepared(
            &self.context.device,
            &self.context.queue,
            &prepared.tube_vertices,
            &prepared.tube_indices,
            prepared.tube_index_count,
            prepared.backbone_chains.clone(),
            prepared.ss_types.clone(),
        );

        self.ribbon_renderer.apply_prepared(
            &self.context.device,
            &self.context.queue,
            &prepared.ribbon_vertices,
            &prepared.ribbon_indices,
            prepared.ribbon_index_count,
            prepared.sheet_offsets.clone(),
            prepared.backbone_chains.clone(),
            prepared.ss_types.clone(),
        );

        self.sidechain_renderer.apply_prepared(
            &self.context.device,
            &self.context.queue,
            &prepared.sidechain_instances,
            prepared.sidechain_instance_count,
        );

        self.ball_and_stick_renderer.apply_prepared(
            &self.context.device,
            &self.context.queue,
            &prepared.bns_sphere_instances,
            prepared.bns_sphere_count,
            &prepared.bns_capsule_instances,
            prepared.bns_capsule_count,
            &prepared.bns_picking_capsules,
            prepared.bns_picking_count,
        );

        self.nucleic_acid_renderer.apply_prepared(
            &self.context.device,
            &self.context.queue,
            &prepared.na_vertices,
            &prepared.na_indices,
            prepared.na_index_count,
        );

        // Recreate picking bind groups (buffers may have been reallocated)
        self.capsule_picking_bind_group = Some(self.picking.create_capsule_bind_group(
            &self.context.device,
            self.sidechain_renderer.capsule_buffer(),
        ));
        self.bns_picking_bind_group = if self.ball_and_stick_renderer.picking_count() > 0 {
            Some(self.picking.create_capsule_bind_group(
                &self.context.device,
                self.ball_and_stick_renderer.picking_buffer(),
            ))
        } else {
            None
        };
    }

    /// Set up animation targets from prepared scene data.
    ///
    /// This is the fast path extracted from `animate_to_full_pose_with_action`:
    /// only array copies and animator state, no mesh generation.
    fn setup_animation_targets_from_prepared(
        &mut self,
        prepared: &PreparedScene,
        action: AnimationAction,
    ) {
        let new_backbone = &prepared.backbone_chains;
        let sidechain_positions = &prepared.sidechain_positions;
        let sidechain_bonds = &prepared.sidechain_bonds;
        let sidechain_hydrophobicity = &prepared.sidechain_hydrophobicity;
        let sidechain_residue_indices = &prepared.sidechain_residue_indices;
        let sidechain_atom_names = &prepared.sidechain_atom_names;
        let backbone_sidechain_bonds = &prepared.backbone_sidechain_bonds;

        // Capture current VISUAL positions as start (for smooth preemption)
        if self.target_sidechain_positions.len() == sidechain_positions.len() {
            if self.animator.is_animating() && self.animator.has_sidechain_data() {
                self.start_sidechain_positions = self.animator.get_sidechain_positions();
                let ctx = self.animator.interpolation_context();
                self.start_backbone_sidechain_bonds = self
                    .start_backbone_sidechain_bonds
                    .iter()
                    .zip(self.target_backbone_sidechain_bonds.iter())
                    .map(|((start_pos, idx), (target_pos, _))| {
                        let pos = *start_pos + (*target_pos - *start_pos) * ctx.eased_t;
                        (pos, *idx)
                    })
                    .collect();
            } else {
                self.start_sidechain_positions = self.target_sidechain_positions.clone();
                self.start_backbone_sidechain_bonds =
                    self.target_backbone_sidechain_bonds.clone();
            }
        } else {
            self.start_sidechain_positions = sidechain_positions.to_vec();
            self.start_backbone_sidechain_bonds = backbone_sidechain_bonds.to_vec();
        }

        // Set new targets and cached data
        self.target_sidechain_positions = sidechain_positions.to_vec();
        self.target_backbone_sidechain_bonds = backbone_sidechain_bonds.to_vec();
        self.cached_sidechain_bonds = sidechain_bonds.to_vec();
        self.cached_sidechain_hydrophobicity = sidechain_hydrophobicity.to_vec();
        self.cached_sidechain_residue_indices = sidechain_residue_indices.to_vec();
        self.cached_sidechain_atom_names = sidechain_atom_names.to_vec();

        // Cache secondary structure types
        if let Some(ref ss) = prepared.ss_types {
            self.cached_ss_types = ss.clone();
        } else {
            self.cached_ss_types = self.compute_ss_types(new_backbone);
        }

        // Cache per-residue colors (derived from scores by scene processor)
        self.cached_per_residue_colors = prepared.per_residue_colors.clone();

        // Extract CA positions for sidechain collapse animation
        let ca_positions: Vec<Vec3> = new_backbone
            .iter()
            .flat_map(|chain| chain.chunks(3).filter_map(|chunk| chunk.get(1).copied()))
            .collect();

        // Pass sidechain data to animator FIRST (before set_target)
        self.animator.set_sidechain_target_with_action(
            sidechain_positions,
            sidechain_residue_indices,
            &ca_positions,
            Some(action),
        );

        // Set backbone target (starts the animation)
        self.animator.set_target(new_backbone, action);

        // Ensure selection buffer has capacity
        let total_residues: usize = new_backbone.iter().map(|c| c.len() / 3).sum();
        self.selection_buffer
            .ensure_capacity(&self.context.device, total_residues);
    }

    /// Snap update from prepared scene data (no animation).
    fn snap_from_prepared(&mut self, prepared: &PreparedScene) {
        // Cache all passthrough data
        self.target_sidechain_positions = prepared.sidechain_positions.clone();
        self.start_sidechain_positions = prepared.sidechain_positions.clone();
        self.target_backbone_sidechain_bonds = prepared.backbone_sidechain_bonds.clone();
        self.start_backbone_sidechain_bonds = prepared.backbone_sidechain_bonds.clone();
        self.cached_sidechain_bonds = prepared.sidechain_bonds.clone();
        self.cached_sidechain_hydrophobicity = prepared.sidechain_hydrophobicity.clone();
        self.cached_sidechain_residue_indices = prepared.sidechain_residue_indices.clone();
        self.cached_sidechain_atom_names = prepared.sidechain_atom_names.clone();

        // Cache secondary structure types
        if let Some(ref ss) = prepared.ss_types {
            self.cached_ss_types = ss.clone();
        } else {
            self.cached_ss_types = self.compute_ss_types(&prepared.backbone_chains);
        }

        // Cache per-residue colors (derived from scores by scene processor)
        self.cached_per_residue_colors = prepared.per_residue_colors.clone();

        // Ensure selection buffer has capacity
        let total_residues: usize = prepared.backbone_chains.iter().map(|c| c.len() / 3).sum();
        self.selection_buffer
            .ensure_capacity(&self.context.device, total_residues);
    }

    /// Shut down the background scene processor thread.
    pub fn shutdown_scene_processor(&mut self) {
        self.scene_processor.shutdown();
    }

    /// Submit an animation frame to the background thread for mesh generation.
    ///
    /// Gets interpolated backbone and sidechain positions from the animator
    /// and sends them to the SceneProcessor. Returns immediately (~0.3ms).
    fn submit_animation_frame(&mut self) {
        let visual_backbone = self.animator.get_backbone();
        let has_sidechains = self.animator.has_sidechain_data();
        self.submit_animation_frame_with_backbone(visual_backbone, has_sidechains);
    }

    /// Submit an animation frame with explicit backbone chains.
    ///
    /// Used by both the animator path and trajectory playback.
    fn submit_animation_frame_with_backbone(
        &mut self,
        backbone_chains: Vec<Vec<Vec3>>,
        include_sidechains: bool,
    ) {
        let sidechains = if include_sidechains {
            let interpolated_positions = self.animator.get_sidechain_positions();

            // Compute interpolated backbone-sidechain bond CA positions
            let interpolated_bs_bonds: Vec<(Vec3, u32)> = self
                .target_backbone_sidechain_bonds
                .iter()
                .map(|(target_ca_pos, cb_idx)| {
                    let res_idx = self
                        .cached_sidechain_residue_indices
                        .get(*cb_idx as usize)
                        .copied()
                        .unwrap_or(0) as usize;
                    let ca_pos = self
                        .animator
                        .get_ca_position(res_idx)
                        .unwrap_or(*target_ca_pos);
                    (ca_pos, *cb_idx)
                })
                .collect();

            Some(AnimationSidechainData {
                sidechain_positions: interpolated_positions,
                sidechain_bonds: self.cached_sidechain_bonds.clone(),
                backbone_sidechain_bonds: interpolated_bs_bonds,
                sidechain_hydrophobicity: self.cached_sidechain_hydrophobicity.clone(),
                sidechain_residue_indices: self.cached_sidechain_residue_indices.clone(),
            })
        } else {
            None
        };

        let ss_types = if self.cached_ss_types.is_empty() {
            None
        } else {
            Some(self.cached_ss_types.clone())
        };

        self.scene_processor.submit(SceneRequest::AnimationFrame {
            backbone_chains,
            sidechains,
            ss_types,
            per_residue_colors: self.cached_per_residue_colors.clone(),
        });
    }

    /// Apply any pending animation frame from the background thread.
    ///
    /// Called every frame before the animator update. If the background thread
    /// has finished generating mesh, uploads it to the GPU (<0.5ms).
    fn apply_pending_animation(&mut self) {
        let prepared = match self.scene_processor.try_recv_animation() {
            Some(p) => p,
            None => return,
        };

        // Upload tube mesh
        self.tube_renderer.apply_mesh(
            &self.context.device,
            &self.context.queue,
            &prepared.tube_vertices,
            &prepared.tube_indices,
            prepared.tube_index_count,
        );

        // Upload ribbon mesh
        self.ribbon_renderer.apply_mesh(
            &self.context.device,
            &self.context.queue,
            &prepared.ribbon_vertices,
            &prepared.ribbon_indices,
            prepared.ribbon_index_count,
            prepared.sheet_offsets,
        );

        // Upload sidechain instances if present
        if let Some(ref instances) = prepared.sidechain_instances {
            let reallocated = self.sidechain_renderer.apply_prepared(
                &self.context.device,
                &self.context.queue,
                instances,
                prepared.sidechain_instance_count,
            );
            // Recreate capsule picking bind group if buffer was reallocated
            if reallocated {
                self.capsule_picking_bind_group = Some(self.picking.create_capsule_bind_group(
                    &self.context.device,
                    self.sidechain_renderer.capsule_buffer(),
                ));
            }
        }
    }

    /// Apply combined Rosetta update to all groups.
    pub fn apply_combined_update(
        &mut self,
        bytes: &[u8],
        chain_ids: &[(GroupId, Vec<u8>)],
        action: AnimationAction,
    ) -> Result<(), String> {
        self.scene.apply_combined_update(bytes, chain_ids)?;
        self.sync_scene_to_renderers(Some(action));
        Ok(())
    }

    /// Update protein coords for a specific group.
    pub fn update_group_coords(
        &mut self,
        id: GroupId,
        coords: foldit_conv::coords::Coords,
        action: AnimationAction,
    ) {
        self.scene.update_group_protein_coords(id, coords);
        self.sync_scene_to_renderers(Some(action));
    }
}
