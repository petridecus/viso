use crate::animation::{AnimationConfig, AnimationTimeline, InterpolatedResidue};
use crate::camera::controller::CameraController;
use crate::camera::input::InputHandler;
use crate::capsule_sidechain_renderer::CapsuleSidechainRenderer;
use crate::composite::CompositePass;
use crate::frame_timing::FrameTiming;
use crate::lighting::Lighting;
use crate::picking::{Picking, SelectionBuffer};
use crate::protein_data::ProteinData;
use crate::render_context::RenderContext;
use crate::ribbon_renderer::RibbonRenderer;
use crate::secondary_structure::SSType;
use crate::ssao::SsaoRenderer;
use crate::text_renderer::TextRenderer;
use crate::tube_renderer::TubeRenderer;

use glam::{Vec2, Vec3};
use std::collections::HashSet;
use std::time::{Duration, Instant};
use winit::event::MouseButton;
use winit::keyboard::ModifiersState;

/// View mode for backbone rendering
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewMode {
    /// Uniform-radius tubes for all residues
    #[default]
    Tube,
    /// Secondary structure ribbons for helices/sheets, tubes for coils
    Ribbon,
}

/// Target FPS limit
const TARGET_FPS: u32 = 300;

pub struct ProteinRenderEngine {
    pub context: RenderContext,
    pub camera_controller: CameraController,
    pub lighting: Lighting,
    pub sidechain_renderer: CapsuleSidechainRenderer,
    pub tube_renderer: TubeRenderer,
    pub ribbon_renderer: RibbonRenderer,
    pub view_mode: ViewMode,
    pub text_renderer: TextRenderer,
    pub frame_timing: FrameTiming,
    pub input_handler: InputHandler,
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    pub ssao_renderer: SsaoRenderer,
    pub composite_pass: CompositePass,
    pub picking: Picking,
    pub selection_buffer: SelectionBuffer,
    /// Animation timeline for smooth transitions
    pub animation_timeline: Option<AnimationTimeline>,
    /// Current backbone state: Vec[chain][residue] = [N, CA, C]
    current_backbone_state: Vec<Vec<[Vec3; 3]>>,
    /// Current sidechain chi angles: Vec[residue] = chis
    current_sidechain_state: Vec<Vec<f32>>,
    /// Animation configuration
    pub animation_config: AnimationConfig,
    /// Map from global residue index to chain index
    residue_to_chain: Vec<usize>,
}

impl ProteinRenderEngine {
    /// Create a new engine with a default molecule path
    pub async fn new(window: std::sync::Arc<winit::window::Window>) -> Self {
        Self::new_with_path(window, "assets/models/6ta1.cif").await
    }

    /// Create a new engine with a specified molecule path
    pub async fn new_with_path(window: std::sync::Arc<winit::window::Window>, cif_path: &str) -> Self {
        let context = RenderContext::new(window).await;
        let mut camera_controller = CameraController::new(&context);
        let lighting = Lighting::new(&context);
        let input_handler = InputHandler::new();

        // Load protein data from mmCIF file
        let protein_data = ProteinData::from_mmcif(cif_path)
            .expect("Failed to load protein data from CIF file");

        // Count total residues for selection buffer sizing
        // Must use backbone_chains (not backbone_residue_chains) for consistent indexing
        // with tube_renderer and picking
        let total_residues: usize = protein_data.backbone_chains.iter().map(|c| c.len() / 3).sum();

        // Create selection buffer (shared by all renderers)
        let selection_buffer = SelectionBuffer::new(&context.device, total_residues.max(1));

        // Create backbone renderer (B-spline tube through backbone atoms)
        let tube_renderer = TubeRenderer::new(
            &context,
            &camera_controller.layout,
            &lighting.layout,
            &selection_buffer.layout,
            &protein_data.backbone_chains,
        );

        // Create ribbon renderer for secondary structure visualization
        // Use the new Foldit-style renderer if we have full backbone residue data (N, CA, C, O)
        let ribbon_renderer = if !protein_data.backbone_residue_chains.is_empty() {
            RibbonRenderer::new_from_residues(
                &context,
                &camera_controller.layout,
                &lighting.layout,
                &selection_buffer.layout,
                &protein_data.backbone_residue_chains,
            )
        } else {
            // Fallback to legacy renderer if only backbone_chains available
            RibbonRenderer::new(
                &context,
                &camera_controller.layout,
                &lighting.layout,
                &selection_buffer.layout,
                &protein_data.backbone_chains,
            )
        };

        // Get sidechain data
        let sidechain_positions = protein_data.sidechain_positions();
        let sidechain_hydrophobicity: Vec<bool> = protein_data
            .sidechain_atoms
            .iter()
            .map(|a| a.is_hydrophobic)
            .collect();
        let sidechain_residue_indices: Vec<u32> = protein_data
            .sidechain_atoms
            .iter()
            .map(|a| a.residue_idx as u32)
            .collect();

        let sidechain_renderer = CapsuleSidechainRenderer::new(
            &context,
            &camera_controller.layout,
            &lighting.layout,
            &selection_buffer.layout,
            &sidechain_positions,
            &protein_data.sidechain_bonds,
            &protein_data.backbone_sidechain_bonds,
            &sidechain_hydrophobicity,
            &sidechain_residue_indices,
        );

        // Create shared depth texture (bindable for SSAO)
        let (depth_texture, depth_view) = Self::create_depth_texture(&context);

        // Create SSAO renderer
        let ssao_renderer = SsaoRenderer::new(&context, &depth_view);

        // Create composite pass (applies SSAO and outlines to final image)
        let composite_pass = CompositePass::new(&context, ssao_renderer.get_ssao_view(), &depth_view);

        // Create text renderer for FPS display
        let text_renderer = TextRenderer::new(&context);

        // Create frame timing with 300 FPS limit
        let frame_timing = FrameTiming::new(TARGET_FPS);

        // Fit camera to all atom positions
        camera_controller.fit_to_positions(&protein_data.all_positions);

        // Initialize picking with CA positions from backbone_chains
        // (must match what tube_renderer uses for consistent residue indexing)
        let mut picking = Picking::new();
        picking.update_from_backbone_chains(&protein_data.backbone_chains);

        // Build animation state
        // Create animation timeline with capacity for total residues
        let animation_timeline = Some(AnimationTimeline::new(total_residues));
        let animation_config = AnimationConfig::default();

        // Build residue_to_chain mapping
        let mut residue_to_chain = Vec::with_capacity(total_residues);
        for (chain_idx, chain) in protein_data.backbone_chains.iter().enumerate() {
            let residues_in_chain = chain.len() / 3; // Each residue has N, CA, C
            for _ in 0..residues_in_chain {
                residue_to_chain.push(chain_idx);
            }
        }

        // Initialize backbone and sidechain state as empty (populated on first animate call)
        let current_backbone_state = Vec::new();
        let current_sidechain_state = Vec::new();

        Self {
            context,
            camera_controller,
            lighting,
            tube_renderer,
            ribbon_renderer,
            view_mode: ViewMode::default(),
            sidechain_renderer,
            text_renderer,
            frame_timing,
            input_handler,
            depth_texture,
            depth_view,
            ssao_renderer,
            composite_pass,
            picking,
            selection_buffer,
            animation_timeline,
            current_backbone_state,
            current_sidechain_state,
            animation_config,
            residue_to_chain,
        }
    }

    fn create_depth_texture(context: &RenderContext) -> (wgpu::Texture, wgpu::TextureView) {
        let size = wgpu::Extent3d {
            width: context.config.width,
            height: context.config.height,
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

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // Check if we should render based on FPS limit
        if !self.frame_timing.should_render() {
            return Ok(());
        }

        // Update animations for this frame
        self.update_animations();

        // Update hover state in camera uniform
        self.camera_controller.uniform.hovered_residue = self.picking.hovered_residue;
        self.camera_controller.update_gpu(&self.context.queue);

        // Update selection buffer
        self.selection_buffer.update(&self.context.queue, &self.picking.selected_residues);

        // Update lighting to follow camera (headlamp mode)
        let camera = &self.camera_controller.camera;
        let forward = (camera.target - camera.eye).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward);
        self.lighting.update_headlamp(right, up, forward);
        self.lighting.update_gpu(&self.context.queue);

        // Update FPS display
        self.text_renderer.update_fps(self.frame_timing.fps());
        self.text_renderer.prepare(&self.context);

        let frame = self.context.get_next_frame()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.context.create_encoder();

        // Geometry pass - render to intermediate color texture (for SSAO compositing)
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.composite_pass.get_color_view(),
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
                })],
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
            match self.view_mode {
                ViewMode::Tube => {
                    // Tube mode: render all SS types as tubes
                    self.tube_renderer.draw(
                        &mut rp,
                        &self.camera_controller.bind_group,
                        &self.lighting.bind_group,
                        &self.selection_buffer.bind_group,
                    );
                }
                ViewMode::Ribbon => {
                    // Ribbon mode: tubes for coils, ribbons for helices/sheets
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
                }
            }

            self.sidechain_renderer.draw(
                &mut rp,
                &self.camera_controller.bind_group,
                &self.lighting.bind_group,
                &self.selection_buffer.bind_group,
            );
        }

        // Update SSAO projection matrices
        let proj = self.camera_controller.camera.build_projection();
        self.ssao_renderer.update_projection(
            &self.context.queue,
            proj,
            self.camera_controller.camera.znear,
            self.camera_controller.camera.zfar,
        );

        // SSAO pass (compute ambient occlusion from depth buffer)
        self.ssao_renderer.render_ssao(&mut encoder);

        // Composite pass - apply SSAO to the geometry and output to swapchain
        self.composite_pass.render(&mut encoder, &view);

        // Text rendering pass (no depth testing needed, renders on top of composited image)
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("text render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Don't clear - render on top
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            self.text_renderer.render(&mut rp);
        }

        self.context.submit(encoder);
        frame.present();

        // Update frame timing
        self.frame_timing.end_frame();

        Ok(())
    }

    pub fn resize(&mut self, newsize: winit::dpi::PhysicalSize<u32>) {
        if newsize.width > 0 && newsize.height > 0 {
            self.context.resize(newsize);
            self.camera_controller.resize(newsize.width, newsize.height);
            let (depth_texture, depth_view) = Self::create_depth_texture(&self.context);
            self.depth_texture = depth_texture;
            self.depth_view = depth_view;
            self.ssao_renderer.resize(&self.context, &self.depth_view);
            self.composite_pass.resize(&self.context, self.ssao_renderer.get_ssao_view(), &self.depth_view);
            self.text_renderer.resize(newsize.width, newsize.height);
        }
    }

    pub fn handle_mouse_move(&mut self, delta_x: f32, delta_y: f32) {
        if self.camera_controller.mouse_pressed {
            let delta = Vec2::new(delta_x, delta_y);
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

    pub fn handle_mouse_button(&mut self, button: MouseButton, pressed: bool) {
        if button == MouseButton::Left {
            self.camera_controller.mouse_pressed = pressed;
        }
    }

    /// Handle mouse position update for hover detection
    pub fn handle_mouse_position(&mut self, x: f32, y: f32) {
        // Both cursor position (from winit PhysicalPosition) and context.config dimensions
        // (from inner_size()) are in physical pixels - use them directly
        let width = self.context.config.width as f32;
        let height = self.context.config.height as f32;
        let camera = &self.camera_controller.camera;

        // Diagnostic logging (once) to verify coordinate spaces match
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            eprintln!("=== PICKING COORDINATE DIAGNOSTICS ===");
            eprintln!("  cursor: ({:.0}, {:.0})", x, y);
            eprintln!("  screen: ({:.0}, {:.0})", width, height);
            eprintln!("  ratio cursor/screen: ({:.3}, {:.3})", x / width, y / height);
            // At center of screen, ratio should be ~0.5
        });

        self.picking.update_hover_from_camera(x, y, width, height, camera);
    }

    /// Handle click for residue selection
    /// Returns true if selection changed
    pub fn handle_click(&mut self, x: f32, y: f32) -> bool {
        // Both cursor position and context.config dimensions are in physical pixels
        let width = self.context.config.width as f32;
        let height = self.context.config.height as f32;
        let camera = &self.camera_controller.camera;
        let shift_held = self.camera_controller.shift_pressed;

        self.picking.handle_click_from_camera(x, y, width, height, camera, shift_held)
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
        );
    }

    /// Set the view mode for backbone rendering
    pub fn set_view_mode(&mut self, mode: ViewMode) {
        if self.view_mode == mode {
            return;
        }
        self.view_mode = mode;

        // Update tube renderer filter based on mode
        match mode {
            ViewMode::Tube => {
                // Tube mode: render all SS types as tubes
                self.tube_renderer.set_ss_filter(None);
            }
            ViewMode::Ribbon => {
                // Ribbon mode: tube renderer only renders coils
                let mut coil_only = HashSet::new();
                coil_only.insert(SSType::Coil);
                self.tube_renderer.set_ss_filter(Some(coil_only));
            }
        }

        // Regenerate mesh with new filter
        self.tube_renderer.regenerate(&self.context.device, &self.context.queue);
    }

    /// Toggle between tube and ribbon view modes
    pub fn toggle_view_mode(&mut self) {
        let new_mode = match self.view_mode {
            ViewMode::Tube => ViewMode::Ribbon,
            ViewMode::Ribbon => ViewMode::Tube,
        };
        self.set_view_mode(new_mode);
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
        sidechain_bonds: &[(u32, u32)],
        backbone_sidechain_bonds: &[(Vec3, u32)], // (CA position, CB index)
        all_positions: &[Vec3],
        fit_camera: bool,
    ) {
        use crate::protein_data::BackboneSidechainBond;

        // Update backbone tubes
        self.tube_renderer.update(
            &self.context.device,
            &self.context.queue,
            backbone_chains,
        );

        // Update ribbon renderer
        self.ribbon_renderer.update(
            &self.context.device,
            &self.context.queue,
            backbone_chains,
        );

        // Convert backbone_sidechain_bonds to the format CylinderRenderer expects
        let bs_bonds: Vec<BackboneSidechainBond> = backbone_sidechain_bonds
            .iter()
            .map(|(ca_pos, cb_idx)| BackboneSidechainBond {
                ca_position: *ca_pos,
                cb_index: *cb_idx,
            })
            .collect();

        self.sidechain_renderer.update(
            &self.context.device,
            &self.context.queue,
            sidechain_positions,
            sidechain_bonds,
            &bs_bonds,
            sidechain_hydrophobicity,
            sidechain_residue_indices,
        );

        // Fit camera if requested and we have positions
        if fit_camera && !all_positions.is_empty() {
            self.camera_controller.fit_to_positions(all_positions);
        }
    }

    /// Capture current backbone state from protein data.
    ///
    /// Converts Vec<Vec3> (N, CA, C flat) to Vec<Vec<[Vec3; 3]>> (per residue).
    fn capture_backbone_state(&mut self, backbone_chains: &[Vec<Vec3>]) {
        self.current_backbone_state.clear();
        self.current_backbone_state.reserve(backbone_chains.len());

        for chain in backbone_chains {
            let num_residues = chain.len() / 3;
            let mut residue_states = Vec::with_capacity(num_residues);

            for i in 0..num_residues {
                let base = i * 3;
                if base + 2 < chain.len() {
                    residue_states.push([chain[base], chain[base + 1], chain[base + 2]]);
                }
            }

            self.current_backbone_state.push(residue_states);
        }
    }

    /// Get chain index for a residue.
    #[inline]
    fn chain_for_residue(&self, residue_idx: usize) -> usize {
        self.residue_to_chain.get(residue_idx).copied().unwrap_or(0)
    }

    // =========================================================================
    // Animation Trigger Methods
    // =========================================================================

    /// Animate from current state to new pose.
    ///
    /// Creates animations for all residues that differ between the current
    /// backbone state and the new backbone positions.
    ///
    /// If an animation is already in progress, the new animation starts from
    /// the current interpolated (visual) position, ensuring smooth continuity.
    ///
    /// # Arguments
    /// * `new_backbone` - New backbone positions: Vec[chain][residue * 3 + atom] where atom is N=0, CA=1, C=2
    pub fn animate_to_pose(&mut self, new_backbone: &[Vec<Vec3>]) {
        use crate::animation::state::ResidueAnimationState;

        // 1. Capture current state if empty
        if self.current_backbone_state.is_empty() {
            // First call - just set state, no animation
            self.capture_backbone_state(new_backbone);
            return;
        }

        // Check if animation is enabled
        if !self.animation_config.enabled {
            // Animation disabled - just update state directly
            self.capture_backbone_state(new_backbone);
            return;
        }

        // 2. If animation is in progress, sync current_backbone_state with visual position
        // This ensures the new animation starts from where we visually are, not the previous target
        if self.is_animating() {
            self.sync_current_state_with_interpolated();
        }

        const EPSILON: f32 = 0.0001;

        // 2. Compare and create ResidueAnimationState for differing residues
        let mut animation_states = Vec::new();
        let mut global_residue_idx = 0usize;

        for (chain_idx, chain_positions) in new_backbone.iter().enumerate() {
            let num_residues = chain_positions.len() / 3;

            // Get current chain state (if it exists)
            let current_chain = self.current_backbone_state.get(chain_idx);

            for residue_in_chain in 0..num_residues {
                let base_idx = residue_in_chain * 3;

                // Extract new backbone positions for this residue
                let new_n = chain_positions.get(base_idx).copied().unwrap_or(Vec3::ZERO);
                let new_ca = chain_positions.get(base_idx + 1).copied().unwrap_or(Vec3::ZERO);
                let new_c = chain_positions.get(base_idx + 2).copied().unwrap_or(Vec3::ZERO);
                let end_backbone = [new_n, new_ca, new_c];

                // Get current backbone positions for this residue
                let start_backbone = current_chain
                    .and_then(|c| c.get(residue_in_chain))
                    .copied()
                    .unwrap_or(end_backbone); // Default to end if no current state

                // Check if any position differs by more than epsilon
                let needs_animation = (0..3).any(|i| {
                    (end_backbone[i] - start_backbone[i]).length() > EPSILON
                });

                if needs_animation {
                    // Create animation state for this residue
                    let mut state = ResidueAnimationState::new(
                        global_residue_idx,
                        start_backbone,
                        end_backbone,
                        &[], // Empty chi angles for now (backbone only)
                        &[], // Empty chi angles for now (backbone only)
                    );
                    state.needs_animation = true;
                    animation_states.push(state);
                }

                global_residue_idx += 1;
            }
        }

        // 3. Add animations to timeline
        if !animation_states.is_empty() {
            if let Some(timeline) = &mut self.animation_timeline {
                timeline.add(animation_states, Some(self.animation_config.duration), None);
            }
        }

        // 4. Update current_backbone_state to new state
        self.capture_backbone_state(new_backbone);
    }

    /// Skip all animations to final state immediately.
    pub fn skip_animations(&mut self) {
        if let Some(timeline) = &mut self.animation_timeline {
            timeline.skip();
        }
    }

    /// Cancel all animations (stay at current interpolated state).
    pub fn cancel_animations(&mut self) {
        if let Some(timeline) = &mut self.animation_timeline {
            timeline.cancel();
        }
    }

    /// Check if any animations are active.
    #[inline]
    pub fn is_animating(&self) -> bool {
        self.animation_timeline
            .as_ref()
            .map(|t| t.is_animating())
            .unwrap_or(false)
    }

    /// Set animation enabled/disabled.
    pub fn set_animation_enabled(&mut self, enabled: bool) {
        self.animation_config.enabled = enabled;
    }

    /// Set animation duration.
    pub fn set_animation_duration(&mut self, duration: Duration) {
        self.animation_config.duration = duration;
    }

    // =========================================================================
    // Animation Update Methods
    // =========================================================================

    /// Update animations for current frame.
    ///
    /// Returns true if render is needed (animations active).
    /// Call this at the start of each render frame.
    pub fn update_animations(&mut self) -> bool {
        // Check if we have a timeline and it's animating
        let has_updates = {
            let Some(timeline) = &mut self.animation_timeline else {
                return false;
            };

            let now = Instant::now();
            timeline.update(now)
        };

        if !has_updates {
            return false;
        }

        // Copy interpolated state to avoid borrow conflicts
        let interpolated: Vec<InterpolatedResidue> = {
            let Some(timeline) = &self.animation_timeline else {
                return false;
            };
            timeline.get_interpolated().to_vec()
        };

        if interpolated.is_empty() {
            return false;
        }

        // Convert interpolated residues back to backbone_chains format
        // and update the renderers
        self.apply_interpolated_state(&interpolated);

        true
    }

    /// Apply interpolated state to renderers.
    ///
    /// Groups interpolated residues by chain using residue_to_chain mapping,
    /// builds new backbone_chains Vec<Vec<Vec3>> by flattening [N, CA, C] arrays,
    /// and calls update_backbone() to update tube and ribbon renderers.
    fn apply_interpolated_state(&mut self, interpolated: &[InterpolatedResidue]) {
        // 1. Determine the number of chains and residues per chain
        // We need to build the full backbone structure, updating only the animated residues

        // First, figure out how many chains we have
        let num_chains = self.residue_to_chain.iter().max().map(|&m| m + 1).unwrap_or(0);
        if num_chains == 0 {
            return;
        }

        // Count residues per chain
        let mut residues_per_chain = vec![0usize; num_chains];
        for &chain_idx in &self.residue_to_chain {
            residues_per_chain[chain_idx] += 1;
        }

        // 2. Build new backbone_chains from current_backbone_state, updating with interpolated values
        // First, create a map of residue_idx -> interpolated backbone
        let mut interpolated_map: std::collections::HashMap<usize, [Vec3; 3]> =
            std::collections::HashMap::with_capacity(interpolated.len());
        for interp in interpolated {
            interpolated_map.insert(interp.residue_idx, interp.backbone);
        }

        // 3. Build the new backbone chains
        let mut new_backbone_chains: Vec<Vec<Vec3>> = Vec::with_capacity(num_chains);
        let mut global_residue_idx = 0usize;

        for (chain_idx, &num_residues) in residues_per_chain.iter().enumerate() {
            // Pre-allocate for N, CA, C per residue
            let mut chain_positions: Vec<Vec3> = Vec::with_capacity(num_residues * 3);

            for residue_in_chain in 0..num_residues {
                // Check if this residue has interpolated data
                let backbone = if let Some(&interp_backbone) = interpolated_map.get(&global_residue_idx) {
                    // Use interpolated backbone
                    interp_backbone
                } else if let Some(chain_state) = self.current_backbone_state.get(chain_idx) {
                    // Use current backbone state
                    chain_state.get(residue_in_chain).copied().unwrap_or([Vec3::ZERO; 3])
                } else {
                    // Fallback to zeros (shouldn't happen in practice)
                    [Vec3::ZERO; 3]
                };

                // Flatten [N, CA, C] into the chain positions
                chain_positions.push(backbone[0]); // N
                chain_positions.push(backbone[1]); // CA
                chain_positions.push(backbone[2]); // C

                global_residue_idx += 1;
            }

            new_backbone_chains.push(chain_positions);
        }

        // 4. Update the backbone renderers
        self.update_backbone(&new_backbone_chains);

        // Note: For now, sidechains are not animated.
        // They would need similar treatment when implemented.
    }

    /// Sync current_backbone_state with the current interpolated (visual) positions.
    ///
    /// Called before starting a new animation when one is already in progress.
    /// This ensures the new animation starts from where we visually are,
    /// not from the previous animation's target, avoiding visual jumps.
    fn sync_current_state_with_interpolated(&mut self) {
        let Some(timeline) = &self.animation_timeline else {
            return;
        };

        let interpolated = timeline.get_interpolated();
        if interpolated.is_empty() {
            return;
        }

        // Build a map of residue_idx -> interpolated backbone
        let interpolated_map: std::collections::HashMap<usize, [Vec3; 3]> = interpolated
            .iter()
            .map(|i| (i.residue_idx, i.backbone))
            .collect();

        // Update current_backbone_state with interpolated positions
        let mut global_residue_idx = 0usize;
        for chain in &mut self.current_backbone_state {
            for residue_backbone in chain.iter_mut() {
                if let Some(&interp_backbone) = interpolated_map.get(&global_residue_idx) {
                    *residue_backbone = interp_backbone;
                }
                global_residue_idx += 1;
            }
        }
    }
}
