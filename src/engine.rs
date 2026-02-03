use crate::animation::{AnimationAction, StructureAnimator};
use crate::camera::controller::CameraController;
use crate::camera::input::InputHandler;
use crate::capsule_sidechain_renderer::CapsuleSidechainRenderer;
use crate::composite::CompositePass;
use crate::frame_timing::FrameTiming;
use crate::picking::{Picking, SelectionBuffer};
use crate::lighting::Lighting;

use crate::protein_data::ProteinData;
use crate::render_context::RenderContext;
use crate::ribbon_renderer::RibbonRenderer;
use crate::secondary_structure::SSType;
use crate::ssao::SsaoRenderer;
use crate::text_renderer::TextRenderer;
use crate::tube_renderer::TubeRenderer;

use glam::{Vec2, Vec3};
use std::collections::HashSet;
use std::time::Instant;
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
}

impl ProteinRenderEngine {
    /// Create a new engine with a default molecule path
    pub async fn new(window: std::sync::Arc<winit::window::Window>) -> Self {
        Self::new_with_path(window, "assets/models/4pnk.cif").await
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

        // Create GPU-based picking
        let picking = Picking::new(&context, &camera_controller.layout);

        // Create initial capsule picking bind group
        let capsule_picking_bind_group = Some(picking.create_capsule_bind_group(
            &context.device,
            sidechain_renderer.capsule_buffer(),
        ));

        // Create structure animator
        let animator = StructureAnimator::new();

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
            capsule_picking_bind_group,
            mouse_pos: (0.0, 0.0),
            mouse_down_residue: -1,
            is_dragging: false,
            last_click_time: Instant::now(),
            last_click_residue: -1,
            cached_ss_types: Vec::new(),
            animator,
            start_sidechain_positions: Vec::new(),
            target_sidechain_positions: Vec::new(),
            start_backbone_sidechain_bonds: Vec::new(),
            target_backbone_sidechain_bonds: Vec::new(),
            cached_sidechain_bonds: Vec::new(),
            cached_sidechain_hydrophobicity: Vec::new(),
            cached_sidechain_residue_indices: Vec::new(),
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
        let animating = self.animator.update(Instant::now());

        // If animation is active, update renderers with interpolated positions
        if animating {
            let visual_backbone = self.animator.get_backbone();
            self.update_backbone(&visual_backbone);

            // Interpolate sidechain positions
            let t = self.animator.progress();
            self.update_sidechains_interpolated(t);
        }

        // Update hover state in camera uniform (from GPU picking)
        self.camera_controller.uniform.hovered_residue = self.picking.hovered_residue;
        self.camera_controller.update_gpu(&self.context.queue);

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

        // GPU Picking pass - render residue IDs to picking buffer
        // In Ribbon mode, also render ribbons for picking (helices/sheets)
        let (ribbon_vb, ribbon_ib, ribbon_count) = match self.view_mode {
            ViewMode::Ribbon => (
                Some(self.ribbon_renderer.vertex_buffer()),
                Some(self.ribbon_renderer.index_buffer()),
                self.ribbon_renderer.index_count,
            ),
            ViewMode::Tube => (None, None, 0),
        };

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
            self.sidechain_renderer.instance_count,
            self.mouse_pos.0 as u32,
            self.mouse_pos.1 as u32,
        );

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

        // Complete GPU picking readback (after submit)
        self.picking.complete_readback(&self.context.device);

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
            self.picking.resize(&self.context.device, newsize.width, newsize.height);
        }
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

    /// Update sidechains with interpolated positions.
    fn update_sidechains_interpolated(&mut self, t: f32) {
        // Interpolate sidechain atom positions
        let interpolated_positions: Vec<Vec3> = self
            .start_sidechain_positions
            .iter()
            .zip(self.target_sidechain_positions.iter())
            .map(|(start, target)| *start + (*target - *start) * t)
            .collect();

        // Interpolate backbone-sidechain CA positions
        let interpolated_bs_bonds: Vec<(Vec3, u32)> = self
            .start_backbone_sidechain_bonds
            .iter()
            .zip(self.target_backbone_sidechain_bonds.iter())
            .map(|((start_pos, idx), (target_pos, _))| {
                let pos = *start_pos + (*target_pos - *start_pos) * t;
                (pos, *idx)
            })
            .collect();

        // Update sidechain renderer
        use crate::protein_data::BackboneSidechainBond;
        let bs_bonds: Vec<BackboneSidechainBond> = interpolated_bs_bonds
            .iter()
            .map(|(ca_pos, cb_idx)| BackboneSidechainBond {
                ca_position: *ca_pos,
                cb_index: *cb_idx,
            })
            .collect();

        self.sidechain_renderer.update(
            &self.context.device,
            &self.context.queue,
            &interpolated_positions,
            &self.cached_sidechain_bonds,
            &bs_bonds,
            &self.cached_sidechain_hydrophobicity,
            &self.cached_sidechain_residue_indices,
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

        // Update capsule picking bind group (buffer may have been reallocated)
        self.capsule_picking_bind_group = Some(self.picking.create_capsule_bind_group(
            &self.context.device,
            self.sidechain_renderer.capsule_buffer(),
        ));

        // Cache secondary structure types for double-click segment selection
        self.cached_ss_types = self.compute_ss_types(backbone_chains);

        // Fit camera if requested and we have positions
        if fit_camera && !all_positions.is_empty() {
            self.camera_controller.fit_to_positions(all_positions);
        }
    }

    /// Compute secondary structure types for all residues across all chains
    fn compute_ss_types(&self, backbone_chains: &[Vec<Vec3>]) -> Vec<SSType> {
        use crate::secondary_structure::detect_secondary_structure;

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
        backbone_sidechain_bonds: &[(Vec3, u32)],
    ) {
        self.animate_to_full_pose_with_action(
            new_backbone,
            sidechain_positions,
            sidechain_bonds,
            sidechain_hydrophobicity,
            sidechain_residue_indices,
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
        backbone_sidechain_bonds: &[(Vec3, u32)],
        action: AnimationAction,
    ) {
        // Capture current positions as start (for interpolation)
        // If sizes match, use current target; otherwise use new positions (snap)
        if self.target_sidechain_positions.len() == sidechain_positions.len() {
            self.start_sidechain_positions = self.target_sidechain_positions.clone();
            self.start_backbone_sidechain_bonds = self.target_backbone_sidechain_bonds.clone();
        } else {
            self.start_sidechain_positions = sidechain_positions.to_vec();
            self.start_backbone_sidechain_bonds = backbone_sidechain_bonds.to_vec();
        }

        // Set new targets
        self.target_sidechain_positions = sidechain_positions.to_vec();
        self.target_backbone_sidechain_bonds = backbone_sidechain_bonds.to_vec();
        self.cached_sidechain_bonds = sidechain_bonds.to_vec();
        self.cached_sidechain_hydrophobicity = sidechain_hydrophobicity.to_vec();
        self.cached_sidechain_residue_indices = sidechain_residue_indices.to_vec();

        // Set backbone target (this starts the animation)
        self.animator.set_target(new_backbone, action);

        // Update renderers with START visual state (animation will interpolate from here)
        if self.animator.residue_count() > 0 {
            let visual_backbone = self.animator.get_backbone();
            self.update_backbone(&visual_backbone);
        }

        // Update sidechain renderer with start positions
        use crate::protein_data::BackboneSidechainBond;
        let bs_bonds: Vec<BackboneSidechainBond> = self.start_backbone_sidechain_bonds
            .iter()
            .map(|(ca_pos, cb_idx)| BackboneSidechainBond {
                ca_position: *ca_pos,
                cb_index: *cb_idx,
            })
            .collect();

        self.sidechain_renderer.update(
            &self.context.device,
            &self.context.queue,
            &self.start_sidechain_positions,
            sidechain_bonds,
            &bs_bonds,
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
}
