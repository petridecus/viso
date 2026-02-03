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
        let total_residues: usize = if !protein_data.backbone_residue_chains.is_empty() {
            protein_data.backbone_residue_chains.iter().map(|c| c.residues.len()).sum()
        } else {
            protein_data.backbone_chains.iter().map(|c| c.len() / 3).sum()
        };

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

        // Initialize picking with CA positions
        let mut picking = Picking::new();
        if !protein_data.backbone_residue_chains.is_empty() {
            picking.update_from_residue_chains(&protein_data.backbone_residue_chains);
        } else {
            picking.update_from_backbone_chains(&protein_data.backbone_chains);
        }

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
        let width = self.context.config.width as f32;
        let height = self.context.config.height as f32;
        let camera = &self.camera_controller.camera;

        self.picking.update_hover_from_camera(x, y, width, height, camera);
    }

    /// Handle click for residue selection
    /// Returns true if selection changed
    pub fn handle_click(&mut self, x: f32, y: f32) -> bool {
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
}
