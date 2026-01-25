use crate::atom_renderer::AtomRenderer;
use crate::backbone_renderer::BackboneRenderer;
use crate::camera::controller::CameraController;
use crate::camera::input::InputHandler;
use crate::cylinder_renderer::CylinderRenderer;
use crate::frame_timing::FrameTiming;
use crate::lighting::Lighting;
use crate::protein_data::ProteinData;
use crate::render_context::RenderContext;
use crate::ssao::SsaoRenderer;
use crate::text_renderer::TextRenderer;
use glam::{Vec2, Vec3};
use winit::event::MouseButton;
use winit::keyboard::ModifiersState;

/// Target FPS limit
const TARGET_FPS: u32 = 300;

pub struct ProteinRenderEngine {
    pub context: RenderContext,
    pub camera_controller: CameraController,
    pub lighting: Lighting,
    pub backbone_renderer: BackboneRenderer,
    pub cylinder_renderer: CylinderRenderer,
    pub atom_renderer: AtomRenderer,
    pub text_renderer: TextRenderer,
    pub frame_timing: FrameTiming,
    pub input_handler: InputHandler,
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    pub ssao_renderer: SsaoRenderer,
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

        // Create backbone renderer (B-spline tube through backbone atoms)
        let backbone_renderer = BackboneRenderer::new(
            &context,
            &camera_controller.layout,
            &lighting.layout,
            &protein_data.backbone_chains,
        );

        // Get sidechain data
        let sidechain_positions = protein_data.sidechain_positions();
        let sidechain_hydrophobicity: Vec<bool> = protein_data
            .sidechain_atoms
            .iter()
            .map(|a| a.is_hydrophobic)
            .collect();

        // Create cylinder renderer for sidechain bonds AND backbone-sidechain bonds
        let cylinder_renderer = CylinderRenderer::new(
            &context,
            &camera_controller.layout,
            &lighting.layout,
            &sidechain_positions,
            &protein_data.sidechain_bonds,
            &protein_data.backbone_sidechain_bonds,
            &sidechain_hydrophobicity,
        );

        // Create atom renderer for sidechain atoms with hydrophobicity coloring
        let atom_renderer = AtomRenderer::new(
            &context,
            &camera_controller.layout,
            &lighting.layout,
            sidechain_positions,
            sidechain_hydrophobicity,
        );

        // Create shared depth texture (bindable for SSAO)
        let (depth_texture, depth_view) = Self::create_depth_texture(&context);

        // Create SSAO renderer
        let ssao_renderer = SsaoRenderer::new(&context, &depth_view);

        // Create text renderer for FPS display
        let text_renderer = TextRenderer::new(&context);

        // Create frame timing with 300 FPS limit
        let frame_timing = FrameTiming::new(TARGET_FPS);

        // Fit camera to all atom positions
        camera_controller.fit_to_positions(&protein_data.all_positions);

        Self {
            context,
            camera_controller,
            lighting,
            backbone_renderer,
            cylinder_renderer,
            atom_renderer,
            text_renderer,
            frame_timing,
            input_handler,
            depth_texture,
            depth_view,
            ssao_renderer,
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

        self.camera_controller.update_gpu(&self.context.queue);

        // Update FPS display
        self.text_renderer.update_fps(self.frame_timing.fps());
        self.text_renderer.prepare(&self.context);

        let frame = self.context.get_next_frame()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.context.create_encoder();

        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
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

            // Render order: backbone tubes -> cylinders -> spheres (all opaque)
            self.backbone_renderer
                .draw(&mut rp, &self.camera_controller.bind_group, &self.lighting.bind_group);
            self.cylinder_renderer
                .draw(&mut rp, &self.camera_controller.bind_group, &self.lighting.bind_group);
            self.atom_renderer
                .draw(&mut rp, &self.camera_controller.bind_group, &self.lighting.bind_group);
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

        // Text rendering pass (no depth testing needed)
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

    pub fn update_modifiers(&mut self, modifiers: ModifiersState) {
        self.camera_controller.shift_pressed = modifiers.shift_key();
    }

    /// Update protein atom positions for animation
    /// Handles dynamic buffer resizing if new positions exceed current capacity
    pub fn update_positions(&mut self, positions: &[Vec3], hydrophobicity: &[bool]) {
        self.atom_renderer.update_positions(
            &self.context.device,
            &self.context.queue,
            positions,
            hydrophobicity,
        );
    }

    /// Update backbone with new chains (regenerates the tube mesh)
    /// Use this for designed backbones from ML models like RFDiffusion3
    pub fn update_backbone(&mut self, backbone_chains: &[Vec<Vec3>]) {
        self.backbone_renderer
            .update_chains(&self.context.device, backbone_chains);
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
        sidechain_bonds: &[(u32, u32)],
        backbone_sidechain_bonds: &[(Vec3, u32)], // (CA position, CB index)
        all_positions: &[Vec3],
        fit_camera: bool,
    ) {
        use crate::protein_data::BackboneSidechainBond;

        // Update backbone tubes
        self.backbone_renderer.update(
            &self.context.device,
            &self.context.queue,
            backbone_chains,
        );

        // Update sidechain spheres
        self.atom_renderer.update_positions(
            &self.context.device,
            &self.context.queue,
            sidechain_positions,
            sidechain_hydrophobicity,
        );

        // Convert backbone_sidechain_bonds to the format CylinderRenderer expects
        let bs_bonds: Vec<BackboneSidechainBond> = backbone_sidechain_bonds
            .iter()
            .map(|(ca_pos, cb_idx)| BackboneSidechainBond {
                ca_position: *ca_pos,
                cb_index: *cb_idx,
            })
            .collect();

        // Update bonds
        self.cylinder_renderer.update(
            &self.context.device,
            &self.context.queue,
            sidechain_positions,
            sidechain_bonds,
            &bs_bonds,
            sidechain_hydrophobicity,
        );

        // Fit camera if requested and we have positions
        if fit_camera && !all_positions.is_empty() {
            self.camera_controller.fit_to_positions(all_positions);
        }
    }

    pub fn handle_mouse_position(&mut self, position: (f32, f32)) {
        let (x, y) = position;
        let width = self.context.config.width as f32;
        let height = self.context.config.height as f32;

        // Normalized device coordinates (-1 to 1), y flipped
        let ndc_x = (x / width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (y / height) * 2.0;

        // Compute ray direction in world space
        let view_proj = self.camera_controller.camera.build_matrix();
        let inv_view_proj = view_proj.inverse();
        // Use ndc z = 1 for near plane, 0 for far plane (correction matrix seems to invert)
        let ndc_near = Vec3::new(ndc_x, ndc_y, 1.0);
        let ndc_far = Vec3::new(ndc_x, ndc_y, 0.0);
        let world_near = inv_view_proj.project_point3(ndc_near);
        let world_far = inv_view_proj.project_point3(ndc_far);
        let ray_dir = (world_far - world_near).normalize();
        let ray_origin = world_near;

        // Intersection test with spheres
        let mut selected_index = -1;
        let mut closest_t = f32::INFINITY;
        let sphere_radius = 0.3;

        for (i, &pos) in self.atom_renderer.positions.iter().enumerate() {
            let oc = ray_origin - pos;
            let a = ray_dir.dot(ray_dir);
            let b = 2.0 * oc.dot(ray_dir);
            let c = oc.dot(oc) - sphere_radius * sphere_radius;
            let discriminant = b * b - 4.0 * a * c;
            if discriminant >= 0.0 {
                let t = (-b - discriminant.sqrt()) / (2.0 * a);
                if t > 0.0 && t < closest_t {
                    closest_t = t;
                    selected_index = i as i32;
                }
            }
        }

        // Update selection if changed
        let old_selected = self.camera_controller.uniform.selected_atom_index;
        if old_selected != selected_index {
            self.camera_controller.uniform.selected_atom_index = selected_index;
        }
    }
}
