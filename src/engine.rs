use crate::atom_renderer::AtomRenderer;
use crate::backbone_renderer::BackboneRenderer;
use crate::camera::controller::CameraController;
use crate::camera::input::InputHandler;
use crate::cylinder_renderer::CylinderRenderer;
use crate::frame_timing::FrameTiming;
use crate::lighting::Lighting;
use crate::protein_data::ProteinData;
use crate::render_context::RenderContext;
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
    pub depth_view: wgpu::TextureView,
}

impl ProteinRenderEngine {
    pub async fn new(window: std::sync::Arc<winit::window::Window>) -> Self {
        let context = RenderContext::new(window).await;
        let mut camera_controller = CameraController::new(&context);
        let lighting = Lighting::new(&context);
        let input_handler = InputHandler::new();

        // Load protein data from mmCIF file
        let cif_path = "assets/models/6ta1.cif";
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

        // Create shared depth view
        let depth_view = Self::create_depth_view(&context);

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
            depth_view,
        }
    }

    fn create_depth_view(context: &RenderContext) -> wgpu::TextureView {
        let size = wgpu::Extent3d {
            width: context.config.width,
            height: context.config.height,
            depth_or_array_layers: 1,
        };

        let desc = wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        };

        context
            .device
            .create_texture(&desc)
            .create_view(&wgpu::TextureViewDescriptor::default())
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
            self.depth_view = Self::create_depth_view(&self.context);
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
