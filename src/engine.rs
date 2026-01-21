use crate::atom_renderer::AtomRenderer;
use crate::camera::controller::CameraController;
use crate::camera::input::InputHandler;
use crate::render_context::RenderContext;
use glam::{Vec2, Vec3};
use winit::event::MouseButton;
use winit::keyboard::ModifiersState;

pub struct ProteinRenderEngine {
    pub context: RenderContext,
    pub camera_controller: CameraController,
    pub atom_renderer: AtomRenderer,
    pub input_handler: InputHandler,
}

impl ProteinRenderEngine {
    pub async fn new(window: std::sync::Arc<winit::window::Window>) -> Self {
        let context = RenderContext::new(window).await;
        let camera_controller = CameraController::new(&context);
        let atom_renderer = AtomRenderer::new(&context, &camera_controller.layout).await;
        let input_handler = InputHandler::new();

        Self {
            context,
            camera_controller,
            atom_renderer,
            input_handler,
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.camera_controller.update_gpu(&self.context.queue);

        let frame = self.context.get_next_frame()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.context.create_encoder();

        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear pass"),
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
                    view: &self.atom_renderer.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            self.atom_renderer
                .draw(&mut rp, &self.camera_controller.bind_group);
        }

        self.context.submit(encoder);
        frame.present();
        Ok(())
    }

    pub fn resize(&mut self, newsize: winit::dpi::PhysicalSize<u32>) {
        if newsize.width > 0 && newsize.height > 0 {
            self.context.resize(newsize);
            self.camera_controller.resize(newsize.width, newsize.height);
            self.atom_renderer.depth_view = AtomRenderer::create_depth_view(&self.context);
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
        let sphere_radius = 2.0;

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
