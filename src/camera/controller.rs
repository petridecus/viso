use crate::camera::core::{Camera, CameraUniform};
use crate::render_context::RenderContext;
use glam::{Quat, Vec2, Vec3};
use wgpu::util::DeviceExt;

pub struct CameraController {
    orientation: Quat,
    distance: f32,
    focus_point: Vec3,

    pub camera: Camera,
    pub uniform: CameraUniform,
    pub buffer: wgpu::Buffer,
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,

    pub mouse_pressed: bool,
    pub shift_pressed: bool,
    rotate_speed: f32,
    pan_speed: f32,
    zoom_speed: f32,
}

impl CameraController {
    pub fn new(context: &RenderContext) -> Self {
        let focus_point = Vec3::new(50.0, 50.0, 50.0);
        let distance = 150.0;
        let orientation = Quat::IDENTITY;

        let camera = Camera {
            eye: focus_point + Vec3::new(0.0, 0.0, distance),
            target: focus_point,
            up: Vec3::Y,
            aspect: context.config.width as f32 / context.config.height as f32,
            fovy: 45.0,
            znear: 5.0,
            zfar: 2000.0,
        };

        let mut uniform = CameraUniform::new();
        uniform.update_view_proj(&camera);

        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let layout = context
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
                label: Some("Camera Bind Group"),
            });

        Self {
            orientation,
            distance,
            focus_point,
            camera,
            uniform,
            buffer,
            layout,
            bind_group,
            mouse_pressed: false,
            shift_pressed: false,
            rotate_speed: 0.01,
            pan_speed: 0.1,
            zoom_speed: 0.05,
        }
    }

    fn update_camera_pos(&mut self) {
        let dir = self.orientation * Vec3::Z;

        self.camera.eye = self.focus_point + (dir * self.distance);
        self.camera.target = self.focus_point;
        self.camera.up = self.orientation * Vec3::Y;
    }

    pub fn update_gpu(&mut self, queue: &wgpu::Queue) {
        self.uniform.update_view_proj(&self.camera);
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[self.uniform]));
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.camera.aspect = width as f32 / height as f32;
    }

    pub fn rotate(&mut self, delta: Vec2) {
        // Horizontal rotation around camera's up vector
        let up = self.orientation * Vec3::Y;
        let horizontal_rotation = Quat::from_axis_angle(up, -delta.x * self.rotate_speed);

        // Apply horizontal rotation
        self.orientation = horizontal_rotation * self.orientation;

        // Vertical rotation around camera's right vector (after horizontal rotation)
        let right = self.orientation * Vec3::X;
        let vertical_rotation = Quat::from_axis_angle(right, -delta.y * self.rotate_speed);

        // Apply vertical rotation
        self.orientation = vertical_rotation * self.orientation;

        self.update_camera_pos();
    }

    pub fn pan(&mut self, delta: Vec2) {
        let right = self.orientation * Vec3::X;
        let up = self.orientation * Vec3::Y;

        let translation = right * (-delta.x * self.pan_speed) + up * (delta.y * self.pan_speed);

        self.focus_point += translation;
        self.update_camera_pos();
    }

    pub fn zoom(&mut self, delta: f32) {
        self.distance *= 1.0 - delta * self.zoom_speed;
        self.distance = self.distance.clamp(1.0, 1000.0);
        self.update_camera_pos();
    }

    /// Adjust camera to fit the given positions, centering on their centroid
    /// and setting distance so all points are visible.
    pub fn fit_to_positions(&mut self, positions: &[Vec3]) {
        if positions.is_empty() {
            return;
        }

        // Calculate centroid
        let centroid: Vec3 = positions.iter().copied().sum::<Vec3>() / positions.len() as f32;

        // Calculate bounding sphere radius from centroid
        let radius = positions
            .iter()
            .map(|p| (*p - centroid).length())
            .fold(0.0f32, f32::max);

        self.focus_point = centroid;

        // Set distance to fit the bounding sphere in view
        // Using fovy and some padding factor
        let fovy_rad = self.camera.fovy.to_radians();
        let fit_distance = radius / (fovy_rad / 2.0).tan();
        self.distance = fit_distance * 1.5; // 1.5x padding for comfortable view

        self.update_camera_pos();
    }
}
