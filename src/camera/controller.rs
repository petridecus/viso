use crate::camera::core::{Camera, CameraUniform};
use crate::render_context::RenderContext;
use glam::{Quat, Vec2, Vec3};
use wgpu::util::DeviceExt;

/// Speed of camera animation (higher = faster, 1.0 = instant)
const CAMERA_ANIMATION_SPEED: f32 = 3.0;

pub struct CameraController {
    orientation: Quat,
    distance: f32,
    focus_point: Vec3,
    bounding_radius: f32,  // Protein bounding sphere radius for fog computation

    // Animation targets (None = no animation in progress)
    target_focus_point: Option<Vec3>,
    target_distance: Option<f32>,
    target_bounding_radius: Option<f32>,

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
            bounding_radius: 50.0, // Default, will be updated by fit_to_positions
            target_focus_point: None,
            target_distance: None,
            target_bounding_radius: None,
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

    /// Update camera animation. Call this every frame.
    /// Returns true if animation is still in progress.
    pub fn update_animation(&mut self, dt: f32) -> bool {
        let mut animating = false;
        let t = (CAMERA_ANIMATION_SPEED * dt).min(1.0);

        // Animate focus point
        if let Some(target) = self.target_focus_point {
            let diff = target - self.focus_point;
            if diff.length() < 0.01 {
                self.focus_point = target;
                self.target_focus_point = None;
            } else {
                self.focus_point = self.focus_point.lerp(target, t);
                animating = true;
            }
        }

        // Animate distance
        if let Some(target) = self.target_distance {
            let diff = (target - self.distance).abs();
            if diff < 0.01 {
                self.distance = target;
                self.target_distance = None;
            } else {
                self.distance = self.distance + (target - self.distance) * t;
                animating = true;
            }
        }

        // Animate bounding radius (for fog)
        if let Some(target) = self.target_bounding_radius {
            let diff = (target - self.bounding_radius).abs();
            if diff < 0.01 {
                self.bounding_radius = target;
                self.target_bounding_radius = None;
            } else {
                self.bounding_radius = self.bounding_radius + (target - self.bounding_radius) * t;
                animating = true;
            }
        }

        if animating {
            self.update_camera_pos();
            self.update_fog_params();
        }

        animating
    }

    /// Check if camera is currently animating
    pub fn is_animating(&self) -> bool {
        self.target_focus_point.is_some()
            || self.target_distance.is_some()
            || self.target_bounding_radius.is_some()
    }

    /// Update fog parameters based on current camera distance and protein bounding radius.
    /// Called after any camera transform (zoom, pan, fit_to_positions).
    fn update_fog_params(&mut self) {
        // Fog starts at the back of the protein (center + some offset)
        // This keeps the front crisp and only fades the back
        let fog_start = self.distance + self.bounding_radius * 0.5;

        // Density calibrated so fog reaches ~90% at 4x bounding radius
        // Larger proteins get gentler fog falloff
        let fog_density = 0.5 / self.bounding_radius.max(10.0);

        self.uniform.fog_start = fog_start;
        self.uniform.fog_density = fog_density;
    }

    /// Get the camera's right vector from the orientation quaternion.
    #[inline]
    pub fn right(&self) -> Vec3 {
        self.orientation * Vec3::X
    }

    /// Get the camera's up vector from the orientation quaternion.
    #[inline]
    pub fn up(&self) -> Vec3 {
        self.orientation * Vec3::Y
    }

    /// Get the camera's forward vector from the orientation quaternion.
    #[inline]
    pub fn forward(&self) -> Vec3 {
        -(self.orientation * Vec3::Z)
    }

    fn update_camera_pos(&mut self) {
        // Renormalize quaternion to prevent accumulated floating-point drift
        // after many rotation operations
        self.orientation = self.orientation.normalize();

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
        self.update_fog_params();
    }

    /// Calculate fit parameters for the given positions.
    /// Returns (centroid, radius, fit_distance).
    fn calculate_fit_params(&self, positions: &[Vec3]) -> Option<(Vec3, f32, f32)> {
        if positions.is_empty() {
            return None;
        }

        // Calculate centroid
        let centroid: Vec3 = positions.iter().copied().sum::<Vec3>() / positions.len() as f32;

        // Calculate bounding sphere radius from centroid
        let radius = positions
            .iter()
            .map(|p| (*p - centroid).length())
            .fold(0.0f32, f32::max);

        // Set distance to fit the bounding sphere in view
        // Account for both vertical and horizontal FOV to fill viewport maximally
        let fovy_rad = self.camera.fovy.to_radians();
        let fovx_rad = fovy_rad * self.camera.aspect;

        // Calculate required distance for each axis
        let fit_distance_y = radius / (fovy_rad / 2.0).tan();
        let fit_distance_x = radius / (fovx_rad / 2.0).tan();

        // Use the larger distance (tighter constraint) to ensure fit on both axes
        let fit_distance = fit_distance_y.max(fit_distance_x);

        // Minimal padding (1.05x) to fill viewport without clipping
        Some((centroid, radius, fit_distance * 1.05))
    }

    /// Adjust camera to fit the given positions instantly (no animation).
    /// Used for initial load.
    pub fn fit_to_positions(&mut self, positions: &[Vec3]) {
        if let Some((centroid, radius, fit_distance)) = self.calculate_fit_params(positions) {
            self.focus_point = centroid;
            self.bounding_radius = radius;
            self.distance = fit_distance;

            // Clear any pending animation
            self.target_focus_point = None;
            self.target_distance = None;
            self.target_bounding_radius = None;

            self.update_camera_pos();
            self.update_fog_params();
        }
    }

    /// Adjust camera to fit the given positions with smooth animation.
    /// Used when new designs are added to the scene.
    pub fn fit_to_positions_animated(&mut self, positions: &[Vec3]) {
        if let Some((centroid, radius, fit_distance)) = self.calculate_fit_params(positions) {
            self.target_focus_point = Some(centroid);
            self.target_bounding_radius = Some(radius);
            self.target_distance = Some(fit_distance);
        }
    }

    /// Convert screen delta (pixels) to world-space offset.
    /// Uses camera orientation to map 2D mouse movement to 3D space.
    pub fn screen_delta_to_world(&self, delta_x: f32, delta_y: f32) -> Vec3 {
        // Scale factor based on distance (further = larger movements)
        let scale = self.distance * 0.002;

        // Use camera right/up vectors to convert 2D delta to 3D offset
        let right = self.right();
        let up = self.up();

        right * (delta_x * scale) + up * (-delta_y * scale)
    }

    /// Unproject screen coordinates to a world-space point on a plane at the given depth.
    ///
    /// # Arguments
    /// * `screen_x`, `screen_y` - Screen coordinates in pixels (origin top-left)
    /// * `screen_width`, `screen_height` - Screen dimensions
    /// * `world_point` - A point in world space; the result will be on a plane
    ///                   parallel to the camera at this point's depth
    pub fn screen_to_world_at_depth(
        &self,
        screen_x: f32,
        screen_y: f32,
        screen_width: u32,
        screen_height: u32,
        world_point: Vec3,
    ) -> Vec3 {
        // Convert screen coords to NDC [-1, 1]
        let ndc_x = (2.0 * screen_x / screen_width as f32) - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_y / screen_height as f32); // flip Y

        // Calculate the depth of the world_point from camera
        let to_point = world_point - self.camera.eye;
        let depth = to_point.dot(self.forward());

        // Use FOV to calculate the half-extents of the view plane at that depth
        let fovy_rad = self.camera.fovy.to_radians();
        let half_height = depth * (fovy_rad / 2.0).tan();
        let half_width = half_height * self.camera.aspect;

        // Calculate world position on the plane at that depth
        let right = self.right();
        let up = self.up();
        let forward = self.forward();

        let center = self.camera.eye + forward * depth;
        center + right * (ndc_x * half_width) + up * (ndc_y * half_height)
    }
}
