use crate::render_context::RenderContext;
use wgpu::util::DeviceExt;

/// Lighting configuration shared across all shaders
/// NOTE: Must match WGSL struct layout exactly (64 bytes)
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightingUniform {
    /// Primary light direction (normalized)
    pub light1_dir: [f32; 3],
    pub _pad1: f32,
    /// Secondary light direction (normalized)
    pub light2_dir: [f32; 3],
    pub _pad2: f32,
    /// Primary light intensity
    pub light1_intensity: f32,
    /// Secondary light intensity
    pub light2_intensity: f32,
    /// Ambient light intensity
    pub ambient: f32,
    /// Specular intensity
    pub specular_intensity: f32,
    /// Specular shininess exponent
    pub shininess: f32,
    /// Fresnel edge falloff power (higher = tighter edge glow)
    pub fresnel_power: f32,
    /// Fresnel edge brightness boost
    pub fresnel_intensity: f32,
    pub _pad3: f32,
}

impl Default for LightingUniform {
    fn default() -> Self {
        Self {
            // Primary light: upper-left for strong directional contrast
            light1_dir: normalize([-0.3, 0.9, -0.3]),
            _pad1: 0.0,
            // Secondary light: upper-right-front for fill
            light2_dir: normalize([0.3, 0.6, -0.4]),
            _pad2: 0.0,
            // Stronger primary light for better depth perception
            light1_intensity: 0.7,
            light2_intensity: 0.3,
            // Slightly reduced ambient to allow darker shadows
            ambient: 0.12,
            // Increased specular for glossy/jewel-like look
            specular_intensity: 0.5,
            // Higher shininess for tight highlights
            shininess: 64.0,
            // Fresnel edge falloff
            fresnel_power: 3.0,
            // Subtle edge glow
            fresnel_intensity: 0.25,
            _pad3: 0.0,
        }
    }
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / len, v[1] / len, v[2] / len]
}

pub struct Lighting {
    pub uniform: LightingUniform,
    pub buffer: wgpu::Buffer,
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
}

impl Lighting {
    pub fn new(context: &RenderContext) -> Self {
        let uniform = LightingUniform::default();

        let buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Lighting Buffer"),
                contents: bytemuck::cast_slice(&[uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let layout = context
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lighting Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
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
                label: Some("Lighting Bind Group"),
            });

        Self {
            uniform,
            buffer,
            layout,
            bind_group,
        }
    }

    pub fn update_gpu(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[self.uniform]));
    }

    /// Update light directions to follow camera (headlamp mode)
    /// Call this each frame after camera updates
    pub fn update_headlamp(
        &mut self,
        camera_right: glam::Vec3,
        camera_up: glam::Vec3,
        camera_forward: glam::Vec3,
    ) {
        // Primary light: upper-left relative to camera view
        // In camera space: (-0.3, 0.9, -0.3) = left, up, toward viewer
        // Negative z ensures surfaces facing camera receive light
        let light1_camera = glam::Vec3::new(-0.3, 0.9, -0.3).normalize();

        // Transform from camera space to world space
        let light1_world = camera_right * light1_camera.x
            + camera_up * light1_camera.y
            + camera_forward * light1_camera.z;

        self.uniform.light1_dir = light1_world.normalize().to_array();

        // Secondary fill light: upper-right relative to camera
        let light2_camera = glam::Vec3::new(0.3, 0.6, -0.4).normalize();
        let light2_world = camera_right * light2_camera.x
            + camera_up * light2_camera.y
            + camera_forward * light2_camera.z;

        self.uniform.light2_dir = light2_world.normalize().to_array();
    }
}
