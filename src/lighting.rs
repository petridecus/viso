use crate::render_context::RenderContext;
use wgpu::util::DeviceExt;

/// Lighting configuration shared across all shaders
/// NOTE: Must match WGSL struct layout exactly (80 bytes with vec3 alignment)
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
    // Padding: shininess ends at offset 52, vec3 _pad3 needs 16-byte alignment (offset 64)
    pub _pad_align: [f32; 3],
    // vec3 in WGSL is 12 bytes but struct pads to 16-byte boundary -> 80 total
    pub _pad3: [f32; 4],
}

impl Default for LightingUniform {
    fn default() -> Self {
        Self {
            // Match Foldit's lighting setup for consistent appearance
            // Light 0: (-100, 300, 0) - upper-left (16.7% ambient + 33.3% diffuse)
            light1_dir: normalize([-100.0, 300.0, 0.0]),
            _pad1: 0.0,
            // Light 1: (100, 300, 100) - upper-right-front (33.3% diffuse)
            light2_dir: normalize([100.0, 300.0, 100.0]),
            _pad2: 0.0,
            light1_intensity: 0.333,
            light2_intensity: 0.333,
            ambient: 0.167,
            // No specular for consistent matte appearance from all angles
            specular_intensity: 0.0,
            shininess: 64.0,
            _pad_align: [0.0; 3],
            _pad3: [0.0; 4],
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

    #[allow(dead_code)]
    pub fn update_gpu(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[self.uniform]));
    }
}
