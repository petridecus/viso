use crate::render_context::RenderContext;
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

/// Sphere rendering parameters
const SPHERE_RADIUS: f32 = 0.3;  // Match cylinder radius
const SUBDIVISIONS: u32 = 2;     // Icosphere subdivision level (1 = 80 triangles, 2 = 320)

/// Vertex for the unit sphere mesh
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SphereVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

/// Per-instance data for sphere rendering
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SphereInstance {
    /// Model matrix to transform unit sphere (4x4 = 16 floats)
    model: [[f32; 4]; 4],
    /// Color for this sphere (RGB)
    color: [f32; 3],
    /// Instance index for selection highlighting
    instance_index: f32,
}

pub struct SphereRenderer {
    pub pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub instance_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub instance_count: u32,
    pub positions: Vec<Vec3>,
}

// Color constants (match cylinder renderer)
const HYDROPHOBIC_COLOR: [f32; 3] = [0.3, 0.5, 0.9];  // Blue
const HYDROPHILIC_COLOR: [f32; 3] = [0.95, 0.6, 0.2]; // Orange

impl SphereRenderer {
    pub fn new(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        positions: Vec<Vec3>,
        hydrophobicity: Vec<bool>,
    ) -> Self {
        // Generate unit icosphere mesh
        let (vertices, indices) = Self::generate_icosphere(SUBDIVISIONS);

        let vertex_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Sphere Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let index_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Sphere Index Buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        // Generate instance data
        let instances: Vec<SphereInstance> = positions
            .iter()
            .zip(hydrophobicity.iter())
            .enumerate()
            .map(|(i, (&pos, &is_hydrophobic))| {
                let model = Mat4::from_scale_rotation_translation(
                    Vec3::splat(SPHERE_RADIUS),
                    glam::Quat::IDENTITY,
                    pos,
                );
                let color = if is_hydrophobic {
                    HYDROPHOBIC_COLOR
                } else {
                    HYDROPHILIC_COLOR
                };
                SphereInstance {
                    model: model.to_cols_array_2d(),
                    color,
                    instance_index: i as f32,
                }
            })
            .collect();

        let instance_count = instances.len() as u32;

        let instance_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Sphere Instance Buffer"),
                contents: bytemuck::cast_slice(&instances),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let pipeline = Self::create_pipeline(context, camera_layout, lighting_layout);

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            instance_buffer,
            index_count: indices.len() as u32,
            instance_count,
            positions,
        }
    }

    fn create_pipeline(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = context
            .device
            .create_shader_module(wgpu::include_wgsl!("../assets/shaders/geometry_spheres.wgsl"));

        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Sphere Pipeline Layout"),
                    bind_group_layouts: &[camera_layout, lighting_layout],
                    immediate_size: 0,
                });

        // Vertex buffer layout for sphere mesh
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SphereVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0, // position
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1, // normal
                },
            ],
        };

        // Instance buffer layout (4x4 matrix as 4 vec4s + color + instance_index)
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SphereInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 2, // model matrix col 0
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 3, // model matrix col 1
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 4, // model matrix col 2
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 48,
                    shader_location: 5, // model matrix col 3
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 64,
                    shader_location: 6, // color
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 76,
                    shader_location: 7, // instance_index
                },
            ],
        };

        context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Sphere Render Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout, instance_layout],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: context.config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
    }

    /// Generate an icosphere mesh with the given subdivision level
    /// Level 0 = icosahedron (20 triangles, 12 vertices)
    /// Level 1 = 80 triangles, 42 vertices
    /// Level 2 = 320 triangles, 162 vertices
    fn generate_icosphere(subdivisions: u32) -> (Vec<SphereVertex>, Vec<u32>) {
        // Golden ratio for icosahedron vertices
        let phi = (1.0 + 5.0_f32.sqrt()) / 2.0;
        let inv_len = 1.0 / (1.0 + phi * phi).sqrt();

        // 12 vertices of icosahedron (normalized to unit sphere)
        let mut positions: Vec<Vec3> = vec![
            Vec3::new(-1.0, phi, 0.0) * inv_len,
            Vec3::new(1.0, phi, 0.0) * inv_len,
            Vec3::new(-1.0, -phi, 0.0) * inv_len,
            Vec3::new(1.0, -phi, 0.0) * inv_len,
            Vec3::new(0.0, -1.0, phi) * inv_len,
            Vec3::new(0.0, 1.0, phi) * inv_len,
            Vec3::new(0.0, -1.0, -phi) * inv_len,
            Vec3::new(0.0, 1.0, -phi) * inv_len,
            Vec3::new(phi, 0.0, -1.0) * inv_len,
            Vec3::new(phi, 0.0, 1.0) * inv_len,
            Vec3::new(-phi, 0.0, -1.0) * inv_len,
            Vec3::new(-phi, 0.0, 1.0) * inv_len,
        ];

        // 20 triangles of icosahedron (CCW winding for outward-facing normals)
        let mut indices: Vec<u32> = vec![
            0, 5, 11,   0, 1, 5,    0, 7, 1,    0, 10, 7,   0, 11, 10,
            1, 9, 5,    5, 4, 11,   11, 2, 10,  10, 6, 7,   7, 8, 1,
            3, 4, 9,    3, 2, 4,    3, 6, 2,    3, 8, 6,    3, 9, 8,
            4, 5, 9,    2, 11, 4,   6, 10, 2,   8, 7, 6,    9, 1, 8,
        ];

        // Subdivide
        use std::collections::HashMap;
        let mut midpoint_cache: HashMap<(u32, u32), u32> = HashMap::new();

        for _ in 0..subdivisions {
            let mut new_indices = Vec::with_capacity(indices.len() * 4);

            for tri in indices.chunks(3) {
                let v0 = tri[0];
                let v1 = tri[1];
                let v2 = tri[2];

                // Get or create midpoint vertices
                let a = Self::get_midpoint(&mut positions, &mut midpoint_cache, v0, v1);
                let b = Self::get_midpoint(&mut positions, &mut midpoint_cache, v1, v2);
                let c = Self::get_midpoint(&mut positions, &mut midpoint_cache, v2, v0);

                // Create 4 new triangles
                new_indices.extend_from_slice(&[v0, a, c]);
                new_indices.extend_from_slice(&[v1, b, a]);
                new_indices.extend_from_slice(&[v2, c, b]);
                new_indices.extend_from_slice(&[a, b, c]);
            }

            indices = new_indices;
        }

        // Convert to vertices with normals (for a unit sphere, normal = position)
        let vertices: Vec<SphereVertex> = positions
            .iter()
            .map(|&p| SphereVertex {
                position: p.to_array(),
                normal: p.to_array(), // Unit sphere: normal = position
            })
            .collect();

        (vertices, indices)
    }

    fn get_midpoint(
        positions: &mut Vec<Vec3>,
        cache: &mut std::collections::HashMap<(u32, u32), u32>,
        v0: u32,
        v1: u32,
    ) -> u32 {
        // Ensure consistent ordering for cache key
        let key = if v0 < v1 { (v0, v1) } else { (v1, v0) };

        if let Some(&idx) = cache.get(&key) {
            return idx;
        }

        // Compute midpoint and normalize to unit sphere
        let p0 = positions[v0 as usize];
        let p1 = positions[v1 as usize];
        let mid = ((p0 + p1) * 0.5).normalize();

        let idx = positions.len() as u32;
        positions.push(mid);
        cache.insert(key, idx);
        idx
    }

    pub fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        lighting_bind_group: &'a wgpu::BindGroup,
    ) {
        if self.instance_count == 0 {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, lighting_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.index_count, 0, 0..self.instance_count);
    }
}
