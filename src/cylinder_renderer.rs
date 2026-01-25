//! Cylinder renderer for bonds between atoms
//!
//! Renders bonds as cylinders with hydrophobicity-based coloring.
//! Uses instanced rendering with a dynamic instance buffer.

use crate::dynamic_buffer::TypedBuffer;
use crate::protein_data::BackboneSidechainBond;
use crate::render_context::RenderContext;
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

/// Cylinder rendering parameters
const CYLINDER_RADIUS: f32 = 0.3; // Match sphere radius
const RADIAL_SEGMENTS: usize = 16;

/// Vertex for the unit cylinder mesh
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CylinderVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

/// Per-instance data for cylinder rendering
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CylinderInstance {
    /// Model matrix to transform unit cylinder (4x4 = 16 floats)
    model: [[f32; 4]; 4],
    /// Color for this cylinder (RGB)
    color: [f32; 3],
    /// Padding to align to 16 bytes
    _pad: f32,
}

pub struct CylinderRenderer {
    pub pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    instance_buffer: TypedBuffer<CylinderInstance>,
    pub index_count: u32,
    pub instance_count: u32,
}

// Color constants
const HYDROPHOBIC_COLOR: [f32; 3] = [0.3, 0.5, 0.9]; // Blue
const HYDROPHILIC_COLOR: [f32; 3] = [0.95, 0.6, 0.2]; // Orange

impl CylinderRenderer {
    pub fn new(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        sidechain_positions: &[Vec3],
        sidechain_bonds: &[(u32, u32)],
        backbone_sidechain_bonds: &[BackboneSidechainBond],
        hydrophobicity: &[bool],
    ) -> Self {
        // Generate unit cylinder mesh (radius 1, height 1, centered at origin, along Y axis)
        let (vertices, indices) = Self::generate_unit_cylinder();

        let vertex_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Cylinder Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let index_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Cylinder Index Buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        // Generate instance data
        let instances = Self::generate_all_instances(
            sidechain_positions,
            sidechain_bonds,
            backbone_sidechain_bonds,
            hydrophobicity,
        );

        let instance_count = instances.len() as u32;

        let instance_buffer = TypedBuffer::new_with_data(
            &context.device,
            "Cylinder Instance Buffer",
            &instances,
            wgpu::BufferUsages::VERTEX,
        );

        let pipeline = Self::create_pipeline(context, camera_layout, lighting_layout);

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            instance_buffer,
            index_count: indices.len() as u32,
            instance_count,
        }
    }

    /// Update bonds dynamically - can handle changing topology
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sidechain_positions: &[Vec3],
        sidechain_bonds: &[(u32, u32)],
        backbone_sidechain_bonds: &[BackboneSidechainBond],
        hydrophobicity: &[bool],
    ) {
        let instances = Self::generate_all_instances(
            sidechain_positions,
            sidechain_bonds,
            backbone_sidechain_bonds,
            hydrophobicity,
        );

        self.instance_buffer.write(device, queue, &instances);
        self.instance_count = instances.len() as u32;
    }

    /// Generate all instances (sidechain bonds + backbone-sidechain bonds)
    fn generate_all_instances(
        sidechain_positions: &[Vec3],
        sidechain_bonds: &[(u32, u32)],
        backbone_sidechain_bonds: &[BackboneSidechainBond],
        hydrophobicity: &[bool],
    ) -> Vec<CylinderInstance> {
        let mut instances = Self::generate_instances_with_color(
            sidechain_positions,
            sidechain_bonds,
            hydrophobicity,
        );

        // Add backbone-sidechain bonds (CA to CB) - use CB's color
        for bond in backbone_sidechain_bonds {
            let cb_idx = bond.cb_index as usize;
            if cb_idx < sidechain_positions.len() {
                let cb_pos = sidechain_positions[cb_idx];
                let model = Self::compute_cylinder_transform(bond.ca_position, cb_pos);
                let is_hydrophobic = hydrophobicity.get(cb_idx).copied().unwrap_or(false);
                let color = if is_hydrophobic {
                    HYDROPHOBIC_COLOR
                } else {
                    HYDROPHILIC_COLOR
                };
                instances.push(CylinderInstance {
                    model: model.to_cols_array_2d(),
                    color,
                    _pad: 0.0,
                });
            }
        }

        instances
    }

    fn create_pipeline(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = context
            .device
            .create_shader_module(wgpu::include_wgsl!("../assets/shaders/cylinder_bonds.wgsl"));

        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Cylinder Pipeline Layout"),
                    bind_group_layouts: &[camera_layout, lighting_layout],
                    immediate_size: 0,
                });

        // Vertex buffer layout for cylinder mesh
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CylinderVertex>() as wgpu::BufferAddress,
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

        // Instance buffer layout (4x4 matrix as 4 vec4s + color)
        let instance_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CylinderInstance>() as wgpu::BufferAddress,
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
            ],
        };

        context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Cylinder Render Pipeline"),
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

    /// Generate a unit cylinder mesh (radius 1, height 1, Y-axis aligned, centered at origin)
    fn generate_unit_cylinder() -> (Vec<CylinderVertex>, Vec<u32>) {
        let mut vertices = Vec::with_capacity(RADIAL_SEGMENTS * 2);
        let mut indices = Vec::new();

        // Generate vertices for bottom and top rings
        for i in 0..RADIAL_SEGMENTS {
            let angle = (i as f32 / RADIAL_SEGMENTS as f32) * std::f32::consts::TAU;
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            // Normal points outward (radial direction)
            let normal = [cos_a, 0.0, sin_a];

            // Bottom ring vertex (y = -0.5)
            vertices.push(CylinderVertex {
                position: [cos_a, -0.5, sin_a],
                normal,
            });

            // Top ring vertex (y = 0.5)
            vertices.push(CylinderVertex {
                position: [cos_a, 0.5, sin_a],
                normal,
            });
        }

        // Generate indices for cylinder sides (quads between rings)
        for i in 0..RADIAL_SEGMENTS {
            let i_next = (i + 1) % RADIAL_SEGMENTS;

            let v0 = (i * 2) as u32; // bottom current
            let v1 = (i * 2 + 1) as u32; // top current
            let v2 = (i_next * 2) as u32; // bottom next
            let v3 = (i_next * 2 + 1) as u32; // top next

            // Two triangles per quad
            indices.extend_from_slice(&[v0, v2, v1]);
            indices.extend_from_slice(&[v1, v2, v3]);
        }

        (vertices, indices)
    }

    /// Generate instance data (transforms + colors) for each bond
    fn generate_instances_with_color(
        positions: &[Vec3],
        bonds: &[(u32, u32)],
        hydrophobicity: &[bool],
    ) -> Vec<CylinderInstance> {
        bonds
            .iter()
            .filter_map(|&(a, b)| {
                let a_idx = a as usize;
                let b_idx = b as usize;
                if a_idx >= positions.len() || b_idx >= positions.len() {
                    return None;
                }

                let start = positions[a_idx];
                let end = positions[b_idx];
                let model = Self::compute_cylinder_transform(start, end);

                // Use color from first atom of the bond
                let is_hydrophobic = hydrophobicity.get(a_idx).copied().unwrap_or(false);
                let color = if is_hydrophobic {
                    HYDROPHOBIC_COLOR
                } else {
                    HYDROPHILIC_COLOR
                };

                Some(CylinderInstance {
                    model: model.to_cols_array_2d(),
                    color,
                    _pad: 0.0,
                })
            })
            .collect()
    }

    /// Compute transform matrix for a cylinder connecting two points
    fn compute_cylinder_transform(start: Vec3, end: Vec3) -> Mat4 {
        let diff = end - start;
        let length = diff.length();

        if length < 1e-6 {
            return Mat4::IDENTITY;
        }

        let direction = diff / length;
        let center = (start + end) * 0.5;

        // Build rotation that transforms Y-axis to the bond direction
        let y_axis = Vec3::Y;
        let rotation = if direction.dot(y_axis).abs() > 0.999 {
            // Nearly parallel to Y, use simple rotation or identity
            if direction.y > 0.0 {
                Mat4::IDENTITY
            } else {
                Mat4::from_axis_angle(Vec3::X, std::f32::consts::PI)
            }
        } else {
            // Compute rotation from Y to direction
            let axis = y_axis.cross(direction).normalize();
            let angle = y_axis.dot(direction).acos();
            Mat4::from_axis_angle(axis, angle)
        };

        // Scale: radius for X/Z, length for Y
        let scale = Mat4::from_scale(Vec3::new(CYLINDER_RADIUS, length, CYLINDER_RADIUS));

        // Translation to center
        let translation = Mat4::from_translation(center);

        // Combined transform: translate * rotate * scale
        translation * rotation * scale
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
        render_pass.set_vertex_buffer(1, self.instance_buffer.buffer().slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.index_count, 0, 0..self.instance_count);
    }
}
