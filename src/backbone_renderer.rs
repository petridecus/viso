use crate::render_context::RenderContext;
use glam::Vec3;
use wgpu::util::DeviceExt;

/// Parameters for backbone tube rendering
const TUBE_RADIUS: f32 = 0.3;
const SEGMENTS_PER_SPAN: usize = 8;
const RADIAL_SEGMENTS: usize = 8;

/// A point along the spline with position, tangent, and frame vectors
#[derive(Clone, Copy)]
struct SplinePoint {
    pos: Vec3,
    tangent: Vec3,
    normal: Vec3,
    binormal: Vec3,
}

/// Vertex for the tube mesh
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TubeVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

pub struct BackboneRenderer {
    pub pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

impl BackboneRenderer {
    pub fn new(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        backbone_chains: &[Vec<Vec3>],
    ) -> Self {
        let (vertices, indices) = Self::generate_tube_mesh(backbone_chains);

        let vertex_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Backbone Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let index_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Backbone Index Buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        let pipeline = Self::create_pipeline(context, camera_layout, lighting_layout);

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
        }
    }

    fn create_pipeline(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = context
            .device
            .create_shader_module(wgpu::include_wgsl!("../assets/shaders/backbone_tube.wgsl"));

        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Backbone Pipeline Layout"),
                    bind_group_layouts: &[camera_layout, lighting_layout],
                    immediate_size: 0,
                });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<TubeVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
            ],
        };

        context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Backbone Render Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[vertex_layout],
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

    fn generate_tube_mesh(chains: &[Vec<Vec3>]) -> (Vec<TubeVertex>, Vec<u32>) {
        let mut all_vertices: Vec<TubeVertex> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();

        for ca_positions in chains {
            // Need at least 4 CA atoms for a smooth spline
            if ca_positions.len() < 4 {
                continue;
            }

            // Generate spline points
            let spline_points = Self::generate_spline_points(ca_positions);

            // Generate tube geometry
            let base_vertex = all_vertices.len() as u32;
            let (vertices, indices) = Self::generate_tube_segment(&spline_points, base_vertex);

            all_vertices.extend(vertices);
            all_indices.extend(indices);
        }

        (all_vertices, all_indices)
    }

    fn generate_spline_points(ca_positions: &[Vec3]) -> Vec<SplinePoint> {
        let n = ca_positions.len();
        let total_segments = (n - 1) * SEGMENTS_PER_SPAN;
        let mut points = Vec::with_capacity(total_segments + 1);

        // Calculate tangents at each CA using Catmull-Rom style
        let tangents: Vec<Vec3> = (0..n)
            .map(|i| {
                if i == 0 {
                    ca_positions[1] - ca_positions[0]
                } else if i == n - 1 {
                    ca_positions[n - 1] - ca_positions[n - 2]
                } else {
                    (ca_positions[i + 1] - ca_positions[i - 1]) * 0.5
                }
            })
            .collect();

        // Generate spline points using cubic Hermite interpolation
        for i in 0..n - 1 {
            let p0 = ca_positions[i];
            let p1 = ca_positions[i + 1];
            let m0 = tangents[i];
            let m1 = tangents[i + 1];

            for j in 0..SEGMENTS_PER_SPAN {
                let t = j as f32 / SEGMENTS_PER_SPAN as f32;
                let pos = hermite_point(p0, m0, p1, m1, t);
                let tangent = hermite_tangent(p0, m0, p1, m1, t).normalize();

                points.push(SplinePoint {
                    pos,
                    tangent,
                    normal: Vec3::ZERO,   // Will be computed by RMF
                    binormal: Vec3::ZERO, // Will be computed by RMF
                });
            }
        }

        // Add final point
        let last = ca_positions[n - 1];
        let last_tangent = tangents[n - 1].normalize();
        points.push(SplinePoint {
            pos: last,
            tangent: last_tangent,
            normal: Vec3::ZERO,
            binormal: Vec3::ZERO,
        });

        // Compute rotation minimizing frames
        compute_rmf(&mut points);

        points
    }

    fn generate_tube_segment(
        points: &[SplinePoint],
        base_vertex: u32,
    ) -> (Vec<TubeVertex>, Vec<u32>) {
        let num_rings = points.len();
        let mut vertices = Vec::with_capacity(num_rings * RADIAL_SEGMENTS);
        let mut indices = Vec::new();

        // Generate vertices for each ring
        for point in points {
            for k in 0..RADIAL_SEGMENTS {
                let angle = (k as f32 / RADIAL_SEGMENTS as f32) * std::f32::consts::TAU;
                let cos_a = angle.cos();
                let sin_a = angle.sin();

                // Position on tube surface
                let offset = point.normal * cos_a + point.binormal * sin_a;
                let pos = point.pos + offset * TUBE_RADIUS;

                // Normal is just the offset direction (points outward from tube center)
                let normal = offset.normalize();

                vertices.push(TubeVertex {
                    position: pos.into(),
                    normal: normal.into(),
                });
            }
        }

        // Generate indices for triangles connecting adjacent rings
        for i in 0..num_rings - 1 {
            let ring_start = i * RADIAL_SEGMENTS;
            let next_ring_start = (i + 1) * RADIAL_SEGMENTS;

            for k in 0..RADIAL_SEGMENTS {
                let k_next = (k + 1) % RADIAL_SEGMENTS;

                let v0 = base_vertex + (ring_start + k) as u32;
                let v1 = base_vertex + (ring_start + k_next) as u32;
                let v2 = base_vertex + (next_ring_start + k) as u32;
                let v3 = base_vertex + (next_ring_start + k_next) as u32;

                // Two triangles per quad
                indices.extend_from_slice(&[v0, v2, v1]);
                indices.extend_from_slice(&[v1, v2, v3]);
            }
        }

        (vertices, indices)
    }

    pub fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        lighting_bind_group: &'a wgpu::BindGroup,
    ) {
        if self.index_count == 0 {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, lighting_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}

/// Cubic Hermite interpolation for position
fn hermite_point(p0: Vec3, m0: Vec3, p1: Vec3, m1: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;

    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;

    p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11
}

/// Cubic Hermite interpolation for tangent (derivative of position)
fn hermite_tangent(p0: Vec3, m0: Vec3, p1: Vec3, m1: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;

    let dh00 = 6.0 * t2 - 6.0 * t;
    let dh10 = 3.0 * t2 - 4.0 * t + 1.0;
    let dh01 = -6.0 * t2 + 6.0 * t;
    let dh11 = 3.0 * t2 - 2.0 * t;

    p0 * dh00 + m0 * dh10 + p1 * dh01 + m1 * dh11
}

/// Compute Rotation Minimizing Frames using the double reflection method
/// (Wang et al. 2008: "Computation of Rotation Minimizing Frames")
fn compute_rmf(points: &mut [SplinePoint]) {
    if points.is_empty() {
        return;
    }

    // Initialize first frame
    let t0 = points[0].tangent;
    let arbitrary = if t0.x.abs() < 0.9 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let n0 = t0.cross(arbitrary).normalize();
    let b0 = t0.cross(n0).normalize();

    points[0].normal = n0;
    points[0].binormal = b0;

    // Propagate frame using double reflection
    for i in 0..points.len() - 1 {
        let x_i = points[i].pos;
        let x_i1 = points[i + 1].pos;
        let t_i = points[i].tangent;
        let t_i1 = points[i + 1].tangent;
        let r_i = points[i].normal;
        let s_i = points[i].binormal;

        let v1 = x_i1 - x_i;
        let c1 = v1.dot(v1);

        if c1 < 1e-10 {
            // Points are coincident, just copy frame
            points[i + 1].normal = r_i;
            points[i + 1].binormal = s_i;
            continue;
        }

        // First reflection (reflect r_i and t_i across plane perpendicular to v1)
        let r_i_l = r_i - (2.0 / c1) * v1.dot(r_i) * v1;
        let t_i_l = t_i - (2.0 / c1) * v1.dot(t_i) * v1;

        // Second reflection
        let v2 = t_i1 - t_i_l;
        let c2 = v2.dot(v2);

        let r_i1 = if c2 < 1e-10 {
            r_i_l
        } else {
            r_i_l - (2.0 / c2) * v2.dot(r_i_l) * v2
        };

        // Ensure orthonormality
        let r_i1 = (r_i1 - t_i1 * t_i1.dot(r_i1)).normalize();
        let s_i1 = t_i1.cross(r_i1).normalize();

        points[i + 1].normal = r_i1;
        points[i + 1].binormal = s_i1;
    }
}
