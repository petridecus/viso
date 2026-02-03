//! Tube renderer for protein backbone
//!
//! Renders protein backbone as smooth tubes using cubic Hermite splines
//! with rotation-minimizing frames for consistent tube orientation.
//!
//! Can render all SS types or filter to specific ones (e.g., coils only
//! when used alongside RibbonRenderer in ribbon view mode).

use crate::dynamic_buffer::DynamicBuffer;
use crate::render_context::RenderContext;
use crate::secondary_structure::{detect_secondary_structure, SSType};
use glam::Vec3;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};

/// Parameters for backbone tube rendering
const TUBE_RADIUS: f32 = 0.3;
const SEGMENTS_PER_SPAN: usize = 16;
const RADIAL_SEGMENTS: usize = 32;

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
    color: [f32; 3],
    residue_idx: u32,
}

pub struct TubeRenderer {
    pub pipeline: wgpu::RenderPipeline,
    vertex_buffer: DynamicBuffer,
    index_buffer: DynamicBuffer,
    pub index_count: u32,
    /// Hash of the last chain data for change detection
    last_chain_hash: u64,
    /// Which SS types to render (None = all types)
    ss_filter: Option<HashSet<SSType>>,
    /// Cached chain data for regeneration when filter changes
    cached_chains: Vec<Vec<Vec3>>,
}

impl TubeRenderer {
    pub fn new(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        selection_layout: &wgpu::BindGroupLayout,
        backbone_chains: &[Vec<Vec3>],
    ) -> Self {
        let ss_filter = None; // Render all SS types by default
        let (vertices, indices) = Self::generate_tube_mesh(backbone_chains, &ss_filter);

        let vertex_buffer = if vertices.is_empty() {
            DynamicBuffer::new(
                &context.device,
                "Backbone Vertex Buffer",
                std::mem::size_of::<TubeVertex>() * 1000,
                wgpu::BufferUsages::VERTEX,
            )
        } else {
            DynamicBuffer::new_with_data(
                &context.device,
                "Backbone Vertex Buffer",
                &vertices,
                wgpu::BufferUsages::VERTEX,
            )
        };

        let index_buffer = if indices.is_empty() {
            DynamicBuffer::new(
                &context.device,
                "Backbone Index Buffer",
                std::mem::size_of::<u32>() * 3000,
                wgpu::BufferUsages::INDEX,
            )
        } else {
            DynamicBuffer::new_with_data(
                &context.device,
                "Backbone Index Buffer",
                &indices,
                wgpu::BufferUsages::INDEX,
            )
        };

        let pipeline = Self::create_pipeline(context, camera_layout, lighting_layout, selection_layout);
        let last_chain_hash = Self::compute_chain_hash(backbone_chains);

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            last_chain_hash,
            ss_filter,
            cached_chains: backbone_chains.to_vec(),
        }
    }

    /// Set which secondary structure types this renderer should render
    /// Pass None to render all types, or Some(set) to filter
    pub fn set_ss_filter(&mut self, filter: Option<HashSet<SSType>>) {
        self.ss_filter = filter;
    }

    /// Regenerate mesh with current filter settings
    pub fn regenerate(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let (vertices, indices) = Self::generate_tube_mesh(&self.cached_chains, &self.ss_filter);
        if !vertices.is_empty() {
            self.vertex_buffer.write(device, queue, &vertices);
            self.index_buffer.write(device, queue, &indices);
        }
        self.index_count = indices.len() as u32;
    }

    /// Compute a hash of the chain data for change detection
    fn compute_chain_hash(chains: &[Vec<Vec3>]) -> u64 {
        let mut hasher = DefaultHasher::new();

        chains.len().hash(&mut hasher);
        for chain in chains {
            chain.len().hash(&mut hasher);
            // Hash first, middle, and last positions for good change detection
            // without hashing every single point
            if let Some(first) = chain.first() {
                first.x.to_bits().hash(&mut hasher);
                first.y.to_bits().hash(&mut hasher);
                first.z.to_bits().hash(&mut hasher);
            }
            if chain.len() > 2 {
                let mid = &chain[chain.len() / 2];
                mid.x.to_bits().hash(&mut hasher);
                mid.y.to_bits().hash(&mut hasher);
                mid.z.to_bits().hash(&mut hasher);
            }
            if let Some(last) = chain.last() {
                last.x.to_bits().hash(&mut hasher);
                last.y.to_bits().hash(&mut hasher);
                last.z.to_bits().hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    /// Update the backbone with new chains (regenerates the mesh if changed)
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        backbone_chains: &[Vec<Vec3>],
    ) {
        // Quick hash check to see if chains actually changed
        let new_hash = Self::compute_chain_hash(backbone_chains);
        if new_hash == self.last_chain_hash {
            return; // No change, skip expensive mesh regeneration
        }
        self.last_chain_hash = new_hash;
        self.cached_chains = backbone_chains.to_vec();

        let (vertices, indices) = Self::generate_tube_mesh(backbone_chains, &self.ss_filter);

        if vertices.is_empty() {
            self.index_count = 0;
            return;
        }

        self.vertex_buffer.write(device, queue, &vertices);
        self.index_buffer.write(device, queue, &indices);
        self.index_count = indices.len() as u32;
    }

    /// Legacy method for compatibility - calls update()
    pub fn update_chains(&mut self, device: &wgpu::Device, backbone_chains: &[Vec<Vec3>]) {
        // Create a temporary queue-less update by regenerating buffers
        let new_hash = Self::compute_chain_hash(backbone_chains);
        if new_hash == self.last_chain_hash {
            return;
        }
        self.last_chain_hash = new_hash;
        self.cached_chains = backbone_chains.to_vec();

        let (vertices, indices) = Self::generate_tube_mesh(backbone_chains, &self.ss_filter);

        if vertices.is_empty() {
            self.index_count = 0;
            return;
        }

        // For legacy compatibility, recreate buffers directly
        self.vertex_buffer = DynamicBuffer::new_with_data(
            device,
            "Backbone Vertex Buffer",
            &vertices,
            wgpu::BufferUsages::VERTEX,
        );

        self.index_buffer = DynamicBuffer::new_with_data(
            device,
            "Backbone Index Buffer",
            &indices,
            wgpu::BufferUsages::INDEX,
        );

        self.index_count = indices.len() as u32;
    }

    fn create_pipeline(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        selection_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = context
            .device
            .create_shader_module(wgpu::include_wgsl!("../assets/shaders/backbone_tube.wgsl"));

        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Backbone Pipeline Layout"),
                    bind_group_layouts: &[camera_layout, lighting_layout, selection_layout],
                    immediate_size: 0,
                });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<TubeVertex>() as wgpu::BufferAddress,
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
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 24,
                    shader_location: 2, // color
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint32,
                    offset: 36,
                    shader_location: 3, // residue_idx
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

    fn generate_tube_mesh(
        chains: &[Vec<Vec3>],
        ss_filter: &Option<HashSet<SSType>>,
    ) -> (Vec<TubeVertex>, Vec<u32>) {
        let mut all_vertices: Vec<TubeVertex> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();
        let mut global_residue_idx: u32 = 0;

        for backbone_atoms in chains {
            // Backbone chains contain N, CA, C atoms per residue (3 atoms per residue)
            // Extract just CA positions for spline and SS detection
            // CA is at index 1, 4, 7, 10, ... (every 3rd starting at 1)
            let ca_positions: Vec<Vec3> = backbone_atoms
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 3 == 1) // CA is the second atom in each N, CA, C triplet
                .map(|(_, &pos)| pos)
                .collect();

            // Need at least 4 CA atoms for a smooth spline
            if ca_positions.len() < 4 {
                global_residue_idx += ca_positions.len() as u32;
                continue;
            }

            // Detect secondary structure from CA positions
            let ss_types = detect_secondary_structure(&ca_positions);

            // Generate spline points from raw CA positions (no smoothing/idealization)
            let spline_points = Self::generate_spline_points(&ca_positions);

            // If no filter, render all. Otherwise, render only matching SS segments.
            if let Some(filter) = ss_filter {
                // Render only segments matching the filter
                Self::generate_filtered_segments(
                    &spline_points,
                    &ss_types,
                    filter,
                    global_residue_idx,
                    &mut all_vertices,
                    &mut all_indices,
                );
            } else {
                // Render everything
                let spline_colors = Self::interpolate_ss_colors(&ss_types, spline_points.len());
                let residue_indices = Self::interpolate_residue_indices(ca_positions.len(), spline_points.len(), global_residue_idx);
                let base_vertex = all_vertices.len() as u32;
                let (vertices, indices) =
                    Self::generate_tube_segment(&spline_points, &spline_colors, &residue_indices, base_vertex);
                all_vertices.extend(vertices);
                all_indices.extend(indices);
            }
            
            global_residue_idx += ca_positions.len() as u32;
        }

        (all_vertices, all_indices)
    }

    /// Generate tube segments only for residues matching the SS filter
    /// Extends segments by 1 residue at each boundary to overlap with ribbons
    fn generate_filtered_segments(
        spline_points: &[SplinePoint],
        ss_types: &[SSType],
        filter: &HashSet<SSType>,
        base_residue_idx: u32,
        all_vertices: &mut Vec<TubeVertex>,
        all_indices: &mut Vec<u32>,
    ) {
        if spline_points.is_empty() || ss_types.is_empty() {
            return;
        }

        let n_residues = ss_types.len();
        let points_per_residue = SEGMENTS_PER_SPAN;

        // Find contiguous runs of residues matching the filter
        let mut i = 0;
        while i < n_residues {
            // Skip residues not in filter
            if !filter.contains(&ss_types[i]) {
                i += 1;
                continue;
            }

            // Found start of a matching segment
            let start_residue = i;
            while i < n_residues && filter.contains(&ss_types[i]) {
                i += 1;
            }
            let end_residue = i;

            // Extend segment by 1 residue on each end to overlap with ribbons
            // This ensures tubes go "under" the ribbons at junctions
            let extended_start = start_residue.saturating_sub(1);
            let extended_end = (end_residue + 1).min(n_residues);

            // Need at least 2 residues for a tube segment
            if extended_end - extended_start < 2 {
                continue;
            }

            // Convert residue range to spline point range
            let start_point = extended_start * points_per_residue;
            let end_point = ((extended_end - 1) * points_per_residue + 1).min(spline_points.len());

            if start_point >= end_point || start_point >= spline_points.len() {
                continue;
            }

            let segment_points = &spline_points[start_point..end_point];
            // Use original (non-extended) SS types for coloring so colors match ribbons
            let segment_ss = &ss_types[extended_start..extended_end];
            let segment_colors = Self::interpolate_ss_colors(segment_ss, segment_points.len());
            let segment_residue_indices = Self::interpolate_residue_indices(
                extended_end - extended_start,
                segment_points.len(),
                base_residue_idx + extended_start as u32,
            );

            let base_vertex = all_vertices.len() as u32;
            let (vertices, indices) =
                Self::generate_tube_segment(segment_points, &segment_colors, &segment_residue_indices, base_vertex);

            all_vertices.extend(vertices);
            all_indices.extend(indices);
        }
    }

    /// Interpolate SS colors to match spline point count
    fn interpolate_ss_colors(ss_types: &[SSType], num_spline_points: usize) -> Vec<[f32; 3]> {
        if ss_types.is_empty() {
            return vec![[0.6, 0.85, 0.6]; num_spline_points];
        }

        let n_residues = ss_types.len();
        let _points_per_residue = if n_residues > 1 {
            (num_spline_points - 1) / (n_residues - 1)
        } else {
            num_spline_points
        };

        let mut colors = Vec::with_capacity(num_spline_points);

        for i in 0..num_spline_points {
            // Map spline point index to residue index
            let residue_idx = if n_residues > 1 {
                ((i as f32 / (num_spline_points - 1) as f32) * (n_residues - 1) as f32) as usize
            } else {
                0
            };
            let residue_idx = residue_idx.min(n_residues - 1);
            colors.push(ss_types[residue_idx].color());
        }

        colors
    }

    /// Interpolate residue indices to match spline point count
    fn interpolate_residue_indices(n_residues: usize, num_spline_points: usize, base_residue: u32) -> Vec<u32> {
        if n_residues == 0 {
            return vec![base_residue; num_spline_points];
        }

        let mut indices = Vec::with_capacity(num_spline_points);

        for i in 0..num_spline_points {
            // Map spline point index to residue index
            let residue_idx = if n_residues > 1 {
                ((i as f32 / (num_spline_points - 1) as f32) * (n_residues - 1) as f32) as usize
            } else {
                0
            };
            let residue_idx = residue_idx.min(n_residues - 1);
            indices.push(base_residue + residue_idx as u32);
        }

        indices
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
        colors: &[[f32; 3]],
        residue_indices: &[u32],
        base_vertex: u32,
    ) -> (Vec<TubeVertex>, Vec<u32>) {
        let num_rings = points.len();
        let mut vertices = Vec::with_capacity(num_rings * RADIAL_SEGMENTS);
        let mut indices = Vec::new();

        // Generate vertices for each ring
        for (i, point) in points.iter().enumerate() {
            let color = colors.get(i).copied().unwrap_or([0.6, 0.85, 0.6]);
            let residue_idx = residue_indices.get(i).copied().unwrap_or(0);

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
                    color,
                    residue_idx,
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
        selection_bind_group: &'a wgpu::BindGroup,
    ) {
        if self.index_count == 0 {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, lighting_bind_group, &[]);
        render_pass.set_bind_group(2, selection_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.buffer().slice(..));
        render_pass.set_index_buffer(self.index_buffer.buffer().slice(..), wgpu::IndexFormat::Uint32);
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
