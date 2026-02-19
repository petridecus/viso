//! Tube renderer for protein backbone
//!
//! Renders protein backbone as smooth tubes using cubic Hermite splines
//! with rotation-minimizing frames for consistent tube orientation.
//!
//! Can render all SS types or filter to specific ones (e.g., coils only
//! when used alongside RibbonRenderer in ribbon view mode).

use std::collections::HashSet;

use foldit_conv::secondary_structure::{resolve, DetectionInput, SSType};
use glam::Vec3;

use crate::{
    gpu::{
        dynamic_buffer::DynamicBuffer, render_context::RenderContext,
        shader_composer::ShaderComposer,
    },
    renderer::pipeline_util,
    util::hash::hash_vec3_slices,
};

/// Parameters for backbone tube rendering
const TUBE_RADIUS: f32 = 0.3;

/// Radial segments around the tube circumference
const RADIAL_SEGMENTS: usize = 8;

/// Axial segments along the tube between CA atoms
const SEGMENTS_PER_SPAN: usize = 4;

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
pub(crate) struct TubeVertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
    residue_idx: u32,
    /// Tube centerline position — enables per-pixel cylindrical normals in
    /// fragment shader
    center_pos: [f32; 3],
}

/// Get the vertex buffer layout for TubeVertex (for use with picking pipeline)
pub fn tube_vertex_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
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
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 40,
                shader_location: 4, // center_pos
            },
        ],
    }
}

/// Pre-computed tube mesh data for GPU upload.
pub struct PreparedTubeData<'a> {
    pub vertices: &'a [u8],
    pub indices: &'a [u8],
    pub index_count: u32,
    pub cached_chains: Vec<Vec<Vec3>>,
    pub ss_override: Option<Vec<SSType>>,
}

pub struct TubeRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: DynamicBuffer,
    index_buffer: DynamicBuffer,
    pub index_count: u32,
    /// Hash of the last chain data for change detection
    last_chain_hash: u64,
    /// Which SS types to render (None = all types)
    ss_filter: Option<HashSet<SSType>>,
    /// Cached chain data for regeneration when filter changes
    cached_chains: Vec<Vec<Vec3>>,
    /// Pre-computed SS types override (from puzzle.toml annotation)
    ss_override: Option<Vec<SSType>>,
}

impl TubeRenderer {
    pub fn new(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        selection_layout: &wgpu::BindGroupLayout,
        color_layout: &wgpu::BindGroupLayout,
        backbone_chains: &[Vec<Vec3>],
        shader_composer: &mut ShaderComposer,
    ) -> Self {
        let ss_filter = None; // Render all SS types by default

        let (vertices, indices) =
            Self::generate_tube_mesh(backbone_chains, &ss_filter, None);

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

        let pipeline = Self::create_pipeline(
            context,
            camera_layout,
            lighting_layout,
            selection_layout,
            color_layout,
            shader_composer,
        );
        let last_chain_hash = hash_vec3_slices(backbone_chains);

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            last_chain_hash,
            ss_filter,
            cached_chains: backbone_chains.to_vec(),
            ss_override: None,
        }
    }

    /// Set which secondary structure types this renderer should render
    /// Pass None to render all types, or Some(set) to filter
    pub fn set_ss_filter(&mut self, filter: Option<HashSet<SSType>>) {
        self.ss_filter = filter;
    }

    /// Set pre-computed SS types (from puzzle.toml annotation or DSSP).
    /// When set, these are used instead of auto-detection.
    pub fn set_ss_override(&mut self, ss_types: Option<Vec<SSType>>) {
        self.ss_override = ss_types;
    }

    /// Regenerate mesh with current filter settings
    pub fn regenerate(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let (vertices, indices) = Self::generate_tube_mesh(
            &self.cached_chains,
            &self.ss_filter,
            self.ss_override.as_deref(),
        );
        if !vertices.is_empty() {
            self.vertex_buffer.write(device, queue, &vertices);
            self.index_buffer.write(device, queue, &indices);
        }
        self.index_count = indices.len() as u32;
    }

    /// Update the backbone with new chains (regenerates the mesh if changed)
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        backbone_chains: &[Vec<Vec3>],
        ss_types: Option<&[SSType]>,
    ) {
        // Quick hash check to see if chains actually changed
        let new_hash = hash_vec3_slices(backbone_chains);
        if new_hash == self.last_chain_hash && ss_types.is_none() {
            return; // No change, skip expensive mesh regeneration
        }
        self.last_chain_hash = new_hash;
        self.cached_chains = backbone_chains.to_vec();
        if let Some(ss) = ss_types {
            self.ss_override = Some(ss.to_vec());
        }

        let (vertices, indices) = Self::generate_tube_mesh(
            backbone_chains,
            &self.ss_filter,
            self.ss_override.as_deref(),
        );

        if vertices.is_empty() {
            self.index_count = 0;
            return;
        }

        self.vertex_buffer.write(device, queue, &vertices);
        self.index_buffer.write(device, queue, &indices);
        self.index_count = indices.len() as u32;
    }

    /// Legacy method for compatibility - calls update()
    pub fn update_chains(
        &mut self,
        device: &wgpu::Device,
        backbone_chains: &[Vec<Vec3>],
    ) {
        // Create a temporary queue-less update by regenerating buffers
        let new_hash = hash_vec3_slices(backbone_chains);
        if new_hash == self.last_chain_hash {
            return;
        }
        self.last_chain_hash = new_hash;
        self.cached_chains = backbone_chains.to_vec();

        let (vertices, indices) = Self::generate_tube_mesh(
            backbone_chains,
            &self.ss_filter,
            self.ss_override.as_deref(),
        );

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
        color_layout: &wgpu::BindGroupLayout,
        shader_composer: &mut ShaderComposer,
    ) -> wgpu::RenderPipeline {
        let shader = shader_composer.compose(
            &context.device,
            "Backbone Tube Shader",
            include_str!(
                "../../../assets/shaders/raster/mesh/backbone_tube.wgsl"
            ),
            "backbone_tube.wgsl",
        );

        let pipeline_layout = context.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Backbone Pipeline Layout"),
                bind_group_layouts: &[
                    camera_layout,
                    lighting_layout,
                    selection_layout,
                    color_layout,
                ],
                immediate_size: 0,
            },
        );

        let vertex_layout = tube_vertex_buffer_layout();

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
                    targets: &pipeline_util::hdr_fragment_targets(),
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(pipeline_util::depth_stencil_state()),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
    }

    /// Generate tube mesh for all chains
    pub(crate) fn generate_tube_mesh(
        chains: &[Vec<Vec3>],
        ss_filter: &Option<HashSet<SSType>>,
        ss_override: Option<&[SSType]>,
    ) -> (Vec<TubeVertex>, Vec<u32>) {
        Self::generate_tube_mesh_colored(chains, ss_filter, ss_override, None)
    }

    /// Generate tube mesh with optional per-residue color override
    pub(crate) fn generate_tube_mesh_colored(
        chains: &[Vec<Vec3>],
        ss_filter: &Option<HashSet<SSType>>,
        ss_override: Option<&[SSType]>,
        per_residue_colors: Option<&[[f32; 3]]>,
    ) -> (Vec<TubeVertex>, Vec<u32>) {
        let mut all_vertices: Vec<TubeVertex> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();
        let mut global_residue_idx: u32 = 0;

        for backbone_atoms in chains {
            // Backbone chains contain N, CA, C atoms per residue (3 atoms per
            // residue) Extract just CA positions for spline and SS
            // detection CA is at index 1, 4, 7, 10, ... (every 3rd
            // starting at 1)
            let ca_positions: Vec<Vec3> = backbone_atoms
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 3 == 1) // CA is the second atom in each N, CA, C triplet
                .map(|(_, &pos)| pos)
                .collect();

            // Need at least 2 CA atoms for a spline segment
            if ca_positions.len() < 2 {
                global_residue_idx += ca_positions.len() as u32;
                continue;
            }

            let n_residues = ca_positions.len();
            let chain_override = ss_override.and_then(|o| {
                let start = global_residue_idx as usize;
                let end = (start + n_residues).min(o.len());
                (start < o.len()).then(|| &o[start..end])
            });
            let ss_types = resolve(
                chain_override,
                DetectionInput::CaPositions(&ca_positions),
            );

            // Generate spline points from raw CA positions
            let spline_points = Self::generate_spline_points(&ca_positions);

            // If no filter, render all. Otherwise, render only matching SS
            // segments.
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
                let spline_colors =
                    Self::interpolate_ss_colors(&ss_types, spline_points.len());
                let residue_indices = Self::interpolate_residue_indices(
                    ca_positions.len(),
                    spline_points.len(),
                    global_residue_idx,
                );

                let base_vertex = all_vertices.len() as u32;
                let (vertices, indices) = Self::generate_tube_segment(
                    &spline_points,
                    &spline_colors,
                    &residue_indices,
                    base_vertex,
                );
                all_vertices.extend(vertices);
                all_indices.extend(indices);
            }

            global_residue_idx += ca_positions.len() as u32;
        }

        // Apply per-residue color override if provided
        if let Some(colors) = per_residue_colors {
            for vert in &mut all_vertices {
                let idx = vert.residue_idx as usize;
                if idx < colors.len() {
                    vert.color = colors[idx];
                }
            }
        }

        (all_vertices, all_indices)
    }

    /// Generate tube segments only for residues matching the SS filter
    /// Generate tube segments only for residues matching the SS filter.
    /// Ribbon handles the taper into coil regions, so tubes render only their
    /// exact range.
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

            // Need at least 2 residues for a tube segment
            if end_residue - start_residue < 2 {
                continue;
            }

            // Convert residue range to spline point range
            let start_point = start_residue * points_per_residue;
            let end_point = ((end_residue - 1) * points_per_residue + 1)
                .min(spline_points.len());

            if start_point >= end_point || start_point >= spline_points.len() {
                continue;
            }

            let segment_points = &spline_points[start_point..end_point];
            let segment_ss = &ss_types[start_residue..end_residue];
            let segment_colors =
                Self::interpolate_ss_colors(segment_ss, segment_points.len());
            let segment_residue_indices = Self::interpolate_residue_indices(
                end_residue - start_residue,
                segment_points.len(),
                base_residue_idx + start_residue as u32,
            );

            let base_vertex = all_vertices.len() as u32;
            let (vertices, indices) = Self::generate_tube_segment(
                segment_points,
                &segment_colors,
                &segment_residue_indices,
                base_vertex,
            );

            all_vertices.extend(vertices);
            all_indices.extend(indices);
        }
    }

    /// Interpolate SS colors to match spline point count
    fn interpolate_ss_colors(
        ss_types: &[SSType],
        num_spline_points: usize,
    ) -> Vec<[f32; 3]> {
        if ss_types.is_empty() {
            return vec![[0.6, 0.85, 0.6]; num_spline_points];
        }

        let n_residues = ss_types.len();
        let mut colors = Vec::with_capacity(num_spline_points);

        for i in 0..num_spline_points {
            // Map spline point index to residue index
            let residue_idx = if n_residues > 1 {
                ((i as f32 / (num_spline_points - 1) as f32)
                    * (n_residues - 1) as f32) as usize
            } else {
                0
            };
            let residue_idx = residue_idx.min(n_residues - 1);
            colors.push(ss_types[residue_idx].color());
        }

        colors
    }

    /// Interpolate residue indices to match spline point count
    fn interpolate_residue_indices(
        n_residues: usize,
        num_spline_points: usize,
        base_residue: u32,
    ) -> Vec<u32> {
        if n_residues == 0 {
            return vec![base_residue; num_spline_points];
        }

        let mut indices = Vec::with_capacity(num_spline_points);

        for i in 0..num_spline_points {
            // Map spline point index to residue index
            let residue_idx = if n_residues > 1 {
                ((i as f32 / (num_spline_points - 1) as f32)
                    * (n_residues - 1) as f32) as usize
            } else {
                0
            };
            let residue_idx = residue_idx.min(n_residues - 1);
            indices.push(base_residue + residue_idx as u32);
        }

        indices
    }

    /// Generate spline points from CA positions
    fn generate_spline_points(ca_positions: &[Vec3]) -> Vec<SplinePoint> {
        let n = ca_positions.len();
        if n < 2 {
            return Vec::new();
        }

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
                    normal: Vec3::ZERO, // Will be computed by RMF
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

    /// Generate tube segment geometry
    fn generate_tube_segment(
        points: &[SplinePoint],
        colors: &[[f32; 3]],
        residue_indices: &[u32],
        base_vertex: u32,
    ) -> (Vec<TubeVertex>, Vec<u32>) {
        if points.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let num_rings = points.len();
        let total_vertices = num_rings * RADIAL_SEGMENTS;
        let mut vertices = Vec::with_capacity(total_vertices);
        let mut indices = Vec::new();

        // Generate vertices for each ring
        for (i, point) in points.iter().enumerate() {
            let color = colors.get(i).copied().unwrap_or([0.6, 0.85, 0.6]);
            let residue_idx = residue_indices.get(i).copied().unwrap_or(0);

            for k in 0..RADIAL_SEGMENTS {
                let angle =
                    (k as f32 / RADIAL_SEGMENTS as f32) * std::f32::consts::TAU;
                let cos_a = angle.cos();
                let sin_a = angle.sin();

                // Position on tube surface
                let offset = point.normal * cos_a + point.binormal * sin_a;
                let pos = point.pos + offset * TUBE_RADIUS;

                // Normal is just the offset direction (points outward from tube
                // center)
                let normal = offset.normalize();

                vertices.push(TubeVertex {
                    position: pos.into(),
                    normal: normal.into(),
                    color,
                    residue_idx,
                    center_pos: point.pos.into(),
                });
            }
        }

        // Generate indices for triangles connecting adjacent rings
        for i in 0..num_rings - 1 {
            let ring_offset = i * RADIAL_SEGMENTS;
            let next_ring_offset = (i + 1) * RADIAL_SEGMENTS;

            for k in 0..RADIAL_SEGMENTS {
                let k_next = (k + 1) % RADIAL_SEGMENTS;

                let v0 = base_vertex + (ring_offset + k) as u32;
                let v1 = base_vertex + (ring_offset + k_next) as u32;
                let v2 = base_vertex + (next_ring_offset + k) as u32;
                let v3 = base_vertex + (next_ring_offset + k_next) as u32;

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
        bind_groups: &super::draw_context::DrawBindGroups<'a>,
    ) {
        if self.index_count == 0 {
            return;
        }

        let color_bind_group = match bind_groups.color {
            Some(bg) => bg,
            None => return,
        };

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, bind_groups.camera, &[]);
        render_pass.set_bind_group(1, bind_groups.lighting, &[]);
        render_pass.set_bind_group(2, bind_groups.selection, &[]);
        render_pass.set_bind_group(3, color_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.buffer().slice(..));
        render_pass.set_index_buffer(
            self.index_buffer.buffer().slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(0..self.index_count, 0, 0..1);
    }

    /// Get the vertex buffer for picking
    pub fn vertex_buffer(&self) -> &wgpu::Buffer {
        self.vertex_buffer.buffer()
    }

    /// Get the index buffer for picking
    pub fn index_buffer(&self) -> &wgpu::Buffer {
        self.index_buffer.buffer()
    }

    /// Apply pre-computed mesh data (GPU upload only, no CPU generation).
    ///
    /// Called from `apply_pending_scene` with data produced by the background
    /// SceneProcessor.
    pub fn apply_prepared(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: PreparedTubeData,
    ) {
        if !data.vertices.is_empty() {
            self.vertex_buffer.write_bytes(device, queue, data.vertices);
            self.index_buffer.write_bytes(device, queue, data.indices);
        }
        self.index_count = data.index_count;
        self.cached_chains = data.cached_chains;
        self.last_chain_hash = hash_vec3_slices(&self.cached_chains);
        if let Some(ss) = data.ss_override {
            self.ss_override = Some(ss);
        }
    }

    /// Update only metadata (chains, SS types) without uploading vertex data.
    ///
    /// Used when a FullRebuild arrives during animation: we need the metadata
    /// for subsequent animation frames, but vertex data should come from the
    /// animation frame path to avoid a one-frame jump to target positions.
    pub fn update_metadata(
        &mut self,
        cached_chains: Vec<Vec<Vec3>>,
        ss_override: Option<Vec<SSType>>,
    ) {
        self.cached_chains = cached_chains;
        self.last_chain_hash = hash_vec3_slices(&self.cached_chains);
        if let Some(ss) = ss_override {
            self.ss_override = Some(ss);
        }
    }

    /// Apply pre-computed animation frame mesh (GPU upload only).
    ///
    /// Lighter-weight than `apply_prepared`: only writes vertex/index buffers
    /// without updating `cached_chains` or `ss_override` (those are set by
    /// FullRebuild, not animation frames).
    pub fn apply_mesh(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[u8],
        indices: &[u8],
        index_count: u32,
    ) {
        if !vertices.is_empty() {
            self.vertex_buffer.write_bytes(device, queue, vertices);
            self.index_buffer.write_bytes(device, queue, indices);
        }
        self.index_count = index_count;
    }

    /// Get a reference to the cached backbone chains.
    pub fn cached_chains(&self) -> &[Vec<Vec3>] {
        &self.cached_chains
    }
}

impl super::MolecularRenderer for TubeRenderer {
    fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &super::draw_context::DrawBindGroups<'a>,
    ) {
        self.draw(render_pass, bind_groups);
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
    let arbitrary = if t0.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
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

        // First reflection (reflect r_i and t_i across plane perpendicular to
        // v1)
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
