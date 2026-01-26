//! Ribbon renderer for secondary structure visualization
//!
//! Renders helices and sheets as ribbons:
//! - Helices: Flat ribbons oriented by peptide plane (C=O direction)
//! - Sheets: Constant-width quads from N/O atom positions
//!
//! Coils/loops are NOT rendered here - they are handled by TubeRenderer as tubes.

use crate::dynamic_buffer::DynamicBuffer;
use crate::protein_data::{BackboneChain, BackboneResidue};
use crate::render_context::RenderContext;
use crate::secondary_structure::{detect_secondary_structure, SSType};
use glam::Vec3;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Parameters for ribbon rendering (Foldit-style)
/// Only applies to helices and sheets - coils are rendered by TubeRenderer
#[derive(Clone, Copy)]
pub struct RibbonParams {
    // Helix parameters - flat ribbon oriented by peptide plane
    pub helix_width: f32,
    pub helix_thickness: f32,

    // Sheet parameters - constant width quads from N/O atoms
    pub sheet_thickness: f32,

    // Shared
    pub segments_per_residue: usize,
}

impl Default for RibbonParams {
    fn default() -> Self {
        Self {
            helix_width: 1.5,       // Flat ribbon width
            helix_thickness: 0.3,   // Thin flat ribbon

            sheet_thickness: 0.4,   // Sheet extrusion thickness

            segments_per_residue: 8,
        }
    }
}

/// Segments per span between control points
const SEGMENTS_PER_SPAN: usize = 8;
/// Unified cross-section vertex count for all SS types (enables smooth transitions)
/// Helix: 4 corners + 4 edge midpoints (rounded rectangle)
/// Coil: 8-sided polygon (octagon approximating circle)
/// Sheet: 4 quad corners + 4 for thickness edges
const UNIFIED_CROSS_SECTION_VERTS: usize = 8;

/// A point along the spline with position, tangent, and frame vectors
#[derive(Clone, Copy)]
struct SplinePoint {
    pos: Vec3,
    tangent: Vec3,
    normal: Vec3,
    binormal: Vec3,
    /// Residue index this spline point belongs to
    residue_idx: usize,
    /// Parameter t within the residue (0.0 to 1.0)
    residue_t: f32,
}

/// Vertex for the ribbon mesh (same layout as TubeVertex for shader compatibility)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RibbonVertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
}

/// A segment of the backbone with consistent secondary structure
#[derive(Debug)]
struct SSSegment {
    ss_type: SSType,
    start_residue: usize,
    end_residue: usize, // Exclusive
}

pub struct RibbonRenderer {
    pub pipeline: wgpu::RenderPipeline,
    vertex_buffer: DynamicBuffer,
    index_buffer: DynamicBuffer,
    pub index_count: u32,
    last_chain_hash: u64,
    params: RibbonParams,
}

impl RibbonRenderer {
    pub fn new(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        backbone_chains: &[Vec<Vec3>],
    ) -> Self {
        let params = RibbonParams::default();
        let (vertices, indices) = Self::generate_ribbon_mesh(backbone_chains, &params);

        let vertex_buffer = if vertices.is_empty() {
            DynamicBuffer::new(
                &context.device,
                "Ribbon Vertex Buffer",
                std::mem::size_of::<RibbonVertex>() * 1000,
                wgpu::BufferUsages::VERTEX,
            )
        } else {
            DynamicBuffer::new_with_data(
                &context.device,
                "Ribbon Vertex Buffer",
                &vertices,
                wgpu::BufferUsages::VERTEX,
            )
        };

        let index_buffer = if indices.is_empty() {
            DynamicBuffer::new(
                &context.device,
                "Ribbon Index Buffer",
                std::mem::size_of::<u32>() * 3000,
                wgpu::BufferUsages::INDEX,
            )
        } else {
            DynamicBuffer::new_with_data(
                &context.device,
                "Ribbon Index Buffer",
                &indices,
                wgpu::BufferUsages::INDEX,
            )
        };

        let pipeline = Self::create_pipeline(context, camera_layout, lighting_layout);
        let last_chain_hash = Self::compute_chain_hash(backbone_chains);

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            last_chain_hash,
            params,
        }
    }

    /// Compute a hash of the chain data for change detection
    fn compute_chain_hash(chains: &[Vec<Vec3>]) -> u64 {
        let mut hasher = DefaultHasher::new();

        chains.len().hash(&mut hasher);
        for chain in chains {
            chain.len().hash(&mut hasher);
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

    /// Update the ribbon with new chains
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        backbone_chains: &[Vec<Vec3>],
    ) {
        let new_hash = Self::compute_chain_hash(backbone_chains);
        if new_hash == self.last_chain_hash {
            return;
        }
        self.last_chain_hash = new_hash;

        let (vertices, indices) = Self::generate_ribbon_mesh(backbone_chains, &self.params);

        if vertices.is_empty() {
            self.index_count = 0;
            return;
        }

        self.vertex_buffer.write(device, queue, &vertices);
        self.index_buffer.write(device, queue, &indices);
        self.index_count = indices.len() as u32;
    }

    /// Set ribbon parameters
    pub fn set_params(&mut self, params: RibbonParams) {
        self.params = params;
        // Force regeneration on next update
        self.last_chain_hash = 0;
    }

    /// Create a new ribbon renderer using full backbone residue data (Foldit-style)
    /// This provides peptide plane orientation for helices and N/O atom positions for sheets
    pub fn new_from_residues(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        backbone_chains: &[BackboneChain],
    ) -> Self {
        let params = RibbonParams::default();
        let (vertices, indices) = Self::generate_ribbon_mesh_from_residues(backbone_chains, &params);

        let vertex_buffer = if vertices.is_empty() {
            DynamicBuffer::new(
                &context.device,
                "Ribbon Vertex Buffer",
                std::mem::size_of::<RibbonVertex>() * 1000,
                wgpu::BufferUsages::VERTEX,
            )
        } else {
            DynamicBuffer::new_with_data(
                &context.device,
                "Ribbon Vertex Buffer",
                &vertices,
                wgpu::BufferUsages::VERTEX,
            )
        };

        let index_buffer = if indices.is_empty() {
            DynamicBuffer::new(
                &context.device,
                "Ribbon Index Buffer",
                std::mem::size_of::<u32>() * 3000,
                wgpu::BufferUsages::INDEX,
            )
        } else {
            DynamicBuffer::new_with_data(
                &context.device,
                "Ribbon Index Buffer",
                &indices,
                wgpu::BufferUsages::INDEX,
            )
        };

        let pipeline = Self::create_pipeline(context, camera_layout, lighting_layout);
        let last_chain_hash = Self::compute_residue_chain_hash(backbone_chains);

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            last_chain_hash,
            params,
        }
    }

    /// Compute hash for BackboneChain data
    fn compute_residue_chain_hash(chains: &[BackboneChain]) -> u64 {
        let mut hasher = DefaultHasher::new();
        chains.len().hash(&mut hasher);
        for chain in chains {
            chain.residues.len().hash(&mut hasher);
            if let Some(first) = chain.residues.first() {
                first.ca_pos.x.to_bits().hash(&mut hasher);
                first.ca_pos.y.to_bits().hash(&mut hasher);
                first.ca_pos.z.to_bits().hash(&mut hasher);
            }
            if let Some(last) = chain.residues.last() {
                last.ca_pos.x.to_bits().hash(&mut hasher);
                last.ca_pos.y.to_bits().hash(&mut hasher);
                last.ca_pos.z.to_bits().hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    /// Update the ribbon with new backbone residue chains (Foldit-style)
    pub fn update_from_residues(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        backbone_chains: &[BackboneChain],
    ) {
        let new_hash = Self::compute_residue_chain_hash(backbone_chains);
        if new_hash == self.last_chain_hash {
            return;
        }
        self.last_chain_hash = new_hash;

        let (vertices, indices) = Self::generate_ribbon_mesh_from_residues(backbone_chains, &self.params);

        if vertices.is_empty() {
            self.index_count = 0;
            return;
        }

        self.vertex_buffer.write(device, queue, &vertices);
        self.index_buffer.write(device, queue, &indices);
        self.index_count = indices.len() as u32;
    }

    fn create_pipeline(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        // Reuse the backbone_tube shader since vertex format is identical
        let shader = context
            .device
            .create_shader_module(wgpu::include_wgsl!("../assets/shaders/backbone_tube.wgsl"));

        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Ribbon Pipeline Layout"),
                    bind_group_layouts: &[camera_layout, lighting_layout],
                    immediate_size: 0,
                });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<RibbonVertex>() as wgpu::BufferAddress,
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
            ],
        };

        context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Ribbon Render Pipeline"),
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
                    cull_mode: None,  // Disable culling - ribbons visible from both sides
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

    fn generate_ribbon_mesh(
        chains: &[Vec<Vec3>],
        params: &RibbonParams,
    ) -> (Vec<RibbonVertex>, Vec<u32>) {
        let mut all_vertices: Vec<RibbonVertex> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();

        for backbone_atoms in chains {
            // Extract CA positions (every 3rd starting at index 1)
            let ca_positions: Vec<Vec3> = backbone_atoms
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 3 == 1)
                .map(|(_, &pos)| pos)
                .collect();

            if ca_positions.len() < 4 {
                continue;
            }

            // Detect secondary structure from CA positions
            let ss_types = detect_secondary_structure(&ca_positions);

            // Generate spline points from raw CA positions (no smoothing/idealization)
            let spline_points = Self::generate_spline_points(&ca_positions);

            // Segment by secondary structure type
            let segments = Self::segment_by_ss(&ss_types);

            // Generate geometry for each segment
            for segment in &segments {
                let base_vertex = all_vertices.len() as u32;

                // Find spline points for this segment
                let start_spline = segment.start_residue * SEGMENTS_PER_SPAN;
                let end_spline = if segment.end_residue >= ca_positions.len() {
                    spline_points.len()
                } else {
                    segment.end_residue * SEGMENTS_PER_SPAN
                };

                if start_spline >= spline_points.len() || end_spline <= start_spline {
                    continue;
                }

                let segment_spline = &spline_points[start_spline..end_spline.min(spline_points.len())];
                let segment_ss = &ss_types[segment.start_residue..segment.end_residue.min(ss_types.len())];

                // Only render helices and sheets - coils are handled by TubeRenderer
                let (vertices, indices) = match segment.ss_type {
                    SSType::Helix => Self::generate_helix_ribbon(
                        segment_spline,
                        segment_ss,
                        params,
                        base_vertex,
                        segment.start_residue,
                    ),
                    SSType::Sheet => Self::generate_sheet_ribbon(
                        segment_spline,
                        segment_ss,
                        params,
                        base_vertex,
                    ),
                    SSType::Coil => continue, // Skip coils - rendered by TubeRenderer
                };

                all_vertices.extend(vertices);
                all_indices.extend(indices);
            }
        }

        (all_vertices, all_indices)
    }

    /// Generate ribbon mesh from full backbone residue data (Foldit-style)
    /// Uses peptide plane (C=O) orientation for helices and N/O atoms for sheets
    fn generate_ribbon_mesh_from_residues(
        chains: &[BackboneChain],
        params: &RibbonParams,
    ) -> (Vec<RibbonVertex>, Vec<u32>) {
        let mut all_vertices: Vec<RibbonVertex> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();

        for chain in chains {
            if chain.residues.len() < 4 {
                continue;
            }

            // Get CA positions for secondary structure detection
            let ca_positions = chain.ca_positions();
            let ss_types = detect_secondary_structure(&ca_positions);

            // Segment by secondary structure type
            let segments = Self::segment_by_ss(&ss_types);

            // Generate geometry for each segment
            for segment in &segments {
                let base_vertex = all_vertices.len() as u32;

                let segment_residues = &chain.residues[segment.start_residue..segment.end_residue.min(chain.residues.len())];
                let segment_ss = &ss_types[segment.start_residue..segment.end_residue.min(ss_types.len())];

                if segment_residues.is_empty() {
                    continue;
                }

                // Only render helices and sheets - coils are handled by TubeRenderer
                let (vertices, indices) = match segment.ss_type {
                    SSType::Helix => Self::generate_helix_ribbon_peptide_plane(
                        segment_residues,
                        segment_ss,
                        params,
                        base_vertex,
                    ),
                    SSType::Sheet => Self::generate_sheet_quads_from_atoms(
                        segment_residues,
                        segment_ss,
                        params,
                        base_vertex,
                    ),
                    SSType::Coil => continue, // Skip coils - rendered by TubeRenderer
                };

                all_vertices.extend(vertices);
                all_indices.extend(indices);
            }
        }

        (all_vertices, all_indices)
    }

    /// Generate helix ribbon using peptide plane orientation (Foldit-style)
    /// The C=O vector defines the ribbon's normal, creating natural helical twist
    fn generate_helix_ribbon_peptide_plane(
        residues: &[BackboneResidue],
        ss_types: &[SSType],
        params: &RibbonParams,
        base_vertex: u32,
    ) -> (Vec<RibbonVertex>, Vec<u32>) {
        if residues.len() < 2 {
            return (Vec::new(), Vec::new());
        }

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let half_width = params.helix_width * 0.5;
        let half_thickness = params.helix_thickness * 0.5;

        // Generate cross-sections at each residue and interpolate between them
        for (i, residue) in residues.iter().enumerate() {
            let color = if i < ss_types.len() {
                ss_types[i].color()
            } else {
                SSType::Helix.color()
            };

            // Compute tangent along backbone
            let tangent = if i == 0 {
                (residues[1].ca_pos - residue.ca_pos).normalize()
            } else if i == residues.len() - 1 {
                (residue.ca_pos - residues[i - 1].ca_pos).normalize()
            } else {
                (residues[i + 1].ca_pos - residues[i - 1].ca_pos).normalize()
            };

            // Compute normal from C=O vector (peptide plane orientation)
            // The C=O vector points roughly perpendicular to the helix backbone
            let co_vec = (residue.o_pos - residue.c_pos).normalize();

            // Project C=O onto plane perpendicular to tangent to get ribbon normal
            let normal = (co_vec - tangent * tangent.dot(co_vec)).normalize();
            let binormal = tangent.cross(normal).normalize();

            // Center the ribbon on the CA position
            let center = residue.ca_pos;

            // Generate 8-vertex cross-section (rounded rectangle for smooth transitions)
            // 4 corners + 4 edge midpoints
            Self::add_rounded_rectangle_cross_section(
                &mut vertices,
                center,
                normal,
                binormal,
                half_width,
                half_thickness,
                color,
            );
        }

        // Generate indices connecting adjacent cross-sections
        Self::generate_tube_indices(&mut indices, residues.len(), base_vertex);

        (vertices, indices)
    }

    /// Generate sheet quads directly from N and O atom positions (Foldit-style)
    /// Creates natural pleating from hydrogen bonding geometry, no arrows
    fn generate_sheet_quads_from_atoms(
        residues: &[BackboneResidue],
        ss_types: &[SSType],
        params: &RibbonParams,
        base_vertex: u32,
    ) -> (Vec<RibbonVertex>, Vec<u32>) {
        if residues.len() < 2 {
            return (Vec::new(), Vec::new());
        }

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let half_thickness = params.sheet_thickness * 0.5;

        // For sheets, we generate quads using N and O positions
        // The natural N-O-N-O pattern follows hydrogen bonding geometry
        for (i, residue) in residues.iter().enumerate() {
            let color = if i < ss_types.len() {
                ss_types[i].color()
            } else {
                SSType::Sheet.color()
            };

            // Compute tangent along backbone
            let tangent = if i == 0 {
                (residues[1].ca_pos - residue.ca_pos).normalize()
            } else if i == residues.len() - 1 {
                (residue.ca_pos - residues[i - 1].ca_pos).normalize()
            } else {
                (residues[i + 1].ca_pos - residues[i - 1].ca_pos).normalize()
            };

            // Compute width direction from N to O (across the strand)
            // This creates the characteristic sheet width
            let n_to_o = residue.o_pos - residue.n_pos;
            let width_dir = (n_to_o - tangent * tangent.dot(n_to_o)).normalize();

            // Normal is perpendicular to both tangent and width
            let normal = tangent.cross(width_dir).normalize();

            // Sheet width comes from N-O distance (typically ~2.8Å in beta sheets)
            let sheet_half_width = n_to_o.length() * 0.5;

            // Center on CA position
            let center = residue.ca_pos;

            // Generate 8-vertex cross-section (flat rectangle with thickness)
            Self::add_flat_rectangle_cross_section(
                &mut vertices,
                center,
                normal,
                width_dir,
                sheet_half_width,
                half_thickness,
                color,
            );
        }

        // Generate indices connecting adjacent cross-sections
        Self::generate_tube_indices(&mut indices, residues.len(), base_vertex);

        (vertices, indices)
    }

    /// Add an 8-vertex rounded rectangle cross-section (for helices)
    fn add_rounded_rectangle_cross_section(
        vertices: &mut Vec<RibbonVertex>,
        center: Vec3,
        normal: Vec3,
        binormal: Vec3,
        half_width: f32,
        half_thickness: f32,
        color: [f32; 3],
    ) {
        // 8 vertices: 4 corners + 4 edge midpoints
        // Arranged as: bottom-left, bottom, bottom-right, right, top-right, top, top-left, left
        let positions = [
            (-half_width, -half_thickness),   // 0: bottom-left corner
            (0.0, -half_thickness),           // 1: bottom edge midpoint
            (half_width, -half_thickness),    // 2: bottom-right corner
            (half_width, 0.0),                // 3: right edge midpoint
            (half_width, half_thickness),     // 4: top-right corner
            (0.0, half_thickness),            // 5: top edge midpoint
            (-half_width, half_thickness),    // 6: top-left corner
            (-half_width, 0.0),               // 7: left edge midpoint
        ];

        for &(w, h) in &positions {
            let offset = binormal * w + normal * h;
            let pos = center + offset;
            // Normal points outward from center
            let vert_normal = offset.normalize();

            vertices.push(RibbonVertex {
                position: pos.into(),
                normal: vert_normal.into(),
                color,
            });
        }
    }

    /// Add an 8-vertex flat rectangle cross-section (for sheets)
    fn add_flat_rectangle_cross_section(
        vertices: &mut Vec<RibbonVertex>,
        center: Vec3,
        normal: Vec3,
        width_dir: Vec3,
        half_width: f32,
        half_thickness: f32,
        color: [f32; 3],
    ) {
        // 8 vertices arranged for sheet geometry
        // Top face: 4 vertices, Bottom face: 4 vertices interleaved
        let positions = [
            (-half_width, -half_thickness),   // 0: bottom-left
            (0.0, -half_thickness),           // 1: bottom-center
            (half_width, -half_thickness),    // 2: bottom-right
            (half_width, 0.0),                // 3: right-center
            (half_width, half_thickness),     // 4: top-right
            (0.0, half_thickness),            // 5: top-center
            (-half_width, half_thickness),    // 6: top-left
            (-half_width, 0.0),               // 7: left-center
        ];

        for &(w, h) in &positions {
            let offset = width_dir * w + normal * h;
            let pos = center + offset;
            // For sheets, normal points primarily up/down (perpendicular to sheet plane)
            let vert_normal = if h.abs() > 0.001 {
                normal * h.signum()
            } else {
                (width_dir * w.signum()).normalize()
            };

            vertices.push(RibbonVertex {
                position: pos.into(),
                normal: vert_normal.into(),
                color,
            });
        }
    }

    /// Generate indices for tube geometry connecting adjacent cross-sections
    fn generate_tube_indices(indices: &mut Vec<u32>, num_rings: usize, base_vertex: u32) {
        for i in 0..num_rings.saturating_sub(1) {
            let ring_start = i * UNIFIED_CROSS_SECTION_VERTS;
            let next_ring_start = (i + 1) * UNIFIED_CROSS_SECTION_VERTS;

            for k in 0..UNIFIED_CROSS_SECTION_VERTS {
                let k_next = (k + 1) % UNIFIED_CROSS_SECTION_VERTS;

                let v0 = base_vertex + (ring_start + k) as u32;
                let v1 = base_vertex + (ring_start + k_next) as u32;
                let v2 = base_vertex + (next_ring_start + k) as u32;
                let v3 = base_vertex + (next_ring_start + k_next) as u32;

                // Two triangles per quad
                indices.extend_from_slice(&[v0, v2, v1]);
                indices.extend_from_slice(&[v1, v2, v3]);
            }
        }
    }

    /// Segment the backbone into runs of same SS type
    fn segment_by_ss(ss_types: &[SSType]) -> Vec<SSSegment> {
        if ss_types.is_empty() {
            return Vec::new();
        }

        let mut segments = Vec::new();
        let mut current_type = ss_types[0];
        let mut start = 0;

        for (i, &ss) in ss_types.iter().enumerate() {
            if ss != current_type {
                segments.push(SSSegment {
                    ss_type: current_type,
                    start_residue: start,
                    end_residue: i,
                });
                current_type = ss;
                start = i;
            }
        }

        // Don't forget the last segment
        segments.push(SSSegment {
            ss_type: current_type,
            start_residue: start,
            end_residue: ss_types.len(),
        });

        segments
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
                    normal: Vec3::ZERO,
                    binormal: Vec3::ZERO,
                    residue_idx: i,
                    residue_t: t,
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
            residue_idx: n - 1,
            residue_t: 0.0,
        });

        // Compute rotation minimizing frames
        compute_rmf(&mut points);

        points
    }

    /// Generate helix ribbon geometry (legacy version for Vec<Vec3> chains)
    /// Uses 8-vertex cross-sections for compatibility with unified rendering
    fn generate_helix_ribbon(
        points: &[SplinePoint],
        ss_types: &[SSType],
        params: &RibbonParams,
        base_vertex: u32,
        start_residue: usize,
    ) -> (Vec<RibbonVertex>, Vec<u32>) {
        let num_rings = points.len();
        let mut vertices = Vec::with_capacity(num_rings * UNIFIED_CROSS_SECTION_VERTS);
        let mut indices = Vec::new();

        let half_width = params.helix_width * 0.5;
        let half_thickness = params.helix_thickness * 0.5;

        for point in points.iter() {
            let residue_idx = point.residue_idx.saturating_sub(start_residue);
            let color = if residue_idx < ss_types.len() {
                ss_types[residue_idx].color()
            } else {
                SSType::Helix.color()
            };

            // Calculate helix phase for spiraling effect
            let global_residue = point.residue_idx as f32 + point.residue_t;
            let phase = global_residue * (std::f32::consts::TAU / 3.6);

            // Rotate the RMF normal/binormal around the tangent by the phase
            let cos_phase = phase.cos();
            let sin_phase = phase.sin();
            let rotated_normal = point.normal * cos_phase + point.binormal * sin_phase;
            let rotated_binormal = -point.normal * sin_phase + point.binormal * cos_phase;

            // Center on spline (no coil offset for flat ribbon style)
            let center = point.pos;

            // Generate 8-vertex rounded rectangle cross-section
            Self::add_rounded_rectangle_cross_section(
                &mut vertices,
                center,
                rotated_normal,
                rotated_binormal,
                half_width,
                half_thickness,
                color,
            );
        }

        // Generate indices
        Self::generate_tube_indices(&mut indices, num_rings, base_vertex);

        (vertices, indices)
    }

    /// Generate sheet ribbon geometry (legacy version for Vec<Vec3> chains)
    /// Constant width, no arrows (Foldit-style)
    fn generate_sheet_ribbon(
        points: &[SplinePoint],
        ss_types: &[SSType],
        params: &RibbonParams,
        base_vertex: u32,
    ) -> (Vec<RibbonVertex>, Vec<u32>) {
        let num_rings = points.len();
        let mut vertices = Vec::with_capacity(num_rings * UNIFIED_CROSS_SECTION_VERTS);
        let mut indices = Vec::new();

        let half_thickness = params.sheet_thickness * 0.5;
        // Constant width for sheets (no arrows)
        let half_width = 1.0; // Default sheet width

        for point in points.iter() {
            let residue_idx = point.residue_idx;
            let color = if residue_idx < ss_types.len() {
                ss_types[residue_idx].color()
            } else {
                SSType::Sheet.color()
            };

            // Sheet ribbons are flat - use RMF normal as "up", binormal as "width"
            let up = point.normal;
            let right = point.binormal;

            // Generate 8-vertex flat rectangle cross-section
            Self::add_flat_rectangle_cross_section(
                &mut vertices,
                point.pos,
                up,
                right,
                half_width,
                half_thickness,
                color,
            );
        }

        // Generate indices
        Self::generate_tube_indices(&mut indices, num_rings, base_vertex);

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
        render_pass.set_vertex_buffer(0, self.vertex_buffer.buffer().slice(..));
        render_pass.set_index_buffer(
            self.index_buffer.buffer().slice(..),
            wgpu::IndexFormat::Uint32,
        );
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
        let _s_i = points[i].binormal;

        let v1 = x_i1 - x_i;
        let c1 = v1.dot(v1);

        if c1 < 1e-10 {
            points[i + 1].normal = r_i;
            points[i + 1].binormal = _s_i;
            continue;
        }

        // First reflection
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
