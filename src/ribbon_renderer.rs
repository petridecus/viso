//! Ribbon renderer for secondary structure visualization
//!
//! Key design decisions matching Molstar's approach:
//! - Helices: Ribbon normal points RADIALLY OUTWARD from helix axis
//! - Sheets: Constant width, smooth RMF-propagated normals (no pleating)
//! - High subdivision (16 segments per residue) for smooth curves
//! - B-spline interpolation for C2 continuity

use crate::dynamic_buffer::DynamicBuffer;
use foldit_conv::coords::RenderBackboneResidue;
use crate::render_context::RenderContext;
use foldit_conv::secondary_structure::auto::detect as detect_secondary_structure;
use foldit_conv::secondary_structure::{SSType, merge_short_segments};
use glam::Vec3;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Tube diameter for taper at ribbon-tube junctions (must match tube_renderer TUBE_RADIUS * 2)
const TUBE_DIAMETER: f32 = 0.6;

/// Parameters for ribbon rendering
#[derive(Clone, Copy)]
pub struct RibbonParams {
    pub helix_width: f32,
    pub helix_thickness: f32,
    pub sheet_width: f32,
    pub sheet_thickness: f32,
    pub segments_per_residue: usize,
}

impl Default for RibbonParams {
    fn default() -> Self {
        Self {
            helix_width: 1.4,
            helix_thickness: 0.25,
            sheet_width: 1.6,
            sheet_thickness: 0.25,
            segments_per_residue: 16, // High subdivision for smooth curves
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RibbonVertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
    residue_idx: u32,
}

struct RibbonFrame {
    position: Vec3,
    tangent: Vec3,
    normal: Vec3,    // Points "up" from ribbon surface
    binormal: Vec3,  // Points along ribbon width
    color: [f32; 3],
    residue_idx: u32,
}

#[derive(Debug)]
struct SSSegment {
    ss_type: SSType,
    start_residue: usize,
    end_residue: usize,
}

pub struct RibbonRenderer {
    pub pipeline: wgpu::RenderPipeline,
    vertex_buffer: DynamicBuffer,
    index_buffer: DynamicBuffer,
    pub index_count: u32,
    last_chain_hash: u64,
    params: RibbonParams,
    /// Cached chain data for regeneration
    cached_chains: Vec<Vec<Vec3>>,
    /// Pre-computed SS types override (from puzzle.toml annotation)
    ss_override: Option<Vec<SSType>>,
    /// Per-residue offsets for sheet residues: (global_residue_idx, offset).
    /// offset = flattened_position - original_CA_position.
    /// Used by the engine to translate sidechains onto the sheet surface.
    sheet_offsets: Vec<(u32, Vec3)>,
}

impl RibbonRenderer {
    pub fn new(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        selection_layout: &wgpu::BindGroupLayout,
        backbone_chains: &[Vec<Vec3>],
    ) -> Self {
        let params = RibbonParams::default();
        let (vertices, indices, offsets) = Self::generate_from_ca_only(backbone_chains, &params, None);

        let vertex_buffer = DynamicBuffer::new_with_data(
            &context.device,
            "Ribbon Vertex Buffer",
            if vertices.is_empty() { &[RibbonVertex { position: [0.0; 3], normal: [0.0; 3], color: [0.0; 3], residue_idx: 0 }] } else { &vertices },
            wgpu::BufferUsages::VERTEX,
        );

        let index_buffer = DynamicBuffer::new_with_data(
            &context.device,
            "Ribbon Index Buffer",
            if indices.is_empty() { &[0u32] } else { &indices },
            wgpu::BufferUsages::INDEX,
        );

        let pipeline = Self::create_pipeline(context, camera_layout, lighting_layout, selection_layout);
        let last_chain_hash = Self::compute_chain_hash(backbone_chains);

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            last_chain_hash,
            params,
            cached_chains: backbone_chains.to_vec(),
            ss_override: None,
            sheet_offsets: offsets,
        }
    }

    pub fn new_from_residues(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        selection_layout: &wgpu::BindGroupLayout,
        backbone_chains: &[Vec<RenderBackboneResidue>],
    ) -> Self {
        let params = RibbonParams::default();
        let (vertices, indices, offsets) = Self::generate_from_residues(backbone_chains, &params, None);

        let vertex_buffer = DynamicBuffer::new_with_data(
            &context.device,
            "Ribbon Vertex Buffer",
            if vertices.is_empty() { &[RibbonVertex { position: [0.0; 3], normal: [0.0; 3], color: [0.0; 3], residue_idx: 0 }] } else { &vertices },
            wgpu::BufferUsages::VERTEX,
        );

        let index_buffer = DynamicBuffer::new_with_data(
            &context.device,
            "Ribbon Index Buffer",
            if indices.is_empty() { &[0u32] } else { &indices },
            wgpu::BufferUsages::INDEX,
        );

        let pipeline = Self::create_pipeline(context, camera_layout, lighting_layout, selection_layout);
        let last_chain_hash = Self::compute_residue_hash(backbone_chains);

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            last_chain_hash,
            params,
            cached_chains: Vec::new(),
            ss_override: None,
            sheet_offsets: offsets,
        }
    }

    fn compute_chain_hash(chains: &[Vec<Vec3>]) -> u64 {
        let mut hasher = DefaultHasher::new();
        chains.len().hash(&mut hasher);
        for chain in chains {
            chain.len().hash(&mut hasher);
            if let Some(f) = chain.first() { f.x.to_bits().hash(&mut hasher); }
        }
        hasher.finish()
    }

    fn compute_residue_hash(chains: &[Vec<RenderBackboneResidue>]) -> u64 {
        let mut hasher = DefaultHasher::new();
        chains.len().hash(&mut hasher);
        for chain in chains {
            chain.len().hash(&mut hasher);
            if let Some(f) = chain.first() { f.ca_pos.x.to_bits().hash(&mut hasher); }
        }
        hasher.finish()
    }

    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, backbone_chains: &[Vec<Vec3>], ss_types: Option<&[SSType]>) {
        let new_hash = Self::compute_chain_hash(backbone_chains);
        if new_hash == self.last_chain_hash && ss_types.is_none() { return; }
        self.last_chain_hash = new_hash;
        self.cached_chains = backbone_chains.to_vec();
        if let Some(ss) = ss_types {
            self.ss_override = Some(ss.to_vec());
        }

        let (vertices, indices, offsets) = Self::generate_from_ca_only(backbone_chains, &self.params, self.ss_override.as_deref());
        if !vertices.is_empty() {
            self.vertex_buffer.write(device, queue, &vertices);
            self.index_buffer.write(device, queue, &indices);
        }
        self.index_count = indices.len() as u32;
        self.sheet_offsets = offsets;
    }

    /// Set pre-computed SS types (from puzzle.toml annotation or DSSP).
    pub fn set_ss_override(&mut self, ss_types: Option<Vec<SSType>>) {
        self.ss_override = ss_types;
    }

    /// Regenerate mesh with current settings.
    pub fn regenerate(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let (vertices, indices, offsets) = Self::generate_from_ca_only(&self.cached_chains, &self.params, self.ss_override.as_deref());
        if !vertices.is_empty() {
            self.vertex_buffer.write(device, queue, &vertices);
            self.index_buffer.write(device, queue, &indices);
        }
        self.index_count = indices.len() as u32;
        self.sheet_offsets = offsets;
    }

    pub fn update_from_residues(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, chains: &[Vec<RenderBackboneResidue>]) {
        let new_hash = Self::compute_residue_hash(chains);
        if new_hash == self.last_chain_hash { return; }
        self.last_chain_hash = new_hash;

        let (vertices, indices, offsets) = Self::generate_from_residues(chains, &self.params, self.ss_override.as_deref());
        if !vertices.is_empty() {
            self.vertex_buffer.write(device, queue, &vertices);
            self.index_buffer.write(device, queue, &indices);
        }
        self.index_count = indices.len() as u32;
        self.sheet_offsets = offsets;
    }

    /// Get per-residue offsets for sheet residues.
    /// Returns (global_residue_idx, offset) pairs where offset = flattened_pos - raw_CA_pos.
    /// Used by the engine to translate entire sidechains onto the sheet surface.
    pub fn sheet_offsets(&self) -> &[(u32, Vec3)] {
        &self.sheet_offsets
    }

    pub fn set_params(&mut self, params: RibbonParams) {
        self.params = params;
        self.last_chain_hash = 0;
    }

    fn create_pipeline(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        selection_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = context.device.create_shader_module(wgpu::include_wgsl!("../assets/shaders/backbone_tube.wgsl"));

        let pipeline_layout = context.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Ribbon Pipeline Layout"),
            bind_group_layouts: &[camera_layout, lighting_layout, selection_layout],
            immediate_size: 0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<RibbonVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 12, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 24, shader_location: 2 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32, offset: 36, shader_location: 3 },
            ],
        };

        context.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Ribbon Pipeline"),
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
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: context.config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
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

    // ==================== MAIN GENERATION ====================

    fn generate_from_residues(chains: &[Vec<RenderBackboneResidue>], params: &RibbonParams, ss_override: Option<&[SSType]>) -> (Vec<RibbonVertex>, Vec<u32>, Vec<(u32, Vec3)>) {
        let mut all_verts = Vec::new();
        let mut all_inds = Vec::new();
        let mut all_surface_pos = Vec::new();
        let mut global_residue_idx: u32 = 0;

        for chain in chains {
            if chain.len() < 2 {
                global_residue_idx += chain.len() as u32;
                continue;
            }

            let ca_positions: Vec<Vec3> = chain.iter().map(|r| r.ca_pos).collect();
            let n_residues = ca_positions.len();
            let ss_types = if let Some(overrides) = ss_override {
                let start = global_residue_idx as usize;
                let end = (start + n_residues).min(overrides.len());
                if start < overrides.len() {
                    overrides[start..end].to_vec()
                } else {
                    detect_secondary_structure(&ca_positions)
                }
            } else {
                detect_secondary_structure(&ca_positions)
            };
            let ss_types = merge_short_segments(&ss_types);
            let segments = segment_by_ss(&ss_types);

            for seg in &segments {
                if seg.ss_type == SSType::Coil { continue; }

                let start = seg.start_residue;
                let end = seg.end_residue.min(chain.len());
                if end <= start + 1 { continue; } // Need at least 2 real residues

                // Extend segment into coil regions at boundaries for smooth taper
                let taper_start = if start > 0 && ss_types[start - 1] == SSType::Coil { 1 } else { 0 };
                let taper_end = if end < chain.len() && end < ss_types.len() && ss_types[end] == SSType::Coil { 1 } else { 0 };
                let ext_start = start - taper_start;
                let ext_end = (end + taper_end).min(chain.len());

                let residues = &chain[ext_start..ext_end];
                // Override taper residues to segment SS type so they get the right color
                let mut ss: Vec<SSType> = ss_types[ext_start..ext_end.min(ss_types.len())].to_vec();
                if taper_start > 0 { ss[0] = seg.ss_type; }
                if taper_end > 0 { if let Some(last) = ss.last_mut() { *last = seg.ss_type; } }
                let base = all_verts.len() as u32;
                let segment_residue_base = global_residue_idx + ext_start as u32;

                let (v, i, sp) = match seg.ss_type {
                    SSType::Helix => generate_helix(residues, &ss, params, base, segment_residue_base, taper_start, taper_end),
                    SSType::Sheet => {
                        let n_pos: Vec<Vec3> = residues.iter().map(|r| r.n_pos).collect();
                        let ca_pos: Vec<Vec3> = residues.iter().map(|r| r.ca_pos).collect();
                        let c_pos: Vec<Vec3> = residues.iter().map(|r| r.c_pos).collect();
                        peptide_plane_sheet(&n_pos, &ca_pos, &c_pos, &ss, params, base, segment_residue_base, taper_start, taper_end)
                    }
                    SSType::Coil => continue,
                };

                all_verts.extend(v);
                all_inds.extend(i);
                all_surface_pos.extend(sp);
            }

            global_residue_idx += chain.len() as u32;
        }

        (all_verts, all_inds, all_surface_pos)
    }

    fn generate_from_ca_only(chains: &[Vec<Vec3>], params: &RibbonParams, ss_override: Option<&[SSType]>) -> (Vec<RibbonVertex>, Vec<u32>, Vec<(u32, Vec3)>) {
        let mut all_verts = Vec::new();
        let mut all_inds = Vec::new();
        let mut all_surface_pos = Vec::new();
        let mut global_residue_idx: u32 = 0;

        for chain in chains {
            // Extract CA positions (every 3rd starting at 1 for N-CA-C pattern)
            let ca_positions: Vec<Vec3> = chain.iter().enumerate()
                .filter(|(i, _)| i % 3 == 1)
                .map(|(_, &p)| p)
                .collect();

            if ca_positions.len() < 2 {
                global_residue_idx += ca_positions.len() as u32;
                continue;
            }

            // Use SS override if available, otherwise auto-detect
            let n_residues = ca_positions.len();
            let ss_types = if let Some(overrides) = ss_override {
                let start = global_residue_idx as usize;
                let end = (start + n_residues).min(overrides.len());
                if start < overrides.len() {
                    overrides[start..end].to_vec()
                } else {
                    detect_secondary_structure(&ca_positions)
                }
            } else {
                detect_secondary_structure(&ca_positions)
            };
            let ss_types = merge_short_segments(&ss_types);
            let segments = segment_by_ss(&ss_types);

            for seg in &segments {
                if seg.ss_type == SSType::Coil { continue; }

                let start = seg.start_residue;
                let end = seg.end_residue.min(ca_positions.len());
                if end <= start + 1 { continue; } // Need at least 2 real residues

                // Extend segment into coil regions at boundaries for smooth taper
                let taper_start = if start > 0 && ss_types[start - 1] == SSType::Coil { 1 } else { 0 };
                let taper_end = if end < ss_types.len() && ss_types[end] == SSType::Coil { 1 } else { 0 };
                let ext_start = start - taper_start;
                let ext_end = (end + taper_end).min(ca_positions.len());

                let positions = &ca_positions[ext_start..ext_end];
                // Override taper residues to segment SS type so they get the right color
                let mut ss: Vec<SSType> = ss_types[ext_start..ext_end.min(ss_types.len())].to_vec();
                if taper_start > 0 { ss[0] = seg.ss_type; }
                if taper_end > 0 { if let Some(last) = ss.last_mut() { *last = seg.ss_type; } }
                let base = all_verts.len() as u32;
                let segment_residue_base = global_residue_idx + ext_start as u32;

                let (v, i, sp) = match seg.ss_type {
                    SSType::Helix => generate_helix_from_ca(positions, &ss, params, base, segment_residue_base, taper_start, taper_end),
                    SSType::Sheet => {
                        // Extract N-CA-C triplets for peptide plane normals
                        let backbone_start = ext_start * 3;
                        let backbone_end = (ext_end * 3).min(chain.len());
                        let backbone_slice = &chain[backbone_start..backbone_end];
                        let n_res = backbone_slice.len() / 3;
                        if n_res < 2 { continue; }
                        let n_pos: Vec<Vec3> = (0..n_res).map(|i| backbone_slice[i * 3]).collect();
                        let ca_pos: Vec<Vec3> = (0..n_res).map(|i| backbone_slice[i * 3 + 1]).collect();
                        let c_pos: Vec<Vec3> = (0..n_res).map(|i| backbone_slice[i * 3 + 2]).collect();
                        peptide_plane_sheet(&n_pos, &ca_pos, &c_pos, &ss, params, base, segment_residue_base, taper_start, taper_end)
                    }
                    SSType::Coil => continue,
                };

                all_verts.extend(v);
                all_inds.extend(i);
                all_surface_pos.extend(sp);
            }

            global_residue_idx += ca_positions.len() as u32;
        }

        (all_verts, all_inds, all_surface_pos)
    }

    pub fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        lighting_bind_group: &'a wgpu::BindGroup,
        selection_bind_group: &'a wgpu::BindGroup,
    ) {
        if self.index_count == 0 { return; }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, camera_bind_group, &[]);
        render_pass.set_bind_group(1, lighting_bind_group, &[]);
        render_pass.set_bind_group(2, selection_bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.buffer().slice(..));
        render_pass.set_index_buffer(self.index_buffer.buffer().slice(..), wgpu::IndexFormat::Uint32);
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
}

// ==================== HELIX GENERATION ====================
// Key: Normal points RADIALLY OUTWARD from helix axis

fn generate_helix(residues: &[RenderBackboneResidue], ss_types: &[SSType], params: &RibbonParams, base: u32, global_residue_base: u32, taper_start: usize, taper_end: usize) -> (Vec<RibbonVertex>, Vec<u32>, Vec<(u32, Vec3)>) {
    let ca_positions: Vec<Vec3> = residues.iter().map(|r| r.ca_pos).collect();
    generate_helix_from_ca(&ca_positions, ss_types, params, base, global_residue_base, taper_start, taper_end)
}

fn generate_helix_from_ca(ca_positions: &[Vec3], ss_types: &[SSType], params: &RibbonParams, base: u32, global_residue_base: u32, taper_start: usize, taper_end: usize) -> (Vec<RibbonVertex>, Vec<u32>, Vec<(u32, Vec3)>) {
    let n = ca_positions.len();
    if n < 2 { return (Vec::new(), Vec::new(), Vec::new()); }

    // Step 1: Compute the helix axis using a sliding window average
    let helix_centers = compute_helix_axis_points(ca_positions);
    
    // Step 2: Catmull-Rom for positions (passes through each CA exactly)
    //         B-spline for axis centers (smooth approximation)
    let spline_points = catmull_rom(ca_positions, params.segments_per_residue);
    let spline_centers = cubic_bspline(&helix_centers, params.segments_per_residue);
    
    // Step 3: For each spline point, compute frame with radial outward normal
    let mut frames = Vec::with_capacity(spline_points.len());
    
    for (i, &pos) in spline_points.iter().enumerate() {
        let center = spline_centers[i.min(spline_centers.len() - 1)];
        
        // Tangent along the spline
        let tangent = if i == 0 {
            (spline_points[1] - pos).normalize()
        } else if i == spline_points.len() - 1 {
            (pos - spline_points[i - 1]).normalize()
        } else {
            (spline_points[i + 1] - spline_points[i - 1]).normalize()
        };
        
        // Normal points radially outward from helix axis
        let to_surface = pos - center;
        let radial = (to_surface - tangent * tangent.dot(to_surface)).normalize();
        
        // Handle degenerate case
        let normal = if radial.length_squared() > 0.01 {
            radial
        } else {
            let arbitrary = if tangent.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
            tangent.cross(arbitrary).normalize()
        };
        
        let binormal = tangent.cross(normal).normalize();
        
        // Color and residue index from SS type
        let local_residue_idx = (i * (n - 1)) / spline_points.len().max(1);
        let color = ss_types.get(local_residue_idx).map(|s| s.color()).unwrap_or(SSType::Helix.color());
        let residue_idx = global_residue_base + local_residue_idx as u32;
        
        frames.push(RibbonFrame { position: pos, tangent, normal, binormal, color, residue_idx });
    }
    
    let widths = compute_taper_widths(frames.len(), params.helix_width, params.segments_per_residue, taper_start, taper_end);
    let (v, i) = build_ribbon_mesh(&frames, &widths, params.helix_thickness, base, true);
    (v, i, Vec::new())
}

/// Compute approximate helix axis points (center of helix at each residue)
fn compute_helix_axis_points(ca_positions: &[Vec3]) -> Vec<Vec3> {
    let n = ca_positions.len();
    let window = 4; // Average over 4 residues (roughly one helix turn)
    
    let mut centers = Vec::with_capacity(n);
    
    for i in 0..n {
        let start = i.saturating_sub(window / 2);
        let end = (i + window / 2 + 1).min(n);
        
        let mut sum = Vec3::ZERO;
        for j in start..end {
            sum += ca_positions[j];
        }
        centers.push(sum / (end - start) as f32);
    }
    
    centers
}

// ==================== SHEET GENERATION ====================

/// Peptide-plane sheet: derives ribbon normals from backbone chemistry
/// (cross(CA→N, CA→C)), then applies PyMOL-style iterative flattening.
/// Produces the flat, planar appearance expected for beta sheets.
/// Returns offsets (flattened_pos - original_CA) for sidechain adjustment.
fn peptide_plane_sheet(
    n_pos: &[Vec3], ca_pos: &[Vec3], c_pos: &[Vec3],
    ss_types: &[SSType], params: &RibbonParams, base: u32, global_residue_base: u32,
    taper_start: usize, taper_end: usize,
) -> (Vec<RibbonVertex>, Vec<u32>, Vec<(u32, Vec3)>) {
    let n = ca_pos.len();
    if n < 2 { return (Vec::new(), Vec::new(), Vec::new()); }

    // Peptide plane normal: cross(CA→N, CA→C)
    let mut normals: Vec<Vec3> = (0..n).map(|i| {
        let ca_n = (n_pos[i] - ca_pos[i]).normalize_or_zero();
        let ca_c = (c_pos[i] - ca_pos[i]).normalize_or_zero();
        let normal = ca_n.cross(ca_c);
        if normal.length_squared() > 1e-6 { normal.normalize() } else { Vec3::Y }
    }).collect();

    // Ensure consistent orientation (flip normals opposing the previous)
    for i in 1..n {
        if normals[i].dot(normals[i - 1]) < 0.0 {
            normals[i] = -normals[i];
        }
    }

    // Iterative flattening (4 cycles, PyMOL-style)
    let mut positions = ca_pos.to_vec();
    flatten_sheet(&mut positions, &mut normals, 4);

    // Compute offsets: how much each CA moved during flattening
    // Skip taper residues (coil extensions) — their sidechains shouldn't be adjusted
    let offsets: Vec<(u32, Vec3)> = positions.iter().enumerate()
        .filter(|(i, _)| *i >= taper_start && *i < n.saturating_sub(taper_end))
        .map(|(i, &pos)| (global_residue_base + i as u32, pos - ca_pos[i]))
        .collect();

    let (v, i) = sheet_from_normals(&positions, &normals, ss_types, params, base, global_residue_base, taper_start, taper_end);
    (v, i, offsets)
}

// ==================== SHARED SHEET HELPERS ====================

/// Build sheet mesh from positions with pre-computed per-residue normals.
/// Shared by both peptide_plane_sheet and any backend that provides
/// flattened positions + normals.
fn sheet_from_normals(
    ca_positions: &[Vec3], normals: &[Vec3],
    ss_types: &[SSType], params: &RibbonParams, base: u32, global_residue_base: u32,
    taper_start: usize, taper_end: usize,
) -> (Vec<RibbonVertex>, Vec<u32>) {
    let n = ca_positions.len();
    if n < 2 { return (Vec::new(), Vec::new()); }

    let spline_points = catmull_rom(ca_positions, params.segments_per_residue);

    let tangents: Vec<Vec3> = spline_points.iter().enumerate().map(|(i, _)| {
        if i == 0 {
            (spline_points[1] - spline_points[0]).normalize()
        } else if i == spline_points.len() - 1 {
            (spline_points[i] - spline_points[i - 1]).normalize()
        } else {
            (spline_points[i + 1] - spline_points[i - 1]).normalize()
        }
    }).collect();

    // Interpolate per-residue normals along spline, re-orthogonalize vs tangent
    let spline_normals: Vec<Vec3> = (0..spline_points.len()).map(|i| {
        let frac = i as f32 / (spline_points.len() - 1).max(1) as f32;
        let rf = frac * (n - 1) as f32;
        let r0 = (rf.floor() as usize).min(n - 1);
        let r1 = (r0 + 1).min(n - 1);
        let t = rf - r0 as f32;

        let interp = normals[r0].lerp(normals[r1], t).normalize_or_zero();
        let tangent = tangents[i];
        let proj = interp - tangent * interp.dot(tangent);
        if proj.length_squared() > 1e-6 { proj.normalize() } else { interp }
    }).collect();

    let mut frames = Vec::with_capacity(spline_points.len());
    for (i, &pos) in spline_points.iter().enumerate() {
        let tangent = tangents[i];
        let normal = spline_normals[i];
        let binormal = tangent.cross(normal).normalize();

        let local_residue_idx = (i * (n - 1)) / spline_points.len().max(1);
        let color = ss_types.get(local_residue_idx).map(|s| s.color()).unwrap_or(SSType::Sheet.color());
        let residue_idx = global_residue_base + local_residue_idx as u32;

        frames.push(RibbonFrame { position: pos, tangent, normal, binormal, color, residue_idx });
    }

    let widths = compute_taper_widths(frames.len(), params.sheet_width, params.segments_per_residue, taper_start, taper_end);
    build_ribbon_mesh(&frames, &widths, params.sheet_thickness, base, false)
}

/// Iterative flattening of sheet positions and normals (PyMOL-style).
/// Each cycle averages each point/normal with its neighbors using a
/// weighted kernel (1, 2, 1)/4, then re-orthogonalizes the normal
/// against the backbone tangent.
fn flatten_sheet(positions: &mut [Vec3], normals: &mut [Vec3], cycles: usize) {
    let n = positions.len();
    if n < 3 { return; }

    for _ in 0..cycles {
        // Average positions with neighbors (skip endpoints)
        let mut new_pos = positions.to_vec();
        for i in 1..n - 1 {
            new_pos[i] = (positions[i - 1] + positions[i] * 2.0 + positions[i + 1]) * 0.25;
        }
        positions.copy_from_slice(&new_pos);

        // Average normals with neighbors (skip endpoints)
        let mut new_normals = normals.to_vec();
        for i in 1..n - 1 {
            let avg = normals[i - 1] + normals[i] * 2.0 + normals[i + 1];
            new_normals[i] = if avg.length_squared() > 1e-6 { avg.normalize() } else { normals[i] };
        }
        normals.copy_from_slice(&new_normals);

        // Re-orthogonalize normals against backbone tangent
        for i in 1..n - 1 {
            let tangent = (positions[i + 1] - positions[i - 1]).normalize_or_zero();
            let proj = normals[i] - tangent * normals[i].dot(tangent);
            normals[i] = if proj.length_squared() > 1e-6 { proj.normalize() } else { normals[i] };
        }
    }
}

/// CA-only fallback for sheet generation. Estimates peptide plane normals
/// from backbone path curvature when N/C atoms are unavailable.
#[allow(dead_code)]
fn generate_sheet_from_ca(ca_positions: &[Vec3], ss_types: &[SSType], params: &RibbonParams, base: u32, global_residue_base: u32) -> (Vec<RibbonVertex>, Vec<u32>) {
    let n = ca_positions.len();
    if n < 2 { return (Vec::new(), Vec::new()); }

    // Without real N/C positions, approximate peptide plane normals from
    // backbone path curvature: cross(prev→curr, curr→next)
    let mut normals: Vec<Vec3> = Vec::with_capacity(n);
    for i in 0..n {
        let prev = if i > 0 { ca_positions[i - 1] } else { ca_positions[0] * 2.0 - ca_positions[1] };
        let next = if i + 1 < n { ca_positions[i + 1] } else { ca_positions[n - 1] * 2.0 - ca_positions[n - 2] };
        let d1 = (ca_positions[i] - prev).normalize_or_zero();
        let d2 = (next - ca_positions[i]).normalize_or_zero();
        let normal = d1.cross(d2);
        normals.push(if normal.length_squared() > 1e-6 { normal.normalize() } else { Vec3::Y });
    }

    for i in 1..n {
        if normals[i].dot(normals[i - 1]) < 0.0 {
            normals[i] = -normals[i];
        }
    }

    let mut positions = ca_positions.to_vec();
    flatten_sheet(&mut positions, &mut normals, 4);

    sheet_from_normals(&positions, &normals, ss_types, params, base, global_residue_base, 0, 0)
}

// ==================== MESH BUILDING ====================

/// Compute per-frame ribbon widths with linear taper at segment boundaries.
/// `taper_start` and `taper_end` are the number of taper residues at each end.
fn compute_taper_widths(
    n_frames: usize,
    full_width: f32,
    segments_per_residue: usize,
    taper_start: usize,
    taper_end: usize,
) -> Vec<f32> {
    let taper_start_frames = taper_start * segments_per_residue;
    let taper_end_frames = taper_end * segments_per_residue;
    (0..n_frames).map(|i| {
        if taper_start_frames > 0 && i < taper_start_frames {
            let t = i as f32 / taper_start_frames as f32;
            TUBE_DIAMETER + (full_width - TUBE_DIAMETER) * t
        } else if taper_end_frames > 0 && i >= n_frames.saturating_sub(taper_end_frames) {
            let frames_from_end = n_frames - 1 - i;
            let t = frames_from_end as f32 / taper_end_frames.max(1) as f32;
            TUBE_DIAMETER + (full_width - TUBE_DIAMETER) * t
        } else {
            full_width
        }
    }).collect()
}

fn build_ribbon_mesh(frames: &[RibbonFrame], widths: &[f32], thickness: f32, base: u32, smooth_normals: bool) -> (Vec<RibbonVertex>, Vec<u32>) {
    let mut vertices = Vec::with_capacity(frames.len() * 4);
    let mut indices = Vec::new();

    let ht = thickness * 0.5;

    for (idx, frame) in frames.iter().enumerate() {
        let hw = widths[idx] * 0.5;
        // 4 vertices per frame: top-left, top-right, bottom-right, bottom-left
        let tl = frame.position + frame.normal * ht - frame.binormal * hw;
        let tr = frame.position + frame.normal * ht + frame.binormal * hw;
        let br = frame.position - frame.normal * ht + frame.binormal * hw;
        let bl = frame.position - frame.normal * ht - frame.binormal * hw;

        let (n_tl, n_tr, n_br, n_bl) = if smooth_normals {
            // Smooth normals: treat cross-section as an ellipse with semi-axes (hw, ht).
            // The outward normal at offset (b, n) is proportional to (b/hw², n/ht²).
            // This gives top vertices normals mostly pointing up with subtle side tilt,
            // creating gradual specular falloff across the surface.
            let inv_hw2 = 1.0 / (hw * hw).max(1e-6);
            let inv_ht2 = 1.0 / (ht * ht).max(1e-6);
            // TL: binormal_offset = -hw, normal_offset = +ht
            let n_tl = (frame.binormal * (-hw * inv_hw2) + frame.normal * (ht * inv_ht2)).normalize();
            // TR: binormal_offset = +hw, normal_offset = +ht
            let n_tr = (frame.binormal * (hw * inv_hw2) + frame.normal * (ht * inv_ht2)).normalize();
            // BR: binormal_offset = +hw, normal_offset = -ht
            let n_br = (frame.binormal * (hw * inv_hw2) + frame.normal * (-ht * inv_ht2)).normalize();
            // BL: binormal_offset = -hw, normal_offset = -ht
            let n_bl = (frame.binormal * (-hw * inv_hw2) + frame.normal * (-ht * inv_ht2)).normalize();
            (n_tl, n_tr, n_br, n_bl)
        } else {
            // Flat normals: top face up, bottom face down (for sheets)
            let up = frame.normal;
            let down = -frame.normal;
            (up, up, down, down)
        };

        vertices.push(RibbonVertex { position: tl.into(), normal: n_tl.into(), color: frame.color, residue_idx: frame.residue_idx });
        vertices.push(RibbonVertex { position: tr.into(), normal: n_tr.into(), color: frame.color, residue_idx: frame.residue_idx });
        vertices.push(RibbonVertex { position: br.into(), normal: n_br.into(), color: frame.color, residue_idx: frame.residue_idx });
        vertices.push(RibbonVertex { position: bl.into(), normal: n_bl.into(), color: frame.color, residue_idx: frame.residue_idx });
    }
    
    // Connect adjacent frames
    for i in 0..frames.len() - 1 {
        let v0 = base + (i * 4) as u32;     // current TL
        let v1 = base + (i * 4 + 1) as u32; // current TR
        let v2 = base + (i * 4 + 2) as u32; // current BR
        let v3 = base + (i * 4 + 3) as u32; // current BL
        
        let v4 = base + ((i + 1) * 4) as u32;     // next TL
        let v5 = base + ((i + 1) * 4 + 1) as u32; // next TR
        let v6 = base + ((i + 1) * 4 + 2) as u32; // next BR
        let v7 = base + ((i + 1) * 4 + 3) as u32; // next BL
        
        // Top face
        indices.extend_from_slice(&[v0, v4, v1, v1, v4, v5]);
        // Bottom face
        indices.extend_from_slice(&[v3, v2, v7, v7, v2, v6]);
        // Right edge
        indices.extend_from_slice(&[v1, v5, v2, v2, v5, v6]);
        // Left edge
        indices.extend_from_slice(&[v0, v3, v4, v4, v3, v7]);
    }
    
    (vertices, indices)
}

// ==================== SPLINE UTILITIES ====================

/// Catmull-Rom spline interpolation (passes through all control points)
fn catmull_rom(points: &[Vec3], segments_per_span: usize) -> Vec<Vec3> {
    let n = points.len();
    if n < 2 { return points.to_vec(); }
    if n < 3 { return linear_interpolate(points, segments_per_span); }
    
    let mut result = Vec::new();
    
    // Catmull-Rom with tension = 0.5 (standard)
    for i in 0..n - 1 {
        let p0 = if i == 0 { points[0] * 2.0 - points[1] } else { points[i - 1] };
        let p1 = points[i];
        let p2 = points[i + 1];
        let p3 = if i + 2 >= n { points[n - 1] * 2.0 - points[n - 2] } else { points[i + 2] };
        
        for j in 0..segments_per_span {
            let t = j as f32 / segments_per_span as f32;
            let t2 = t * t;
            let t3 = t2 * t;
            
            // Catmull-Rom basis (passes through p1 at t=0, p2 at t=1)
            let pos = 0.5 * (
                (2.0 * p1) +
                (-p0 + p2) * t +
                (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
                (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            );
            result.push(pos);
        }
    }
    
    // Add final point (exactly at last control point)
    result.push(points[n - 1]);
    
    result
}

// Keep B-spline available for helix axis smoothing (where we DON'T want to pass through points)
fn cubic_bspline(points: &[Vec3], segments_per_span: usize) -> Vec<Vec3> {
    let n = points.len();
    if n < 2 { return points.to_vec(); }
    if n < 4 { return linear_interpolate(points, segments_per_span); }
    
    let mut result = Vec::new();
    
    fn b0(t: f32) -> f32 { (1.0 - t).powi(3) / 6.0 }
    fn b1(t: f32) -> f32 { (3.0 * t.powi(3) - 6.0 * t.powi(2) + 4.0) / 6.0 }
    fn b2(t: f32) -> f32 { (-3.0 * t.powi(3) + 3.0 * t.powi(2) + 3.0 * t + 1.0) / 6.0 }
    fn b3(t: f32) -> f32 { t.powi(3) / 6.0 }
    
    let mut padded = Vec::with_capacity(n + 2);
    padded.push(points[0] * 2.0 - points[1]);
    padded.extend_from_slice(points);
    padded.push(points[n - 1] * 2.0 - points[n - 2]);
    
    for i in 0..n - 1 {
        let p0 = padded[i];
        let p1 = padded[i + 1];
        let p2 = padded[i + 2];
        let p3 = padded[i + 3];
        
        for j in 0..segments_per_span {
            let t = j as f32 / segments_per_span as f32;
            let pos = p0 * b0(t) + p1 * b1(t) + p2 * b2(t) + p3 * b3(t);
            result.push(pos);
        }
    }
    
    result.push(points[n - 1]);
    result
}

fn linear_interpolate(points: &[Vec3], segments_per_span: usize) -> Vec<Vec3> {
    let mut result = Vec::new();
    for i in 0..points.len() - 1 {
        for j in 0..segments_per_span {
            let t = j as f32 / segments_per_span as f32;
            result.push(points[i].lerp(points[i + 1], t));
        }
    }
    result.push(*points.last().unwrap());
    result
}


fn segment_by_ss(ss_types: &[SSType]) -> Vec<SSSegment> {
    if ss_types.is_empty() { return Vec::new(); }
    
    let mut segments = Vec::new();
    let mut current = ss_types[0];
    let mut start = 0;
    
    for (i, &ss) in ss_types.iter().enumerate() {
        if ss != current {
            segments.push(SSSegment { ss_type: current, start_residue: start, end_residue: i });
            current = ss;
            start = i;
        }
    }
    segments.push(SSSegment { ss_type: current, start_residue: start, end_residue: ss_types.len() });
    
    segments
}
