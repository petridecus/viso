//! Nucleic acid backbone renderer.
//!
//! Renders DNA/RNA backbones as narrow flat ribbons tracing phosphorus (P) atoms,
//! smoothed with B-splines and oriented with rotation-minimizing frames.

use crate::gpu::dynamic_buffer::DynamicBuffer;
use crate::gpu::render_context::RenderContext;
use crate::gpu::shader_composer::ShaderComposer;
use crate::renderer::pipeline_util;
use foldit_conv::coords::entity::NucleotideRing;
use glam::Vec3;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Ribbon half-width for nucleic acid backbone (narrower than protein sheets/helices)
const NA_RIBBON_WIDTH: f32 = 1.2;

/// Ribbon thickness (same as protein ribbons)
const NA_RIBBON_THICKNESS: f32 = 0.25;

/// Spline subdivision per P-atom span
const SEGMENTS_PER_RESIDUE: usize = 16;

/// Default light blue-violet color for nucleic acid backbone (overridden by ColorOptions)
const NA_COLOR: [f32; 3] = [0.45, 0.55, 0.85];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct NaVertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
    residue_idx: u32,
    /// Encodes normal direction as position - normal, so the shared shader computes
    /// normalize(position - center_pos) = normalize(normal) for flat geometry.
    center_pos: [f32; 3],
}

#[derive(Clone, Copy)]
struct SplinePoint {
    pos: Vec3,
    tangent: Vec3,
    normal: Vec3,
    binormal: Vec3,
}

struct RibbonFrame {
    position: Vec3,
    normal: Vec3,
    binormal: Vec3,
    color: [f32; 3],
    residue_idx: u32,
}

pub struct NucleicAcidRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: DynamicBuffer,
    index_buffer: DynamicBuffer,
    pub index_count: u32,
    last_chain_hash: u64,
}

impl NucleicAcidRenderer {
    pub fn new(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        selection_layout: &wgpu::BindGroupLayout,
        na_chains: &[Vec<Vec3>],
        rings: &[NucleotideRing],
        shader_composer: &mut ShaderComposer,
    ) -> Self {
        let (vertices, indices) = Self::generate_mesh(na_chains, rings, None);

        let vertex_buffer = if vertices.is_empty() {
            DynamicBuffer::new(
                &context.device,
                "NA Vertex Buffer",
                std::mem::size_of::<NaVertex>() * 1000,
                wgpu::BufferUsages::VERTEX,
            )
        } else {
            DynamicBuffer::new_with_data(
                &context.device,
                "NA Vertex Buffer",
                &vertices,
                wgpu::BufferUsages::VERTEX,
            )
        };

        let index_buffer = if indices.is_empty() {
            DynamicBuffer::new(
                &context.device,
                "NA Index Buffer",
                std::mem::size_of::<u32>() * 3000,
                wgpu::BufferUsages::INDEX,
            )
        } else {
            DynamicBuffer::new_with_data(
                &context.device,
                "NA Index Buffer",
                &indices,
                wgpu::BufferUsages::INDEX,
            )
        };

        let pipeline = Self::create_pipeline(
            context,
            camera_layout,
            lighting_layout,
            selection_layout,
            shader_composer,
        );
        let last_chain_hash = Self::compute_combined_hash(na_chains, rings);

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            last_chain_hash,
        }
    }

    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        na_chains: &[Vec<Vec3>],
        rings: &[NucleotideRing],
    ) {
        let new_hash = Self::compute_combined_hash(na_chains, rings);
        if new_hash == self.last_chain_hash {
            return;
        }
        self.last_chain_hash = new_hash;

        let (vertices, indices) = Self::generate_mesh(na_chains, rings, None);

        if vertices.is_empty() {
            self.index_count = 0;
            return;
        }

        self.vertex_buffer.write(device, queue, &vertices);
        self.index_buffer.write(device, queue, &indices);
        self.index_count = indices.len() as u32;
    }

    pub fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &super::draw_context::DrawBindGroups<'a>,
    ) {
        if self.index_count == 0 {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, bind_groups.camera, &[]);
        render_pass.set_bind_group(1, bind_groups.lighting, &[]);
        render_pass.set_bind_group(2, bind_groups.selection, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.buffer().slice(..));
        render_pass.set_index_buffer(
            self.index_buffer.buffer().slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(0..self.index_count, 0, 0..1);
    }

    pub fn vertex_buffer(&self) -> &wgpu::Buffer {
        self.vertex_buffer.buffer()
    }

    pub fn index_buffer(&self) -> &wgpu::Buffer {
        self.index_buffer.buffer()
    }

    /// Apply pre-computed mesh data (GPU upload only, no CPU generation).
    pub fn apply_prepared(
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
        self.last_chain_hash = 0; // Invalidate hash so next synchronous update doesn't skip
    }

    // ── Pipeline ──

    fn create_pipeline(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        selection_layout: &wgpu::BindGroupLayout,
        shader_composer: &mut ShaderComposer,
    ) -> wgpu::RenderPipeline {
        let shader = shader_composer.compose(
            &context.device,
            "Nucleic Acid Shader",
            include_str!(
                "../../../assets/shaders/raster/mesh/backbone_na.wgsl"
            ),
            "backbone_na.wgsl",
        );

        let pipeline_layout = context.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("NA Pipeline Layout"),
                bind_group_layouts: &[
                    camera_layout,
                    lighting_layout,
                    selection_layout,
                ],
                immediate_size: 0,
            },
        );

        let vertex_layout = super::tube::tube_vertex_buffer_layout();

        context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("NA Render Pipeline"),
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
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(pipeline_util::depth_stencil_state()),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
    }

    // ── Mesh generation ──

    pub(crate) fn generate_mesh(
        na_chains: &[Vec<Vec3>],
        rings: &[NucleotideRing],
        na_color: Option<[f32; 3]>,
    ) -> (Vec<NaVertex>, Vec<u32>) {
        let na_color = na_color.unwrap_or(NA_COLOR);
        let mut all_vertices: Vec<NaVertex> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();
        let mut all_spline_points: Vec<Vec3> = Vec::new();

        for chain in na_chains {
            if chain.len() < 2 {
                continue;
            }

            // Smooth the P-atom chain with Catmull-Rom (interpolates through control points)
            let spline = catmull_rom(chain, SEGMENTS_PER_RESIDUE);
            if spline.len() < 2 {
                continue;
            }

            all_spline_points.extend_from_slice(&spline);

            // Build SplinePoints with tangents via finite differences
            let mut points: Vec<SplinePoint> = Vec::with_capacity(spline.len());
            for i in 0..spline.len() {
                let tangent = if i == 0 {
                    (spline[1] - spline[0]).normalize()
                } else if i == spline.len() - 1 {
                    (spline[i] - spline[i - 1]).normalize()
                } else {
                    (spline[i + 1] - spline[i - 1]).normalize()
                };
                points.push(SplinePoint {
                    pos: spline[i],
                    tangent,
                    normal: Vec3::ZERO,
                    binormal: Vec3::ZERO,
                });
            }

            // Use Frenet frames so the ribbon twists with the helix curvature
            // (RMF would keep the ribbon flat like a sheet)
            compute_frenet_frames(&mut points);

            // Convert to RibbonFrames with constant color and interpolated residue indices
            let n_residues = chain.len();
            let frames: Vec<RibbonFrame> = points
                .iter()
                .enumerate()
                .map(|(i, sp)| {
                    let residue_idx = if n_residues > 1 {
                        let t = i as f32 / (points.len() - 1) as f32;
                        (t * (n_residues - 1) as f32).round() as u32
                    } else {
                        0
                    };
                    RibbonFrame {
                        position: sp.pos,
                        normal: sp.normal,
                        binormal: sp.binormal,
                        color: na_color,
                        residue_idx,
                    }
                })
                .collect();

            // Constant width for nucleic acid ribbon
            let widths = vec![NA_RIBBON_WIDTH; frames.len()];

            let base_vertex = all_vertices.len() as u32;
            let (verts, inds) = build_ribbon_mesh(
                &frames,
                &widths,
                NA_RIBBON_THICKNESS,
                base_vertex,
                false,
            );
            all_vertices.extend(verts);
            all_indices.extend(inds);
        }

        // Append ring geometry for nucleotide bases
        for ring in rings {
            // Stem: closest spline point to C1' → ring centroid
            if let Some(c1p) = ring.c1_prime {
                if let Some(&anchor) = closest_point(&all_spline_points, c1p) {
                    let centroid = ring.hex_ring.iter().copied().sum::<Vec3>()
                        / ring.hex_ring.len() as f32;
                    append_stem_tube(
                        anchor,
                        centroid,
                        ring.color,
                        &mut all_vertices,
                        &mut all_indices,
                    );
                }
            }
            append_ring_triangles(
                &ring.hex_ring,
                ring.color,
                &mut all_vertices,
                &mut all_indices,
            );
            if !ring.pent_ring.is_empty() {
                append_ring_triangles(
                    &ring.pent_ring,
                    ring.color,
                    &mut all_vertices,
                    &mut all_indices,
                );
            }
        }

        (all_vertices, all_indices)
    }

    fn compute_combined_hash(
        chains: &[Vec<Vec3>],
        rings: &[NucleotideRing],
    ) -> u64 {
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
        // Hash ring data for change detection
        rings.len().hash(&mut hasher);
        if let Some(first_ring) = rings.first() {
            if let Some(p) = first_ring.hex_ring.first() {
                p.x.to_bits().hash(&mut hasher);
                p.y.to_bits().hash(&mut hasher);
                p.z.to_bits().hash(&mut hasher);
            }
        }
        hasher.finish()
    }
}

impl super::MolecularRenderer for NucleicAcidRenderer {
    fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &super::draw_context::DrawBindGroups<'a>,
    ) {
        self.draw(render_pass, bind_groups);
    }
}

/// Catmull-Rom spline: interpolates through every control point.
fn catmull_rom(points: &[Vec3], segments_per_span: usize) -> Vec<Vec3> {
    let n = points.len();
    if n < 2 {
        return points.to_vec();
    }
    if n < 3 {
        return linear_interpolate(points, segments_per_span);
    }

    // Pad with reflected endpoints so the curve starts/ends at the first/last point
    let mut padded = Vec::with_capacity(n + 2);
    padded.push(points[0] * 2.0 - points[1]);
    padded.extend_from_slice(points);
    padded.push(points[n - 1] * 2.0 - points[n - 2]);

    let mut result = Vec::new();
    for i in 0..n - 1 {
        let p0 = padded[i];
        let p1 = padded[i + 1];
        let p2 = padded[i + 2];
        let p3 = padded[i + 3];

        for j in 0..segments_per_span {
            let t = j as f32 / segments_per_span as f32;
            let t2 = t * t;
            let t3 = t2 * t;
            // Catmull-Rom basis (tau = 0.5)
            let pos = 0.5
                * ((2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
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

// ── Frenet frames (curvature-based) ──
//
// For helical traces (DNA/RNA P-atoms), the Frenet normal naturally points
// toward the helix axis.  Negating it gives an outward-pointing ribbon normal
// so the flat face tracks the helix curvature instead of staying sheet-like.

fn compute_frenet_frames(points: &mut [SplinePoint]) {
    if points.len() < 2 {
        return;
    }

    // Compute curvature vector (dT/ds) at each point via finite differences of tangent
    let n = points.len();
    let mut curvatures: Vec<Vec3> = Vec::with_capacity(n);
    for i in 0..n {
        let curv = if i == 0 {
            points[1].tangent - points[0].tangent
        } else if i == n - 1 {
            points[n - 1].tangent - points[n - 2].tangent
        } else {
            (points[i + 1].tangent - points[i - 1].tangent) * 0.5
        };
        curvatures.push(curv);
    }

    for i in 0..n {
        let t = points[i].tangent;
        let curv = curvatures[i];
        let curv_len = curv.length();

        if curv_len > 1e-6 {
            // Frenet normal points toward center of curvature (inward);
            // negate for outward-pointing ribbon normal.
            let normal = -curv.normalize();
            // Project out tangent component to ensure orthogonality
            let normal = (normal - t * t.dot(normal)).normalize();
            let binormal = t.cross(normal).normalize();
            points[i].normal = normal;
            points[i].binormal = binormal;
        } else {
            // Near-zero curvature (straight segment): fall back to arbitrary frame
            let arbitrary = if t.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
            let normal = t.cross(arbitrary).normalize();
            let binormal = t.cross(normal).normalize();
            points[i].normal = normal;
            points[i].binormal = binormal;
        }
    }
}

// ── Closest-point lookup ──

fn closest_point(points: &[Vec3], target: Vec3) -> Option<&Vec3> {
    points.iter().min_by(|a, b| {
        a.distance_squared(target)
            .partial_cmp(&b.distance_squared(target))
            .unwrap()
    })
}

// ── Stem tube (thin cylinder connecting backbone P to base ring) ──

const STEM_RADIUS: f32 = 0.25;
const STEM_SIDES: usize = 6;

fn append_stem_tube(
    start: Vec3,
    end: Vec3,
    color: [f32; 3],
    vertices: &mut Vec<NaVertex>,
    indices: &mut Vec<u32>,
) {
    let axis = end - start;
    let len = axis.length();
    if len < 1e-4 {
        return;
    }
    let dir = axis / len;

    // Build an orthonormal frame around the axis
    let arbitrary = if dir.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let u = dir.cross(arbitrary).normalize();
    let v = dir.cross(u);

    let base = vertices.len() as u32;
    let n = STEM_SIDES;

    // Two rings of vertices (start cap and end cap)
    for &center in &[start, end] {
        for i in 0..n {
            let angle = (i as f32 / n as f32) * std::f32::consts::TAU;
            let (sin, cos) = angle.sin_cos();
            let offset = u * cos * STEM_RADIUS + v * sin * STEM_RADIUS;
            let pos = center + offset;
            let normal = (u * cos + v * sin).normalize();
            vertices.push(NaVertex {
                position: pos.into(),
                normal: normal.into(),
                color,
                residue_idx: 0,
                center_pos: center.into(),
            });
        }
    }

    // Quads connecting the two rings
    for i in 0..n {
        let i0 = base + i as u32;
        let i1 = base + ((i + 1) % n) as u32;
        let i2 = base + n as u32 + i as u32;
        let i3 = base + n as u32 + ((i + 1) % n) as u32;
        indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
    }
}

// ── Ring slab mesh (extruded polygon matching stem diameter) ──

fn append_ring_triangles(
    ring_positions: &[Vec3],
    color: [f32; 3],
    vertices: &mut Vec<NaVertex>,
    indices: &mut Vec<u32>,
) {
    if ring_positions.len() < 3 {
        return;
    }

    let n = ring_positions.len();
    let half_thickness = STEM_RADIUS;

    // Compute face normal via Newell's method
    let mut normal = Vec3::ZERO;
    for i in 0..n {
        let curr = ring_positions[i];
        let next = ring_positions[(i + 1) % n];
        normal.x += (curr.y - next.y) * (curr.z + next.z);
        normal.y += (curr.z - next.z) * (curr.x + next.x);
        normal.z += (curr.x - next.x) * (curr.y + next.y);
    }
    normal = normal.normalize_or_zero();
    if normal == Vec3::ZERO {
        return;
    }

    let offset = normal * half_thickness;

    // --- Top face (triangle fan) ---
    let top_base = vertices.len() as u32;
    let top_centroid: Vec3 =
        ring_positions.iter().copied().sum::<Vec3>() / n as f32 + offset;
    vertices.push(NaVertex {
        position: top_centroid.into(),
        normal: normal.into(),
        color,
        residue_idx: 0,
        center_pos: (top_centroid - normal).into(),
    });
    for &pos in ring_positions {
        let p = pos + offset;
        vertices.push(NaVertex {
            position: p.into(),
            normal: normal.into(),
            color,
            residue_idx: 0,
            center_pos: (p - normal).into(),
        });
    }
    for i in 0..n {
        indices.extend_from_slice(&[
            top_base,
            top_base + 1 + i as u32,
            top_base + 1 + ((i + 1) % n) as u32,
        ]);
    }

    // --- Bottom face (triangle fan, reversed winding) ---
    let bot_base = vertices.len() as u32;
    let bot_centroid = top_centroid - offset * 2.0;
    let neg_normal: [f32; 3] = (-normal).into();
    let neg_normal_v: Vec3 = -normal;
    vertices.push(NaVertex {
        position: bot_centroid.into(),
        normal: neg_normal,
        color,
        residue_idx: 0,
        center_pos: (bot_centroid - neg_normal_v).into(),
    });
    for &pos in ring_positions {
        let p = pos - offset;
        vertices.push(NaVertex {
            position: p.into(),
            normal: neg_normal,
            color,
            residue_idx: 0,
            center_pos: (p - neg_normal_v).into(),
        });
    }
    for i in 0..n {
        indices.extend_from_slice(&[
            bot_base,
            bot_base + 1 + ((i + 1) % n) as u32,
            bot_base + 1 + i as u32,
        ]);
    }

    // --- Side walls (quads connecting top and bottom edges) ---
    let side_base = vertices.len() as u32;
    for i in 0..n {
        let next = (i + 1) % n;
        // Outward normal for this edge
        let edge = ring_positions[next] - ring_positions[i];
        let side_normal = edge.cross(normal).normalize();

        let t0 = ring_positions[i] + offset;
        let t1 = ring_positions[next] + offset;
        let b0 = ring_positions[i] - offset;
        let b1 = ring_positions[next] - offset;

        let si = side_base + (i as u32) * 4;
        vertices.push(NaVertex {
            position: t0.into(),
            normal: side_normal.into(),
            color,
            residue_idx: 0,
            center_pos: (t0 - side_normal).into(),
        });
        vertices.push(NaVertex {
            position: t1.into(),
            normal: side_normal.into(),
            color,
            residue_idx: 0,
            center_pos: (t1 - side_normal).into(),
        });
        vertices.push(NaVertex {
            position: b0.into(),
            normal: side_normal.into(),
            color,
            residue_idx: 0,
            center_pos: (b0 - side_normal).into(),
        });
        vertices.push(NaVertex {
            position: b1.into(),
            normal: side_normal.into(),
            color,
            residue_idx: 0,
            center_pos: (b1 - side_normal).into(),
        });
        indices.extend_from_slice(&[
            si,
            si + 1,
            si + 2,
            si + 2,
            si + 1,
            si + 3,
        ]);
    }
}

// ── Ribbon mesh building ──

fn build_ribbon_mesh(
    frames: &[RibbonFrame],
    widths: &[f32],
    thickness: f32,
    base: u32,
    smooth_normals: bool,
) -> (Vec<NaVertex>, Vec<u32>) {
    let mut vertices = Vec::with_capacity(frames.len() * 4);
    let mut indices = Vec::new();

    let ht = thickness * 0.5;

    for (idx, frame) in frames.iter().enumerate() {
        let hw = widths[idx] * 0.5;
        let tl = frame.position + frame.normal * ht - frame.binormal * hw;
        let tr = frame.position + frame.normal * ht + frame.binormal * hw;
        let br = frame.position - frame.normal * ht + frame.binormal * hw;
        let bl = frame.position - frame.normal * ht - frame.binormal * hw;

        let (n_tl, n_tr, n_br, n_bl) = if smooth_normals {
            let inv_hw2 = 1.0 / (hw * hw).max(1e-6);
            let inv_ht2 = 1.0 / (ht * ht).max(1e-6);
            let n_tl = (frame.binormal * (-hw * inv_hw2)
                + frame.normal * (ht * inv_ht2))
                .normalize();
            let n_tr = (frame.binormal * (hw * inv_hw2)
                + frame.normal * (ht * inv_ht2))
                .normalize();
            let n_br = (frame.binormal * (hw * inv_hw2)
                + frame.normal * (-ht * inv_ht2))
                .normalize();
            let n_bl = (frame.binormal * (-hw * inv_hw2)
                + frame.normal * (-ht * inv_ht2))
                .normalize();
            (n_tl, n_tr, n_br, n_bl)
        } else {
            let up = frame.normal;
            let down = -frame.normal;
            (up, up, down, down)
        };

        vertices.push(NaVertex {
            position: tl.into(),
            normal: n_tl.into(),
            color: frame.color,
            residue_idx: frame.residue_idx,
            center_pos: (tl - n_tl).into(),
        });
        vertices.push(NaVertex {
            position: tr.into(),
            normal: n_tr.into(),
            color: frame.color,
            residue_idx: frame.residue_idx,
            center_pos: (tr - n_tr).into(),
        });
        vertices.push(NaVertex {
            position: br.into(),
            normal: n_br.into(),
            color: frame.color,
            residue_idx: frame.residue_idx,
            center_pos: (br - n_br).into(),
        });
        vertices.push(NaVertex {
            position: bl.into(),
            normal: n_bl.into(),
            color: frame.color,
            residue_idx: frame.residue_idx,
            center_pos: (bl - n_bl).into(),
        });
    }

    for i in 0..frames.len() - 1 {
        let v0 = base + (i * 4) as u32;
        let v1 = base + (i * 4 + 1) as u32;
        let v2 = base + (i * 4 + 2) as u32;
        let v3 = base + (i * 4 + 3) as u32;

        let v4 = base + ((i + 1) * 4) as u32;
        let v5 = base + ((i + 1) * 4 + 1) as u32;
        let v6 = base + ((i + 1) * 4 + 2) as u32;
        let v7 = base + ((i + 1) * 4 + 3) as u32;

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
