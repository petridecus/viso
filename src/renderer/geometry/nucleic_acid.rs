//! Nucleic acid ring + stem renderer.
//!
//! Renders base rings as extruded polygon impostors and stem tubes as capsule
//! impostors connecting the backbone spline to nucleotide ring centers.
//! Backbone ribbons are handled by `BackboneRenderer`.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use foldit_conv::types::entity::NucleotideRing;
use glam::Vec3;

use crate::error::VisoError;
use crate::gpu::render_context::RenderContext;
use crate::gpu::shader_composer::{Shader, ShaderComposer};
use crate::renderer::impostor::capsule::CapsuleInstance;
use crate::renderer::impostor::polygon::ExtrudedPolygonInstance;
use crate::renderer::impostor::{ImpostorPass, ShaderDef};
use crate::util::hash::{hash_vec3, hash_vec3_slice_summary};

/// Spline subdivision per P-atom span (used for stem anchor lookup)
const SEGMENTS_PER_RESIDUE: usize = 16;

/// Default light blue-violet color for nucleic acid backbone (overridden by
/// ColorOptions)
const NA_COLOR: [f32; 3] = [0.45, 0.55, 0.85];

/// Stem capsule radius (matches old STEM_RADIUS)
const STEM_RADIUS: f32 = 0.25;

/// Renders DNA/RNA base rings and stem tubes as impostors.
pub struct NucleicAcidRenderer {
    stem_pass: ImpostorPass<CapsuleInstance>,
    ring_pass: ImpostorPass<ExtrudedPolygonInstance>,
    last_chain_hash: u64,
}

impl NucleicAcidRenderer {
    /// Create a new nucleic acid renderer from phosphorus atom chains.
    pub fn new(
        context: &RenderContext,
        layouts: &crate::renderer::PipelineLayouts,
        na_chains: &[Vec<Vec3>],
        rings: &[NucleotideRing],
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, VisoError> {
        let mut stem_pass = ImpostorPass::new(
            context,
            &ShaderDef {
                label: "NA Stem",
                shader: Shader::Capsule,
            },
            layouts,
            6,
            shader_composer,
        )?;

        let mut ring_pass = ImpostorPass::new(
            context,
            &ShaderDef {
                label: "NA Ring",
                shader: Shader::Polygon,
            },
            layouts,
            72,
            shader_composer,
        )?;

        let (stems, ring_instances) =
            Self::generate_instances(na_chains, rings, None);
        let _ =
            stem_pass.write_instances(&context.device, &context.queue, &stems);
        let _ = ring_pass.write_instances(
            &context.device,
            &context.queue,
            &ring_instances,
        );

        let last_chain_hash = Self::compute_combined_hash(na_chains, rings);

        Ok(Self {
            stem_pass,
            ring_pass,
            last_chain_hash,
        })
    }

    /// Draw nucleic acid geometry into the given render pass.
    pub fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &crate::renderer::draw_context::DrawBindGroups<'a>,
    ) {
        self.stem_pass.draw(render_pass, bind_groups);
        self.ring_pass.draw(render_pass, bind_groups);
    }

    /// Apply pre-computed instance data (GPU upload only, no CPU generation).
    pub fn apply_prepared(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        na: &crate::scene::prepared::NucleicAcidInstances,
    ) {
        let _ = self.stem_pass.write_bytes(
            device,
            queue,
            &na.stem_instances,
            na.stem_count,
        );
        let _ = self.ring_pass.write_bytes(
            device,
            queue,
            &na.ring_instances,
            na.ring_count,
        );
        self.last_chain_hash = 0;
    }

    /// GPU buffer sizes: `(label, used_bytes, allocated_bytes)`.
    pub fn buffer_info(&self) -> Vec<(&'static str, usize, usize)> {
        vec![
            self.stem_pass.buffer_info("NA Stems"),
            self.ring_pass.buffer_info("NA Rings"),
        ]
    }

    // ── Instance generation ──

    /// Generate capsule + polygon instances for stems and rings.
    pub(crate) fn generate_instances(
        na_chains: &[Vec<Vec3>],
        rings: &[NucleotideRing],
        na_color: Option<[f32; 3]>,
    ) -> (Vec<CapsuleInstance>, Vec<ExtrudedPolygonInstance>) {
        let _na_color = na_color.unwrap_or(NA_COLOR);
        let mut stems = Vec::new();
        let mut ring_instances = Vec::new();

        // Compute spline points for stem anchoring
        let mut all_spline_points: Vec<Vec3> = Vec::new();
        for chain in na_chains {
            if chain.len() < 2 {
                continue;
            }
            let spline = catmull_rom(chain, SEGMENTS_PER_RESIDUE);
            all_spline_points.extend_from_slice(&spline);
        }

        for ring in rings {
            // Stem: closest spline point to C1' → ring centroid
            if let Some(c1p) = ring.c1_prime {
                if let Some(&anchor) = closest_point(&all_spline_points, c1p) {
                    let centroid = ring.hex_ring.iter().copied().sum::<Vec3>()
                        / ring.hex_ring.len() as f32;
                    stems.push(CapsuleInstance {
                        endpoint_a: [anchor.x, anchor.y, anchor.z, STEM_RADIUS],
                        endpoint_b: [centroid.x, centroid.y, centroid.z, 0.0],
                        color_a: [
                            ring.color[0],
                            ring.color[1],
                            ring.color[2],
                            0.0,
                        ],
                        color_b: [
                            ring.color[0],
                            ring.color[1],
                            ring.color[2],
                            0.0,
                        ],
                    });
                }
            }

            // Hex ring
            make_polygon_instance(
                &ring.hex_ring,
                ring.color,
                STEM_RADIUS,
                &mut ring_instances,
            );

            // Pent ring (purines only)
            if !ring.pent_ring.is_empty() {
                make_polygon_instance(
                    &ring.pent_ring,
                    ring.color,
                    STEM_RADIUS,
                    &mut ring_instances,
                );
            }
        }

        (stems, ring_instances)
    }

    fn compute_combined_hash(
        chains: &[Vec<Vec3>],
        rings: &[NucleotideRing],
    ) -> u64 {
        let mut hasher = DefaultHasher::new();
        chains.len().hash(&mut hasher);
        for chain in chains {
            hash_vec3_slice_summary(chain, &mut hasher);
        }
        rings.len().hash(&mut hasher);
        if let Some(first_ring) = rings.first() {
            if let Some(p) = first_ring.hex_ring.first() {
                hash_vec3(p, &mut hasher);
            }
        }
        hasher.finish()
    }
}

/// Build an `ExtrudedPolygonInstance` from a ring of 3–6 coplanar positions.
fn make_polygon_instance(
    positions: &[Vec3],
    color: [f32; 3],
    half_thickness: f32,
    out: &mut Vec<ExtrudedPolygonInstance>,
) {
    let n = positions.len();
    if !(3..=6).contains(&n) {
        return;
    }

    // Compute face normal via Newell's method
    let mut normal = Vec3::ZERO;
    for i in 0..n {
        let curr = positions[i];
        let next = positions[(i + 1) % n];
        normal.x += (curr.y - next.y) * (curr.z + next.z);
        normal.y += (curr.z - next.z) * (curr.x + next.x);
        normal.z += (curr.x - next.x) * (curr.y + next.y);
    }
    normal = normal.normalize_or_zero();
    if normal == Vec3::ZERO {
        return;
    }

    // Centroid for degenerate padding
    let centroid: Vec3 = positions.iter().copied().sum::<Vec3>() / n as f32;

    // Pad to exactly 6 verts (unused slots = centroid = degenerate triangle)
    let v = |i: usize| -> [f32; 4] {
        let p = if i < n { positions[i] } else { centroid };
        [p.x, p.y, p.z, 0.0]
    };

    out.push(ExtrudedPolygonInstance {
        v0: {
            let p = positions[0];
            [p.x, p.y, p.z, n as f32]
        },
        v1: {
            let p = positions[1];
            [p.x, p.y, p.z, half_thickness]
        },
        v2: v(2),
        v3: v(3),
        v4: v(4),
        v5: v(5),
        normal: [normal.x, normal.y, normal.z, 0.0],
        color: [color[0], color[1], color[2], 0.0],
    });
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

    // Pad with reflected endpoints so the curve starts/ends at the first/last
    // point
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
    if let Some(&last) = points.last() {
        result.push(last);
    }
    result
}

fn closest_point(points: &[Vec3], target: Vec3) -> Option<&Vec3> {
    points.iter().min_by(|a, b| {
        a.distance_squared(target)
            .partial_cmp(&b.distance_squared(target))
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}
