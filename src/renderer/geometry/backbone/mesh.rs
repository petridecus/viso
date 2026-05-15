//! Backbone mesh generation: ties spline, profile, and sheet modules together
//! into final vertex/index buffers for both protein and nucleic acid chains.

use glam::Vec3;
use molex::SSType;

use super::arrows::apply_sheet_arrows;
use super::curve::{cubic_bspline, sliding_window_centroids};
use super::index::{extrude_and_index, MeshParams};
use super::path::{
    compute_sheet_geometry, interpolate_per_residue_normals, SheetGeometry,
};
use super::profile::{
    interpolate_profiles, resolve_na_profile, resolve_profile,
    CrossSectionProfile,
};
use super::spline::{
    dual_hermite_spline, frenet_frames, helix_aware_spline, rmf_frames,
    SplinePoint, SplineTrace,
};
use super::BackboneMeshOutput;
use crate::options::{ChainLod, GeometryOptions};

/// Per-chain index range and bounding sphere for frustum culling.
///
/// The tube/ribbon index spans are half-open `Range<u32>` into the
/// shared index buffer. The range fields are private so the only
/// construction path is [`ChainRange::new`], which asserts the two
/// invariants every consumer relies on: `start <= end` and triangle
/// alignment (`% 3 == 0`, since indices come in triples).
#[derive(Clone, Debug)]
pub(crate) struct ChainRange {
    tube: std::ops::Range<u32>,
    ribbon: std::ops::Range<u32>,
    pub(crate) bounding_center: Vec3,
    pub(crate) bounding_radius: f32,
}

impl ChainRange {
    /// Build a chain range, asserting the index-span invariants.
    pub(crate) fn new(
        tube: std::ops::Range<u32>,
        ribbon: std::ops::Range<u32>,
        bounding_center: Vec3,
        bounding_radius: f32,
    ) -> Self {
        debug_assert!(
            tube.start <= tube.end && ribbon.start <= ribbon.end,
            "ChainRange index span: start must not exceed end (tube {tube:?}, \
             ribbon {ribbon:?})",
        );
        debug_assert!(
            tube.start.is_multiple_of(3)
                && tube.end.is_multiple_of(3)
                && ribbon.start.is_multiple_of(3)
                && ribbon.end.is_multiple_of(3),
            "ChainRange index span must be triangle-aligned (multiple of 3): \
             tube {tube:?}, ribbon {ribbon:?}",
        );
        Self {
            tube,
            ribbon,
            bounding_center,
            bounding_radius,
        }
    }

    /// Half-open index span for the tube (round cross-section) pass.
    pub(crate) fn tube(&self) -> std::ops::Range<u32> {
        self.tube.clone()
    }

    /// Half-open index span for the ribbon (flat cross-section) pass.
    pub(crate) fn ribbon(&self) -> std::ops::Range<u32> {
        self.ribbon.clone()
    }
}

/// Default nucleic acid backbone color (light blue-violet).
const NA_COLOR: [f32; 3] = [0.45, 0.55, 0.85];

/// Generate unified backbone mesh from protein and nucleic acid chains.
pub(crate) fn generate_mesh_colored(
    protein: &[crate::renderer::entity_topology::ProteinBackboneChain],
    na: &[crate::renderer::entity_topology::NaBackboneChain],
    ss_override: Option<&[SSType]>,
    per_residue_colors: Option<&[[f32; 3]]>,
    geo: &GeometryOptions,
    per_chain_lod: Option<&[ChainLod]>,
    na_residue_colors: Option<&[[f32; 3]]>,
) -> BackboneMeshOutput {
    let (mut out, global_residue_idx) = process_protein_chains(
        protein,
        ss_override,
        per_residue_colors,
        geo,
        per_chain_lod,
    );
    let na_lod = per_chain_lod.and_then(|l| l.get(protein.len()..));
    process_na_chains(
        na,
        geo,
        na_lod,
        &mut out,
        global_residue_idx,
        na_residue_colors,
    );

    out
}

fn process_protein_chains(
    chains: &[crate::renderer::entity_topology::ProteinBackboneChain],
    ss_override: Option<&[SSType]>,
    per_residue_colors: Option<&[[f32; 3]]>,
    geo: &GeometryOptions,
    per_chain_lod: Option<&[ChainLod]>,
) -> (BackboneMeshOutput, u32) {
    let mut out = BackboneMeshOutput::default();
    let mut global_residue_idx: u32 = 0;
    let spr = geo.segments_per_residue;
    let csv = geo.cross_section_verts;

    for (chain_idx, atoms) in chains.iter().enumerate() {
        if atoms.ca().len() < 2 {
            global_residue_idx += atoms.ca().len() as u32;
            continue;
        }

        let n_residues = atoms.ca().len();
        let chain_slice = ss_override.and_then(|o| {
            let start = global_residue_idx as usize;
            let end = (start + n_residues).min(o.len());
            (end.saturating_sub(start) == n_residues).then(|| &o[start..end])
        });
        // Engine sync always installs per-entity SS via
        // `Assembly::ss_types`, so every protein chain with >= 2 CA atoms
        // has a matching slice. If that invariant is ever violated the
        // chain renders as coil -- it doesn't recompute DSSP here.
        let ss_types = chain_slice.map_or_else(
            || vec![SSType::Coil; n_residues],
            molex::analysis::merge_short_segments,
        );

        let mut profiles: Vec<CrossSectionProfile> = (0..n_residues)
            .map(|i| {
                let color = per_residue_colors
                    .and_then(|c| {
                        c.get(global_residue_idx as usize + i).copied()
                    })
                    .unwrap_or_else(|| ss_types[i].color());
                resolve_profile(
                    ss_types[i],
                    global_residue_idx + i as u32,
                    color,
                    geo,
                )
            })
            .collect();

        if geo.sheet_arrows {
            apply_sheet_arrows(&ss_types, &mut profiles, geo);
        }

        let lod = per_chain_lod
            .and_then(|l| l.get(chain_idx).copied())
            .unwrap_or(ChainLod {
                segments_per_residue: spr,
                cross_section_verts: csv,
            });

        let params = MeshParams {
            base_vertex: out.vertices.len() as u32,
            cross_section_verts: lod.cross_section_verts,
            segments_per_residue: lod.segments_per_residue,
        };
        // Widest the extruded ribbon/tube can sit off the CA spline:
        // the largest configured half-width/thickness, scaled by the
        // x1.5 sheet-arrow shoulder, plus Catmull-Rom overshoot.
        let max_extent = geo
            .sheet_width
            .max(geo.helix_width)
            .max(geo.coil_width)
            .max(geo.sheet_thickness)
            .max(geo.helix_thickness)
            .max(geo.coil_thickness)
            * 1.5;
        let (center, radius) =
            bounding_sphere(atoms.ca(), max_extent + SPLINE_OVERSHOOT_SLACK);

        let chain_mesh = generate_protein_chain_mesh(
            atoms,
            &ss_types,
            &profiles,
            global_residue_idx,
            &params,
        );
        out.push_chain(chain_mesh, center, radius);

        global_residue_idx += n_residues as u32;
    }

    (out, global_residue_idx)
}

fn process_na_chains(
    chains: &[crate::renderer::entity_topology::NaBackboneChain],
    geo: &GeometryOptions,
    per_chain_lod: Option<&[ChainLod]>,
    out: &mut BackboneMeshOutput,
    mut global_residue_idx: u32,
    na_residue_colors: Option<&[[f32; 3]]>,
) {
    let spr = geo.segments_per_residue;
    let csv = geo.cross_section_verts;

    // Running index into the flat na_residue_colors slice.
    let mut na_residue_offset: usize = 0;

    for (na_idx, chain) in chains.iter().enumerate() {
        let points = chain.p();
        if points.len() < 2 {
            global_residue_idx += points.len() as u32;
            na_residue_offset += points.len();
            continue;
        }

        let n_residues = points.len();
        let profiles: Vec<CrossSectionProfile> = (0..n_residues)
            .map(|i| {
                let color = na_residue_colors
                    .and_then(|c| c.get(na_residue_offset + i).copied())
                    .unwrap_or(NA_COLOR);
                resolve_na_profile(global_residue_idx + i as u32, color, geo)
            })
            .collect();

        let lod = per_chain_lod
            .and_then(|l| l.get(na_idx).copied())
            .unwrap_or(ChainLod {
                segments_per_residue: spr,
                cross_section_verts: csv,
            });

        let params = MeshParams {
            base_vertex: out.vertices.len() as u32,
            cross_section_verts: lod.cross_section_verts,
            segments_per_residue: lod.segments_per_residue,
        };
        let na_extent = geo.na_width.max(geo.na_thickness);
        let (center, radius) =
            bounding_sphere(points, na_extent + SPLINE_OVERSHOOT_SLACK);
        let chain_mesh = generate_na_chain_mesh(points, &profiles, &params);
        out.push_chain(chain_mesh, center, radius);

        global_residue_idx += n_residues as u32;
        na_residue_offset += n_residues;
    }
}

/// Catmull-Rom interpolation can bow outside the CA control hull at
/// sharp turns; this bounds that overshoot for the culling sphere so a
/// chain isn't culled while its extruded curve is still on-screen.
const SPLINE_OVERSHOOT_SLACK: f32 = 1.0;

/// Compute bounding sphere (centroid + max distance) from a set of
/// positions, padded by `slack`.
///
/// The sphere is fit to the raw control points (CA / P atoms), but the
/// drawn mesh is an extruded tube/ribbon that extends past them by up to
/// the cross-section half-extent plus spline overshoot. `slack` widens
/// the radius to cover that, eliminating false-negative frustum culls at
/// sharp turns. Over-padding only makes culling more conservative.
fn bounding_sphere(positions: &[Vec3], slack: f32) -> (Vec3, f32) {
    if positions.is_empty() {
        return (Vec3::ZERO, 0.0);
    }
    let center =
        positions.iter().copied().sum::<Vec3>() / positions.len() as f32;
    let radius = positions
        .iter()
        .map(|p| (*p - center).length())
        .fold(0.0f32, f32::max);
    (center, radius + slack)
}

/// Generate mesh for a single protein chain (with SS detection, sheet
/// geometry, and RMF/radial/sheet normal blending). Takes the SoA
/// backbone-atom view directly from the topology -- no interleaved
/// stride shuffling.
fn generate_protein_chain_mesh(
    atoms: &crate::renderer::entity_topology::ProteinBackboneChain,
    ss_types: &[SSType],
    profiles: &[CrossSectionProfile],
    global_residue_base: u32,
    params: &MeshParams,
) -> BackboneMeshOutput {
    let n = atoms.ca().len();
    if n < 2 {
        return BackboneMeshOutput::default();
    }

    let SheetGeometry {
        flat_ca,
        normals: sheet_normals,
        offsets: sheet_offsets,
    } = compute_sheet_geometry(atoms, ss_types, global_residue_base);

    let spr = params.segments_per_residue;
    let spline_points = helix_aware_spline(&flat_ca, ss_types, spr);
    let total = spline_points.len();
    if total < 2 {
        return BackboneMeshOutput::default();
    }

    let tangents = compute_tangents(&spline_points);

    let helix_centers = sliding_window_centroids(atoms.ca());
    let spline_helix_centers = cubic_bspline(&helix_centers, spr);

    let traces = build_traces(&spline_points, &tangents);
    // Seed the RMF roll from the first residue's peptide-plane normal so
    // the whole chain's roll is fixed by backbone geometry rather than a
    // world axis. `compute_rmf` projects this perpendicular to the first
    // tangent and falls back to an axis only if it is zero/absent.
    let frames = rmf_frames(&traces, sheet_normals.first().copied());

    let spline_sheet_normals =
        interpolate_per_residue_normals(&sheet_normals, total, n);
    let spline_profiles = interpolate_profiles(profiles, total, n);

    let final_frames = compute_final_frames(
        &frames,
        &spline_helix_centers,
        &spline_sheet_normals,
        &spline_profiles,
    );

    if super::sheet_trace::enabled() {
        super::sheet_trace::trace_final_frames(
            global_residue_base,
            n,
            &tangents,
            &frames,
            &spline_sheet_normals,
            &final_frames,
            &spline_profiles,
        );
    }

    let (verts, tube_inds, ribbon_inds) =
        extrude_and_index(&final_frames, &spline_profiles, params);

    BackboneMeshOutput {
        vertices: verts,
        tube_indices: tube_inds,
        ribbon_indices: ribbon_inds,
        sheet_offsets,
        ..Default::default()
    }
}

// ==================== NUCLEIC ACID CHAIN MESH ====================

/// Generate mesh for a single NA chain (P-atom positions, Frenet frames,
/// no sheet geometry).
fn generate_na_chain_mesh(
    positions: &[Vec3],
    profiles: &[CrossSectionProfile],
    params: &MeshParams,
) -> BackboneMeshOutput {
    let n = positions.len();
    if n < 2 {
        return BackboneMeshOutput::default();
    }

    let spr = params.segments_per_residue;
    let spline_points = dual_hermite_spline(positions, spr);
    let total = spline_points.len();
    if total < 2 {
        return BackboneMeshOutput::default();
    }

    let tangents = compute_tangents(&spline_points);

    let traces = build_traces(&spline_points, &tangents);
    let frames = frenet_frames(&traces);

    let spline_profiles = interpolate_profiles(profiles, total, n);
    let (verts, tube_inds, ribbon_inds) =
        extrude_and_index(&frames, &spline_profiles, params);

    BackboneMeshOutput {
        vertices: verts,
        tube_indices: tube_inds,
        ribbon_indices: ribbon_inds,
        ..Default::default()
    }
}

// ==================== SHARED HELPERS ====================

/// Compute tangents from spline positions via central differences.
fn compute_tangents(spline: &[Vec3]) -> Vec<Vec3> {
    let n = spline.len();
    (0..n)
        .map(|i| {
            if i == 0 {
                (spline[1] - spline[0]).normalize_or_zero()
            } else if i == n - 1 {
                (spline[i] - spline[i - 1]).normalize_or_zero()
            } else {
                (spline[i + 1] - spline[i - 1]).normalize_or_zero()
            }
        })
        .collect()
}

/// Build position+tangent traces from spline positions and tangents.
fn build_traces(spline: &[Vec3], tangents: &[Vec3]) -> Vec<SplineTrace> {
    spline
        .iter()
        .zip(tangents.iter())
        .map(|(&pos, &tangent)| SplineTrace { pos, tangent })
        .collect()
}

// ==================== NORMAL BLENDING (protein only) ====================

fn compute_final_frames(
    rmf_frames: &[SplinePoint],
    helix_centers: &[Vec3],
    sheet_normals: &[Vec3],
    profiles: &[CrossSectionProfile],
) -> Vec<SplinePoint> {
    let total_spline = rmf_frames.len();
    let mut result: Vec<SplinePoint> = Vec::with_capacity(total_spline);

    for i in 0..total_spline {
        let frame = &rmf_frames[i];
        let profile = &profiles[i];

        let tangent = frame.tangent;
        let rmf_normal = frame.normal;

        // Radial candidate: outward from helix axis, projected perp to
        // tangent. Falls back to rmf_normal when degenerate.
        let radial_normal = if profile.radial_blend > 0.01 {
            let ci = i.min(helix_centers.len().saturating_sub(1));
            let to_surface = frame.pos - helix_centers[ci];
            let radial = (to_surface - tangent * tangent.dot(to_surface))
                .normalize_or_zero();
            if radial.length_squared() > 0.01 {
                radial
            } else {
                rmf_normal
            }
        } else {
            rmf_normal
        };

        // Non-sheet candidate: RMF blended toward radial by radial_blend.
        let non_sheet_candidate = {
            let blended = rmf_normal
                .lerp(radial_normal, profile.radial_blend)
                .normalize_or_zero();
            if blended.length_squared() > 0.01 {
                blended
            } else {
                rmf_normal
            }
        };

        // Sheet candidate: peptide-plane normal projected perp to
        // tangent. Falls back to the non-sheet candidate when
        // degenerate.
        let sheet_n = sheet_normals[i];
        let sheet_candidate = {
            let proj = sheet_n - tangent * sheet_n.dot(tangent);
            if proj.length_squared() > 1e-6 {
                proj.normalize()
            } else {
                non_sheet_candidate
            }
        };

        // Smooth blend between the two candidates via sheet_blend,
        // which `interpolate_profiles` already ramps 0->1 across sheet
        // boundaries. Replaces the old binary `has_sheet` switch that
        // caused one-sample ~90deg flips at every sheet<->non-sheet
        // transition.
        let normal = {
            // The broad-face normal has no geometrically meaningful sign
            // for a flat ribbon, but `propagate_segment_signs` aligns
            // peptide normals to their own strand neighbor, not to the
            // RMF chain. Flip the sheet candidate into the RMF hemisphere
            // so the blend can't pass through zero when the two are
            // opposed.
            let sheet_candidate =
                if non_sheet_candidate.dot(sheet_candidate) < 0.0 {
                    -sheet_candidate
                } else {
                    sheet_candidate
                };
            let blended = non_sheet_candidate
                .lerp(sheet_candidate, profile.sheet_blend)
                .normalize_or_zero();
            if blended.length_squared() > 0.01 {
                blended
            } else {
                non_sheet_candidate
            }
        };

        // The within-sample alignment above keeps the blend well-defined
        // but its branch can toggle on float noise when the RMF normal is
        // ~perpendicular to the sheet candidate, producing isolated
        // single-sample 180deg spikes. The broad-face normal sign is
        // geometrically free, so force each frame into the previous
        // frame's hemisphere: consecutive samples are densely spaced, so
        // a sign opposition between neighbors is always spurious.
        let normal = match result.last() {
            Some(prev) if normal.dot(prev.normal) < 0.0 => -normal,
            _ => normal,
        };

        let binormal = tangent.cross(normal).normalize_or_zero();

        result.push(SplinePoint {
            pos: frame.pos,
            tangent,
            normal,
            binormal,
        });
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile_with_sheet_blend(sheet_blend: f32) -> CrossSectionProfile {
        CrossSectionProfile {
            width: 1.0,
            thickness: 0.2,
            roundness: 0.0,
            radial_blend: 0.0,
            sheet_blend,
            color: [0.5, 0.5, 0.5],
            residue_idx: 0,
        }
    }

    /// At a strand entry the RMF normal and the peptide-plane sheet normal
    /// can be anti-parallel. Blending across the sheet ramp must not let the
    /// broad-face normal swing through zero into the opposite hemisphere:
    /// consecutive output normals must keep a positive dot product.
    #[test]
    fn sheet_blend_does_not_flip_hemisphere() {
        let n = 5;
        // Straight chain along +Z so every tangent is +Z.
        let rmf_frames: Vec<SplinePoint> = (0..n)
            .map(|i| SplinePoint {
                pos: Vec3::new(0.0, 0.0, i as f32),
                tangent: Vec3::Z,
                normal: Vec3::X,
                binormal: Vec3::Y,
            })
            .collect();
        let helix_centers = vec![Vec3::ZERO; n];
        // Peptide-plane normal anti-parallel to the RMF normal.
        let sheet_normals = vec![-Vec3::X; n];
        // sheet_blend ramps 0 -> 1 across the strand entry.
        let profiles: Vec<CrossSectionProfile> = (0..n)
            .map(|i| profile_with_sheet_blend(i as f32 / (n - 1) as f32))
            .collect();

        let result = compute_final_frames(
            &rmf_frames,
            &helix_centers,
            &sheet_normals,
            &profiles,
        );

        for i in 0..result.len() - 1 {
            let d = result[i].normal.dot(result[i + 1].normal);
            assert!(
                d > 0.0,
                "frame {i}->{}: normal flipped hemisphere (dot = {d}, {:?} -> \
                 {:?})",
                i + 1,
                result[i].normal,
                result[i + 1].normal,
            );
        }
    }

    /// Isolates the sequential hemisphere-coherence step (the
    /// `result.last()` alignment). Inputs are continuous, but the
    /// within-sample T0-A branch toggles its flip across samples (the
    /// RMF normal is ~perpendicular to the sheet candidate), so the
    /// pre-coherence normal sequence is +Y, -Y, +Y. Only the
    /// cross-sample step removes the single-sample 180deg spike; remove it
    /// and this test goes red while every other test stays green.
    #[test]
    fn seq_coherence_fixes_isolated_sign_toggle() {
        let rmf_frames: Vec<SplinePoint> = (0..3)
            .map(|i| SplinePoint {
                pos: Vec3::new(0.0, 0.0, i as f32),
                tangent: Vec3::Z,
                normal: Vec3::X,
                binormal: Vec3::Y,
            })
            .collect();
        let helix_centers = vec![Vec3::ZERO; 3];
        // Sheet normal ~ +Y with a tiny +/-x tilt: its sign vs the RMF
        // normal (X) alternates, toggling the T0-A flip sample to sample.
        let sheet_normals = vec![
            Vec3::new(0.02, 0.9998, 0.0),
            Vec3::new(-0.02, 0.9998, 0.0),
            Vec3::new(0.02, 0.9998, 0.0),
        ];
        let profiles: Vec<CrossSectionProfile> =
            (0..3).map(|_| profile_with_sheet_blend(1.0)).collect();

        let result = compute_final_frames(
            &rmf_frames,
            &helix_centers,
            &sheet_normals,
            &profiles,
        );

        for i in 1..result.len() {
            let d = result[i].normal.dot(result[i - 1].normal);
            assert!(
                d > 0.5,
                "sample {i}: isolated sign toggle not absorbed (dot = {d})",
            );
        }
    }

    /// Isolates T0-A's distinct job: keeping the within-sample blend
    /// pointing the right way. With near-opposed candidates at
    /// `sheet_blend = 0.5`, removing the T0-A flip makes the lerp a
    /// near-zero residual that `normalize_or_zero` rescues into a wild
    /// ~90deg-off direction (~ +Y here). The flip keeps the blended normal
    /// aligned with the intended broad face (~ the RMF/X hemisphere).
    /// Disable the flip and this test goes red while the others stay
    /// green.
    #[test]
    fn toa_blend_avoids_fallback_collapse() {
        let rmf_frames = vec![SplinePoint {
            pos: Vec3::ZERO,
            tangent: Vec3::Z,
            normal: Vec3::X,
            binormal: Vec3::Y,
        }];
        let helix_centers = vec![Vec3::ZERO];
        // ~179deg from the RMF normal: lerp at 0.5 collapses without the
        // flip, stays unit-length with it.
        let sheet_normals = vec![Vec3::new(-0.999_847_7, 0.017_452_4, 0.0)];
        let profiles = vec![profile_with_sheet_blend(0.5)];

        let result = compute_final_frames(
            &rmf_frames,
            &helix_centers,
            &sheet_normals,
            &profiles,
        );

        assert!(
            result[0].normal.dot(Vec3::X) > 0.9,
            "blend swung off the intended broad face (normal = {:?})",
            result[0].normal,
        );
    }
}
