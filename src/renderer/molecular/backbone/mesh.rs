//! Backbone mesh generation: ties spline, profile, and sheet modules together
//! into final vertex/index buffers for both protein and nucleic acid chains.

use foldit_conv::secondary_structure::{resolve, DetectionInput, SSType};
use glam::Vec3;

use super::spline::{
    SplinePoint, catmull_rom, compute_frenet_frames,
    compute_helix_axis_points, compute_rmf, cubic_bspline,
};
use super::profile::{
    CrossSectionProfile, cap_offset, extrude_cross_section,
    interpolate_profiles, resolve_na_profile, resolve_profile,
};
use super::sheet::{compute_sheet_geometry, interpolate_per_residue_normals};
use super::BackboneVertex;
use crate::options::GeometryOptions;

/// Default nucleic acid backbone color (light blue-violet).
const NA_COLOR: [f32; 3] = [0.45, 0.55, 0.85];

/// Generate unified backbone mesh from protein and nucleic acid chains.
///
/// Returns `(vertices, tube_indices, ribbon_indices, sheet_offsets)`.
pub(crate) fn generate_mesh_colored(
    protein_chains: &[Vec<Vec3>],
    na_chains: &[Vec<Vec3>],
    ss_override: Option<&[SSType]>,
    per_residue_colors: Option<&[[f32; 3]]>,
    geo: &GeometryOptions,
) -> (Vec<BackboneVertex>, Vec<u32>, Vec<u32>, Vec<(u32, Vec3)>) {
    let mut all_vertices: Vec<BackboneVertex> = Vec::new();
    let mut all_tube_indices: Vec<u32> = Vec::new();
    let mut all_ribbon_indices: Vec<u32> = Vec::new();
    let mut all_sheet_offsets: Vec<(u32, Vec3)> = Vec::new();
    let mut global_residue_idx: u32 = 0;

    let spr = geo.segments_per_residue;
    let csv = geo.cross_section_verts;

    // ── Protein chains (N-CA-C triplets) ──

    for chain in protein_chains {
        let ca_positions: Vec<Vec3> = chain
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 3 == 1)
            .map(|(_, &pos)| pos)
            .collect();

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

        let n_positions: Vec<Vec3> = chain
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 3 == 0)
            .map(|(_, &pos)| pos)
            .collect();
        let c_positions: Vec<Vec3> = chain
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 3 == 2)
            .map(|(_, &pos)| pos)
            .collect();

        let profiles: Vec<CrossSectionProfile> = (0..n_residues)
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

        let base_vertex = all_vertices.len() as u32;
        let (verts, tube_inds, ribbon_inds, offsets) =
            generate_protein_chain_mesh(
                &ca_positions,
                &n_positions,
                &c_positions,
                &ss_types,
                &profiles,
                global_residue_idx,
                base_vertex,
                spr,
                csv,
            );

        all_vertices.extend(verts);
        all_tube_indices.extend(tube_inds);
        all_ribbon_indices.extend(ribbon_inds);
        all_sheet_offsets.extend(offsets);
        global_residue_idx += n_residues as u32;
    }

    // ── Nucleic acid chains (P-atom positions) ──

    for chain in na_chains {
        if chain.len() < 2 {
            global_residue_idx += chain.len() as u32;
            continue;
        }

        let n_residues = chain.len();
        let profiles: Vec<CrossSectionProfile> = (0..n_residues)
            .map(|i| {
                resolve_na_profile(
                    global_residue_idx + i as u32,
                    NA_COLOR,
                    geo,
                )
            })
            .collect();

        let base_vertex = all_vertices.len() as u32;
        let (verts, tube_inds, ribbon_inds) = generate_na_chain_mesh(
            chain,
            &profiles,
            base_vertex,
            spr,
            csv,
        );

        all_vertices.extend(verts);
        all_tube_indices.extend(tube_inds);
        all_ribbon_indices.extend(ribbon_inds);
        global_residue_idx += n_residues as u32;
    }

    (
        all_vertices,
        all_tube_indices,
        all_ribbon_indices,
        all_sheet_offsets,
    )
}

// ==================== PROTEIN CHAIN MESH ====================

/// Generate mesh for a single protein chain (with SS detection, sheet
/// geometry, and RMF/radial/sheet normal blending).
fn generate_protein_chain_mesh(
    ca_positions: &[Vec3],
    n_positions: &[Vec3],
    c_positions: &[Vec3],
    ss_types: &[SSType],
    profiles: &[CrossSectionProfile],
    global_residue_base: u32,
    base_vertex: u32,
    spr: usize,
    csv: usize,
) -> (Vec<BackboneVertex>, Vec<u32>, Vec<u32>, Vec<(u32, Vec3)>) {
    let n = ca_positions.len();
    if n < 2 {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    let (flat_ca, sheet_normals, sheet_offsets) = compute_sheet_geometry(
        ca_positions,
        n_positions,
        c_positions,
        ss_types,
        global_residue_base,
    );

    let spline_points = catmull_rom(&flat_ca, spr);
    let total = spline_points.len();
    if total < 2 {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    let tangents = compute_tangents(&spline_points);

    let helix_centers = compute_helix_axis_points(ca_positions);
    let spline_helix_centers = cubic_bspline(&helix_centers, spr);

    let mut frames = build_frames(&spline_points, &tangents);
    compute_rmf(&mut frames);

    let spline_sheet_normals =
        interpolate_per_residue_normals(&sheet_normals, total, n);
    let spline_profiles = interpolate_profiles(profiles, total, n);

    let final_frames = compute_final_frames(
        &frames,
        &spline_helix_centers,
        &spline_sheet_normals,
        ss_types,
        &spline_profiles,
        total,
        n,
    );

    let (verts, tube_inds, ribbon_inds) =
        extrude_and_index(&final_frames, &spline_profiles, base_vertex, csv);

    (verts, tube_inds, ribbon_inds, sheet_offsets)
}

// ==================== NUCLEIC ACID CHAIN MESH ====================

/// Generate mesh for a single NA chain (P-atom positions, Frenet frames,
/// no sheet geometry).
fn generate_na_chain_mesh(
    positions: &[Vec3],
    profiles: &[CrossSectionProfile],
    base_vertex: u32,
    spr: usize,
    csv: usize,
) -> (Vec<BackboneVertex>, Vec<u32>, Vec<u32>) {
    let n = positions.len();
    if n < 2 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let spline_points = catmull_rom(positions, spr);
    let total = spline_points.len();
    if total < 2 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let tangents = compute_tangents(&spline_points);

    let mut frames = build_frames(&spline_points, &tangents);
    compute_frenet_frames(&mut frames);

    let spline_profiles = interpolate_profiles(profiles, total, n);

    extrude_and_index(&frames, &spline_profiles, base_vertex, csv)
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

/// Build SplinePoint shells (position + tangent, normals zeroed).
fn build_frames(spline: &[Vec3], tangents: &[Vec3]) -> Vec<SplinePoint> {
    spline
        .iter()
        .zip(tangents.iter())
        .map(|(&pos, &tangent)| SplinePoint {
            pos,
            tangent,
            normal: Vec3::ZERO,
            binormal: Vec3::ZERO,
        })
        .collect()
}

/// Extrude cross-sections and generate partitioned indices + end caps.
fn extrude_and_index(
    frames: &[SplinePoint],
    profiles: &[CrossSectionProfile],
    base_vertex: u32,
    csv: usize,
) -> (Vec<BackboneVertex>, Vec<u32>, Vec<u32>) {
    let total = frames.len();
    let mut vertices = Vec::with_capacity(total * csv);
    for (i, frame) in frames.iter().enumerate() {
        let p = &profiles[i];
        extrude_cross_section(
            frame,
            p.width * 0.5,
            p.thickness * 0.5,
            p.roundness,
            p.color,
            p.residue_idx,
            csv,
            &mut vertices,
        );
    }

    let mut tube_indices = Vec::new();
    let mut ribbon_indices = Vec::new();
    generate_partitioned_indices(
        frames, profiles, base_vertex, csv, &mut tube_indices,
        &mut ribbon_indices,
    );
    generate_end_caps(
        frames, profiles, base_vertex, csv, &mut vertices,
        &mut tube_indices, &mut ribbon_indices,
    );

    (vertices, tube_indices, ribbon_indices)
}

// ==================== NORMAL BLENDING (protein only) ====================

fn compute_final_frames(
    rmf_frames: &[SplinePoint],
    helix_centers: &[Vec3],
    sheet_normals: &[Vec3],
    ss_types: &[SSType],
    profiles: &[CrossSectionProfile],
    total_spline: usize,
    n_residues: usize,
) -> Vec<SplinePoint> {
    let mut result = Vec::with_capacity(total_spline);

    for i in 0..total_spline {
        let frame = &rmf_frames[i];
        let profile = &profiles[i];

        let residue_frac = if total_spline > 1 {
            i as f32 / (total_spline - 1) as f32
        } else {
            0.0
        };
        let residue_idx =
            ((residue_frac * (n_residues - 1) as f32) as usize)
                .min(n_residues - 1);
        let ss = ss_types[residue_idx];

        let tangent = frame.tangent;
        let rmf_normal = frame.normal;

        let radial_normal = if profile.radial_blend > 0.01 {
            let ci = i.min(helix_centers.len().saturating_sub(1));
            let to_surface = frame.pos - helix_centers[ci];
            let radial = (to_surface
                - tangent * tangent.dot(to_surface))
            .normalize_or_zero();
            if radial.length_squared() > 0.01 {
                radial
            } else {
                rmf_normal
            }
        } else {
            rmf_normal
        };

        let sheet_n = sheet_normals[i];
        let has_sheet =
            ss == SSType::Sheet && sheet_n.length_squared() > 0.01;

        let normal = if has_sheet {
            let proj = sheet_n - tangent * sheet_n.dot(tangent);
            if proj.length_squared() > 1e-6 {
                proj.normalize()
            } else {
                rmf_normal
            }
        } else {
            let blended = rmf_normal
                .lerp(radial_normal, profile.radial_blend)
                .normalize_or_zero();
            if blended.length_squared() > 0.01 {
                blended
            } else {
                rmf_normal
            }
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

// ==================== INDEX GENERATION ====================

fn generate_partitioned_indices(
    frames: &[SplinePoint],
    profiles: &[CrossSectionProfile],
    base_vertex: u32,
    csv: usize,
    tube_indices: &mut Vec<u32>,
    ribbon_indices: &mut Vec<u32>,
) {
    if frames.len() < 2 {
        return;
    }

    for i in 0..frames.len() - 1 {
        let is_tube =
            profiles[i].roundness > 0.5 && profiles[i + 1].roundness > 0.5;

        let ring_a = base_vertex + (i * csv) as u32;
        let ring_b = base_vertex + ((i + 1) * csv) as u32;

        for k in 0..csv {
            let k_next = (k + 1) % csv;
            let v0 = ring_a + k as u32;
            let v1 = ring_a + k_next as u32;
            let v2 = ring_b + k as u32;
            let v3 = ring_b + k_next as u32;
            let target = if is_tube {
                &mut *tube_indices
            } else {
                &mut *ribbon_indices
            };
            target.extend_from_slice(&[v0, v2, v1]);
            target.extend_from_slice(&[v1, v2, v3]);
        }
    }
}

fn generate_end_caps(
    frames: &[SplinePoint],
    profiles: &[CrossSectionProfile],
    base_vertex: u32,
    csv: usize,
    vertices: &mut Vec<BackboneVertex>,
    tube_indices: &mut Vec<u32>,
    ribbon_indices: &mut Vec<u32>,
) {
    if frames.len() < 2 {
        return;
    }

    emit_cap(
        &frames[0],
        &profiles[0],
        -frames[0].tangent,
        base_vertex,
        csv,
        vertices,
        tube_indices,
        ribbon_indices,
        false,
    );

    let last = frames.len() - 1;
    emit_cap(
        &frames[last],
        &profiles[last],
        frames[last].tangent,
        base_vertex,
        csv,
        vertices,
        tube_indices,
        ribbon_indices,
        true,
    );
}

fn emit_cap(
    frame: &SplinePoint,
    profile: &CrossSectionProfile,
    cap_normal: Vec3,
    base_vertex: u32,
    csv: usize,
    vertices: &mut Vec<BackboneVertex>,
    tube_indices: &mut Vec<u32>,
    ribbon_indices: &mut Vec<u32>,
    forward: bool,
) {
    let hw = profile.width * 0.5;
    let ht = profile.thickness * 0.5;
    let is_tube = profile.roundness > 0.5;

    let center_idx = base_vertex + vertices.len() as u32;
    vertices.push(BackboneVertex {
        position: frame.pos.into(),
        normal: cap_normal.into(),
        color: profile.color,
        residue_idx: profile.residue_idx,
        center_pos: (frame.pos - cap_normal).into(),
    });

    let edge_base = base_vertex + vertices.len() as u32;
    for k in 0..csv {
        let offset = cap_offset(frame, hw, ht, profile.roundness, csv, k);
        let pos = frame.pos + offset;
        vertices.push(BackboneVertex {
            position: pos.into(),
            normal: cap_normal.into(),
            color: profile.color,
            residue_idx: profile.residue_idx,
            center_pos: (pos - cap_normal).into(),
        });
    }

    let target = if is_tube { tube_indices } else { ribbon_indices };
    for k in 0..csv {
        let k_next = (k + 1) % csv;
        if forward {
            target.extend_from_slice(&[
                center_idx,
                edge_base + k as u32,
                edge_base + k_next as u32,
            ]);
        } else {
            target.extend_from_slice(&[
                center_idx,
                edge_base + k_next as u32,
                edge_base + k as u32,
            ]);
        }
    }
}
