//! Sheet-specific backbone geometry: peptide-plane normals, iterative
//! flattening (PyMOL-style), and sidechain offset computation.

use glam::Vec3;
use molex::SSType;

/// Segment a residue SS-type array into contiguous runs.
#[derive(Debug)]
pub(crate) struct SSSegment {
    pub(crate) ss_type: SSType,
    pub(crate) start_residue: usize,
    pub(crate) end_residue: usize,
}

pub(crate) fn segment_by_ss(ss_types: &[SSType]) -> Vec<SSSegment> {
    if ss_types.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut current = ss_types[0];
    let mut start = 0;

    for (i, &ss) in ss_types.iter().enumerate() {
        if ss != current {
            segments.push(SSSegment {
                ss_type: current,
                start_residue: start,
                end_residue: i,
            });
            current = ss;
            start = i;
        }
    }
    segments.push(SSSegment {
        ss_type: current,
        start_residue: start,
        end_residue: ss_types.len(),
    });

    segments
}

/// Compute sheet-specific geometry: flattened CA positions, peptide-plane
/// normals, and position offsets for sidechain adjustment.
///
/// Returns `(flat_ca, normals, sheet_offsets)`.
pub(crate) fn compute_sheet_geometry(
    ca_positions: &[Vec3],
    n_positions: &[Vec3],
    c_positions: &[Vec3],
    ss_types: &[SSType],
    global_residue_base: u32,
    sheet_plane_normals: &[(u32, Vec3)],
) -> (Vec<Vec3>, Vec<Vec3>, Vec<(u32, Vec3)>) {
    let n = ca_positions.len();
    let mut flat_ca = ca_positions.to_vec();
    let mut normals = vec![Vec3::ZERO; n];
    let mut offsets = Vec::new();

    // Start from per-residue peptide plane normals: cross(CA→N, CA→C).
    // These are used everywhere the fitted sheet plane is absent.
    for i in 0..n {
        if i < n_positions.len() && i < c_positions.len() {
            let ca_n = (n_positions[i] - ca_positions[i]).normalize_or_zero();
            let ca_c = (c_positions[i] - ca_positions[i]).normalize_or_zero();
            let normal = ca_n.cross(ca_c);
            normals[i] = if normal.length_squared() > 1e-6 {
                normal.normalize()
            } else {
                Vec3::Y
            };
        }
    }

    // Ensure consistent orientation of the local peptide-plane normals
    // along the chain (within-strand sidechain alternation gives raw
    // normals that flip every residue — fix that by propagating sign).
    for i in 1..n {
        if normals[i].dot(normals[i - 1]) < 0.0 {
            normals[i] = -normals[i];
        }
    }

    // Overlay fitted sheet-plane normals where available. Each residue
    // in a multi-strand β-sheet gets the sheet's global plane normal
    // instead of its local peptide plane, so every strand in the sheet
    // shares one face direction. `sheet_plane_normals` is already
    // sliced to this chain's [base, base+n) window and sorted by
    // global residue idx.
    for &(global_idx, normal) in sheet_plane_normals {
        if global_idx < global_residue_base {
            continue;
        }
        let local = (global_idx - global_residue_base) as usize;
        if local < n {
            normals[local] = normal;
        }
    }

    // Find sheet segments and apply flattening
    let segments = segment_by_ss(ss_types);
    for seg in &segments {
        if seg.ss_type != SSType::Sheet {
            continue;
        }
        let start = seg.start_residue;
        let end = seg.end_residue.min(n);
        if end <= start + 1 {
            continue;
        }

        let mut seg_pos = flat_ca[start..end].to_vec();
        let mut seg_normals = normals[start..end].to_vec();
        flatten_sheet(&mut seg_pos, &mut seg_normals, 4);

        for (j, i) in (start..end).enumerate() {
            let offset = seg_pos[j] - ca_positions[i];
            flat_ca[i] = seg_pos[j];
            normals[i] = seg_normals[j];
            offsets.push((global_residue_base + i as u32, offset));
        }
    }

    (flat_ca, normals, offsets)
}

/// Iterative flattening of sheet positions and normals (PyMOL-style).
///
/// Each cycle averages each point/normal with its neighbors using a
/// weighted kernel (1,2,1)/4, then re-orthogonalizes the normal against
/// the backbone tangent.
fn flatten_sheet(positions: &mut [Vec3], normals: &mut [Vec3], cycles: usize) {
    let n = positions.len();
    if n < 3 {
        return;
    }

    for _ in 0..cycles {
        // Average positions with neighbors (skip endpoints)
        let mut new_pos = positions.to_vec();
        for i in 1..n - 1 {
            new_pos[i] =
                (positions[i - 1] + positions[i] * 2.0 + positions[i + 1])
                    * 0.25;
        }
        positions.copy_from_slice(&new_pos);

        // Average normals with neighbors (skip endpoints)
        let mut new_normals = normals.to_vec();
        for i in 1..n - 1 {
            let avg = normals[i - 1] + normals[i] * 2.0 + normals[i + 1];
            new_normals[i] = if avg.length_squared() > 1e-6 {
                avg.normalize()
            } else {
                normals[i]
            };
        }
        normals.copy_from_slice(&new_normals);

        // Re-orthogonalize normals against backbone tangent
        for i in 1..n - 1 {
            let tangent =
                (positions[i + 1] - positions[i - 1]).normalize_or_zero();
            let proj = normals[i] - tangent * normals[i].dot(tangent);
            normals[i] = if proj.length_squared() > 1e-6 {
                proj.normalize()
            } else {
                normals[i]
            };
        }
    }
}

/// Interpolate per-residue normals to spline resolution.
pub(crate) fn interpolate_per_residue_normals(
    normals: &[Vec3],
    total_spline: usize,
    n_residues: usize,
) -> Vec<Vec3> {
    (0..total_spline)
        .map(|i| {
            let frac = i as f32 / (total_spline - 1).max(1) as f32;
            let rf = frac * (n_residues - 1) as f32;
            let r0 = (rf.floor() as usize).min(n_residues - 1);
            let r1 = (r0 + 1).min(n_residues - 1);
            let t = rf - r0 as f32;
            normals[r0].lerp(normals[r1], t).normalize_or_zero()
        })
        .collect()
}
