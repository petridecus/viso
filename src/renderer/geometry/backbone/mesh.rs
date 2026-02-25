//! Backbone mesh generation: ties spline, profile, and sheet modules together
//! into final vertex/index buffers for both protein and nucleic acid chains.

use foldit_conv::secondary_structure::{resolve, DetectionInput, SSType};
use glam::Vec3;

use super::path::{compute_sheet_geometry, interpolate_per_residue_normals};
use super::profile::{
    cap_offset, extrude_cross_section, interpolate_profiles,
    resolve_na_profile, resolve_profile, CrossSectionProfile,
};
use super::spline::{
    catmull_rom, compute_frenet_frames, compute_helix_axis_points, compute_rmf,
    cubic_bspline, SplinePoint,
};
use super::{BackboneMeshOutput, BackboneVertex};
use crate::options::GeometryOptions;

/// Per-chain index range and bounding sphere for frustum culling.
#[derive(Clone, Debug)]
pub struct ChainRange {
    pub tube_index_start: u32,
    pub tube_index_end: u32,
    pub ribbon_index_start: u32,
    pub ribbon_index_end: u32,
    pub bounding_center: Vec3,
    pub bounding_radius: f32,
}

/// Mesh generation parameters that always travel together.
struct MeshParams {
    base_vertex: u32,
    cross_section_verts: usize,
    segments_per_residue: usize,
}

/// Mutable mesh generation context: parameters + output buffers.
struct MeshWriter<'a> {
    params: &'a MeshParams,
    vertices: &'a mut Vec<BackboneVertex>,
    tube_indices: &'a mut Vec<u32>,
    ribbon_indices: &'a mut Vec<u32>,
}

/// Default nucleic acid backbone color (light blue-violet).
const NA_COLOR: [f32; 3] = [0.45, 0.55, 0.85];

/// Generate unified backbone mesh from protein and nucleic acid chains.
pub(crate) fn generate_mesh_colored(
    chains: &super::ChainPair,
    ss_override: Option<&[SSType]>,
    per_residue_colors: Option<&[[f32; 3]]>,
    geo: &GeometryOptions,
    per_chain_lod: Option<&[(usize, usize)]>,
) -> BackboneMeshOutput {
    let (mut out, global_residue_idx) = process_protein_chains(
        chains.protein,
        ss_override,
        per_residue_colors,
        geo,
        per_chain_lod,
    );
    let na_lod = per_chain_lod.map(|l| &l[chains.protein.len()..]);
    process_na_chains(chains.na, geo, na_lod, &mut out, global_residue_idx);

    out
}

fn process_protein_chains(
    chains: &[Vec<Vec3>],
    ss_override: Option<&[SSType]>,
    per_residue_colors: Option<&[[f32; 3]]>,
    geo: &GeometryOptions,
    per_chain_lod: Option<&[(usize, usize)]>,
) -> (BackboneMeshOutput, u32) {
    let mut out = BackboneMeshOutput::default();
    let mut global_residue_idx: u32 = 0;
    let spr = geo.segments_per_residue;
    let csv = geo.cross_section_verts;

    for (chain_idx, chain) in chains.iter().enumerate() {
        let atoms = ChainAtoms::from_interleaved(chain);

        if atoms.ca.len() < 2 {
            global_residue_idx += atoms.ca.len() as u32;
            continue;
        }

        let n_residues = atoms.ca.len();
        let chain_override = ss_override.and_then(|o| {
            let start = global_residue_idx as usize;
            let end = (start + n_residues).min(o.len());
            (start < o.len()).then(|| &o[start..end])
        });
        let ss_types =
            resolve(chain_override, DetectionInput::CaPositions(&atoms.ca));

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

        let (chain_spr, chain_csv) =
            per_chain_lod.map_or((spr, csv), |l| l[chain_idx]);

        let params = MeshParams {
            base_vertex: out.vertices.len() as u32,
            cross_section_verts: chain_csv,
            segments_per_residue: chain_spr,
        };
        let (center, radius) = bounding_sphere(&atoms.ca);
        let chain_mesh = generate_protein_chain_mesh(
            &atoms,
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
    chains: &[Vec<Vec3>],
    geo: &GeometryOptions,
    per_chain_lod: Option<&[(usize, usize)]>,
    out: &mut BackboneMeshOutput,
    mut global_residue_idx: u32,
) {
    let spr = geo.segments_per_residue;
    let csv = geo.cross_section_verts;

    for (na_idx, chain) in chains.iter().enumerate() {
        if chain.len() < 2 {
            global_residue_idx += chain.len() as u32;
            continue;
        }

        let n_residues = chain.len();
        let profiles: Vec<CrossSectionProfile> = (0..n_residues)
            .map(|i| {
                resolve_na_profile(global_residue_idx + i as u32, NA_COLOR, geo)
            })
            .collect();

        let (chain_spr, chain_csv) =
            per_chain_lod.map_or((spr, csv), |l| l[na_idx]);

        let params = MeshParams {
            base_vertex: out.vertices.len() as u32,
            cross_section_verts: chain_csv,
            segments_per_residue: chain_spr,
        };
        let (center, radius) = bounding_sphere(chain);
        let chain_mesh = generate_na_chain_mesh(chain, &profiles, &params);
        out.push_chain(chain_mesh, center, radius);

        global_residue_idx += n_residues as u32;
    }
}

/// Compute bounding sphere (centroid + max distance) from a set of positions.
fn bounding_sphere(positions: &[Vec3]) -> (Vec3, f32) {
    if positions.is_empty() {
        return (Vec3::ZERO, 0.0);
    }
    let center =
        positions.iter().copied().sum::<Vec3>() / positions.len() as f32;
    let radius = positions
        .iter()
        .map(|p| (*p - center).length())
        .fold(0.0f32, f32::max);
    (center, radius)
}

// ==================== PROTEIN CHAIN MESH ====================

/// Owned backbone atom positions split from an interleaved N-CA-C chain.
struct ChainAtoms {
    n: Vec<Vec3>,
    ca: Vec<Vec3>,
    c: Vec<Vec3>,
}

impl ChainAtoms {
    /// Split a flat interleaved `[N, CA, C, N, CA, C, ...]` chain into
    /// separate N, CA, and C position vectors.
    fn from_interleaved(chain: &[Vec3]) -> Self {
        let cap = chain.len() / 3;
        let mut n = Vec::with_capacity(cap);
        let mut ca = Vec::with_capacity(cap);
        let mut c = Vec::with_capacity(cap);
        for (i, &pos) in chain.iter().enumerate() {
            match i % 3 {
                0 => n.push(pos),
                1 => ca.push(pos),
                _ => c.push(pos),
            }
        }
        Self { n, ca, c }
    }
}

/// Generate mesh for a single protein chain (with SS detection, sheet
/// geometry, and RMF/radial/sheet normal blending).
fn generate_protein_chain_mesh(
    atoms: &ChainAtoms,
    ss_types: &[SSType],
    profiles: &[CrossSectionProfile],
    global_residue_base: u32,
    params: &MeshParams,
) -> BackboneMeshOutput {
    let n = atoms.ca.len();
    if n < 2 {
        return BackboneMeshOutput::default();
    }

    let (flat_ca, sheet_normals, sheet_offsets) = compute_sheet_geometry(
        &atoms.ca,
        &atoms.n,
        &atoms.c,
        ss_types,
        global_residue_base,
    );

    let spr = params.segments_per_residue;
    let spline_points = catmull_rom(&flat_ca, spr);
    let total = spline_points.len();
    if total < 2 {
        return BackboneMeshOutput::default();
    }

    let tangents = compute_tangents(&spline_points);

    let helix_centers = compute_helix_axis_points(&atoms.ca);
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
    );

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
    let spline_points = catmull_rom(positions, spr);
    let total = spline_points.len();
    if total < 2 {
        return BackboneMeshOutput::default();
    }

    let tangents = compute_tangents(&spline_points);

    let mut frames = build_frames(&spline_points, &tangents);
    compute_frenet_frames(&mut frames);

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
    params: &MeshParams,
) -> (Vec<BackboneVertex>, Vec<u32>, Vec<u32>) {
    let csv = params.cross_section_verts;
    let total = frames.len();
    let mut vertices = Vec::with_capacity(total * csv);
    for (i, frame) in frames.iter().enumerate() {
        extrude_cross_section(frame, &profiles[i], csv, &mut vertices);
    }

    let mut tube_indices = Vec::new();
    let mut ribbon_indices = Vec::new();
    generate_partitioned_indices(
        frames,
        profiles,
        params,
        &mut tube_indices,
        &mut ribbon_indices,
    );
    let mut writer = MeshWriter {
        params,
        vertices: &mut vertices,
        tube_indices: &mut tube_indices,
        ribbon_indices: &mut ribbon_indices,
    };
    generate_end_caps(frames, profiles, &mut writer);

    (vertices, tube_indices, ribbon_indices)
}

// ==================== NORMAL BLENDING (protein only) ====================

fn compute_final_frames(
    rmf_frames: &[SplinePoint],
    helix_centers: &[Vec3],
    sheet_normals: &[Vec3],
    ss_types: &[SSType],
    profiles: &[CrossSectionProfile],
) -> Vec<SplinePoint> {
    let total_spline = rmf_frames.len();
    let n_residues = ss_types.len();
    let mut result = Vec::with_capacity(total_spline);

    for i in 0..total_spline {
        let frame = &rmf_frames[i];
        let profile = &profiles[i];

        let residue_frac = if total_spline > 1 {
            i as f32 / (total_spline - 1) as f32
        } else {
            0.0
        };
        let residue_idx = ((residue_frac * (n_residues - 1) as f32) as usize)
            .min(n_residues - 1);
        let ss = ss_types[residue_idx];

        let tangent = frame.tangent;
        let rmf_normal = frame.normal;

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

        let sheet_n = sheet_normals[i];
        let has_sheet = ss == SSType::Sheet && sheet_n.length_squared() > 0.01;

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
    params: &MeshParams,
    tube_indices: &mut Vec<u32>,
    ribbon_indices: &mut Vec<u32>,
) {
    if frames.len() < 2 {
        return;
    }

    let base_vertex = params.base_vertex;
    let csv = params.cross_section_verts;

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
    w: &mut MeshWriter,
) {
    if frames.len() < 2 {
        return;
    }

    emit_cap(&frames[0], &profiles[0], -frames[0].tangent, w, false);

    let last = frames.len() - 1;
    emit_cap(
        &frames[last],
        &profiles[last],
        frames[last].tangent,
        w,
        true,
    );
}

fn emit_cap(
    frame: &SplinePoint,
    profile: &CrossSectionProfile,
    cap_normal: Vec3,
    w: &mut MeshWriter,
    forward: bool,
) {
    let is_tube = profile.roundness > 0.5;
    let base_vertex = w.params.base_vertex;
    let csv = w.params.cross_section_verts;

    let center_idx = base_vertex + w.vertices.len() as u32;
    w.vertices.push(BackboneVertex {
        position: frame.pos.into(),
        normal: cap_normal.into(),
        color: profile.color,
        residue_idx: profile.residue_idx,
        center_pos: (frame.pos - cap_normal).into(),
    });

    let edge_base = base_vertex + w.vertices.len() as u32;
    for k in 0..csv {
        let offset = cap_offset(frame, profile, csv, k);
        let pos = frame.pos + offset;
        w.vertices.push(BackboneVertex {
            position: pos.into(),
            normal: cap_normal.into(),
            color: profile.color,
            residue_idx: profile.residue_idx,
            center_pos: (pos - cap_normal).into(),
        });
    }

    let target = if is_tube {
        &mut *w.tube_indices
    } else {
        &mut *w.ribbon_indices
    };
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
