//! Per-cavity mesh generation from the shared SDF pipeline.
//!
//! Pipeline:
//! 1. Voxelize SAS + probe-carve (same as SES)
//! 2. Detect cavity mask (shared with SES sealing step)
//! 3. Connected-component label the cavity mask → per-cavity IDs + bboxes
//! 4. For each cavity: extract a padded sub-grid, build a per-cavity
//!    binary mask in sub-grid coordinates, run [`binary_to_sdf`] on the
//!    sub-grid, extract marching-cubes triangles
//!
//! Each cavity gets an independent sub-grid SDF, so cavities near each
//! other cannot cross-contaminate each other's meshes. The sub-grids
//! are small (bounded by the cavity size + 1 voxel padding), so the
//! per-cavity SDF cost is negligible compared to the shared SAS rasterization.
//!
//! Metadata (volume, depth, lining residues, buried/pocket flag) lands on
//! [`CavitySet`] in a later pass.

use glam::Vec3;

use super::cpu_marching_cubes::extract_isosurface;
use super::mesh_smooth::taubin_smooth;
use super::sdf_grid::{binary_to_sdf, detect_cavity_mask, edt_3d, voxelize_sas};
use super::IsosurfaceVertex;

/// Number of Taubin smoothing iterations applied to each cavity mesh
/// after marching cubes. Each iteration is one λ pass + one μ pass.
/// Operates on triangles after extraction so it can never lose cavities
/// (unlike SDF-side smoothing, which blurs small features below the
/// iso-threshold and makes them disappear).
const CAVITY_SMOOTHING_ITERATIONS: usize = 8;

/// Default solvent probe radius (water) in Angstroms.
const DEFAULT_PROBE_RADIUS: f32 = 1.4;

/// A single detected cavity with its extracted mesh.
#[derive(Clone)]
pub struct CavityMesh {
    /// Cavity ID (1-based; 0 is reserved for "not a cavity").
    pub id: u32,
    /// Isosurface vertices for this cavity.
    pub vertices: Vec<IsosurfaceVertex>,
    /// Triangle indices into `vertices`.
    pub indices: Vec<u32>,
}

/// A collection of cavities detected in a single pose.
#[derive(Clone, Default)]
pub struct CavitySet {
    /// One entry per distinct cavity, in label order.
    pub meshes: Vec<CavityMesh>,
}

/// Axis-aligned bounding box in voxel coordinates (inclusive on both ends).
#[derive(Debug, Clone, Copy)]
struct VoxelBbox {
    min: [usize; 3],
    max: [usize; 3],
}

/// Generate cavity meshes from atom positions.
///
/// - `positions`: atom world-space positions (Angstroms)
/// - `radii`: per-atom van der Waals radii (Angstroms)
/// - `probe_radius`: solvent probe radius; defaults to 1.4 Å
/// - `resolution`: grid spacing in Angstroms (lower = finer, typ. 0.5–1.0)
/// - `color`: RGBA color for all cavity vertices (alpha for translucency)
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn generate_cavities(
    positions: &[Vec3],
    radii: &[f32],
    probe_radius: Option<f32>,
    resolution: f32,
    color: [f32; 4],
) -> CavitySet {
    if positions.is_empty() {
        return CavitySet::default();
    }

    let probe = probe_radius.unwrap_or(DEFAULT_PROBE_RADIUS);
    let padding = probe + 2.0;

    // Compute grid bounds from atom positions
    let mut min_bound = positions[0];
    let mut max_bound = positions[0];
    for &p in positions {
        min_bound = min_bound.min(p);
        max_bound = max_bound.max(p);
    }
    let max_radius = radii.iter().copied().fold(0.0f32, f32::max);
    let expand = max_radius + probe + padding;
    min_bound -= Vec3::splat(expand);
    max_bound += Vec3::splat(expand);

    let extent = max_bound - min_bound;
    let nx = ((extent.x / resolution).ceil() as usize).max(2);
    let ny = ((extent.y / resolution).ceil() as usize).max(2);
    let nz = ((extent.z / resolution).ceil() as usize).max(2);
    let spacing = [
        extent.x / (nx - 1) as f32,
        extent.y / (ny - 1) as f32,
        extent.z / (nz - 1) as f32,
    ];
    let origin = [min_bound.x, min_bound.y, min_bound.z];
    let dims = [nx, ny, nz];

    // Step 1: Binary SAS solid
    let sas_solid =
        voxelize_sas(positions, radii, probe, dims, &origin, &spacing);

    // Step 2: SES carve — voxels where EDT < probe become outside
    let interior_edt = edt_3d(&sas_solid, dims, &spacing);
    let total = nx * ny * nz;
    let mut ses_solid = vec![false; total];
    for i in 0..total {
        ses_solid[i] = sas_solid[i] && interior_edt[i] >= probe;
    }

    // Step 3: Extract cavity mask (non-solid voxels not reachable from
    // the grid exterior)
    let cavity_mask = detect_cavity_mask(&ses_solid, dims);

    // Step 4: Label connected components → per-cavity IDs + bboxes
    let (labels, bboxes) = label_connected_components(&cavity_mask, dims);
    if bboxes.is_empty() {
        return CavitySet::default();
    }

    // Step 5: Extract one mesh per cavity
    let meshes = bboxes
        .iter()
        .enumerate()
        .filter_map(|(cavity_idx, bbox)| {
            let id = cavity_idx as u32 + 1;
            extract_cavity_mesh(
                id, bbox, &labels, dims, &origin, &spacing, color,
            )
        })
        .collect();

    CavitySet { meshes }
}

/// Extract a single cavity's isosurface mesh on a padded sub-grid.
///
/// Builds a local binary mask containing only voxels whose label matches
/// the target cavity, converts to an SDF, then marching-cubes the
/// sub-grid. The result is in world-space thanks to the per-cavity
/// sub-origin.
#[allow(clippy::too_many_arguments)]
fn extract_cavity_mesh(
    id: u32,
    bbox: &VoxelBbox,
    labels: &[u32],
    dims: [usize; 3],
    origin: &[f32; 3],
    spacing: &[f32; 3],
    color: [f32; 4],
) -> Option<CavityMesh> {
    let [nx, ny, nz] = dims;

    // Pad the bbox by 1 voxel in each direction so marching cubes has a
    // neighboring "outside" cell to close the surface against.
    let sub_min = [
        bbox.min[0].saturating_sub(1),
        bbox.min[1].saturating_sub(1),
        bbox.min[2].saturating_sub(1),
    ];
    let sub_max = [
        (bbox.max[0] + 1).min(nx - 1),
        (bbox.max[1] + 1).min(ny - 1),
        (bbox.max[2] + 1).min(nz - 1),
    ];
    let sub_dims = [
        sub_max[0] - sub_min[0] + 1,
        sub_max[1] - sub_min[1] + 1,
        sub_max[2] - sub_min[2] + 1,
    ];

    // Build per-cavity binary mask in local sub-grid coordinates. Only
    // voxels carrying this cavity's label become solid — other cavities
    // and exterior are treated as outside.
    let sub_total = sub_dims[0] * sub_dims[1] * sub_dims[2];
    let mut sub_mask = vec![false; sub_total];
    let cells = (0..sub_dims[0]).flat_map(|lx| {
        (0..sub_dims[1])
            .flat_map(move |ly| (0..sub_dims[2]).map(move |lz| (lx, ly, lz)))
    });
    for (lx, ly, lz) in cells {
        let gx = sub_min[0] + lx;
        let gy = sub_min[1] + ly;
        let gz = sub_min[2] + lz;
        if labels[gx * ny * nz + gy * nz + gz] == id {
            sub_mask[lx * sub_dims[1] * sub_dims[2] + ly * sub_dims[2] + lz] =
                true;
        }
    }

    // binary_to_sdf returns negative-inside / positive-outside. Negate
    // so inside-cavity is positive (matches the marching-cubes gradient
    // convention used elsewhere in this crate). The voxel-facet
    // appearance gets smoothed away on the triangle side after MC,
    // not by blurring the field — blurring the field would shrink
    // small cavities below the iso-threshold and lose them entirely.
    let mut sub_sdf = binary_to_sdf(&sub_mask, sub_dims, spacing);
    for v in &mut sub_sdf {
        *v = -*v;
    }

    // Sub-grid origin in world coordinates
    let sub_origin = [
        sub_min[0] as f32 * spacing[0] + origin[0],
        sub_min[1] as f32 * spacing[1] + origin[1],
        sub_min[2] as f32 * spacing[2] + origin[2],
    ];

    let (mut vertices, indices) = extract_isosurface(
        &sub_sdf,
        sub_dims,
        0.0,
        [0, 0, 0],
        sub_dims,
        |gx, gy, gz| {
            [
                gx.mul_add(spacing[0], sub_origin[0]),
                gy.mul_add(spacing[1], sub_origin[1]),
                gz.mul_add(spacing[2], sub_origin[2]),
            ]
        },
        color,
    );

    if vertices.is_empty() || indices.is_empty() {
        return None;
    }

    // Smooth voxel facets out of the extracted triangles. Taubin
    // alternating λ/μ passes prevent the volume shrinkage that plain
    // Laplacian smoothing would cause for small cavities.
    taubin_smooth(&mut vertices, &indices, CAVITY_SMOOTHING_ITERATIONS);

    Some(CavityMesh {
        id,
        vertices,
        indices,
    })
}

/// 6-connected flood-fill labeling over a binary cavity mask.
///
/// Returns `(labels, bboxes)` where `labels[i]` is 0 for non-cavity voxels
/// and a positive 1-based cavity ID for cavity voxels. `bboxes[id-1]` is
/// the inclusive voxel-space bounding box of cavity `id`.
fn label_connected_components(
    cavity_mask: &[bool],
    dims: [usize; 3],
) -> (Vec<u32>, Vec<VoxelBbox>) {
    let [nx, ny, nz] = dims;
    let total = nx * ny * nz;
    let idx = |x: usize, y: usize, z: usize| x * ny * nz + y * nz + z;

    let mut labels = vec![0u32; total];
    let mut bboxes: Vec<VoxelBbox> = Vec::new();

    let cells = (0..nx).flat_map(|ix| {
        (0..ny).flat_map(move |iy| (0..nz).map(move |iz| (ix, iy, iz)))
    });
    for (ix, iy, iz) in cells {
        let i = idx(ix, iy, iz);
        if !cavity_mask[i] || labels[i] != 0 {
            continue;
        }
        let label = bboxes.len() as u32 + 1;
        let bbox =
            flood_fill((ix, iy, iz), label, cavity_mask, &mut labels, dims);
        bboxes.push(bbox);
    }

    (labels, bboxes)
}

/// Flood-fill from `start`, marking every connected cavity voxel with
/// `label` and returning the resulting bounding box.
fn flood_fill(
    start: (usize, usize, usize),
    label: u32,
    cavity_mask: &[bool],
    labels: &mut [u32],
    dims: [usize; 3],
) -> VoxelBbox {
    let [nx, ny, nz] = dims;
    let idx = |x: usize, y: usize, z: usize| x * ny * nz + y * nz + z;

    let (sx, sy, sz) = start;
    let mut bbox = VoxelBbox {
        min: [sx, sy, sz],
        max: [sx, sy, sz],
    };
    labels[idx(sx, sy, sz)] = label;

    const NEIGHBORS: [(isize, isize, isize); 6] = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];

    let mut stack = vec![start];
    while let Some((x, y, z)) = stack.pop() {
        bbox.min[0] = bbox.min[0].min(x);
        bbox.min[1] = bbox.min[1].min(y);
        bbox.min[2] = bbox.min[2].min(z);
        bbox.max[0] = bbox.max[0].max(x);
        bbox.max[1] = bbox.max[1].max(y);
        bbox.max[2] = bbox.max[2].max(z);

        for (dx, dy, dz) in NEIGHBORS {
            let nx2 = x as isize + dx;
            let ny2 = y as isize + dy;
            let nz2 = z as isize + dz;
            if nx2 < 0 || ny2 < 0 || nz2 < 0 {
                continue;
            }
            let (nx2, ny2, nz2) =
                (nx2 as usize, ny2 as usize, nz2 as usize);
            if nx2 >= nx || ny2 >= ny || nz2 >= nz {
                continue;
            }
            let j = idx(nx2, ny2, nz2);
            if cavity_mask[j] && labels[j] == 0 {
                labels[j] = label;
                stack.push((nx2, ny2, nz2));
            }
        }
    }

    bbox
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lin(dims: [usize; 3], x: usize, y: usize, z: usize) -> usize {
        x * dims[1] * dims[2] + y * dims[2] + z
    }

    fn mask_from<const N: usize>(
        dims: [usize; 3],
        cells: [(usize, usize, usize); N],
    ) -> Vec<bool> {
        let mut m = vec![false; dims[0] * dims[1] * dims[2]];
        for (x, y, z) in cells {
            m[lin(dims, x, y, z)] = true;
        }
        m
    }

    #[test]
    fn label_empty_mask() {
        let dims = [4usize; 3];
        let mask = vec![false; 64];
        let (labels, bboxes) = label_connected_components(&mask, dims);
        assert!(labels.iter().all(|&l| l == 0));
        assert!(bboxes.is_empty());
    }

    #[test]
    fn label_single_cavity() {
        let dims = [4usize; 3];
        let mask = mask_from(dims, [(1, 1, 1), (1, 1, 2), (1, 2, 1)]);
        let (labels, bboxes) = label_connected_components(&mask, dims);
        assert_eq!(bboxes.len(), 1);
        assert_eq!(labels[lin(dims, 1, 1, 1)], 1);
        assert_eq!(labels[lin(dims, 1, 1, 2)], 1);
        assert_eq!(labels[lin(dims, 1, 2, 1)], 1);
        assert_eq!(bboxes[0].min, [1, 1, 1]);
        assert_eq!(bboxes[0].max, [1, 2, 2]);
    }

    #[test]
    fn label_two_separated_cavities() {
        // Two single-voxel cavities with a gap between them
        let dims = [5usize; 3];
        let mask = mask_from(dims, [(1, 1, 1), (3, 3, 3)]);
        let (labels, bboxes) = label_connected_components(&mask, dims);
        assert_eq!(bboxes.len(), 2);
        assert_eq!(labels[lin(dims, 1, 1, 1)], 1);
        assert_eq!(labels[lin(dims, 3, 3, 3)], 2);
        assert_eq!(bboxes[0].min, [1, 1, 1]);
        assert_eq!(bboxes[1].min, [3, 3, 3]);
    }

    #[test]
    fn label_adjacent_cavities_merge() {
        // Two cavity voxels that share a face should share a label
        let dims = [4usize; 3];
        let mask = mask_from(dims, [(1, 1, 1), (2, 1, 1)]);
        let (labels, bboxes) = label_connected_components(&mask, dims);
        assert_eq!(bboxes.len(), 1);
        assert_eq!(labels[lin(dims, 1, 1, 1)], 1);
        assert_eq!(labels[lin(dims, 2, 1, 1)], 1);
        assert_eq!(bboxes[0].min, [1, 1, 1]);
        assert_eq!(bboxes[0].max, [2, 1, 1]);
    }

    #[test]
    fn generate_cavities_empty_atoms() {
        let set = generate_cavities(&[], &[], None, 1.0, [1.0; 4]);
        assert!(set.meshes.is_empty());
    }

    #[test]
    fn generate_cavities_single_atom_has_none() {
        // A lone atom is a solid blob with no interior voids.
        let set = generate_cavities(
            &[Vec3::ZERO],
            &[1.5],
            Some(1.4),
            0.5,
            [1.0; 4],
        );
        assert!(set.meshes.is_empty());
    }
}
