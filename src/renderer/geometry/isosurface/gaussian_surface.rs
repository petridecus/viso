//! Gaussian molecular surface generation.
//!
//! Each atom contributes a 3D Gaussian blob to a scalar field. The
//! isosurface of the summed field produces a smooth, blobby surface
//! that envelops the molecule. Resolution controls smoothness — low
//! values give a coarse overview, high values resolve individual atoms.

use glam::Vec3;

use super::cpu_marching_cubes::extract_isosurface;
use super::IsosurfaceVertex;

/// Generate a Gaussian molecular surface from atom positions.
///
/// - `positions`: atom world-space positions (Angstroms)
/// - `radii`: per-atom van der Waals radii (Angstroms)
/// - `resolution`: grid spacing in Angstroms (lower = finer, typ. 0.5–2.0)
/// - `level`: isosurface threshold (default ~0.5)
/// - `color`: RGBA color for the surface
///
/// Returns `(vertices, indices)` for an indexed triangle mesh.
pub fn generate_gaussian_surface(
    positions: &[Vec3],
    radii: &[f32],
    resolution: f32,
    level: f32,
    color: [f32; 4],
) -> (Vec<IsosurfaceVertex>, Vec<u32>) {
    if positions.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let padding = 3.0 * resolution;
    let (grid, dims, origin, spacing) =
        build_gaussian_field(positions, radii, resolution, padding);

    extract_isosurface(
        &grid,
        dims,
        level,
        [0, 0, 0],
        dims,
        |gx, gy, gz| {
            [
                gx.mul_add(spacing[0], origin[0]),
                gy.mul_add(spacing[1], origin[1]),
                gz.mul_add(spacing[2], origin[2]),
            ]
        },
        color,
    )
}

/// Build a 3D scalar field by summing Gaussian contributions from all
/// atoms. Each atom's Gaussian has width proportional to its vdW
/// radius.
fn build_gaussian_field(
    positions: &[Vec3],
    radii: &[f32],
    resolution: f32,
    padding: f32,
) -> (Vec<f32>, [usize; 3], [f32; 3], [f32; 3]) {
    // Compute bounding box
    let mut min = positions[0];
    let mut max = positions[0];
    for &p in positions {
        min = min.min(p);
        max = max.max(p);
    }
    // Expand by max radius + padding
    let max_radius = radii.iter().copied().fold(0.0f32, f32::max);
    let expand = max_radius + padding;
    min -= Vec3::splat(expand);
    max += Vec3::splat(expand);

    let extent = max - min;
    let nx = ((extent.x / resolution).ceil() as usize).max(2);
    let ny = ((extent.y / resolution).ceil() as usize).max(2);
    let nz = ((extent.z / resolution).ceil() as usize).max(2);
    let spacing = [
        extent.x / (nx - 1) as f32,
        extent.y / (ny - 1) as f32,
        extent.z / (nz - 1) as f32,
    ];
    let origin = [min.x, min.y, min.z];

    let mut grid = vec![0.0f32; nx * ny * nz];

    // For each atom, splat its Gaussian onto nearby grid cells
    for (i, &pos) in positions.iter().enumerate() {
        splat_gaussian(
            &mut grid,
            [nx, ny, nz],
            &origin,
            &spacing,
            pos,
            radii[i],
        );
    }

    (grid, [nx, ny, nz], origin, spacing)
}

/// Splat a single atom's Gaussian blob onto the grid.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn splat_gaussian(
    grid: &mut [f32],
    dims: [usize; 3],
    origin: &[f32; 3],
    spacing: &[f32; 3],
    pos: Vec3,
    vdw_radius: f32,
) {
    let [nx, ny, nz] = dims;
    let sigma = vdw_radius * 0.7;
    let inv_2sigma2 = 1.0 / (2.0 * sigma * sigma);
    let cutoff = 3.0 * sigma;
    let cutoff2 = cutoff * cutoff;
    let amplitude = vdw_radius;

    let gx0 =
        ((pos.x - cutoff - origin[0]) / spacing[0]).floor().max(0.0) as usize;
    let gy0 =
        ((pos.y - cutoff - origin[1]) / spacing[1]).floor().max(0.0) as usize;
    let gz0 =
        ((pos.z - cutoff - origin[2]) / spacing[2]).floor().max(0.0) as usize;
    let gx1 = (((pos.x + cutoff - origin[0]) / spacing[0]).ceil() as usize)
        .min(nx - 1);
    let gy1 = (((pos.y + cutoff - origin[1]) / spacing[1]).ceil() as usize)
        .min(ny - 1);
    let gz1 = (((pos.z + cutoff - origin[2]) / spacing[2]).ceil() as usize)
        .min(nz - 1);

    for ix in gx0..=gx1 {
        let dx = origin[0] + ix as f32 * spacing[0] - pos.x;
        for iy in gy0..=gy1 {
            let dy = origin[1] + iy as f32 * spacing[1] - pos.y;
            let dxy2 = dx * dx + dy * dy;
            if dxy2 > cutoff2 {
                continue;
            }
            for iz in gz0..=gz1 {
                let dz = origin[2] + iz as f32 * spacing[2] - pos.z;
                let r2 = dxy2 + dz * dz;
                if r2 <= cutoff2 {
                    grid[ix * ny * nz + iy * nz + iz] +=
                        amplitude * (-r2 * inv_2sigma2).exp();
                }
            }
        }
    }
}
