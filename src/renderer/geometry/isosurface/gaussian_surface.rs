//! Gaussian molecular surface mesh rendering.
//!
//! The analytical half — summing atom Gaussians into a scalar voxel
//! field — lives in [`molex::analysis::volumetric::gaussian`]. This
//! file only runs marching cubes on the returned field and formats
//! the result as [`IsosurfaceVertex`].

use glam::Vec3;
use molex::analysis::volumetric::compute_gaussian_field;

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

    let grid = compute_gaussian_field(positions, radii, resolution);
    if grid.data.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let origin = grid.origin;
    let spacing = grid.spacing;

    extract_isosurface(
        &grid.data,
        grid.dims,
        level,
        [0, 0, 0],
        grid.dims,
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
