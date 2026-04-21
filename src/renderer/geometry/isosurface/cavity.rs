//! Per-cavity mesh generation from molex's cavity detector.
//!
//! The analytical half of this module — voxelization, SES erosion,
//! cavity mask flood fill, connected-component labeling, per-cavity
//! sub-grid construction — lives in
//! [`molex::analysis::volumetric::cavity`]. This file only covers the
//! rendering-bound half: taking each `DetectedCavity` sub-grid mask,
//! converting it to an SDF, running marching cubes, smoothing the
//! triangles, and baking cavity-specific vertex attributes
//! (`CAVITY_RGBA`, `cavity_center`).

use molex::analysis::volumetric::{
    binary_to_sdf, detect_cavities, DetectedCavity,
};

use super::cpu_marching_cubes::extract_isosurface;
use super::mesh_smooth::taubin_smooth;
use super::{isosurface_kind, IsosurfaceVertex};

/// Number of Taubin smoothing iterations applied to each cavity mesh
/// after marching cubes. Each iteration is one λ pass + one μ pass.
/// Operates on triangles after extraction so it can never lose cavities
/// (unlike SDF-side smoothing, which blurs small features below the
/// iso-threshold and makes them disappear).
const CAVITY_SMOOTHING_ITERATIONS: usize = 8;

/// Unified RGBA tint baked into every cavity vertex. All cavities share
/// this color so the visual reads as a property of the negative space
/// itself, not of any particular chain. The alpha is a baseline that
/// gets modulated by Beer-Lambert thickness in the fragment shader.
const CAVITY_RGBA: [f32; 4] = [0.22, 0.30, 1.0, 0.90];

/// A single detected cavity with its extracted mesh.
#[derive(Clone)]
pub struct CavityMesh {
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

/// Generate cavity meshes from atom positions.
///
/// Delegates detection to [`molex::analysis::volumetric::detect_cavities`]
/// (voxelization + SES erosion + connected components) and wraps each
/// returned [`DetectedCavity`] with a per-cavity mesh extracted via
/// marching cubes + Taubin smoothing.
///
/// All cavities share the unified [`CAVITY_RGBA`] tint — the color is
/// not a parameter because cavities are meant to read as "negative
/// space", not as a per-entity visual.
///
/// - `positions`: atom world-space positions (Angstroms)
/// - `radii`: per-atom van der Waals radii (Angstroms)
/// - `probe_radius`: solvent probe radius; defaults to 1.4 Å
/// - `resolution`: grid spacing in Angstroms (lower = finer, typ. 0.5–1.0)
#[must_use]
pub fn generate_cavities(
    positions: &[glam::Vec3],
    radii: &[f32],
    probe_radius: Option<f32>,
    resolution: f32,
) -> CavitySet {
    let detected = detect_cavities(positions, radii, probe_radius, resolution);

    let meshes = detected.iter().filter_map(extract_cavity_mesh).collect();

    CavitySet { meshes }
}

/// Extract a single cavity's isosurface mesh on its sub-grid.
///
/// `binary_to_sdf` returns negative-inside / positive-outside. Negate
/// so inside-cavity is positive (matches the marching-cubes gradient
/// convention used elsewhere in this crate). The voxel-facet
/// appearance gets smoothed away on the triangle side after marching
/// cubes, not by blurring the field — blurring the field would shrink
/// small cavities below the iso-threshold and lose them entirely.
fn extract_cavity_mesh(cavity: &DetectedCavity) -> Option<CavityMesh> {
    let mut sub_sdf =
        binary_to_sdf(&cavity.sub_mask, cavity.sub_dims, &cavity.spacing);
    for v in &mut sub_sdf {
        *v = -*v;
    }

    let (mut vertices, indices) = extract_isosurface(
        &sub_sdf,
        cavity.sub_dims,
        0.0,
        [0, 0, 0],
        cavity.sub_dims,
        |gx, gy, gz| {
            [
                gx.mul_add(cavity.spacing[0], cavity.sub_origin[0]),
                gy.mul_add(cavity.spacing[1], cavity.sub_origin[1]),
                gz.mul_add(cavity.spacing[2], cavity.sub_origin[2]),
            ]
        },
        CAVITY_RGBA,
    );

    if vertices.is_empty() || indices.is_empty() {
        return None;
    }

    taubin_smooth(&mut vertices, &indices, CAVITY_SMOOTHING_ITERATIONS);

    // Tag every vertex as a cavity (so the isosurface shader can apply
    // cavity-specific effects without inspecting color) and bake the
    // centroid for radial-breath displacement.
    for v in &mut vertices {
        v.kind = isosurface_kind::CAVITY;
        v.cavity_center = cavity.centroid;
    }

    Some(CavityMesh { vertices, indices })
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;

    #[test]
    fn generate_cavities_empty_atoms() {
        let set = generate_cavities(&[], &[], None, 1.0);
        assert!(set.meshes.is_empty());
    }

    #[test]
    fn generate_cavities_single_atom_has_none() {
        // A lone atom is a solid blob with no interior voids.
        let set = generate_cavities(&[Vec3::ZERO], &[1.5], Some(1.4), 0.5);
        assert!(set.meshes.is_empty());
    }
}
