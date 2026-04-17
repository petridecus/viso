//! Solvent-Excluded Surface (SES / Connolly surface) mesh rendering.
//!
//! The analytical half — voxelization, SES erosion, cavity sealing,
//! signed distance field — lives in
//! [`molex::analysis::volumetric::ses`]. This file only covers the
//! rendering-bound half: marching cubes on the returned SDF and
//! per-vertex mean-curvature coloring (blue = convex, red = concave).

use glam::Vec3;
use molex::analysis::volumetric::compute_ses_sdf;

use super::cpu_marching_cubes::extract_isosurface;
use super::IsosurfaceVertex;

/// Generate a solvent-excluded (Connolly) surface from atom positions.
///
/// Vertices are colored by mean curvature: blue (convex) → white (flat)
/// → red (concave). The `color` alpha channel controls opacity.
pub fn generate_ses(
    positions: &[Vec3],
    radii: &[f32],
    probe_radius: Option<f32>,
    resolution: f32,
    color: [f32; 4],
) -> (Vec<IsosurfaceVertex>, Vec<u32>) {
    if positions.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let grid = compute_ses_sdf(positions, radii, probe_radius, resolution);
    if grid.data.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // molex returns the standard SDF convention (negative inside,
    // positive outside). Our marching cubes extractor expects
    // positive-inside to match the density pipeline, so flip the sign.
    let mut sdf = grid.data;
    for v in &mut sdf {
        *v = -*v;
    }

    let origin = grid.origin;
    let spacing = grid.spacing;

    let (mut verts, idxs) = extract_isosurface(
        &sdf,
        grid.dims,
        0.0,
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
    );

    apply_curvature_coloring(&mut verts, &idxs, color[3]);

    (verts, idxs)
}

/// Compute per-vertex mean curvature via the Laplacian–Beltrami
/// cotangent-weight formula and map to a blue→white→red color scale.
///
/// Mean curvature H > 0 = convex (blue), H < 0 = concave (red).
fn apply_curvature_coloring(
    vertices: &mut [IsosurfaceVertex],
    indices: &[u32],
    alpha: f32,
) {
    if vertices.is_empty() || indices.is_empty() {
        return;
    }

    let n = vertices.len();

    // Accumulate cotangent-weighted Laplacian per vertex
    let mut laplacian = vec![[0.0f32; 3]; n];
    let mut area = vec![0.0f32; n];

    for tri in indices.chunks_exact(3) {
        let (ai, bi, ci) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let a = Vec3::from(vertices[ai].position);
        let b = Vec3::from(vertices[bi].position);
        let c = Vec3::from(vertices[ci].position);

        let ab = b - a;
        let ac = c - a;
        let bc = c - b;

        // Cotangents of each angle
        let cot_a = safe_cot(ab, ac);
        let cot_b = safe_cot(-ab, bc);
        let cot_c = safe_cot(-ac, -bc);

        // Triangle area (for mixed Voronoi area estimate)
        let tri_area = ab.cross(ac).length() * 0.5;
        let third_area = tri_area / 3.0;

        // Accumulate cotangent-weighted displacement for each edge.
        // Edge bc opposite vertex a: contributes to b and c
        let d_bc = c - b;
        laplacian[bi][0] += cot_a * d_bc.x;
        laplacian[bi][1] += cot_a * d_bc.y;
        laplacian[bi][2] += cot_a * d_bc.z;
        laplacian[ci][0] -= cot_a * d_bc.x;
        laplacian[ci][1] -= cot_a * d_bc.y;
        laplacian[ci][2] -= cot_a * d_bc.z;

        // Edge ac opposite vertex b: contributes to a and c
        let d_ac = c - a;
        laplacian[ai][0] += cot_b * d_ac.x;
        laplacian[ai][1] += cot_b * d_ac.y;
        laplacian[ai][2] += cot_b * d_ac.z;
        laplacian[ci][0] -= cot_b * d_ac.x;
        laplacian[ci][1] -= cot_b * d_ac.y;
        laplacian[ci][2] -= cot_b * d_ac.z;

        // Edge ab opposite vertex c: contributes to a and b
        let d_ab = b - a;
        laplacian[ai][0] += cot_c * d_ab.x;
        laplacian[ai][1] += cot_c * d_ab.y;
        laplacian[ai][2] += cot_c * d_ab.z;
        laplacian[bi][0] -= cot_c * d_ab.x;
        laplacian[bi][1] -= cot_c * d_ab.y;
        laplacian[bi][2] -= cot_c * d_ab.z;

        area[ai] += third_area;
        area[bi] += third_area;
        area[ci] += third_area;
    }

    // Compute mean curvature: H = |Δf| / (4A) with sign from normal
    let mut curvatures = vec![0.0f32; n];
    for i in 0..n {
        let a = area[i].max(1e-10);
        let lx = laplacian[i][0] / (2.0 * a);
        let ly = laplacian[i][1] / (2.0 * a);
        let lz = laplacian[i][2] / (2.0 * a);
        let mag = (lx * lx + ly * ly + lz * lz).sqrt();

        // Sign: dot Laplacian with normal. Positive = convex, negative =
        // concave.
        let nx = vertices[i].normal[0];
        let ny = vertices[i].normal[1];
        let nz = vertices[i].normal[2];
        let sign = if lx * nx + ly * ny + lz * nz >= 0.0 {
            1.0
        } else {
            -1.0
        };
        curvatures[i] = sign * mag;
    }

    // Map curvature to color. Clamp to a reasonable range for visualization
    // using percentile-based robust clamping.
    let mut sorted: Vec<f32> = curvatures.clone();
    sorted.sort_unstable_by(|a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let lo = sorted[sorted.len() / 20]; // 5th percentile
    let hi = sorted[sorted.len() - sorted.len() / 20 - 1]; // 95th percentile
    let range = (hi - lo).max(1e-6);

    for (i, v) in vertices.iter_mut().enumerate() {
        let t = ((curvatures[i] - lo) / range).clamp(0.0, 1.0);
        // 0 = concave (red), 0.5 = flat (white), 1 = convex (blue)
        let (r, g, b) = if t < 0.5 {
            // Red → white
            let s = t * 2.0;
            (1.0, s, s)
        } else {
            // White → blue
            let s = (t - 0.5) * 2.0;
            (1.0 - s, 1.0 - s, 1.0)
        };
        v.color = [r, g, b, alpha];
    }
}

/// Cotangent of the angle between two vectors, clamped for stability.
fn safe_cot(u: Vec3, v: Vec3) -> f32 {
    let dot = u.dot(v);
    let cross_len = u.cross(v).length();
    if cross_len < 1e-10 {
        return 0.0;
    }
    (dot / cross_len).clamp(-100.0, 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ses_empty_input() {
        let (v, i) = generate_ses(&[], &[], None, 1.0, [1.0; 4]);
        assert!(v.is_empty());
        assert!(i.is_empty());
    }

    #[test]
    fn ses_single_atom_produces_surface() {
        let pos = vec![Vec3::ZERO];
        let radii = vec![1.5];
        let (verts, idxs) =
            generate_ses(&pos, &radii, Some(1.4), 0.5, [1.0; 4]);
        assert!(!verts.is_empty(), "SES should produce vertices");
        assert!(!idxs.is_empty(), "SES should produce indices");
    }

    #[test]
    fn ses_two_atoms_produces_surface() {
        // Two atoms close enough that the probe fills the gap
        let pos = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0)];
        let radii = vec![1.5, 1.5];
        let (verts, idxs) =
            generate_ses(&pos, &radii, Some(1.4), 0.5, [1.0; 4]);
        assert!(
            !verts.is_empty(),
            "SES should produce vertices for two atoms"
        );
        assert!(idxs.len() > 6, "SES should produce more than 2 triangles");
    }
}
