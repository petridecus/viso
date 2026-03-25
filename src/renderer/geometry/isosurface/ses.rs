//! Solvent-Excluded Surface (SES / Connolly surface) generation.
//!
//! Uses the EDTSurf algorithm:
//! 1. Voxelize the SAS solid (binary: inside any atom's vdW + probe)
//! 2. 3D Euclidean Distance Transform on the solid interior
//! 3. Carve voxels where EDT < probe (probe can reach from outside)
//! 4. Convert the resulting SES solid to a signed distance field
//! 5. Extract isosurface at threshold 0

use glam::Vec3;

use super::cpu_marching_cubes::extract_isosurface;
use super::IsosurfaceVertex;

/// Default solvent probe radius (water) in Angstroms.
const DEFAULT_PROBE_RADIUS: f32 = 1.4;

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

    let probe = probe_radius.unwrap_or(DEFAULT_PROBE_RADIUS);
    let padding = probe + 2.0;

    // Compute grid bounds
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

    // Step 1: Build binary SAS solid
    let sas_solid =
        voxelize_sas(positions, radii, probe, dims, &origin, &spacing);

    // Step 2: EDT on interior → distance to nearest outside voxel
    let interior_edt = edt_3d(&sas_solid, dims, &spacing);

    // Step 3: Carve — voxels where EDT < probe become outside
    let total = nx * ny * nz;
    let mut ses_solid = vec![false; total];
    for i in 0..total {
        ses_solid[i] = sas_solid[i] && interior_edt[i] >= probe;
    }

    // Step 3b: Fill interior voids. Flood-fill from the grid boundary
    // to find the connected exterior. Any non-solid voxel NOT reached
    // by the flood is an internal cavity — fill it in as solid so
    // marching cubes only produces the outer envelope.
    fill_interior_voids(&mut ses_solid, dims);

    // Step 4: Convert SES solid to signed distance field.
    // Negate so the field is positive-inside (matches density convention
    // expected by marching cubes gradient_normal).
    let mut sdf = binary_to_sdf(&ses_solid, dims, &spacing);
    for v in &mut sdf {
        *v = -*v;
    }

    // Step 5: Extract isosurface at threshold 0
    let (mut verts, idxs) = extract_isosurface(
        &sdf,
        dims,
        0.0,
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
    );

    // Step 6: Color by mean curvature (blue=convex, white=flat, red=concave)
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

        // Accumulate cotangent-weighted displacement for each edge
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

    // Map curvature to color. Clamp to a reasonable range for visualization.
    // Use percentile-based clamping for robustness.
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

/// Voxelize the SAS: voxel is `true` if within any atom's (vdW + probe).
fn voxelize_sas(
    positions: &[Vec3],
    radii: &[f32],
    probe: f32,
    dims: [usize; 3],
    origin: &[f32; 3],
    spacing: &[f32; 3],
) -> Vec<bool> {
    let total = dims[0] * dims[1] * dims[2];
    let mut solid = vec![false; total];

    for (i, &pos) in positions.iter().enumerate() {
        splat_sas_atom(
            &mut solid,
            dims,
            origin,
            spacing,
            pos,
            radii[i] + probe,
        );
    }

    solid
}

/// Mark voxels within radius `r` of `pos` as solid.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn splat_sas_atom(
    solid: &mut [bool],
    dims: [usize; 3],
    origin: &[f32; 3],
    spacing: &[f32; 3],
    pos: Vec3,
    r: f32,
) {
    let [nx, ny, nz] = dims;
    let r2 = r * r;

    let gx0 = ((pos.x - r - origin[0]) / spacing[0]).floor().max(0.0) as usize;
    let gy0 = ((pos.y - r - origin[1]) / spacing[1]).floor().max(0.0) as usize;
    let gz0 = ((pos.z - r - origin[2]) / spacing[2]).floor().max(0.0) as usize;
    let gx1 =
        (((pos.x + r - origin[0]) / spacing[0]).ceil() as usize).min(nx - 1);
    let gy1 =
        (((pos.y + r - origin[1]) / spacing[1]).ceil() as usize).min(ny - 1);
    let gz1 =
        (((pos.z + r - origin[2]) / spacing[2]).ceil() as usize).min(nz - 1);

    for ix in gx0..=gx1 {
        let dx = origin[0] + ix as f32 * spacing[0] - pos.x;
        for iy in gy0..=gy1 {
            let dy = origin[1] + iy as f32 * spacing[1] - pos.y;
            let dxy2 = dx * dx + dy * dy;
            if dxy2 > r2 {
                continue;
            }
            for iz in gz0..=gz1 {
                let dz = origin[2] + iz as f32 * spacing[2] - pos.z;
                if dxy2 + dz * dz <= r2 {
                    solid[ix * ny * nz + iy * nz + iz] = true;
                }
            }
        }
    }
}

/// Felzenszwalb & Huttenlocher 1D EDT on squared distances.
///
/// `f[i]` holds the squared distance value at position `i` (with positions
/// scaled by `spacing`). On return, `f[i]` holds the minimum over all `q`
/// of `f[q] + (spacing * (i - q))^2`.
#[allow(clippy::while_float)]
fn edt_1d(f: &mut [f32], spacing: f32) {
    let n = f.len();
    if n <= 1 {
        return;
    }

    let sp2 = spacing * spacing;
    // Parabola envelope
    let mut v = vec![0usize; n]; // locations of parabolas
    let mut z = vec![0.0f32; n + 1]; // boundaries between parabolas
    let mut d = vec![0.0f32; n]; // output

    let mut k = 0;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;

    for q in 1..n {
        loop {
            let vk = v[k];
            let diff = q as f32 - vk as f32;
            let s = (f[q] - f[vk] + sp2 * diff * (q as f32 + vk as f32))
                / (2.0 * sp2 * diff);
            if s > z[k] {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f32::INFINITY;
                break;
            }
            if k == 0 {
                v[0] = q;
                z[1] = f32::INFINITY;
                break;
            }
            k -= 1;
        }
    }

    k = 0;
    for (q, d_q) in d.iter_mut().enumerate().take(n) {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let diff = q as f32 - v[k] as f32;
        *d_q = sp2 * diff * diff + f[v[k]];
    }

    f.copy_from_slice(&d);
}

/// 3D EDT on a binary solid. Returns Euclidean distance from each inside
/// voxel to the nearest outside voxel (outside voxels get 0).
fn edt_3d(solid: &[bool], dims: [usize; 3], spacing: &[f32; 3]) -> Vec<f32> {
    let [nx, ny, nz] = dims;
    let inf = (nx + ny + nz) as f32 * (spacing[0] + spacing[1] + spacing[2]);
    let inf2 = inf * inf;

    // Initialize: inside = INF^2, outside = 0 (squared distances)
    let mut dt: Vec<f32> =
        solid.iter().map(|&s| if s { inf2 } else { 0.0 }).collect();

    // Pass 1: along X for each (y, z) line
    let mut buf = vec![0.0f32; nx.max(ny).max(nz)];
    for iy in 0..ny {
        for iz in 0..nz {
            for ix in 0..nx {
                buf[ix] = dt[ix * ny * nz + iy * nz + iz];
            }
            edt_1d(&mut buf[..nx], spacing[0]);
            for ix in 0..nx {
                dt[ix * ny * nz + iy * nz + iz] = buf[ix];
            }
        }
    }

    // Pass 2: along Y for each (x, z) line
    for ix in 0..nx {
        for iz in 0..nz {
            for iy in 0..ny {
                buf[iy] = dt[ix * ny * nz + iy * nz + iz];
            }
            edt_1d(&mut buf[..ny], spacing[1]);
            for iy in 0..ny {
                dt[ix * ny * nz + iy * nz + iz] = buf[iy];
            }
        }
    }

    // Pass 3: along Z for each (x, y) line
    for ix in 0..nx {
        for iy in 0..ny {
            let base = ix * ny * nz + iy * nz;
            buf[..nz].copy_from_slice(&dt[base..base + nz]);
            edt_1d(&mut buf[..nz], spacing[2]);
            dt[base..base + nz].copy_from_slice(&buf[..nz]);
        }
    }

    // sqrt to get actual Euclidean distances
    for v in &mut dt {
        *v = v.sqrt();
    }

    dt
}

/// Flood-fill from all boundary (face) voxels to find the connected
/// exterior. Any non-solid voxel NOT reached is an internal cavity —
/// mark it as solid so only the outer surface is extracted.
fn fill_interior_voids(solid: &mut [bool], dims: [usize; 3]) {
    let [nx, ny, nz] = dims;
    let total = nx * ny * nz;
    let idx = |x: usize, y: usize, z: usize| x * ny * nz + y * nz + z;

    // `exterior[i] = true` means this voxel is reachable from the grid
    // boundary through non-solid voxels.
    let mut exterior = vec![false; total];
    let mut stack: Vec<(usize, usize, usize)> = Vec::new();

    // Seed: all non-solid voxels on the 6 grid faces
    for ix in 0..nx {
        for iy in 0..ny {
            for &iz in &[0, nz - 1] {
                let i = idx(ix, iy, iz);
                if !solid[i] && !exterior[i] {
                    exterior[i] = true;
                    stack.push((ix, iy, iz));
                }
            }
        }
    }
    for ix in 0..nx {
        for &iy in &[0, ny - 1] {
            for iz in 1..nz - 1 {
                let i = idx(ix, iy, iz);
                if !solid[i] && !exterior[i] {
                    exterior[i] = true;
                    stack.push((ix, iy, iz));
                }
            }
        }
    }
    for &ix in &[0, nx - 1] {
        for iy in 1..ny - 1 {
            for iz in 1..nz - 1 {
                let i = idx(ix, iy, iz);
                if !solid[i] && !exterior[i] {
                    exterior[i] = true;
                    stack.push((ix, iy, iz));
                }
            }
        }
    }

    // BFS/DFS flood fill
    while let Some((x, y, z)) = stack.pop() {
        let neighbors: [(isize, isize, isize); 6] = [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ];
        for (dx, dy, dz) in neighbors {
            let nx2 = x as isize + dx;
            let ny2 = y as isize + dy;
            let nz2 = z as isize + dz;
            if nx2 < 0 || ny2 < 0 || nz2 < 0 {
                continue;
            }
            let (nx2, ny2, nz2) = (nx2 as usize, ny2 as usize, nz2 as usize);
            if nx2 >= nx || ny2 >= ny || nz2 >= nz {
                continue;
            }
            let i = idx(nx2, ny2, nz2);
            if !solid[i] && !exterior[i] {
                exterior[i] = true;
                stack.push((nx2, ny2, nz2));
            }
        }
    }

    // Any non-solid, non-exterior voxel is an interior void — fill it
    for i in 0..total {
        if !solid[i] && !exterior[i] {
            solid[i] = true;
        }
    }
}

/// Convert a binary solid to a signed distance field.
///
/// Runs EDT from both sides of the boundary:
/// - Outside voxels: +distance to nearest inside
/// - Inside voxels: -distance to nearest outside
fn binary_to_sdf(
    solid: &[bool],
    dims: [usize; 3],
    spacing: &[f32; 3],
) -> Vec<f32> {
    let total = solid.len();

    // EDT of inside (distance from outside to nearest inside)
    let inverted: Vec<bool> = solid.iter().map(|&s| !s).collect();
    let dist_outside = edt_3d(&inverted, dims, spacing);

    // EDT of outside (distance from inside to nearest outside)
    let dist_inside = edt_3d(solid, dims, spacing);

    // SDF: negative inside, positive outside
    let mut sdf = vec![0.0f32; total];
    for i in 0..total {
        sdf[i] = if solid[i] {
            -dist_inside[i]
        } else {
            dist_outside[i]
        };
    }

    sdf
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
    fn edt_1d_basic() {
        // Single inside voxel at position 2 in a line of 5
        let inf2 = 1e10f32;
        let mut f = [0.0, 0.0, inf2, 0.0, 0.0];
        edt_1d(&mut f, 1.0);
        // Position 2 should have distance 1.0 squared = 1.0
        assert!((f[2] - 1.0).abs() < 1e-6, "got {}", f[2]);
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
