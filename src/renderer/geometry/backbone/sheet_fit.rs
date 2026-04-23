//! β-sheet topology detection and per-sheet plane fitting.
//!
//! Groups sheet-classified residues into connected components by
//! walking the backbone hydrogen-bond graph, then fits a best-fit plane
//! to each component via a closed-form 3×3 symmetric eigendecomposition
//! of the CA covariance matrix. The resulting per-residue plane normals
//! replace the single-residue peptide-plane cross product that
//! `compute_sheet_geometry` otherwise uses, so every strand in a given
//! sheet gets the same face orientation.

use std::cmp::Ordering;
use std::collections::HashMap;

use glam::Vec3;
use molex::{HBond, SSType};

/// A β-sheet: a connected set of H-bonded strands.
pub(crate) struct SheetGroup {
    /// Residue indices (flat, matching `ss_types` indexing) that
    /// belong to this sheet.
    pub(crate) residues: Vec<usize>,
    /// How many contiguous strands merged into this group. A value of
    /// 1 means the group is a single isolated strand (no cross-strand
    /// H-bonds), and its CA positions are nearly collinear — plane
    /// fitting would be degenerate. Sheets proper have ≥ 2.
    pub(crate) strand_count: usize,
}

/// Group sheet-classified residues into β-sheets.
///
/// First collapses contiguous runs of Sheet residues into strands
/// (adjacent-residue connectivity, not H-bond connectivity — residues
/// within one strand are *not* H-bonded to each other), then merges
/// strands via union-find over the backbone H-bond graph restricted to
/// H-bonds between sheet residues. Each resulting component is one
/// β-sheet.
pub(crate) fn detect_sheet_groups(
    hbonds: &[HBond],
    ss_types: &[SSType],
) -> Vec<SheetGroup> {
    let n = ss_types.len();
    if n == 0 {
        return Vec::new();
    }

    // 1. Assign each sheet residue to a strand (contiguous run of Sheet
    //    residues).
    let mut strand_of: Vec<Option<usize>> = vec![None; n];
    let mut num_strands = 0_usize;
    let mut i = 0;
    while i < n {
        if ss_types[i] != SSType::Sheet {
            i += 1;
            continue;
        }
        let strand_id = num_strands;
        num_strands += 1;
        while i < n && ss_types[i] == SSType::Sheet {
            strand_of[i] = Some(strand_id);
            i += 1;
        }
    }

    if num_strands == 0 {
        return Vec::new();
    }

    // 2. Union-find over strands, merging pairs connected by H-bonds.
    let mut parent: Vec<usize> = (0..num_strands).collect();
    for hb in hbonds {
        if hb.donor >= n || hb.acceptor >= n {
            continue;
        }
        let (Some(a), Some(b)) = (strand_of[hb.donor], strand_of[hb.acceptor])
        else {
            continue;
        };
        union(&mut parent, a, b);
    }

    // 3. Gather residues by merged-strand root, and count how many distinct
    //    strand ids ended up in each component.
    let mut residues_by_root: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut strands_by_root: HashMap<usize, std::collections::HashSet<usize>> =
        HashMap::new();
    for (res, strand) in strand_of.iter().enumerate() {
        if let Some(sid) = *strand {
            let root = find(&mut parent, sid);
            residues_by_root.entry(root).or_default().push(res);
            let _ = strands_by_root.entry(root).or_default().insert(sid);
        }
    }

    residues_by_root
        .into_iter()
        .map(|(root, residues)| SheetGroup {
            residues,
            strand_count: strands_by_root
                .get(&root)
                .map_or(0, std::collections::HashSet::len),
        })
        .collect()
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        parent[ra] = rb;
    }
}

/// Fit a best-fit plane to a set of points and return its unit normal.
///
/// Uses the closed-form trigonometric solution for eigenvalues of a
/// 3×3 symmetric covariance matrix (Smith, "Eigenvalues of a symmetric
/// 3×3 matrix"), then extracts the eigenvector of the smallest
/// eigenvalue as the null space of `(A − λ·I)` via the largest of the
/// three row cross products.
///
/// Returns `Vec3::Y` for degenerate inputs (fewer than 3 points, or
/// colinear points).
pub(crate) fn fit_plane_normal(points: &[Vec3]) -> Vec3 {
    if points.len() < 3 {
        return Vec3::Y;
    }

    let centroid = points.iter().copied().sum::<Vec3>() / points.len() as f32;

    let mut a00 = 0.0_f32;
    let mut a01 = 0.0_f32;
    let mut a02 = 0.0_f32;
    let mut a11 = 0.0_f32;
    let mut a12 = 0.0_f32;
    let mut a22 = 0.0_f32;
    for p in points {
        let d = *p - centroid;
        a00 += d.x * d.x;
        a01 += d.x * d.y;
        a02 += d.x * d.z;
        a11 += d.y * d.y;
        a12 += d.y * d.z;
        a22 += d.z * d.z;
    }

    let lambda_min = smallest_eigenvalue(a00, a01, a02, a11, a12, a22);
    let normal =
        null_space_of_shifted(a00, a01, a02, a11, a12, a22, lambda_min);
    if normal.length_squared() > 0.25 {
        normal.normalize()
    } else {
        Vec3::Y
    }
}

/// Compute the smallest eigenvalue of a 3×3 symmetric matrix with
/// closed-form trigonometric solution.
fn smallest_eigenvalue(
    a00: f32,
    a01: f32,
    a02: f32,
    a11: f32,
    a12: f32,
    a22: f32,
) -> f32 {
    let p1 = a01 * a01 + a02 * a02 + a12 * a12;
    if p1 < 1e-12 {
        // Diagonal matrix — eigenvalues are the diagonal entries.
        return a00.min(a11).min(a22);
    }
    let q = (a00 + a11 + a22) / 3.0;
    let p2 =
        (a00 - q).powi(2) + (a11 - q).powi(2) + (a22 - q).powi(2) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    if p < 1e-6 {
        return q;
    }

    // B = (A − q·I) / p, det(B) / 2
    let b00 = (a00 - q) / p;
    let b11 = (a11 - q) / p;
    let b22 = (a22 - q) / p;
    let b01 = a01 / p;
    let b02 = a02 / p;
    let b12 = a12 / p;
    let m00 = b11 * b22 - b12.powi(2);
    let m01 = b01 * b22 - b12 * b02;
    let m02 = b01 * b12 - b11 * b02;
    let det_b = b00 * m00 - b01 * m01 + b02 * m02;
    let r = (det_b / 2.0).clamp(-1.0, 1.0);
    let phi = r.acos() / 3.0;

    let lam_max = q + 2.0 * p * phi.cos();
    // 2π/3 shift gives the smallest of the three eigenvalues when the
    // trigonometric solution is indexed by the principal cube root.
    let two_pi_over_3 = 2.0 * std::f32::consts::FRAC_PI_3;
    let lam_min = q + 2.0 * p * (phi + two_pi_over_3).cos();
    let lam_mid = 3.0 * q - lam_max - lam_min;
    lam_min.min(lam_mid).min(lam_max)
}

/// Find a non-zero vector in the null space of `(A − λ·I)` for a
/// symmetric 3×3 matrix `A` with eigenvalue `lambda`. Returns the
/// cross product of the two rows of `M = A − λ·I` that produces the
/// largest magnitude, which avoids picking degenerate rows.
fn null_space_of_shifted(
    a00: f32,
    a01: f32,
    a02: f32,
    a11: f32,
    a12: f32,
    a22: f32,
    lambda: f32,
) -> Vec3 {
    let r0 = Vec3::new(a00 - lambda, a01, a02);
    let r1 = Vec3::new(a01, a11 - lambda, a12);
    let r2 = Vec3::new(a02, a12, a22 - lambda);

    let c01 = r0.cross(r1);
    let c02 = r0.cross(r2);
    let c12 = r1.cross(r2);

    let candidates = [
        (c01, c01.length_squared()),
        (c02, c02.length_squared()),
        (c12, c12.length_squared()),
    ];
    candidates
        .iter()
        .copied()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        .map_or(Vec3::ZERO, |(v, _)| v)
}

/// Produce per-residue sheet plane normals.
///
/// Walks the H-bond graph to group sheet residues into β-sheets
/// (multi-strand connected components), fits a plane to each sheet's
/// CA positions, and emits one `(residue_idx, normal)` pair per
/// residue that landed in a successfully-fitted sheet.
///
/// Sparse by design: residues that don't appear in the returned vector
/// (non-sheet residues, lone strands, degenerate fits) will fall back
/// to the local peptide-plane computation in `compute_sheet_geometry`.
/// Residue indices match the flat indexing of `ss_types` and
/// `ca_positions`. The result is sorted by residue index so callers
/// can build an `O(1)` lookup map in one pass if they want.
pub(crate) fn compute_sheet_plane_normals(
    hbonds: &[HBond],
    ss_types: &[SSType],
    ca_positions: &[Vec3],
) -> Vec<(u32, Vec3)> {
    if ss_types.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    let groups = detect_sheet_groups(hbonds, ss_types);
    for group in &groups {
        // Skip isolated strands: their CAs are ~collinear and the
        // plane fit would be indeterminate. Let the local peptide
        // plane handle them via the fallback path.
        if group.strand_count < 2 || group.residues.len() < 3 {
            continue;
        }
        let points: Vec<Vec3> = group
            .residues
            .iter()
            .filter_map(|&i| ca_positions.get(i).copied())
            .collect();
        if points.len() < 3 {
            continue;
        }
        let normal = fit_plane_normal(&points);
        if normal.length_squared() < 0.5 {
            continue;
        }
        for &i in &group.residues {
            out.push((i as u32, normal));
        }
    }

    out.sort_unstable_by_key(|&(i, _)| i);
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn approx_eq(a: Vec3, b: Vec3, tol: f32) -> bool {
        (a - b).length() < tol
    }

    #[test]
    fn plane_fit_xy_plane() {
        let pts = vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.7, 0.7, 0.0),
        ];
        let normal = fit_plane_normal(&pts);
        // Normal should be ±Z
        assert!(
            approx_eq(normal, Vec3::Z, 1e-3)
                || approx_eq(normal, -Vec3::Z, 1e-3),
            "got {normal:?}"
        );
    }

    #[test]
    fn plane_fit_tilted() {
        // Plane with normal (1, 1, 1).normalize(); points span the plane.
        let n_expected = Vec3::new(1.0, 1.0, 1.0).normalize();
        // Two orthogonal vectors in the plane
        let u = Vec3::new(1.0, -1.0, 0.0).normalize();
        let v = n_expected.cross(u).normalize();
        let pts: Vec<Vec3> = (0..8)
            .map(|i| {
                let t = i as f32 * 0.7;
                u * t.cos() + v * t.sin()
            })
            .collect();
        let normal = fit_plane_normal(&pts);
        let dot = normal.dot(n_expected).abs();
        assert!(dot > 0.999, "dot = {dot}, normal = {normal:?}");
    }

    #[test]
    fn plane_fit_fewer_than_three_returns_fallback() {
        assert!(approx_eq(fit_plane_normal(&[]), Vec3::Y, 1e-6));
        assert!(approx_eq(fit_plane_normal(&[Vec3::ZERO]), Vec3::Y, 1e-6));
        assert!(approx_eq(
            fit_plane_normal(&[Vec3::ZERO, Vec3::X]),
            Vec3::Y,
            1e-6
        ));
    }

    #[test]
    fn detect_sheet_groups_two_strand_sheet() {
        // Residues 0-3 and 6-9 are Sheet; 4-5 are Coil. H-bonds connect
        // 0↔9, 1↔8, 2↔7, 3↔6 (antiparallel pairing).
        let ss = vec![
            SSType::Sheet,
            SSType::Sheet,
            SSType::Sheet,
            SSType::Sheet,
            SSType::Coil,
            SSType::Coil,
            SSType::Sheet,
            SSType::Sheet,
            SSType::Sheet,
            SSType::Sheet,
        ];
        let hbonds = vec![
            HBond {
                donor: 0,
                acceptor: 9,
                energy: -2.0,
            },
            HBond {
                donor: 1,
                acceptor: 8,
                energy: -2.0,
            },
            HBond {
                donor: 2,
                acceptor: 7,
                energy: -2.0,
            },
            HBond {
                donor: 3,
                acceptor: 6,
                energy: -2.0,
            },
        ];
        let groups = detect_sheet_groups(&hbonds, &ss);
        assert_eq!(groups.len(), 1);
        let group = groups.into_iter().next().unwrap();
        assert_eq!(group.strand_count, 2);
        let mut residues = group.residues;
        residues.sort_unstable();
        assert_eq!(residues, vec![0, 1, 2, 3, 6, 7, 8, 9]);
    }

    #[test]
    fn detect_sheet_groups_isolated_strand_is_single_group() {
        let ss = vec![SSType::Sheet, SSType::Sheet, SSType::Sheet];
        let groups = detect_sheet_groups(&[], &ss);
        // Three contiguous Sheet residues = one strand = one group.
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].strand_count, 1);
        assert_eq!(groups[0].residues.len(), 3);
    }

    #[test]
    fn compute_sheet_plane_normals_skips_single_strand() {
        // A lone 4-residue strand: plane fit is degenerate, so no
        // entries should be emitted.
        let ss = vec![SSType::Sheet; 4];
        let cas = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(3.5, 0.0, 0.0),
            Vec3::new(7.0, 0.0, 0.0),
            Vec3::new(10.5, 0.0, 0.0),
        ];
        let normals = compute_sheet_plane_normals(&[], &ss, &cas);
        assert!(normals.is_empty());
    }

    #[test]
    fn compute_sheet_plane_normals_fits_two_strand_sheet() {
        // Two strands in the XZ plane, H-bonded. Plane normal → ±Y.
        let ss = vec![
            SSType::Sheet,
            SSType::Sheet,
            SSType::Sheet,
            SSType::Coil,
            SSType::Sheet,
            SSType::Sheet,
            SSType::Sheet,
        ];
        let cas = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(3.5, 0.0, 0.0),
            Vec3::new(7.0, 0.0, 0.0),
            Vec3::new(7.0, 0.0, 4.0), // coil turn
            Vec3::new(7.0, 0.0, 5.0),
            Vec3::new(3.5, 0.0, 5.0),
            Vec3::new(0.0, 0.0, 5.0),
        ];
        let hbonds = vec![
            HBond {
                donor: 0,
                acceptor: 6,
                energy: -2.0,
            },
            HBond {
                donor: 2,
                acceptor: 4,
                energy: -2.0,
            },
        ];
        let normals = compute_sheet_plane_normals(&hbonds, &ss, &cas);
        // Sheet residues (0,1,2,4,5,6) get entries; coil (3) does not.
        let residue_set: std::collections::HashSet<u32> =
            normals.iter().map(|&(i, _)| i).collect();
        assert_eq!(residue_set, [0, 1, 2, 4, 5, 6].into_iter().collect());
        for (i, n) in &normals {
            assert!(
                n.dot(Vec3::Y).abs() > 0.99,
                "residue {i}: {n:?} not aligned to ±Y"
            );
        }
        // Sorted by residue idx.
        let indices: Vec<u32> = normals.iter().map(|&(i, _)| i).collect();
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        assert_eq!(indices, sorted);
    }
}
