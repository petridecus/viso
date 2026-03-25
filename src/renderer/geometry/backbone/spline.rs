//! Spline math and frame computation for backbone geometry.
//!
//! Pure Vec3 → Vec3 transforms with no GPU or SS-type dependencies.

use glam::Vec3;

/// A point along the spline with position, tangent, and frame vectors.
#[derive(Clone, Copy)]
pub(crate) struct SplinePoint {
    pub pos: Vec3,
    pub tangent: Vec3,
    pub normal: Vec3,
    pub binormal: Vec3,
}

/// Catmull-Rom spline interpolation (passes through all control points).
pub(crate) fn catmull_rom(
    points: &[Vec3],
    segments_per_span: usize,
) -> Vec<Vec3> {
    let n = points.len();
    if n < 2 {
        return points.to_vec();
    }
    if n < 3 {
        return linear_interpolate(points, segments_per_span);
    }

    let mut result = Vec::new();

    for i in 0..n - 1 {
        let p0 = if i == 0 {
            points[0] * 2.0 - points[1]
        } else {
            points[i - 1]
        };
        let p1 = points[i];
        let p2 = points[i + 1];
        let p3 = if i + 2 >= n {
            points[n - 1] * 2.0 - points[n - 2]
        } else {
            points[i + 2]
        };

        for j in 0..segments_per_span {
            let t = j as f32 / segments_per_span as f32;
            let t2 = t * t;
            let t3 = t2 * t;

            let pos = 0.5
                * ((2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
            result.push(pos);
        }
    }

    result.push(points[n - 1]);
    result
}

/// Evaluate a single point on the Catmull-Rom spline.
///
/// `span` selects which pair of control points (0..n-2), and `t` is the
/// parameter within that span (0.0 = start control point, 1.0 = end).
/// Uses the same phantom-point extrapolation as [`catmull_rom`] for
/// boundary spans.
pub(crate) fn catmull_rom_eval(
    points: &[Vec3],
    span: usize,
    t: f32,
) -> Option<Vec3> {
    let n = points.len();
    if n < 2 || span >= n - 1 {
        return None;
    }

    let p0 = if span == 0 {
        points[0] * 2.0 - points[1]
    } else {
        points[span - 1]
    };
    let p1 = points[span];
    let p2 = points[span + 1];
    let p3 = if span + 2 >= n {
        points[n - 1] * 2.0 - points[n - 2]
    } else {
        points[span + 2]
    };

    let t2 = t * t;
    let t3 = t2 * t;

    Some(
        0.5 * ((2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3),
    )
}

/// Project backbone N and C atom positions onto the Catmull-Rom spline
/// defined by CA control points.
///
/// Returns `(n_positions, c_positions)` — one per residue — sitting on
/// the rendered backbone curve. Fractional positions are derived from
/// standard peptide bond lengths (CA→C ≈ 1.52 Å, C→N ≈ 1.33 Å,
/// N→CA ≈ 1.47 Å).
///
/// Edge residues (first N, last C in each chain) use the raw backbone
/// positions since they fall outside the spline's control-point range.
pub(crate) fn project_backbone_atoms(
    backbone_chains: &[Vec<Vec3>],
) -> (Vec<Vec3>, Vec<Vec3>) {
    /// Fraction of CA→CA span where C sits (CA→C / total).
    const C_FRAC: f32 = 0.35;
    /// Fraction of CA→CA span where N sits (measured from previous CA).
    const N_FRAC: f32 = 0.66;

    let mut all_n = Vec::new();
    let mut all_c = Vec::new();

    for chain in backbone_chains {
        let residues: Vec<[Vec3; 3]> = chain
            .chunks_exact(3)
            .map(|tri| [tri[0], tri[1], tri[2]])
            .collect();
        let n = residues.len();
        if n == 0 {
            continue;
        }

        let ca: Vec<Vec3> = residues.iter().map(|r| r[1]).collect();

        for (i, res) in residues.iter().enumerate() {
            // N position: on the spline at N_FRAC into span (i-1 → i).
            let n_pos = if i == 0 || ca.len() < 2 {
                res[0] // raw N for first residue
            } else {
                catmull_rom_eval(&ca, i - 1, N_FRAC).unwrap_or(res[0])
            };

            // C position: on the spline at C_FRAC into span (i → i+1).
            let c_pos = if i >= n - 1 || ca.len() < 2 {
                res[2] // raw C for last residue
            } else {
                catmull_rom_eval(&ca, i, C_FRAC).unwrap_or(res[2])
            };

            all_n.push(n_pos);
            all_c.push(c_pos);
        }
    }

    (all_n, all_c)
}

/// Cubic B-spline (smooth approximation, does not pass through control points).
/// Used for helix axis smoothing.
pub(crate) fn cubic_bspline(
    points: &[Vec3],
    segments_per_span: usize,
) -> Vec<Vec3> {
    let n = points.len();
    if n < 2 {
        return points.to_vec();
    }
    if n < 4 {
        return linear_interpolate(points, segments_per_span);
    }

    let mut result = Vec::new();

    fn b0(t: f32) -> f32 {
        (1.0 - t).powi(3) / 6.0
    }
    fn b1(t: f32) -> f32 {
        (3.0 * t.powi(3) - 6.0 * t.powi(2) + 4.0) / 6.0
    }
    fn b2(t: f32) -> f32 {
        (-3.0 * t.powi(3) + 3.0 * t.powi(2) + 3.0 * t + 1.0) / 6.0
    }
    fn b3(t: f32) -> f32 {
        t.powi(3) / 6.0
    }

    let mut padded = Vec::with_capacity(n + 2);
    padded.push(points[0] * 2.0 - points[1]);
    padded.extend_from_slice(points);
    padded.push(points[n - 1] * 2.0 - points[n - 2]);

    for i in 0..n - 1 {
        let p0 = padded[i];
        let p1 = padded[i + 1];
        let p2 = padded[i + 2];
        let p3 = padded[i + 3];

        for j in 0..segments_per_span {
            let t = j as f32 / segments_per_span as f32;
            let pos = p0 * b0(t) + p1 * b1(t) + p2 * b2(t) + p3 * b3(t);
            result.push(pos);
        }
    }

    result.push(points[n - 1]);
    result
}

/// Linear interpolation fallback for short point sequences.
pub(crate) fn linear_interpolate(
    points: &[Vec3],
    segments_per_span: usize,
) -> Vec<Vec3> {
    let mut result = Vec::new();
    for i in 0..points.len() - 1 {
        for j in 0..segments_per_span {
            let t = j as f32 / segments_per_span as f32;
            result.push(points[i].lerp(points[i + 1], t));
        }
    }
    if let Some(&last) = points.last() {
        result.push(last);
    }
    result
}

/// Compute approximate helix axis points via sliding-window average.
pub(crate) fn compute_helix_axis_points(ca_positions: &[Vec3]) -> Vec<Vec3> {
    let n = ca_positions.len();
    let window = 4; // ~one helix turn

    let mut centers = Vec::with_capacity(n);
    for i in 0..n {
        let start = i.saturating_sub(window / 2);
        let end = (i + window / 2 + 1).min(n);
        let mut sum = Vec3::ZERO;
        for pos in &ca_positions[start..end] {
            sum += *pos;
        }
        centers.push(sum / (end - start) as f32);
    }
    centers
}

/// Compute Frenet frames (curvature-based) for helical traces.
///
/// The Frenet normal points away from the center of curvature, so the
/// ribbon twists naturally with helical geometry (DNA/RNA backbones).
/// Falls back to an arbitrary frame on straight segments.
pub(crate) fn compute_frenet_frames(points: &mut [SplinePoint]) {
    if points.len() < 2 {
        return;
    }

    let n = points.len();

    // Curvature vector (dT/ds) at each point via finite differences
    let curvatures: Vec<Vec3> = (0..n)
        .map(|i| {
            if i == 0 {
                points[1].tangent - points[0].tangent
            } else if i == n - 1 {
                points[n - 1].tangent - points[n - 2].tangent
            } else {
                (points[i + 1].tangent - points[i - 1].tangent) * 0.5
            }
        })
        .collect();

    for i in 0..n {
        let t = points[i].tangent;
        let curv = curvatures[i];

        if curv.length() > 1e-6 {
            // Negate Frenet normal for outward-pointing ribbon normal
            let normal = -curv.normalize();
            let normal = (normal - t * t.dot(normal)).normalize();
            let binormal = t.cross(normal).normalize();
            points[i].normal = normal;
            points[i].binormal = binormal;
        } else {
            let arbitrary = if t.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
            let normal = t.cross(arbitrary).normalize();
            let binormal = t.cross(normal).normalize();
            points[i].normal = normal;
            points[i].binormal = binormal;
        }
    }
}

/// Compute Rotation Minimizing Frames using the double reflection method
/// (Wang et al. 2008).
pub(crate) fn compute_rmf(points: &mut [SplinePoint]) {
    if points.is_empty() {
        return;
    }

    let t0 = points[0].tangent;
    let arbitrary = if t0.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let n0 = t0.cross(arbitrary).normalize();
    let b0 = t0.cross(n0).normalize();

    points[0].normal = n0;
    points[0].binormal = b0;

    for i in 0..points.len() - 1 {
        let x_i = points[i].pos;
        let x_i1 = points[i + 1].pos;
        let t_i = points[i].tangent;
        let t_i1 = points[i + 1].tangent;
        let r_i = points[i].normal;

        let v1 = x_i1 - x_i;
        let c1 = v1.dot(v1);

        if c1 < 1e-10 {
            points[i + 1].normal = r_i;
            points[i + 1].binormal = points[i].binormal;
            continue;
        }

        // First reflection
        let r_i_l = r_i - (2.0 / c1) * v1.dot(r_i) * v1;
        let t_i_l = t_i - (2.0 / c1) * v1.dot(t_i) * v1;

        // Second reflection
        let v2 = t_i1 - t_i_l;
        let c2 = v2.dot(v2);

        let r_i1 = if c2 < 1e-10 {
            r_i_l
        } else {
            r_i_l - (2.0 / c2) * v2.dot(r_i_l) * v2
        };

        // Ensure orthonormality
        let r_i1 = (r_i1 - t_i1 * t_i1.dot(r_i1)).normalize();
        let s_i1 = t_i1.cross(r_i1).normalize();

        points[i + 1].normal = r_i1;
        points[i + 1].binormal = s_i1;
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    fn approx_eq(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < TOL
    }

    #[test]
    fn linear_two_points() {
        let pts = vec![Vec3::ZERO, Vec3::new(10.0, 0.0, 0.0)];
        let result = linear_interpolate(&pts, 4);
        assert_eq!(result.len(), 5); // 4 segments + endpoint
        assert!(approx_eq(result[0], pts[0]));
        assert!(approx_eq(*result.last().unwrap(), pts[1]));
        // midpoint at t=0.5 (index 2)
        assert!(approx_eq(result[2], Vec3::new(5.0, 0.0, 0.0)));
    }

    #[test]
    fn catmull_rom_endpoints() {
        let pts: Vec<Vec3> = (0..5)
            .map(|i| Vec3::new(i as f32 * 2.0, 0.0, 0.0))
            .collect();
        let result = catmull_rom(&pts, 4);
        assert!(approx_eq(result[0], pts[0]));
        assert!(approx_eq(*result.last().unwrap(), *pts.last().unwrap()));
    }

    #[test]
    fn catmull_rom_passes_through() {
        let pts: Vec<Vec3> = (0..5)
            .map(|i| Vec3::new(i as f32 * 3.0, (i as f32).sin(), 0.0))
            .collect();
        let segs = 8;
        let result = catmull_rom(&pts, segs);
        // At span boundaries (every `segs` samples), output ≈ control point
        for (i, pt) in pts.iter().enumerate() {
            let idx = if i == pts.len() - 1 {
                result.len() - 1
            } else {
                i * segs
            };
            assert!(
                approx_eq(result[idx], *pt),
                "span {i}: expected {pt:?}, got {:?}",
                result[idx]
            );
        }
    }

    #[test]
    fn catmull_rom_two_points_linear() {
        let pts = vec![Vec3::ZERO, Vec3::new(4.0, 0.0, 0.0)];
        let result = catmull_rom(&pts, 4);
        let linear = linear_interpolate(&pts, 4);
        assert_eq!(result.len(), linear.len());
        for (a, b) in result.iter().zip(linear.iter()) {
            assert!(approx_eq(*a, *b));
        }
    }

    #[test]
    fn catmull_rom_single_point() {
        let pts = vec![Vec3::new(1.0, 2.0, 3.0)];
        let result = catmull_rom(&pts, 8);
        assert_eq!(result.len(), 1);
        assert!(approx_eq(result[0], pts[0]));
    }

    #[test]
    fn bspline_last_point_matches() {
        let pts: Vec<Vec3> = (0..6)
            .map(|i| Vec3::new(i as f32, (i as f32 * 0.5).sin(), 0.0))
            .collect();
        let result = cubic_bspline(&pts, 4);
        assert!(approx_eq(*result.last().unwrap(), *pts.last().unwrap()));
    }

    #[test]
    fn bspline_few_points_linear() {
        let pts = vec![Vec3::ZERO, Vec3::X, Vec3::new(2.0, 0.0, 0.0)];
        let result = cubic_bspline(&pts, 4);
        let linear = linear_interpolate(&pts, 4);
        assert_eq!(result.len(), linear.len());
    }

    #[test]
    fn helix_axis_preserves_count() {
        let pts: Vec<Vec3> = (0..10)
            .map(|i| {
                let t = i as f32 * 0.5;
                Vec3::new(t.cos(), t.sin(), t * 1.5)
            })
            .collect();
        let axis = compute_helix_axis_points(&pts);
        assert_eq!(axis.len(), pts.len());
    }

    #[test]
    fn rmf_orthonormal() {
        // Build a curved path
        let n = 20;
        let mut points: Vec<SplinePoint> = (0..n)
            .map(|i| {
                let t = i as f32 * 0.3;
                SplinePoint {
                    pos: Vec3::new(t.cos() * 5.0, t.sin() * 5.0, t * 2.0),
                    tangent: Vec3::ZERO,
                    normal: Vec3::ZERO,
                    binormal: Vec3::ZERO,
                }
            })
            .collect();
        // Compute tangents via finite differences
        for i in 0..n {
            let prev = if i == 0 { 0 } else { i - 1 };
            let next = (i + 1).min(n - 1);
            points[i].tangent =
                (points[next].pos - points[prev].pos).normalize();
        }
        compute_rmf(&mut points);

        for (i, p) in points.iter().enumerate() {
            assert!(
                (p.normal.length() - 1.0).abs() < TOL,
                "frame {i}: normal not unit"
            );
            assert!(
                (p.binormal.length() - 1.0).abs() < TOL,
                "frame {i}: binormal not unit"
            );
            assert!(
                p.normal.dot(p.binormal).abs() < TOL,
                "frame {i}: n·b != 0"
            );
            assert!(p.tangent.dot(p.normal).abs() < TOL, "frame {i}: t·n != 0");
        }
    }

    #[test]
    fn frenet_and_rmf_empty() {
        // Empty input
        let mut empty: Vec<SplinePoint> = vec![];
        compute_frenet_frames(&mut empty);
        compute_rmf(&mut empty);

        // Single point
        let mut single = vec![SplinePoint {
            pos: Vec3::ZERO,
            tangent: Vec3::Z,
            normal: Vec3::ZERO,
            binormal: Vec3::ZERO,
        }];
        compute_frenet_frames(&mut single);
        compute_rmf(&mut single);
        // Should not panic — that's the test
    }
}
