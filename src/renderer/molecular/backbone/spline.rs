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
            let arbitrary =
                if t.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
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
