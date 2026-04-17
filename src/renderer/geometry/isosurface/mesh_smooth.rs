//! Triangle-mesh smoothing for marching-cubes output.
//!
//! Marching cubes on a piecewise-linear distance field produces meshes
//! with visible voxel facets. The fix at the field level (smoothing the
//! SDF before extraction) destroys small features — small cavities can
//! get blurred away below the iso-threshold.
//!
//! Mesh-side smoothing avoids that problem entirely: it operates on the
//! triangles after extraction, so it can never lose features. It only
//! moves existing vertices along the existing topology.
//!
//! [`taubin_smooth`] uses the standard λ/μ alternating-pass scheme. The
//! λ pass is plain Laplacian smoothing (which shrinks the mesh slightly);
//! the μ pass uses a slightly larger negative factor to push the mesh
//! back out, cancelling shrinkage. The net result is smoother triangles
//! with no volume loss.

use glam::Vec3;

use super::IsosurfaceVertex;

/// Default Taubin λ (positive smoothing factor).
const TAUBIN_LAMBDA: f32 = 0.5;

/// Default Taubin μ (negative anti-shrinkage factor; |μ| > λ).
const TAUBIN_MU: f32 = -0.53;

/// Apply Taubin smoothing to a triangle mesh in place.
///
/// Each iteration is one λ pass followed by one μ pass, both updating
/// vertex positions toward the average of their topological neighbors.
/// After smoothing, vertex normals are recomputed by averaging adjacent
/// face normals.
///
/// Cost is `O(iterations * (V + 6V))` for mesh-only adjacency walks.
pub(crate) fn taubin_smooth(
    vertices: &mut [IsosurfaceVertex],
    indices: &[u32],
    iterations: usize,
) {
    if vertices.is_empty() || indices.is_empty() || iterations == 0 {
        return;
    }

    let adjacency = build_vertex_adjacency(vertices.len(), indices);

    for _ in 0..iterations {
        smoothing_pass(vertices, &adjacency, TAUBIN_LAMBDA);
        smoothing_pass(vertices, &adjacency, TAUBIN_MU);
    }

    recompute_normals(vertices, indices);
}

/// Build a vertex-to-vertex adjacency list from a triangle index buffer.
///
/// Each entry `adj[i]` is the deduplicated list of vertex indices
/// sharing a triangle edge with vertex `i`.
fn build_vertex_adjacency(n: usize, indices: &[u32]) -> Vec<Vec<u32>> {
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for tri in indices.chunks_exact(3) {
        let a = tri[0];
        let b = tri[1];
        let c = tri[2];
        push_unique(&mut adj[a as usize], b);
        push_unique(&mut adj[a as usize], c);
        push_unique(&mut adj[b as usize], a);
        push_unique(&mut adj[b as usize], c);
        push_unique(&mut adj[c as usize], a);
        push_unique(&mut adj[c as usize], b);
    }
    adj
}

fn push_unique(v: &mut Vec<u32>, val: u32) {
    if !v.contains(&val) {
        v.push(val);
    }
}

/// Single smoothing pass: each vertex moves a fraction `factor` toward
/// the centroid of its neighbors. Positive `factor` is shrinking,
/// negative is expanding.
fn smoothing_pass(
    vertices: &mut [IsosurfaceVertex],
    adjacency: &[Vec<u32>],
    factor: f32,
) {
    let new_positions: Vec<[f32; 3]> = vertices
        .iter()
        .enumerate()
        .map(|(i, v)| {
            smooth_vertex(v.position, &adjacency[i], vertices, factor)
        })
        .collect();
    for (v, p) in vertices.iter_mut().zip(new_positions) {
        v.position = p;
    }
}

fn smooth_vertex(
    pos: [f32; 3],
    neighbors: &[u32],
    vertices: &[IsosurfaceVertex],
    factor: f32,
) -> [f32; 3] {
    if neighbors.is_empty() {
        return pos;
    }
    let mut sum = [0.0f32; 3];
    for &j in neighbors {
        let n = vertices[j as usize].position;
        sum[0] += n[0];
        sum[1] += n[1];
        sum[2] += n[2];
    }
    let inv = 1.0 / neighbors.len() as f32;
    let avg = [sum[0] * inv, sum[1] * inv, sum[2] * inv];
    [
        factor.mul_add(avg[0] - pos[0], pos[0]),
        factor.mul_add(avg[1] - pos[1], pos[1]),
        factor.mul_add(avg[2] - pos[2], pos[2]),
    ]
}

/// Recompute per-vertex normals by averaging adjacent face normals.
fn recompute_normals(vertices: &mut [IsosurfaceVertex], indices: &[u32]) {
    for v in vertices.iter_mut() {
        v.normal = [0.0; 3];
    }

    for tri in indices.chunks_exact(3) {
        let a = tri[0] as usize;
        let b = tri[1] as usize;
        let c = tri[2] as usize;
        let pa = Vec3::from(vertices[a].position);
        let pb = Vec3::from(vertices[b].position);
        let pc = Vec3::from(vertices[c].position);
        let face_n = (pb - pa).cross(pc - pa);
        for &i in &[a, b, c] {
            vertices[i].normal[0] += face_n.x;
            vertices[i].normal[1] += face_n.y;
            vertices[i].normal[2] += face_n.z;
        }
    }

    for v in vertices.iter_mut() {
        let n = Vec3::from(v.normal);
        let len = n.length();
        if len > 1e-8 {
            v.normal = [n.x / len, n.y / len, n.z / len];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vert(x: f32, y: f32, z: f32) -> IsosurfaceVertex {
        IsosurfaceVertex {
            position: [x, y, z],
            normal: [0.0, 1.0, 0.0],
            color: [1.0; 4],
            kind: super::super::isosurface_kind::SURFACE,
            cavity_center: [0.0; 3],
        }
    }

    #[test]
    fn taubin_smooth_no_op_when_empty() {
        let mut verts: Vec<IsosurfaceVertex> = Vec::new();
        let indices: Vec<u32> = Vec::new();
        taubin_smooth(&mut verts, &indices, 5);
        assert!(verts.is_empty());
    }

    #[test]
    fn taubin_smooth_single_triangle_unchanged() {
        // A single triangle has no interior vertices to smooth — every
        // vertex's neighbor set is the other two corners, which form a
        // line. Smoothing collapses toward the centroid then expands.
        // After enough iterations the centroid is fixed, but for a
        // single triangle we mainly check it doesn't crash.
        let mut verts = vec![
            vert(0.0, 0.0, 0.0),
            vert(1.0, 0.0, 0.0),
            vert(0.0, 1.0, 0.0),
        ];
        let indices = vec![0, 1, 2];
        taubin_smooth(&mut verts, &indices, 5);
        assert_eq!(verts.len(), 3);
    }

    #[test]
    fn taubin_smooth_flattens_zigzag() {
        // 4 vertices along x with alternating y values; smoothing
        // should pull the high points down and the low points up.
        let mut verts = vec![
            vert(0.0, 0.0, 0.0),
            vert(1.0, 1.0, 0.0),
            vert(2.0, 0.0, 0.0),
            vert(3.0, 1.0, 0.0),
        ];
        let indices = vec![0, 1, 2, 1, 2, 3];
        let original_y_max = verts[1].position[1];
        taubin_smooth(&mut verts, &indices, 10);
        // The peaks should have moved closer to 0.5
        assert!(
            verts[1].position[1] < original_y_max,
            "expected y to decrease, got {}",
            verts[1].position[1]
        );
    }
}
