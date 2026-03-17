/// Per-instance data for extruded polygon impostor (nucleic acid rings).
///
/// Encodes a convex polygon (5 or 6 vertices) extruded by `half_thickness`
/// along its face normal.  The vertex shader procedurally generates top face,
/// bottom face, and side-wall triangles.
///
/// 128 bytes per instance.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ExtrudedPolygonInstance {
    /// xyz = vertex 0, w = vertex count (5.0 or 6.0)
    pub v0: [f32; 4],
    /// xyz = vertex 1, w = half_thickness
    pub v1: [f32; 4],
    /// xyz = vertex 2, w = unused
    pub v2: [f32; 4],
    /// xyz = vertex 3, w = unused
    pub v3: [f32; 4],
    /// xyz = vertex 4, w = unused
    pub v4: [f32; 4],
    /// xyz = vertex 5 (degenerate = centroid for pentagons), w = unused
    pub v5: [f32; 4],
    /// xyz = face normal, w = residue_idx (packed as float)
    pub normal: [f32; 4],
    /// xyz = RGB, w = unused
    pub color: [f32; 4],
}
