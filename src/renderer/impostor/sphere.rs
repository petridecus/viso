/// Per-instance data for sphere impostor.
/// Must match the WGSL SphereInstance struct layout.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SphereInstance {
    /// xyz = position, w = radius
    pub center: [f32; 4],
    /// xyz = RGB color, w = entity_id (packed as float)
    pub color: [f32; 4],
}
