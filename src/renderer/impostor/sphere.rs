/// Per-instance data for sphere impostor.
/// Must match the WGSL SphereInstance struct layout.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SphereInstance {
    /// xyz = position, w = radius
    pub(crate) center: [f32; 4],
    /// xyz = RGB color, w = entity_id (packed as float)
    pub(crate) color: [f32; 4],
}
