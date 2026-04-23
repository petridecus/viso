/// Per-instance data for cone impostor (arrow tip).
/// Must match the WGSL ConeInstance struct layout.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ConeInstance {
    /// xyz = base position, w = base radius
    pub(crate) base: [f32; 4],
    /// xyz = tip position, w = residue_idx
    pub(crate) tip: [f32; 4],
    /// xyz = RGB, w = unused
    pub(crate) color: [f32; 4],
    /// padding for alignment
    pub(crate) _pad: [f32; 4],
}
