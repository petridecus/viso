/// Per-instance data for cone impostor (arrow tip).
/// Must match the WGSL ConeInstance struct layout.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConeInstance {
    /// xyz = base position, w = base radius
    pub base: [f32; 4],
    /// xyz = tip position, w = residue_idx
    pub tip: [f32; 4],
    /// xyz = RGB, w = unused
    pub color: [f32; 4],
    /// padding for alignment
    pub _pad: [f32; 4],
}
