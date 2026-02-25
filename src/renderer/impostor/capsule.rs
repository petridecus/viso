/// Per-instance data for capsule impostor.
/// Must match the WGSL CapsuleInstance struct layout exactly.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CapsuleInstance {
    /// Endpoint A position (xyz), radius (w)
    pub endpoint_a: [f32; 4],
    /// Endpoint B position (xyz), residue_idx (w) - packed as float
    pub endpoint_b: [f32; 4],
    /// Color at endpoint A (RGB), w unused
    pub color_a: [f32; 4],
    /// Color at endpoint B (RGB), w unused
    pub color_b: [f32; 4],
}
