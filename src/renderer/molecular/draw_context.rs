/// Bind groups shared across all molecular draw calls.
pub struct DrawBindGroups<'a> {
    pub camera: &'a wgpu::BindGroup,
    pub lighting: &'a wgpu::BindGroup,
    pub selection: &'a wgpu::BindGroup,
    /// Per-residue color override (used by tube/ribbon only).
    pub color: Option<&'a wgpu::BindGroup>,
}
