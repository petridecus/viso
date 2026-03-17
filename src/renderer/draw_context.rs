/// Bind groups shared across all molecular draw calls.
pub struct DrawBindGroups<'a> {
    /// Camera uniform bind group (view-projection, position, etc.).
    pub camera: &'a wgpu::BindGroup,
    /// Lighting uniform bind group.
    pub lighting: &'a wgpu::BindGroup,
    /// Selection state storage buffer bind group.
    pub selection: &'a wgpu::BindGroup,
    /// Per-residue color override (used by backbone renderer only).
    pub color: Option<&'a wgpu::BindGroup>,
}
