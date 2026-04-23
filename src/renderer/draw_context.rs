/// Bind groups shared across all molecular draw calls.
pub(crate) struct DrawBindGroups<'a> {
    /// Camera uniform bind group (view-projection, position, etc.).
    pub(crate) camera: &'a wgpu::BindGroup,
    /// Lighting uniform bind group.
    pub(crate) lighting: &'a wgpu::BindGroup,
    /// Selection state storage buffer bind group.
    pub(crate) selection: &'a wgpu::BindGroup,
    /// Per-residue color override (used by backbone renderer only).
    pub(crate) color: Option<&'a wgpu::BindGroup>,
}
