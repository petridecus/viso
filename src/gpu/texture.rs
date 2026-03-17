//! Framework-agnostic render-target texture abstraction.

/// A render-target texture and its default view.
///
/// Used to decouple texture creation from any windowing or GUI framework.
/// The texture is created with `RENDER_ATTACHMENT | TEXTURE_BINDING | COPY_SRC`
/// usage flags, making it suitable for off-screen rendering followed by
/// read-back or compositing.
pub struct RenderTarget {
    /// The underlying GPU texture.
    pub texture: wgpu::Texture,
    /// A default full-texture view.
    pub view: wgpu::TextureView,
}

impl RenderTarget {
    /// Create a new render-target texture with the given dimensions and format.
    #[must_use]
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RenderTarget"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { texture, view }
    }
}
