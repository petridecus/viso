use crate::gpu::RenderContext;

/// Uniform interface for fullscreen post-processing passes.
pub trait ScreenPass {
    /// Encode GPU commands for this pass.
    fn render(&self, encoder: &mut wgpu::CommandEncoder);
    /// Recreate resolution-dependent resources.
    /// External texture views must be updated via pass-specific setters
    /// BEFORE calling this.
    fn resize(&mut self, context: &RenderContext);
}

/// Descriptor for a fullscreen screen-space pass dispatch.
pub(crate) struct ScreenPassDesc<'a> {
    /// Debug label for the render pass.
    pub(crate) label: &'a str,
    /// Render target view.
    pub(crate) view: &'a wgpu::TextureView,
    /// Render pipeline to bind.
    pub(crate) pipeline: &'a wgpu::RenderPipeline,
    /// Bind group (slot 0) containing pass inputs.
    pub(crate) bind_group: &'a wgpu::BindGroup,
    /// Clear color for the render target.
    pub(crate) clear_color: wgpu::Color,
}

/// Run a fullscreen screen-space pass: begin render pass, set pipeline, draw
/// 3 vertices. Covers the common dispatch pattern shared by every post-process
/// pass (SSAO, bloom, composite, FXAA).
pub fn run_screen_pass(
    encoder: &mut wgpu::CommandEncoder,
    desc: &ScreenPassDesc,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(desc.label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: desc.view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(desc.clear_color),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        ..Default::default()
    });
    pass.set_pipeline(desc.pipeline);
    pass.set_bind_group(0, desc.bind_group, &[]);
    pass.draw(0..3, 0..1);
}
