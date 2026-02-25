use crate::gpu::render_context::RenderContext;

/// Uniform interface for fullscreen post-processing passes.
pub trait ScreenPass {
    /// Encode GPU commands for this pass.
    fn render(&self, encoder: &mut wgpu::CommandEncoder);
    /// Recreate resolution-dependent resources.
    /// External texture views must be updated via pass-specific setters
    /// BEFORE calling this.
    fn resize(&mut self, context: &RenderContext);
}
