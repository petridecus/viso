use crate::gpu::render_context::RenderContext;
use crate::gpu::shader_composer::ShaderComposer;
use crate::renderer::postprocess::bloom::BloomPass;
use crate::renderer::postprocess::composite::CompositePass;
use crate::renderer::postprocess::fxaa::FxaaPass;
use crate::renderer::postprocess::ssao::SsaoRenderer;
use crate::util::options::Options;
use glam::Mat4;

/// Camera parameters needed for post-processing passes.
pub struct PostProcessCamera {
    pub proj: Mat4,
    pub view_matrix: Mat4,
    pub znear: f32,
    pub zfar: f32,
}

/// Owns the full post-processing pipeline: depth/normal G-buffers,
/// SSAO, bloom, composite, and FXAA passes.
pub(crate) struct PostProcessStack {
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    pub normal_texture: wgpu::Texture,
    pub normal_view: wgpu::TextureView,
    pub ssao_renderer: SsaoRenderer,
    pub bloom_pass: BloomPass,
    pub composite_pass: CompositePass,
    pub fxaa_pass: FxaaPass,
}

impl PostProcessStack {
    /// Build the full post-processing stack (depth/normal textures + all passes).
    pub fn new(
        context: &RenderContext,
        shader_composer: &mut ShaderComposer,
    ) -> Self {
        let (depth_texture, depth_view) = Self::create_depth_texture(context);
        let (normal_texture, normal_view) =
            Self::create_normal_texture(context);

        let ssao_renderer = SsaoRenderer::new(
            context,
            &depth_view,
            &normal_view,
            shader_composer,
        );

        let mut bloom_pass =
            BloomPass::new(context, &normal_view, shader_composer);

        let mut composite_pass = CompositePass::new(
            context,
            ssao_renderer.get_ssao_view(),
            &depth_view,
            &normal_view,
            bloom_pass.get_output_view(),
            shader_composer,
        );

        bloom_pass.rebind_input(context, composite_pass.get_color_view());

        // If sRGB, hardware does gamma correction → gamma = 1.0
        // If linear, apply gamma = 1/2.2 in shader
        composite_pass.params.gamma = if context.config.format.is_srgb() {
            1.0
        } else {
            1.0 / 2.2
        };

        let fxaa_pass = FxaaPass::new(context, shader_composer);

        Self {
            depth_texture,
            depth_view,
            normal_texture,
            normal_view,
            ssao_renderer,
            bloom_pass,
            composite_pass,
            fxaa_pass,
        }
    }

    /// Recreate all resolution-dependent resources.
    pub fn resize(&mut self, context: &RenderContext) {
        let (depth_texture, depth_view) = Self::create_depth_texture(context);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
        let (normal_texture, normal_view) =
            Self::create_normal_texture(context);
        self.normal_texture = normal_texture;
        self.normal_view = normal_view;
        self.ssao_renderer
            .resize(context, &self.depth_view, &self.normal_view);
        self.bloom_pass.resize(context, &self.normal_view);
        self.composite_pass.resize(
            context,
            self.ssao_renderer.get_ssao_view(),
            &self.depth_view,
            &self.normal_view,
            self.bloom_pass.get_output_view(),
        );
        self.bloom_pass
            .rebind_input(context, self.composite_pass.get_color_view());
        self.fxaa_pass.resize(context);
    }

    /// Run the SSAO → bloom → composite → FXAA sequence.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        camera: &PostProcessCamera,
        final_view: &wgpu::TextureView,
    ) {
        // SSAO pass
        self.ssao_renderer.update_matrices(
            queue,
            camera.proj,
            camera.view_matrix,
            camera.znear,
            camera.zfar,
        );
        self.ssao_renderer.render_ssao(encoder);

        // Bloom pass
        self.bloom_pass.render(encoder);

        // Composite pass → writes into FXAA input texture
        self.composite_pass
            .render(encoder, self.fxaa_pass.get_input_view());

        // FXAA pass → writes to swapchain
        self.fxaa_pass.render(encoder, final_view);
    }

    /// Update fog uniforms.
    pub fn update_fog(
        &mut self,
        queue: &wgpu::Queue,
        fog_start: f32,
        fog_density: f32,
    ) {
        self.composite_pass
            .update_fog(queue, fog_start, fog_density);
    }

    /// Push post-processing option values to GPU.
    pub fn apply_options(&mut self, options: &Options, queue: &wgpu::Queue) {
        let pp = &options.post_processing;
        self.composite_pass.params.outline_thickness = pp.outline_thickness;
        self.composite_pass.params.outline_strength = pp.outline_strength;
        self.composite_pass.params.ao_strength = pp.ao_strength;
        self.composite_pass.params.normal_outline_strength =
            pp.normal_outline_strength;
        self.composite_pass.params.exposure = pp.exposure;
        self.composite_pass.params.bloom_intensity = pp.bloom_intensity;
        self.composite_pass.flush_params(queue);

        self.bloom_pass.threshold = pp.bloom_threshold;
        self.bloom_pass.intensity = pp.bloom_intensity;
        self.bloom_pass.update_params(queue);

        self.ssao_renderer.radius = pp.ao_radius;
        self.ssao_renderer.bias = pp.ao_bias;
        self.ssao_renderer.power = pp.ao_power;
    }

    /// The color texture view used as render target for the geometry pass.
    pub fn color_view(&self) -> &wgpu::TextureView {
        &self.composite_pass.color_view
    }

    fn create_depth_texture(
        context: &RenderContext,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let size = wgpu::Extent3d {
            width: context.render_width(),
            height: context.render_height(),
            depth_or_array_layers: 1,
        };

        let texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_normal_texture(
        context: &RenderContext,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let size = wgpu::Extent3d {
            width: context.render_width(),
            height: context.render_height(),
            depth_or_array_layers: 1,
        };

        let texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Normal G-Buffer"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }
}
