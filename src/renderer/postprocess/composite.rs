//! Composite pass - applies SSAO and outline effects to the rendered scene
//!
//! This pass takes the geometry color buffer, SSAO buffer, and depth buffer,
//! combining them to produce the final image with ambient occlusion and
//! silhouette outlines applied.

use wgpu::util::DeviceExt;

use super::screen_pass::ScreenPass;
use crate::error::VisoError;
use crate::gpu::pipeline_helpers::{
    create_screen_space_pipeline, depth_texture_2d, filtering_sampler,
    linear_sampler, non_filtering_sampler, texture_2d, uniform_buffer,
};
use crate::gpu::render_context::RenderContext;
use crate::gpu::shader_composer::ShaderComposer;

/// External texture view inputs for creating a composite pass.
pub struct CompositeInputs<'a> {
    pub ssao: &'a wgpu::TextureView,
    pub depth: &'a wgpu::TextureView,
    pub normal: &'a wgpu::TextureView,
    pub bloom: &'a wgpu::TextureView,
}

/// Parameters for the composite pass effects (SSAO strength, outlines, etc.)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CompositeParams {
    /// Screen dimensions in pixels `[width, height]`.
    pub screen_size: [f32; 2],
    /// Outline thickness in texels.
    pub outline_thickness: f32,
    /// Outline darkness strength (0.0–1.0).
    pub outline_strength: f32,
    /// SSAO contribution strength.
    pub ao_strength: f32,
    /// Near clipping plane distance.
    pub near: f32,
    /// Far clipping plane distance.
    pub far: f32,
    /// Distance at which depth fog begins.
    pub fog_start: f32,
    /// Fog density factor.
    pub fog_density: f32,
    /// Normal-based outline strength.
    pub normal_outline_strength: f32,
    /// Exposure multiplier for tone mapping.
    pub exposure: f32,
    /// Gamma correction exponent.
    pub gamma: f32,
    /// Bloom blend intensity.
    pub bloom_intensity: f32,
    /// Padding for GPU alignment.
    pub _pad: f32,
    /// Padding for GPU alignment.
    pub _pad2: f32,
    /// Padding for GPU alignment.
    pub _pad3: f32,
}

impl Default for CompositeParams {
    fn default() -> Self {
        Self {
            screen_size: [1920.0, 1080.0],
            outline_thickness: 1.0,
            outline_strength: 0.7,
            ao_strength: 0.85,
            near: 5.0,
            far: 2000.0,
            fog_start: 100.0,
            fog_density: 0.005,
            normal_outline_strength: 0.5,
            exposure: 1.0,
            gamma: 1.0,
            bloom_intensity: 0.0,
            _pad: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
        }
    }
}

struct CompositeViews<'a> {
    pub color: &'a wgpu::TextureView,
    pub ssao: &'a wgpu::TextureView,
    pub depth: &'a wgpu::TextureView,
    pub normal: &'a wgpu::TextureView,
    pub bloom: &'a wgpu::TextureView,
    pub sampler: &'a wgpu::Sampler,
    pub depth_sampler: &'a wgpu::Sampler,
    pub params_buffer: &'a wgpu::Buffer,
}

/// Composite pass renderer
pub struct CompositePass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    depth_sampler: wgpu::Sampler,

    /// Intermediate color texture (geometry renders here instead of
    /// swapchain).
    pub color_texture: wgpu::Texture,
    /// View into the intermediate color texture.
    pub color_view: wgpu::TextureView,

    /// Output view (FXAA input texture), set before render.
    output_view: Option<wgpu::TextureView>,
    /// Stored SSAO view for bind group recreation on resize.
    ssao_view: wgpu::TextureView,
    /// Stored depth view for bind group recreation on resize.
    depth_view: wgpu::TextureView,
    /// Stored normal view for bind group recreation on resize.
    normal_view: wgpu::TextureView,
    /// Stored bloom view for bind group recreation on resize.
    bloom_view: wgpu::TextureView,

    /// Composite effect parameters (outline, AO, fog, tone-mapping).
    pub params: CompositeParams,
    params_buffer: wgpu::Buffer,

    width: u32,
    height: u32,
}

impl CompositePass {
    /// Create a new composite pass with all textures, samplers, and pipeline.
    pub fn new(
        context: &RenderContext,
        inputs: &CompositeInputs,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, VisoError> {
        let width = context.render_width();
        let height = context.render_height();

        let (color_texture, color_view) =
            Self::create_color_texture(context, width, height);

        let (sampler, depth_sampler) = Self::create_samplers(context);

        let params = CompositeParams {
            screen_size: [width as f32, height as f32],
            ..Default::default()
        };
        let params_buffer = context.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Composite Params Buffer"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST,
            },
        );

        let bind_group_layout = Self::create_bind_group_layout(context);

        let bind_group = Self::create_bind_group(
            context,
            &bind_group_layout,
            &CompositeViews {
                color: &color_view,
                ssao: inputs.ssao,
                depth: inputs.depth,
                normal: inputs.normal,
                bloom: inputs.bloom,
                sampler: &sampler,
                depth_sampler: &depth_sampler,
                params_buffer: &params_buffer,
            },
        );

        let pipeline = Self::create_pipeline(
            context,
            shader_composer,
            &bind_group_layout,
        )?;

        Ok(Self {
            pipeline,
            bind_group_layout,
            bind_group,
            sampler,
            depth_sampler,
            color_texture,
            color_view,
            output_view: None,
            ssao_view: inputs.ssao.clone(),
            depth_view: inputs.depth.clone(),
            normal_view: inputs.normal.clone(),
            bloom_view: inputs.bloom.clone(),
            params,
            params_buffer,
            width,
            height,
        })
    }

    fn create_samplers(
        context: &RenderContext,
    ) -> (wgpu::Sampler, wgpu::Sampler) {
        let sampler = linear_sampler(&context.device, "Composite Sampler");
        let depth_sampler =
            context.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Composite Depth Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });
        (sampler, depth_sampler)
    }

    fn create_bind_group_layout(
        context: &RenderContext,
    ) -> wgpu::BindGroupLayout {
        context.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Composite Bind Group Layout"),
                entries: &[
                    texture_2d(0),
                    texture_2d(1),
                    depth_texture_2d(2),
                    filtering_sampler(3),
                    non_filtering_sampler(4),
                    uniform_buffer(5),
                    texture_2d(6),
                    texture_2d(7),
                ],
            },
        )
    }

    fn create_pipeline(
        context: &RenderContext,
        shader_composer: &mut ShaderComposer,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<wgpu::RenderPipeline, VisoError> {
        let shader = shader_composer.compose(
            &context.device,
            "Composite Shader",
            "screen/composite.wgsl",
        )?;
        Ok(create_screen_space_pipeline(
            &context.device,
            "Composite",
            &shader,
            context.config.format,
            None,
            &[bind_group_layout],
        ))
    }

    fn create_color_texture(
        context: &RenderContext,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Intermediate Color Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        (texture, view)
    }

    fn create_bind_group(
        context: &RenderContext,
        layout: &wgpu::BindGroupLayout,
        views: &CompositeViews,
    ) -> wgpu::BindGroup {
        context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Composite Bind Group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            views.color,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            views.ssao,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            views.depth,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(views.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(
                            views.depth_sampler,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: views.params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(
                            views.normal,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(
                            views.bloom,
                        ),
                    },
                ],
            })
    }

    /// Set the output view (FXAA input texture) for this frame.
    pub fn set_output_view(&mut self, view: wgpu::TextureView) {
        self.output_view = Some(view);
    }

    /// Update the external texture views used in bind group recreation.
    pub fn set_external_views(
        &mut self,
        ssao: wgpu::TextureView,
        depth: wgpu::TextureView,
        normal: wgpu::TextureView,
        bloom: wgpu::TextureView,
    ) {
        self.ssao_view = ssao;
        self.depth_view = depth;
        self.normal_view = normal;
        self.bloom_view = bloom;
    }

    /// Get the color view for geometry rendering
    pub fn get_color_view(&self) -> &wgpu::TextureView {
        &self.color_view
    }

    /// Update fog parameters (called each frame from engine)
    pub fn update_fog(
        &mut self,
        queue: &wgpu::Queue,
        fog_start: f32,
        fog_density: f32,
    ) {
        self.params.fog_start = fog_start;
        self.params.fog_density = fog_density;
        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::cast_slice(&[self.params]),
        );
    }

    /// Flush the current params to the GPU buffer.
    pub fn flush_params(&self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::cast_slice(&[self.params]),
        );
    }
}

impl ScreenPass for CompositePass {
    fn render(&self, encoder: &mut wgpu::CommandEncoder) {
        let Some(output_view) = &self.output_view else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Composite Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    fn resize(&mut self, context: &RenderContext) {
        if context.render_width() == self.width
            && context.render_height() == self.height
        {
            return;
        }

        self.width = context.render_width();
        self.height = context.render_height();

        // Recreate color texture
        let (color_texture, color_view) =
            Self::create_color_texture(context, self.width, self.height);
        self.color_texture = color_texture;
        self.color_view = color_view;

        // Update screen_size in params
        self.params.screen_size = [self.width as f32, self.height as f32];
        context.queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::cast_slice(&[self.params]),
        );

        // Recreate bind group with stored views
        self.bind_group = Self::create_bind_group(
            context,
            &self.bind_group_layout,
            &CompositeViews {
                color: &self.color_view,
                ssao: &self.ssao_view,
                depth: &self.depth_view,
                normal: &self.normal_view,
                bloom: &self.bloom_view,
                sampler: &self.sampler,
                depth_sampler: &self.depth_sampler,
                params_buffer: &self.params_buffer,
            },
        );
    }
}
