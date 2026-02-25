//! FXAA post-process pass — screen-space anti-aliasing applied after
//! compositing.
//!
//! Reads the composited color image and outputs an anti-aliased version to the
//! swapchain. Smooths jagged silhouette edges on mesh-based geometry (ribbons,
//! tubes) that supersampling alone doesn't fully resolve.

use wgpu::util::DeviceExt;

use super::screen_pass::ScreenPass;
use crate::error::VisoError;
use crate::gpu::pipeline_helpers::{
    create_screen_space_pipeline, filtering_sampler, linear_sampler,
    texture_2d, uniform_buffer, ScreenSpacePipelineDef,
};
use crate::gpu::render_context::RenderContext;
use crate::gpu::shader_composer::{Shader, ShaderComposer};

/// FXAA (Fast Approximate Anti-Aliasing) post-process pass.
pub struct FxaaPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    screen_size_buffer: wgpu::Buffer,
    /// Intermediate texture: composite renders here, FXAA reads from it
    pub input_texture: wgpu::Texture,
    /// View into the FXAA input texture.
    pub input_view: wgpu::TextureView,
    /// Swapchain surface view, set each frame before render.
    output_view: Option<wgpu::TextureView>,
    width: u32,
    height: u32,
}

impl FxaaPass {
    /// Create a new FXAA pass with input texture, sampler, and pipeline.
    pub fn new(
        context: &RenderContext,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, VisoError> {
        let width = context.render_width();
        let height = context.render_height();

        let (input_texture, input_view) =
            Self::create_input_texture(context, width, height);

        let sampler = linear_sampler(&context.device, "FXAA Sampler");

        let screen_size: [f32; 2] = [width as f32, height as f32];
        let screen_size_buffer = context.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("FXAA Screen Size Buffer"),
                contents: bytemuck::cast_slice(&screen_size),
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST,
            },
        );

        let bind_group_layout = Self::create_bind_group_layout(context);

        let bind_group = Self::create_bind_group(
            context,
            &bind_group_layout,
            &input_view,
            &sampler,
            &screen_size_buffer,
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
            screen_size_buffer,
            input_texture,
            input_view,
            output_view: None,
            width,
            height,
        })
    }

    fn create_bind_group_layout(
        context: &RenderContext,
    ) -> wgpu::BindGroupLayout {
        context.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("FXAA Bind Group Layout"),
                entries: &[
                    texture_2d(0),
                    filtering_sampler(1),
                    uniform_buffer(2),
                ],
            },
        )
    }

    fn create_pipeline(
        context: &RenderContext,
        shader_composer: &mut ShaderComposer,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<wgpu::RenderPipeline, VisoError> {
        let shader = shader_composer.compose(&context.device, Shader::Fxaa)?;
        Ok(create_screen_space_pipeline(
            &context.device,
            &ScreenSpacePipelineDef {
                label: "FXAA",
                shader: &shader,
                format: context.config.format,
                blend: None,
                bind_group_layouts: &[bind_group_layout],
            },
        ))
    }

    fn create_input_texture(
        context: &RenderContext,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("FXAA Input Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: context.config.format,
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
        input_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        screen_size_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("FXAA Bind Group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            input_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: screen_size_buffer.as_entire_binding(),
                    },
                ],
            })
    }

    /// Set the output view (swapchain surface) for this frame.
    pub fn set_output_view(&mut self, view: wgpu::TextureView) {
        self.output_view = Some(view);
    }

    /// Get the input view for composite to render into.
    pub fn get_input_view(&self) -> &wgpu::TextureView {
        &self.input_view
    }
}

impl ScreenPass for FxaaPass {
    fn render(&self, encoder: &mut wgpu::CommandEncoder) {
        let Some(output_view) = &self.output_view else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("FXAA Pass"),
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
        let width = context.render_width();
        let height = context.render_height();
        if width == self.width && height == self.height {
            return;
        }

        self.width = width;
        self.height = height;

        let (input_texture, input_view) =
            Self::create_input_texture(context, width, height);
        self.input_texture = input_texture;
        self.input_view = input_view;

        let screen_size: [f32; 2] = [width as f32, height as f32];
        context.queue.write_buffer(
            &self.screen_size_buffer,
            0,
            bytemuck::cast_slice(&screen_size),
        );

        self.bind_group = Self::create_bind_group(
            context,
            &self.bind_group_layout,
            &self.input_view,
            &self.sampler,
            &self.screen_size_buffer,
        );
    }
}
