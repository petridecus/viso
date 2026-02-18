//! FXAA post-process pass — screen-space anti-aliasing applied after
//! compositing.
//!
//! Reads the composited color image and outputs an anti-aliased version to the
//! swapchain. Smooths jagged silhouette edges on mesh-based geometry (ribbons,
//! tubes) that supersampling alone doesn't fully resolve.

use wgpu::util::DeviceExt;

use crate::gpu::{
    render_context::RenderContext, shader_composer::ShaderComposer,
};

pub struct FxaaPass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    screen_size_buffer: wgpu::Buffer,
    /// Intermediate texture: composite renders here, FXAA reads from it
    pub input_texture: wgpu::Texture,
    pub input_view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl FxaaPass {
    pub fn new(
        context: &RenderContext,
        shader_composer: &mut ShaderComposer,
    ) -> Self {
        let width = context.render_width();
        let height = context.render_height();

        let (input_texture, input_view) =
            Self::create_input_texture(context, width, height);

        let sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("FXAA Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let screen_size: [f32; 2] = [width as f32, height as f32];
        let screen_size_buffer = context.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("FXAA Screen Size Buffer"),
                contents: bytemuck::cast_slice(&screen_size),
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST,
            },
        );

        let bind_group_layout = context.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("FXAA Bind Group Layout"),
                entries: &[
                    // binding 0: input color texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float {
                                filterable: true,
                            },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 1: sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(
                            wgpu::SamplerBindingType::Filtering,
                        ),
                        count: None,
                    },
                    // binding 2: screen_size uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            },
        );

        let bind_group = Self::create_bind_group(
            context,
            &bind_group_layout,
            &input_view,
            &sampler,
            &screen_size_buffer,
        );

        let shader = shader_composer.compose(
            &context.device,
            "FXAA Shader",
            include_str!("../../../assets/shaders/screen/fxaa.wgsl"),
            "fxaa.wgsl",
        );

        let pipeline_layout = context.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("FXAA Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            },
        );

        let pipeline = context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("FXAA Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: context.config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        );

        Self {
            pipeline,
            bind_group_layout,
            bind_group,
            sampler,
            screen_size_buffer,
            input_texture,
            input_view,
            width,
            height,
        }
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

    /// Render FXAA pass: read from input_view, write to output_view
    /// (swapchain).
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
    ) {
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

    /// Get the input view for composite to render into.
    pub fn get_input_view(&self) -> &wgpu::TextureView {
        &self.input_view
    }

    /// Resize textures and recreate bind group on window resize.
    pub fn resize(&mut self, context: &RenderContext) {
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
