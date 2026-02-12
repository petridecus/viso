//! Composite pass - applies SSAO and outline effects to the rendered scene
//!
//! This pass takes the geometry color buffer, SSAO buffer, and depth buffer,
//! combining them to produce the final image with ambient occlusion and
//! silhouette outlines applied.

use crate::render_context::RenderContext;
use wgpu::util::DeviceExt;

/// Parameters for the composite pass effects (SSAO strength, outlines, etc.)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CompositeParams {
    pub screen_size: [f32; 2],
    pub outline_thickness: f32,  // 1.0 = 1 texel
    pub outline_strength: f32,   // 0.0-1.0, how dark
    pub ao_strength: f32,
    pub near: f32,
    pub far: f32,
    pub fog_start: f32,
    pub fog_density: f32,
    pub _pad: f32,
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
            _pad: 0.0,
        }
    }
}

/// Composite pass renderer
pub struct CompositePass {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    depth_sampler: wgpu::Sampler,

    // Intermediate color texture (geometry renders here instead of swapchain)
    pub color_texture: wgpu::Texture,
    pub color_view: wgpu::TextureView,

    // Composite params for outline/SSAO control
    pub params: CompositeParams,
    params_buffer: wgpu::Buffer,

    width: u32,
    height: u32,
}

impl CompositePass {
    pub fn new(
        context: &RenderContext,
        ssao_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
    ) -> Self {
        let width = context.render_width();
        let height = context.render_height();

        // Create intermediate color texture (geometry renders here)
        let (color_texture, color_view) = Self::create_color_texture(context, width, height);

        // Sampler for color and SSAO textures
        let sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Composite Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Sampler for depth texture (comparison sampler not needed, just filtering)
        let depth_sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Composite Depth Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create params buffer
        let mut params = CompositeParams::default();
        params.screen_size = [width as f32, height as f32];
        let params_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Composite Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Bind group layout
        let bind_group_layout = context.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Composite Bind Group Layout"),
            entries: &[
                // binding 0: color texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 1: SSAO texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 2: depth texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 3: sampler (for color/ssao)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 4: depth sampler (NonFiltering — sampler uses Nearest)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // binding 5: params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create bind group
        let bind_group = Self::create_bind_group(
            context,
            &bind_group_layout,
            &color_view,
            ssao_view,
            depth_view,
            &sampler,
            &depth_sampler,
            &params_buffer,
        );

        // Load shader
        let shader = context
            .device
            .create_shader_module(wgpu::include_wgsl!("../assets/shaders/composite.wgsl"));

        // Pipeline layout
        let pipeline_layout = context
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Composite Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            });

        // Render pipeline
        let pipeline = context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Composite Pipeline"),
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
            });

        Self {
            pipeline,
            bind_group_layout,
            bind_group,
            sampler,
            depth_sampler,
            color_texture,
            color_view,
            params,
            params_buffer,
            width,
            height,
        }
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
            format: context.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        (texture, view)
    }

    fn create_bind_group(
        context: &RenderContext,
        layout: &wgpu::BindGroupLayout,
        color_view: &wgpu::TextureView,
        ssao_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        depth_sampler: &wgpu::Sampler,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Composite Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(ssao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(depth_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Render the composite pass to the output view (swapchain)
    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, output_view: &wgpu::TextureView) {
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
        pass.draw(0..3, 0..1); // Full-screen triangle
    }

    /// Get the color view for geometry rendering
    pub fn get_color_view(&self) -> &wgpu::TextureView {
        &self.color_view
    }

    /// Update fog parameters (called each frame from engine)
    pub fn update_fog(&mut self, queue: &wgpu::Queue, fog_start: f32, fog_density: f32) {
        self.params.fog_start = fog_start;
        self.params.fog_density = fog_density;
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
    }

    /// Flush the current params to the GPU buffer.
    pub fn flush_params(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));
    }

    /// Resize textures and recreate bind groups
    pub fn resize(&mut self, context: &RenderContext, ssao_view: &wgpu::TextureView, depth_view: &wgpu::TextureView) {
        if context.render_width() == self.width && context.render_height() == self.height {
            return;
        }

        self.width = context.render_width();
        self.height = context.render_height();

        // Recreate color texture
        let (color_texture, color_view) = Self::create_color_texture(context, self.width, self.height);
        self.color_texture = color_texture;
        self.color_view = color_view;

        // Update screen_size in params (write to existing buffer, no bind group recreation)
        self.params.screen_size = [self.width as f32, self.height as f32];
        context.queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[self.params]));

        // Recreate bind group with new textures
        self.bind_group = Self::create_bind_group(
            context,
            &self.bind_group_layout,
            &self.color_view,
            ssao_view,
            depth_view,
            &self.sampler,
            &self.depth_sampler,
            &self.params_buffer,
        );
    }
}
