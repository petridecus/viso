//! Bloom post-processing pass — extracts bright pixels and creates a soft glow.
//!
//! Pipeline: threshold extraction → downsample chain (4 levels) with separable
//! Gaussian blur at each level → upsample + accumulate → final bloom texture.
//! The composite pass adds the bloom texture to the scene before tone mapping.

use crate::gpu::render_context::RenderContext;
use crate::gpu::shader_composer::ShaderComposer;
use wgpu::util::DeviceExt;

/// Blur direction params — must match WGSL struct
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BlurParams {
    texel_size: [f32; 2],
    horizontal: u32,
    _pad: u32,
}

/// Number of downsample levels in the bloom chain
const MIP_LEVELS: usize = 4;

pub struct BloomPass {
    // Threshold extraction
    threshold_pipeline: wgpu::RenderPipeline,
    threshold_bind_group_layout: wgpu::BindGroupLayout,
    threshold_bind_group: wgpu::BindGroup,
    threshold_buffer: wgpu::Buffer,

    // Blur (separable Gaussian, reused for H and V at each mip)
    blur_pipeline: wgpu::RenderPipeline,
    blur_bind_group_layout: wgpu::BindGroupLayout,

    // Mip chain textures (half-res, quarter-res, etc.)
    mip_textures: Vec<wgpu::Texture>,
    mip_views: Vec<wgpu::TextureView>,
    // Ping-pong textures for H/V blur at each mip level
    ping_textures: Vec<wgpu::Texture>,
    ping_views: Vec<wgpu::TextureView>,

    // Blur bind groups: [level][0=horizontal, 1=vertical]
    blur_bind_groups: Vec<[wgpu::BindGroup; 2]>,
    blur_params_buffers: Vec<[wgpu::Buffer; 2]>,

    // Upsample + accumulate
    _upsample_pipeline: wgpu::RenderPipeline,
    upsample_bind_group_layout: wgpu::BindGroupLayout,
    upsample_bind_groups: Vec<wgpu::BindGroup>,

    // Final bloom output (same size as first mip = half resolution)
    pub output_texture: wgpu::Texture,
    pub output_view: wgpu::TextureView,

    sampler: wgpu::Sampler,

    pub threshold: f32,
    pub intensity: f32,
    width: u32,
    height: u32,
}

impl BloomPass {
    pub fn new(
        context: &RenderContext,
        color_view: &wgpu::TextureView,
        shader_composer: &mut ShaderComposer,
    ) -> Self {
        let width = context.render_width();
        let height = context.render_height();

        let sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Bloom Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let threshold = 1.0f32;
        let intensity = 0.0f32;
        let threshold_buffer = context.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Bloom Threshold Buffer"),
                contents: bytemuck::cast_slice(&[threshold]),
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST,
            },
        );

        // Create mip chain textures
        let (mip_textures, mip_views) =
            Self::create_mip_chain(context, width, height);
        let (ping_textures, ping_views) =
            Self::create_mip_chain(context, width, height);

        // Output texture (half resolution)
        let (output_texture, output_view) = Self::create_texture(
            context,
            (width / 2).max(1),
            (height / 2).max(1),
            "Bloom Output",
        );

        // --- Threshold pipeline ---
        let threshold_bind_group_layout = context
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bloom Threshold Layout"),
                entries: &[
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(
                            wgpu::SamplerBindingType::Filtering,
                        ),
                        count: None,
                    },
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
            });

        let threshold_bind_group = Self::create_threshold_bind_group(
            context,
            &threshold_bind_group_layout,
            color_view,
            &sampler,
            &threshold_buffer,
        );

        let threshold_shader = shader_composer.compose(
            &context.device,
            "Bloom Threshold Shader",
            include_str!("../../../assets/shaders/screen/bloom_threshold.wgsl"),
            "bloom_threshold.wgsl",
        );

        let threshold_pipeline_layout = context.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Bloom Threshold Pipeline Layout"),
                bind_group_layouts: &[&threshold_bind_group_layout],
                immediate_size: 0,
            },
        );

        let threshold_pipeline = context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Bloom Threshold Pipeline"),
                layout: Some(&threshold_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &threshold_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &threshold_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
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

        // --- Blur pipeline ---
        let blur_bind_group_layout = context.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Bloom Blur Layout"),
                entries: &[
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(
                            wgpu::SamplerBindingType::Filtering,
                        ),
                        count: None,
                    },
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

        let blur_shader = shader_composer.compose(
            &context.device,
            "Bloom Blur Shader",
            include_str!("../../../assets/shaders/screen/bloom_blur.wgsl"),
            "bloom_blur.wgsl",
        );

        let blur_pipeline_layout = context.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Bloom Blur Pipeline Layout"),
                bind_group_layouts: &[&blur_bind_group_layout],
                immediate_size: 0,
            },
        );

        let blur_pipeline = context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Bloom Blur Pipeline"),
                layout: Some(&blur_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &blur_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &blur_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
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

        // Create blur bind groups and params buffers for each mip level
        let (blur_bind_groups, blur_params_buffers) =
            Self::create_blur_resources(
                context,
                &blur_bind_group_layout,
                &mip_views,
                &ping_views,
                &sampler,
                width,
                height,
            );

        // --- Upsample + accumulate pipeline (reuses blur shader with additive blend) ---
        let upsample_bind_group_layout = context
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Bloom Upsample Layout"),
                entries: &[
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(
                            wgpu::SamplerBindingType::Filtering,
                        ),
                        count: None,
                    },
                ],
            });

        // Simple passthrough shader for upsample (just bilinear sample + additive blend)
        let upsample_shader = shader_composer.compose(
            &context.device,
            "Bloom Upsample Shader",
            include_str!("../../../assets/shaders/screen/bloom_upsample.wgsl"),
            "bloom_upsample.wgsl",
        );

        let upsample_pipeline_layout = context.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Bloom Upsample Pipeline Layout"),
                bind_group_layouts: &[&upsample_bind_group_layout],
                immediate_size: 0,
            },
        );

        let upsample_pipeline = context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("Bloom Upsample Pipeline"),
                layout: Some(&upsample_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &upsample_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &upsample_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::OVER,
                        }),
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

        let upsample_bind_groups = Self::create_upsample_bind_groups(
            context,
            &upsample_bind_group_layout,
            &mip_views,
            &sampler,
        );

        Self {
            threshold_pipeline,
            threshold_bind_group_layout,
            threshold_bind_group,
            threshold_buffer,
            blur_pipeline,
            blur_bind_group_layout,
            mip_textures,
            mip_views,
            ping_textures,
            ping_views,
            blur_bind_groups,
            blur_params_buffers,
            _upsample_pipeline: upsample_pipeline,
            upsample_bind_group_layout,
            upsample_bind_groups,
            output_texture,
            output_view,
            sampler,
            threshold,
            intensity,
            width,
            height,
        }
    }

    fn create_texture(
        context: &RenderContext,
        width: u32,
        height: u32,
        label: &str,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
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

    fn create_mip_chain(
        context: &RenderContext,
        width: u32,
        height: u32,
    ) -> (Vec<wgpu::Texture>, Vec<wgpu::TextureView>) {
        let mut textures = Vec::with_capacity(MIP_LEVELS);
        let mut views = Vec::with_capacity(MIP_LEVELS);

        let mut w = width;
        let mut h = height;
        for i in 0..MIP_LEVELS {
            w = (w / 2).max(1);
            h = (h / 2).max(1);
            let (tex, view) = Self::create_texture(
                context,
                w,
                h,
                &format!("Bloom Mip {}", i),
            );
            textures.push(tex);
            views.push(view);
        }

        (textures, views)
    }

    fn create_threshold_bind_group(
        context: &RenderContext,
        layout: &wgpu::BindGroupLayout,
        color_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        threshold_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bloom Threshold Bind Group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            color_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: threshold_buffer.as_entire_binding(),
                    },
                ],
            })
    }

    fn create_blur_resources(
        context: &RenderContext,
        layout: &wgpu::BindGroupLayout,
        mip_views: &[wgpu::TextureView],
        ping_views: &[wgpu::TextureView],
        sampler: &wgpu::Sampler,
        width: u32,
        height: u32,
    ) -> (Vec<[wgpu::BindGroup; 2]>, Vec<[wgpu::Buffer; 2]>) {
        let mut bind_groups = Vec::with_capacity(MIP_LEVELS);
        let mut buffers = Vec::with_capacity(MIP_LEVELS);

        let mut w = width;
        let mut h = height;
        for i in 0..MIP_LEVELS {
            w = (w / 2).max(1);
            h = (h / 2).max(1);

            let texel_size = [1.0 / w as f32, 1.0 / h as f32];

            // Horizontal: read from mip[i], write to ping[i]
            let h_params = BlurParams {
                texel_size,
                horizontal: 1,
                _pad: 0,
            };
            let h_buffer = context.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Bloom Blur H Params {}", i)),
                    contents: bytemuck::cast_slice(&[h_params]),
                    usage: wgpu::BufferUsages::UNIFORM,
                },
            );
            let h_bg =
                context
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(&format!("Bloom Blur H BG {}", i)),
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &mip_views[i],
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(
                                    sampler,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: h_buffer.as_entire_binding(),
                            },
                        ],
                    });

            // Vertical: read from ping[i], write back to mip[i]
            let v_params = BlurParams {
                texel_size,
                horizontal: 0,
                _pad: 0,
            };
            let v_buffer = context.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Bloom Blur V Params {}", i)),
                    contents: bytemuck::cast_slice(&[v_params]),
                    usage: wgpu::BufferUsages::UNIFORM,
                },
            );
            let v_bg =
                context
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(&format!("Bloom Blur V BG {}", i)),
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    &ping_views[i],
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(
                                    sampler,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: v_buffer.as_entire_binding(),
                            },
                        ],
                    });

            bind_groups.push([h_bg, v_bg]);
            buffers.push([h_buffer, v_buffer]);
        }

        (bind_groups, buffers)
    }

    fn create_upsample_bind_groups(
        context: &RenderContext,
        layout: &wgpu::BindGroupLayout,
        mip_views: &[wgpu::TextureView],
        sampler: &wgpu::Sampler,
    ) -> Vec<wgpu::BindGroup> {
        // For upsampling: we read from mip[i+1] and additively blend into mip[i]
        // So we need bind groups for mip levels 1..MIP_LEVELS (reading from those levels)
        let mut bind_groups = Vec::with_capacity(MIP_LEVELS - 1);
        for (i, mip_view) in
            mip_views.iter().enumerate().take(MIP_LEVELS).skip(1)
        {
            let bg =
                context
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(&format!("Bloom Upsample BG {}", i)),
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(
                                    mip_view,
                                ),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(
                                    sampler,
                                ),
                            },
                        ],
                    });
            bind_groups.push(bg);
        }
        bind_groups
    }

    /// Render the bloom pass: threshold → downsample+blur → upsample+accumulate
    pub fn render(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.intensity <= 0.0 {
            return;
        }

        // Step 1: Threshold extraction → mip[0] (half-res)
        {
            let mut pass =
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Bloom Threshold"),
                    color_attachments: &[Some(
                        wgpu::RenderPassColorAttachment {
                            view: &self.mip_views[0],
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        },
                    )],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
            pass.set_pipeline(&self.threshold_pipeline);
            pass.set_bind_group(0, &self.threshold_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Step 2: Progressive downsample — copy mip[i-1] into mip[i] via blur
        // For simplicity, the first blur pass at each level also acts as the downsample
        // (the bilinear sampler handles the 2x reduction)
        // We blur mip[0] in place first, then for levels 1+, we downsample from the previous level

        // Blur mip[0] (threshold output)
        self.blur_level(encoder, 0);

        // Single-level bloom: only mip[0] is used. Multi-level downsample
        // can be added later by iterating over levels 1..MIP_LEVELS.

        // Step 3: Copy blurred mip[0] → output
        // For single-level bloom, the output IS mip[0]. We'll just reference mip_views[0].
        // Actually we don't need the output texture at all for single-level — composite can
        // just read from mip_views[0].
    }

    /// Separable Gaussian blur at a given mip level (in-place via ping-pong)
    fn blur_level(&self, encoder: &mut wgpu::CommandEncoder, level: usize) {
        // Horizontal: mip[level] → ping[level]
        {
            let mut pass =
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Bloom Blur H"),
                    color_attachments: &[Some(
                        wgpu::RenderPassColorAttachment {
                            view: &self.ping_views[level],
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        },
                    )],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, &self.blur_bind_groups[level][0], &[]);
            pass.draw(0..3, 0..1);
        }

        // Vertical: ping[level] → mip[level]
        {
            let mut pass =
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Bloom Blur V"),
                    color_attachments: &[Some(
                        wgpu::RenderPassColorAttachment {
                            view: &self.mip_views[level],
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        },
                    )],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, &self.blur_bind_groups[level][1], &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Get the bloom output view for the composite pass to sample
    pub fn get_output_view(&self) -> &wgpu::TextureView {
        // Single-level bloom: output is mip[0] (half-res blurred bright pixels)
        &self.mip_views[0]
    }

    /// Rebind the input color texture (called after composite creates its color texture)
    pub fn rebind_input(
        &mut self,
        context: &RenderContext,
        color_view: &wgpu::TextureView,
    ) {
        self.threshold_bind_group = Self::create_threshold_bind_group(
            context,
            &self.threshold_bind_group_layout,
            color_view,
            &self.sampler,
            &self.threshold_buffer,
        );
    }

    /// Update threshold value on GPU
    pub fn update_params(&self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.threshold_buffer,
            0,
            bytemuck::cast_slice(&[self.threshold]),
        );
    }

    /// Resize all textures on window resize
    pub fn resize(
        &mut self,
        context: &RenderContext,
        color_view: &wgpu::TextureView,
    ) {
        let width = context.render_width();
        let height = context.render_height();
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;

        let (mip_textures, mip_views) =
            Self::create_mip_chain(context, width, height);
        let (ping_textures, ping_views) =
            Self::create_mip_chain(context, width, height);

        self.threshold_bind_group = Self::create_threshold_bind_group(
            context,
            &self.threshold_bind_group_layout,
            color_view,
            &self.sampler,
            &self.threshold_buffer,
        );

        let (blur_bind_groups, blur_params_buffers) =
            Self::create_blur_resources(
                context,
                &self.blur_bind_group_layout,
                &mip_views,
                &ping_views,
                &self.sampler,
                width,
                height,
            );

        let upsample_bind_groups = Self::create_upsample_bind_groups(
            context,
            &self.upsample_bind_group_layout,
            &mip_views,
            &self.sampler,
        );

        let (output_texture, output_view) = Self::create_texture(
            context,
            (width / 2).max(1),
            (height / 2).max(1),
            "Bloom Output",
        );

        self.mip_textures = mip_textures;
        self.mip_views = mip_views;
        self.ping_textures = ping_textures;
        self.ping_views = ping_views;
        self.blur_bind_groups = blur_bind_groups;
        self.blur_params_buffers = blur_params_buffers;
        self.upsample_bind_groups = upsample_bind_groups;
        self.output_texture = output_texture;
        self.output_view = output_view;
    }
}
