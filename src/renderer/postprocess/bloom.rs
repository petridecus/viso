//! Bloom post-processing pass — extracts bright pixels and creates a soft glow.
//!
//! Pipeline: threshold extraction → downsample chain (4 levels) with separable
//! Gaussian blur at each level → upsample + accumulate → final bloom texture.
//! The composite pass adds the bloom texture to the scene before tone mapping.

use wgpu::util::DeviceExt;

use super::screen_pass::ScreenPass;
use crate::error::VisoError;
use crate::gpu::pipeline_helpers::{
    create_screen_space_pipeline, filtering_sampler, linear_sampler,
    texture_2d, uniform_buffer, ScreenSpacePipelineDef,
};
use crate::gpu::render_context::RenderContext;
use crate::gpu::shader_composer::{Shader, ShaderComposer};

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

const BLOOM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Additive blend state used by the upsample pass.
const ADDITIVE_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent::OVER,
};

/// Bloom post-processing pass (threshold + blur chain + upsample).
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

    /// Final bloom output texture (half resolution).
    pub output_texture: wgpu::Texture,
    /// View into the bloom output texture.
    pub output_view: wgpu::TextureView,

    sampler: wgpu::Sampler,
    /// Stored input color view for bind group recreation on resize.
    input_color_view: wgpu::TextureView,

    /// Brightness threshold for bloom extraction.
    pub threshold: f32,
    /// Bloom blend intensity.
    pub intensity: f32,
    width: u32,
    height: u32,
}

/// Configuration for creating blur resources across all mip levels.
struct BlurChainConfig<'a> {
    layout: &'a wgpu::BindGroupLayout,
    mip_views: &'a [wgpu::TextureView],
    ping_views: &'a [wgpu::TextureView],
    sampler: &'a wgpu::Sampler,
    width: u32,
    height: u32,
}

/// Configuration for creating blur resources at a single mip level.
struct BlurLevelConfig<'a> {
    layout: &'a wgpu::BindGroupLayout,
    mip_view: &'a wgpu::TextureView,
    ping_view: &'a wgpu::TextureView,
    sampler: &'a wgpu::Sampler,
    width: u32,
    height: u32,
    level: usize,
}

/// Pipelines and their bind group layouts for the bloom pass.
struct BloomPipelines {
    threshold: wgpu::RenderPipeline,
    blur: wgpu::RenderPipeline,
    upsample: wgpu::RenderPipeline,
    threshold_layout: wgpu::BindGroupLayout,
    blur_layout: wgpu::BindGroupLayout,
    upsample_layout: wgpu::BindGroupLayout,
}

/// Create mip, ping, and output textures for the bloom pass.
fn create_bloom_textures(
    context: &RenderContext,
    width: u32,
    height: u32,
) -> BloomTextures {
    let (mip_textures, mip_views) =
        BloomPass::create_mip_chain(context, width, height);
    let (ping_textures, ping_views) =
        BloomPass::create_mip_chain(context, width, height);
    let (output_texture, output_view) = BloomPass::create_texture(
        context,
        (width / 2).max(1),
        (height / 2).max(1),
        "Bloom Output",
    );
    BloomTextures {
        mip_textures,
        mip_views,
        ping_textures,
        ping_views,
        output_texture,
        output_view,
    }
}

struct BloomTextures {
    mip_textures: Vec<wgpu::Texture>,
    mip_views: Vec<wgpu::TextureView>,
    ping_textures: Vec<wgpu::Texture>,
    ping_views: Vec<wgpu::TextureView>,
    output_texture: wgpu::Texture,
    output_view: wgpu::TextureView,
}

/// Create all three bloom pipelines (threshold, blur, upsample) and their
/// bind group layouts.
fn create_bloom_pipelines(
    context: &RenderContext,
    shader_composer: &mut ShaderComposer,
) -> Result<BloomPipelines, VisoError> {
    let threshold_layout = BloomPass::create_threshold_layout(context);
    let threshold = BloomPass::create_threshold_pipeline(
        context,
        shader_composer,
        &threshold_layout,
    )?;

    let blur_layout = BloomPass::create_blur_layout(context);
    let blur = BloomPass::create_blur_pipeline(
        context,
        shader_composer,
        &blur_layout,
    )?;

    let upsample_layout = BloomPass::create_upsample_layout(context);
    let upsample = BloomPass::create_upsample_pipeline(
        context,
        shader_composer,
        &upsample_layout,
    )?;

    Ok(BloomPipelines {
        threshold,
        blur,
        upsample,
        threshold_layout,
        blur_layout,
        upsample_layout,
    })
}

fn create_threshold_buffer(
    device: &wgpu::Device,
    threshold: f32,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Bloom Threshold Buffer"),
        contents: bytemuck::cast_slice(&[threshold]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

impl BloomPass {
    /// Create a new bloom pass with threshold, blur chain, and upsample
    /// pipelines.
    pub fn new(
        context: &RenderContext,
        color_view: &wgpu::TextureView,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, VisoError> {
        let width = context.render_width();
        let height = context.render_height();

        let sampler = linear_sampler(&context.device, "Bloom Sampler");

        let (threshold, intensity) = (1.0f32, 0.0f32);
        let threshold_buffer =
            create_threshold_buffer(&context.device, threshold);

        let tex = create_bloom_textures(context, width, height);
        let pipelines = create_bloom_pipelines(context, shader_composer)?;

        let threshold_bind_group = Self::create_threshold_bind_group(
            context,
            &pipelines.threshold_layout,
            color_view,
            &sampler,
            &threshold_buffer,
        );
        let blur_config = BlurChainConfig {
            layout: &pipelines.blur_layout,
            mip_views: &tex.mip_views,
            ping_views: &tex.ping_views,
            sampler: &sampler,
            width,
            height,
        };
        let (blur_bind_groups, blur_params_buffers) =
            Self::create_blur_resources(context, &blur_config);
        let upsample_bind_groups = Self::create_upsample_bind_groups(
            context,
            &pipelines.upsample_layout,
            &tex.mip_views,
            &sampler,
        );

        Ok(Self {
            threshold_pipeline: pipelines.threshold,
            threshold_bind_group_layout: pipelines.threshold_layout,
            threshold_bind_group,
            threshold_buffer,
            blur_pipeline: pipelines.blur,
            blur_bind_group_layout: pipelines.blur_layout,
            mip_textures: tex.mip_textures,
            mip_views: tex.mip_views,
            ping_textures: tex.ping_textures,
            ping_views: tex.ping_views,
            blur_bind_groups,
            blur_params_buffers,
            _upsample_pipeline: pipelines.upsample,
            upsample_bind_group_layout: pipelines.upsample_layout,
            upsample_bind_groups,
            output_texture: tex.output_texture,
            output_view: tex.output_view,
            sampler,
            input_color_view: color_view.clone(),
            threshold,
            intensity,
            width,
            height,
        })
    }

    // -- Layout helpers --

    fn create_threshold_layout(
        context: &RenderContext,
    ) -> wgpu::BindGroupLayout {
        context.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Bloom Threshold Layout"),
                entries: &[
                    texture_2d(0),
                    filtering_sampler(1),
                    uniform_buffer(2),
                ],
            },
        )
    }

    fn create_blur_layout(context: &RenderContext) -> wgpu::BindGroupLayout {
        context.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Bloom Blur Layout"),
                entries: &[
                    texture_2d(0),
                    filtering_sampler(1),
                    uniform_buffer(2),
                ],
            },
        )
    }

    fn create_upsample_layout(
        context: &RenderContext,
    ) -> wgpu::BindGroupLayout {
        context.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Bloom Upsample Layout"),
                entries: &[texture_2d(0), filtering_sampler(1)],
            },
        )
    }

    // -- Pipeline helpers --

    fn create_threshold_pipeline(
        context: &RenderContext,
        shader_composer: &mut ShaderComposer,
        layout: &wgpu::BindGroupLayout,
    ) -> Result<wgpu::RenderPipeline, VisoError> {
        let shader =
            shader_composer.compose(&context.device, Shader::BloomThreshold)?;
        Ok(create_screen_space_pipeline(
            &context.device,
            &ScreenSpacePipelineDef {
                label: "Bloom Threshold",
                shader: &shader,
                format: BLOOM_FORMAT,
                blend: None,
                bind_group_layouts: &[layout],
            },
        ))
    }

    fn create_blur_pipeline(
        context: &RenderContext,
        shader_composer: &mut ShaderComposer,
        layout: &wgpu::BindGroupLayout,
    ) -> Result<wgpu::RenderPipeline, VisoError> {
        let shader =
            shader_composer.compose(&context.device, Shader::BloomBlur)?;
        Ok(create_screen_space_pipeline(
            &context.device,
            &ScreenSpacePipelineDef {
                label: "Bloom Blur",
                shader: &shader,
                format: BLOOM_FORMAT,
                blend: None,
                bind_group_layouts: &[layout],
            },
        ))
    }

    fn create_upsample_pipeline(
        context: &RenderContext,
        shader_composer: &mut ShaderComposer,
        layout: &wgpu::BindGroupLayout,
    ) -> Result<wgpu::RenderPipeline, VisoError> {
        let shader =
            shader_composer.compose(&context.device, Shader::BloomUpsample)?;
        Ok(create_screen_space_pipeline(
            &context.device,
            &ScreenSpacePipelineDef {
                label: "Bloom Upsample",
                shader: &shader,
                format: BLOOM_FORMAT,
                blend: Some(ADDITIVE_BLEND),
                bind_group_layouts: &[layout],
            },
        ))
    }

    // -- Texture helpers --

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
            format: BLOOM_FORMAT,
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
            let (tex, view) =
                Self::create_texture(context, w, h, &format!("Bloom Mip {i}"));
            textures.push(tex);
            views.push(view);
        }

        (textures, views)
    }

    // -- Bind group helpers --

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
        config: &BlurChainConfig,
    ) -> (Vec<[wgpu::BindGroup; 2]>, Vec<[wgpu::Buffer; 2]>) {
        let mut bind_groups = Vec::with_capacity(MIP_LEVELS);
        let mut buffers = Vec::with_capacity(MIP_LEVELS);

        let mut w = config.width;
        let mut h = config.height;
        for i in 0..MIP_LEVELS {
            w = (w / 2).max(1);
            h = (h / 2).max(1);
            let level_cfg = BlurLevelConfig {
                layout: config.layout,
                mip_view: &config.mip_views[i],
                ping_view: &config.ping_views[i],
                sampler: config.sampler,
                width: w,
                height: h,
                level: i,
            };
            let (bg, buf) =
                Self::create_blur_level_resources(context, &level_cfg);
            bind_groups.push(bg);
            buffers.push(buf);
        }

        (bind_groups, buffers)
    }

    fn create_blur_level_resources(
        context: &RenderContext,
        cfg: &BlurLevelConfig,
    ) -> ([wgpu::BindGroup; 2], [wgpu::Buffer; 2]) {
        let (h_bg, h_buffer) =
            Self::create_blur_direction(context, cfg, cfg.mip_view, 1, "H");
        let (v_bg, v_buffer) =
            Self::create_blur_direction(context, cfg, cfg.ping_view, 0, "V");

        ([h_bg, v_bg], [h_buffer, v_buffer])
    }

    fn create_blur_direction(
        context: &RenderContext,
        cfg: &BlurLevelConfig,
        source_view: &wgpu::TextureView,
        horizontal: u32,
        dir_label: &str,
    ) -> (wgpu::BindGroup, wgpu::Buffer) {
        let level = cfg.level;
        let params = BlurParams {
            texel_size: [1.0 / cfg.width as f32, 1.0 / cfg.height as f32],
            horizontal,
            _pad: 0,
        };
        let buffer = context.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Bloom Blur {dir_label} Params {level}")),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        let bg = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Bloom Blur {dir_label} BG {level}")),
                layout: cfg.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            source_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(cfg.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buffer.as_entire_binding(),
                    },
                ],
            });
        (bg, buffer)
    }

    fn create_upsample_bind_groups(
        context: &RenderContext,
        layout: &wgpu::BindGroupLayout,
        mip_views: &[wgpu::TextureView],
        sampler: &wgpu::Sampler,
    ) -> Vec<wgpu::BindGroup> {
        // For upsampling: we read from mip[i+1] and additively blend into
        // mip[i] So we need bind groups for mip levels 1..MIP_LEVELS
        // (reading from those levels)
        let mut bind_groups = Vec::with_capacity(MIP_LEVELS - 1);
        for (i, mip_view) in
            mip_views.iter().enumerate().take(MIP_LEVELS).skip(1)
        {
            let bg =
                context
                    .device
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(&format!("Bloom Upsample BG {i}")),
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
    fn render_bloom(&self, encoder: &mut wgpu::CommandEncoder) {
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
        // For simplicity, the first blur pass at each level also acts as the
        // downsample (the bilinear sampler handles the 2x reduction)
        // We blur mip[0] in place first, then for levels 1+, we downsample from
        // the previous level

        // Blur mip[0] (threshold output)
        self.blur_level(encoder, 0);

        // Single-level bloom: only mip[0] is used. Multi-level downsample
        // can be added later by iterating over levels 1..MIP_LEVELS.

        // Step 3: Copy blurred mip[0] → output
        // For single-level bloom, the output IS mip[0]. We'll just reference
        // mip_views[0]. Actually we don't need the output texture at
        // all for single-level — composite can just read from
        // mip_views[0].
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

    /// Rebind the input color texture (called after composite creates its color
    /// texture)
    pub fn rebind_input(
        &mut self,
        context: &RenderContext,
        color_view: &wgpu::TextureView,
    ) {
        self.input_color_view = color_view.clone();
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
}

impl ScreenPass for BloomPass {
    fn render(&self, encoder: &mut wgpu::CommandEncoder) {
        self.render_bloom(encoder);
    }

    fn resize(&mut self, context: &RenderContext) {
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
            &self.input_color_view,
            &self.sampler,
            &self.threshold_buffer,
        );

        let (blur_bind_groups, blur_params_buffers) =
            Self::create_blur_resources(
                context,
                &BlurChainConfig {
                    layout: &self.blur_bind_group_layout,
                    mip_views: &mip_views,
                    ping_views: &ping_views,
                    sampler: &self.sampler,
                    width,
                    height,
                },
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
