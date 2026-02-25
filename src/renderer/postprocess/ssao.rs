use glam::Mat4;
use rand::Rng;
use wgpu::util::DeviceExt;

use super::screen_pass::ScreenPass;
use crate::error::VisoError;
use crate::gpu::pipeline_helpers::{
    create_screen_space_pipeline, depth_texture_2d, filtering_sampler,
    linear_sampler, non_filtering_sampler, texture_2d, texture_2d_unfilterable,
    uniform_buffer, ScreenSpacePipelineDef,
};
use crate::gpu::render_context::RenderContext;
use crate::gpu::shader_composer::{Shader, ShaderComposer};

/// SSAO parameters uniform - must match WGSL struct
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SsaoParams {
    /// Inverse projection matrix.
    pub inv_proj: [[f32; 4]; 4],
    /// Projection matrix.
    pub proj: [[f32; 4]; 4],
    /// View matrix.
    pub view: [[f32; 4]; 4],
    /// Screen dimensions in pixels `[width, height]`.
    pub screen_size: [f32; 2],
    /// Near clipping plane distance.
    pub near: f32,
    /// Far clipping plane distance.
    pub far: f32,
    /// SSAO sampling radius in view space.
    pub radius: f32,
    /// Depth bias to prevent self-occlusion.
    pub bias: f32,
    /// Exponent applied to the AO factor.
    pub power: f32,
    /// Padding for GPU alignment.
    pub _pad: f32,
}

/// SSAO (Screen Space Ambient Occlusion) renderer
struct SsaoViews<'a> {
    pub depth: &'a wgpu::TextureView,
    pub noise: &'a wgpu::TextureView,
    pub normal: &'a wgpu::TextureView,
    pub sampler: &'a wgpu::Sampler,
    pub noise_sampler: &'a wgpu::Sampler,
    pub kernel_buffer: &'a wgpu::Buffer,
    pub params_buffer: &'a wgpu::Buffer,
}

/// Inputs for creating the SSAO blur bind group.
struct SsaoBlurInputs<'a> {
    ssao_view: &'a wgpu::TextureView,
    sampler: &'a wgpu::Sampler,
    depth_view: &'a wgpu::TextureView,
    normal_view: &'a wgpu::TextureView,
    params_buffer: &'a wgpu::Buffer,
}

/// SSAO (Screen-Space Ambient Occlusion) renderer.
pub struct SsaoRenderer {
    /// Raw SSAO output texture (before blur).
    pub ssao_texture: wgpu::Texture,
    /// View into the raw SSAO texture.
    pub ssao_view: wgpu::TextureView,
    /// Blurred SSAO output texture.
    pub ssao_blurred_texture: wgpu::Texture,
    /// View into the blurred SSAO texture.
    pub ssao_blurred_view: wgpu::TextureView,

    /// GPU buffer holding the sample kernel.
    pub kernel_buffer: wgpu::Buffer,
    /// GPU buffer holding SSAO parameters uniform.
    pub params_buffer: wgpu::Buffer,

    /// Random rotation noise texture.
    #[allow(dead_code)] // must stay alive to back noise_view
    pub noise_texture: wgpu::Texture,
    /// View into the noise texture.
    pub noise_view: wgpu::TextureView,
    /// Sampler for the noise texture (repeat addressing).
    pub noise_sampler: wgpu::Sampler,
    /// Sampler for the SSAO texture (clamp-to-edge).
    pub ssao_sampler: wgpu::Sampler,

    /// Render pipeline for the SSAO pass.
    pub ssao_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for the SSAO pass.
    pub ssao_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for the SSAO pass.
    pub ssao_bind_group: wgpu::BindGroup,
    /// Render pipeline for the SSAO blur pass.
    pub blur_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for the blur pass.
    pub blur_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for the blur pass.
    pub blur_bind_group: wgpu::BindGroup,

    /// Stored depth view for bind group recreation on resize.
    depth_view: wgpu::TextureView,
    /// Stored normal view for bind group recreation on resize.
    normal_view: wgpu::TextureView,

    width: u32,
    height: u32,

    /// SSAO sampling radius in view space.
    pub radius: f32,
    /// Depth bias for self-occlusion prevention.
    pub bias: f32,
    /// Exponent applied to the AO factor.
    pub power: f32,
}

const KERNEL_SIZE: usize = 32;
const NOISE_SIZE: u32 = 4;

/// Pipelines and their bind group layouts for the SSAO passes.
struct SsaoPipelines {
    ssao: wgpu::RenderPipeline,
    ssao_layout: wgpu::BindGroupLayout,
    blur: wgpu::RenderPipeline,
    blur_layout: wgpu::BindGroupLayout,
}

/// Create both SSAO pipelines (main + blur) and their bind group layouts.
fn create_ssao_pipelines(
    context: &RenderContext,
    shader_composer: &mut ShaderComposer,
) -> Result<SsaoPipelines, VisoError> {
    let ssao_layout = SsaoRenderer::create_ssao_bind_group_layout(context);
    let ssao = SsaoRenderer::create_ssao_pipeline(
        context,
        shader_composer,
        &ssao_layout,
    )?;

    let blur_layout = SsaoRenderer::create_blur_bind_group_layout(context);
    let blur = SsaoRenderer::create_blur_pipeline(
        context,
        shader_composer,
        &blur_layout,
    )?;

    Ok(SsaoPipelines {
        ssao,
        ssao_layout,
        blur,
        blur_layout,
    })
}

struct SsaoTextures {
    ssao: wgpu::Texture,
    ssao_view: wgpu::TextureView,
    blurred: wgpu::Texture,
    blurred_view: wgpu::TextureView,
}

fn create_ssao_textures(
    context: &RenderContext,
    width: u32,
    height: u32,
) -> SsaoTextures {
    let ssao = SsaoRenderer::create_ssao_texture(context, width, height);
    let ssao_view = ssao.create_view(&Default::default());
    let blurred = SsaoRenderer::create_ssao_texture(context, width, height);
    let blurred_view = blurred.create_view(&Default::default());
    SsaoTextures {
        ssao,
        ssao_view,
        blurred,
        blurred_view,
    }
}

impl SsaoRenderer {
    /// Create a new SSAO renderer with kernel, noise, and pipeline resources.
    pub fn new(
        context: &RenderContext,
        depth_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, VisoError> {
        let width = context.render_width();
        let height = context.render_height();

        let tex = create_ssao_textures(context, width, height);
        let (kernel_buffer, params_buffer, radius, bias, power) =
            Self::create_buffers(context, width, height);
        let (noise_texture, noise_view) = Self::create_noise_texture(context);
        let (noise_sampler, ssao_sampler) = Self::create_samplers(context);
        let pipelines = create_ssao_pipelines(context, shader_composer)?;

        let ssao_bind_group = Self::create_ssao_bind_group(
            context,
            &pipelines.ssao_layout,
            &SsaoViews {
                depth: depth_view,
                noise: &noise_view,
                normal: normal_view,
                sampler: &ssao_sampler,
                noise_sampler: &noise_sampler,
                kernel_buffer: &kernel_buffer,
                params_buffer: &params_buffer,
            },
        );

        let blur_bind_group = Self::create_blur_bind_group(
            context,
            &pipelines.blur_layout,
            &SsaoBlurInputs {
                ssao_view: &tex.ssao_view,
                sampler: &ssao_sampler,
                depth_view,
                normal_view,
                params_buffer: &params_buffer,
            },
        );

        Ok(Self {
            ssao_texture: tex.ssao,
            ssao_view: tex.ssao_view,
            ssao_blurred_texture: tex.blurred,
            ssao_blurred_view: tex.blurred_view,
            kernel_buffer,
            params_buffer,
            noise_texture,
            noise_view,
            noise_sampler,
            ssao_sampler,
            ssao_pipeline: pipelines.ssao,
            ssao_bind_group_layout: pipelines.ssao_layout,
            ssao_bind_group,
            blur_pipeline: pipelines.blur,
            blur_bind_group_layout: pipelines.blur_layout,
            blur_bind_group,
            depth_view: depth_view.clone(),
            normal_view: normal_view.clone(),
            width,
            height,
            radius,
            bias,
            power,
        })
    }

    fn create_buffers(
        context: &RenderContext,
        width: u32,
        height: u32,
    ) -> (wgpu::Buffer, wgpu::Buffer, f32, f32, f32) {
        let kernel = Self::generate_kernel();
        let kernel_buffer = context.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("SSAO Kernel"),
                contents: bytemuck::cast_slice(&kernel),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );

        let radius = 0.5;
        let bias = 0.025;
        let power = 2.0;
        let params = SsaoParams {
            inv_proj: Mat4::IDENTITY.to_cols_array_2d(),
            proj: Mat4::IDENTITY.to_cols_array_2d(),
            view: Mat4::IDENTITY.to_cols_array_2d(),
            screen_size: [width as f32, height as f32],
            near: 0.1,
            far: 1000.0,
            radius,
            bias,
            power,
            _pad: 0.0,
        };
        let params_buffer = context.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("SSAO Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM
                    | wgpu::BufferUsages::COPY_DST,
            },
        );

        (kernel_buffer, params_buffer, radius, bias, power)
    }

    fn create_samplers(
        context: &RenderContext,
    ) -> (wgpu::Sampler, wgpu::Sampler) {
        let noise_sampler =
            context.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("SSAO Noise Sampler"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Nearest,
                min_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            });
        let ssao_sampler = linear_sampler(&context.device, "SSAO Sampler");
        (noise_sampler, ssao_sampler)
    }

    fn create_ssao_bind_group_layout(
        context: &RenderContext,
    ) -> wgpu::BindGroupLayout {
        context.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("SSAO Bind Group Layout"),
                entries: &[
                    depth_texture_2d(0),
                    texture_2d_unfilterable(1),
                    filtering_sampler(2),
                    non_filtering_sampler(3),
                    uniform_buffer(4),
                    uniform_buffer(5),
                    texture_2d(6),
                ],
            },
        )
    }

    fn create_ssao_pipeline(
        context: &RenderContext,
        shader_composer: &mut ShaderComposer,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<wgpu::RenderPipeline, VisoError> {
        let shader = shader_composer.compose(&context.device, Shader::Ssao)?;
        Ok(create_screen_space_pipeline(
            &context.device,
            &ScreenSpacePipelineDef {
                label: "SSAO",
                shader: &shader,
                format: wgpu::TextureFormat::R8Unorm,
                blend: None,
                bind_group_layouts: &[bind_group_layout],
            },
        ))
    }

    fn create_blur_bind_group_layout(
        context: &RenderContext,
    ) -> wgpu::BindGroupLayout {
        context.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("SSAO Blur Layout"),
                entries: &[
                    texture_2d(0),
                    filtering_sampler(1),
                    depth_texture_2d(2),
                    texture_2d(3),
                    uniform_buffer(4),
                ],
            },
        )
    }

    fn create_blur_pipeline(
        context: &RenderContext,
        shader_composer: &mut ShaderComposer,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<wgpu::RenderPipeline, VisoError> {
        let shader =
            shader_composer.compose(&context.device, Shader::SsaoBlur)?;
        Ok(create_screen_space_pipeline(
            &context.device,
            &ScreenSpacePipelineDef {
                label: "SSAO Blur",
                shader: &shader,
                format: wgpu::TextureFormat::R8Unorm,
                blend: None,
                bind_group_layouts: &[bind_group_layout],
            },
        ))
    }

    fn create_ssao_texture(
        context: &RenderContext,
        width: u32,
        height: u32,
    ) -> wgpu::Texture {
        context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSAO Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn generate_kernel() -> [[f32; 4]; KERNEL_SIZE] {
        let mut rng = rand::rng();
        let mut kernel = [[0.0f32; 4]; KERNEL_SIZE];

        for (i, kernel_sample) in kernel.iter_mut().enumerate() {
            // Random point in hemisphere (positive Z)
            let mut sample = [
                rng.random::<f32>() * 2.0 - 1.0,
                rng.random::<f32>() * 2.0 - 1.0,
                rng.random::<f32>(),
                0.0,
            ];

            // Normalize
            let len = (sample[0] * sample[0]
                + sample[1] * sample[1]
                + sample[2] * sample[2])
                .sqrt();
            if len > 0.0 {
                sample[0] /= len;
                sample[1] /= len;
                sample[2] /= len;
            }

            // Scale - more samples closer to origin
            let mut scale = i as f32 / KERNEL_SIZE as f32;
            scale = 0.1 + scale * scale * 0.9;
            sample[0] *= scale;
            sample[1] *= scale;
            sample[2] *= scale;

            *kernel_sample = sample;
        }

        kernel
    }

    fn create_noise_texture(
        context: &RenderContext,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let mut rng = rand::rng();
        let mut noise_data = vec![0u8; (NOISE_SIZE * NOISE_SIZE * 4) as usize];

        for i in 0..(NOISE_SIZE * NOISE_SIZE) as usize {
            let x = rng.random::<f32>() * 2.0 - 1.0;
            let y = rng.random::<f32>() * 2.0 - 1.0;
            let len = x.hypot(y);
            let (nx, ny) = if len > 0.0 {
                (x / len, y / len)
            } else {
                (1.0, 0.0)
            };

            noise_data[i * 4] = ((nx * 0.5 + 0.5) * 255.0) as u8;
            noise_data[i * 4 + 1] = ((ny * 0.5 + 0.5) * 255.0) as u8;
            noise_data[i * 4 + 2] = 128; // Z component
            noise_data[i * 4 + 3] = 255;
        }

        let texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SSAO Noise Texture"),
            size: wgpu::Extent3d {
                width: NOISE_SIZE,
                height: NOISE_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        context.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &noise_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(NOISE_SIZE * 4),
                rows_per_image: Some(NOISE_SIZE),
            },
            wgpu::Extent3d {
                width: NOISE_SIZE,
                height: NOISE_SIZE,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&Default::default());
        (texture, view)
    }

    fn create_ssao_bind_group(
        context: &RenderContext,
        layout: &wgpu::BindGroupLayout,
        views: &SsaoViews,
    ) -> wgpu::BindGroup {
        context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SSAO Bind Group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            views.depth,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            views.noise,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(views.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(
                            views.noise_sampler,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: views.kernel_buffer.as_entire_binding(),
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
                ],
            })
    }

    fn create_blur_bind_group(
        context: &RenderContext,
        layout: &wgpu::BindGroupLayout,
        inputs: &SsaoBlurInputs,
    ) -> wgpu::BindGroup {
        context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SSAO Blur Bind Group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            inputs.ssao_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            inputs.sampler,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            inputs.depth_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(
                            inputs.normal_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: inputs.params_buffer.as_entire_binding(),
                    },
                ],
            })
    }

    /// Update projection and view matrices (call before render_ssao)
    pub fn update_matrices(
        &self,
        queue: &wgpu::Queue,
        camera: &super::post_process::PostProcessCamera,
    ) {
        let inv_proj = camera.proj.inverse();
        let params = SsaoParams {
            inv_proj: inv_proj.to_cols_array_2d(),
            proj: camera.proj.to_cols_array_2d(),
            view: camera.view_matrix.to_cols_array_2d(),
            screen_size: [self.width as f32, self.height as f32],
            near: camera.znear,
            far: camera.zfar,
            radius: self.radius,
            bias: self.bias,
            power: self.power,
            _pad: 0.0,
        };
        queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::cast_slice(&[params]),
        );
    }

    /// Update the external geometry views used in bind group recreation.
    pub fn set_geometry_views(
        &mut self,
        depth: wgpu::TextureView,
        normal: wgpu::TextureView,
    ) {
        self.depth_view = depth;
        self.normal_view = normal;
    }

    /// Render SSAO (call after geometry pass)
    pub fn render_ssao(&self, encoder: &mut wgpu::CommandEncoder) {
        // SSAO pass
        {
            let mut pass =
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("SSAO Pass"),
                    color_attachments: &[Some(
                        wgpu::RenderPassColorAttachment {
                            view: &self.ssao_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        },
                    )],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });

            pass.set_pipeline(&self.ssao_pipeline);
            pass.set_bind_group(0, &self.ssao_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Blur pass
        {
            let mut pass =
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("SSAO Blur Pass"),
                    color_attachments: &[Some(
                        wgpu::RenderPassColorAttachment {
                            view: &self.ssao_blurred_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        },
                    )],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });

            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, &self.blur_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    /// Get the final (blurred) SSAO texture view for the composite pass.
    pub fn get_ssao_view(&self) -> &wgpu::TextureView {
        &self.ssao_blurred_view
    }
}

impl ScreenPass for SsaoRenderer {
    fn render(&self, encoder: &mut wgpu::CommandEncoder) {
        self.render_ssao(encoder);
    }

    fn resize(&mut self, context: &RenderContext) {
        if context.render_width() == self.width
            && context.render_height() == self.height
        {
            return;
        }

        self.width = context.render_width();
        self.height = context.render_height();

        self.ssao_texture =
            Self::create_ssao_texture(context, self.width, self.height);
        self.ssao_view = self.ssao_texture.create_view(&Default::default());
        self.ssao_blurred_texture =
            Self::create_ssao_texture(context, self.width, self.height);
        self.ssao_blurred_view =
            self.ssao_blurred_texture.create_view(&Default::default());

        self.ssao_bind_group = Self::create_ssao_bind_group(
            context,
            &self.ssao_bind_group_layout,
            &SsaoViews {
                depth: &self.depth_view,
                noise: &self.noise_view,
                normal: &self.normal_view,
                sampler: &self.ssao_sampler,
                noise_sampler: &self.noise_sampler,
                kernel_buffer: &self.kernel_buffer,
                params_buffer: &self.params_buffer,
            },
        );

        self.blur_bind_group = Self::create_blur_bind_group(
            context,
            &self.blur_bind_group_layout,
            &SsaoBlurInputs {
                ssao_view: &self.ssao_view,
                sampler: &self.ssao_sampler,
                depth_view: &self.depth_view,
                normal_view: &self.normal_view,
                params_buffer: &self.params_buffer,
            },
        );
    }
}
