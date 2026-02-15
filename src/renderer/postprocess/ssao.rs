use crate::engine::render_context::RenderContext;
use crate::engine::shader_composer::ShaderComposer;
use glam::Mat4;
use rand::Rng;
use wgpu::util::DeviceExt;

/// SSAO parameters uniform - must match WGSL struct
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SsaoParams {
    pub inv_proj: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub screen_size: [f32; 2],
    pub near: f32,
    pub far: f32,
    pub radius: f32,
    pub bias: f32,
    pub power: f32,
    pub _pad: f32,
}

/// SSAO (Screen Space Ambient Occlusion) renderer
pub struct SsaoRenderer {
    // SSAO output textures
    pub ssao_texture: wgpu::Texture,
    pub ssao_view: wgpu::TextureView,
    pub ssao_blurred_texture: wgpu::Texture,
    pub ssao_blurred_view: wgpu::TextureView,

    // Buffers
    pub kernel_buffer: wgpu::Buffer,
    pub params_buffer: wgpu::Buffer,

    // Noise texture
    pub noise_texture: wgpu::Texture,
    pub noise_view: wgpu::TextureView,
    pub noise_sampler: wgpu::Sampler,
    pub ssao_sampler: wgpu::Sampler,

    // Pipelines and bind groups
    pub ssao_pipeline: wgpu::RenderPipeline,
    pub ssao_bind_group_layout: wgpu::BindGroupLayout,
    pub ssao_bind_group: wgpu::BindGroup,
    pub blur_pipeline: wgpu::RenderPipeline,
    pub blur_bind_group_layout: wgpu::BindGroupLayout,
    pub blur_bind_group: wgpu::BindGroup,

    width: u32,
    height: u32,

    // Tunable SSAO parameters
    pub radius: f32,
    pub bias: f32,
    pub power: f32,
}

const KERNEL_SIZE: usize = 32;
const NOISE_SIZE: u32 = 4;

impl SsaoRenderer {
    pub fn new(context: &RenderContext, depth_view: &wgpu::TextureView, normal_view: &wgpu::TextureView, shader_composer: &mut ShaderComposer) -> Self {
        let width = context.render_width();
        let height = context.render_height();

        // Create output textures
        let ssao_texture = Self::create_ssao_texture(context, width, height);
        let ssao_view = ssao_texture.create_view(&Default::default());
        let ssao_blurred_texture = Self::create_ssao_texture(context, width, height);
        let ssao_blurred_view = ssao_blurred_texture.create_view(&Default::default());

        // Generate hemisphere kernel
        let kernel = Self::generate_kernel();
        let kernel_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SSAO Kernel"),
            contents: bytemuck::cast_slice(&kernel),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create params buffer with default values
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
        let params_buffer = context.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SSAO Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create noise texture
        let (noise_texture, noise_view) = Self::create_noise_texture(context);

        // Samplers
        let noise_sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SSAO Noise Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let ssao_sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SSAO Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // SSAO bind group layout
        let ssao_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("SSAO Bind Group Layout"),
                    entries: &[
                        // binding 0: depth texture
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Depth,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // binding 1: noise texture
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // binding 2: depth sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        // binding 3: noise sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                            count: None,
                        },
                        // binding 4: kernel
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 5: params
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
                        // binding 6: normal G-buffer texture
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                    ],
                });

        // SSAO pipeline
        let ssao_shader = shader_composer.compose(&context.device, "SSAO Shader", include_str!("../../../assets/shaders/screen/ssao.wgsl"), "ssao.wgsl");

        let ssao_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("SSAO Pipeline Layout"),
                    bind_group_layouts: &[&ssao_bind_group_layout],
                    immediate_size: 0,
                });

        let ssao_pipeline =
            context
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("SSAO Pipeline"),
                    layout: Some(&ssao_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &ssao_shader,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &ssao_shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::R8Unorm,
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

        // Blur bind group layout (bilateral: needs depth + normal + params)
        let blur_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("SSAO Blur Layout"),
                    entries: &[
                        // binding 0: SSAO texture
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
                        // binding 1: sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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
                        // binding 3: normal texture
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // binding 4: SSAO params (for near/far)
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
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

        // Blur pipeline
        let blur_shader = shader_composer.compose(&context.device, "SSAO Blur Shader", include_str!("../../../assets/shaders/screen/ssao_blur.wgsl"), "ssao_blur.wgsl");

        let blur_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("SSAO Blur Pipeline Layout"),
                    bind_group_layouts: &[&blur_bind_group_layout],
                    immediate_size: 0,
                });

        let blur_pipeline =
            context
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("SSAO Blur Pipeline"),
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
                            format: wgpu::TextureFormat::R8Unorm,
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

        // Create bind groups
        let ssao_bind_group = Self::create_ssao_bind_group(
            context,
            &ssao_bind_group_layout,
            depth_view,
            &noise_view,
            &ssao_sampler,
            &noise_sampler,
            &kernel_buffer,
            &params_buffer,
            normal_view,
        );

        let blur_bind_group = Self::create_blur_bind_group(
            context,
            &blur_bind_group_layout,
            &ssao_view,
            &ssao_sampler,
            depth_view,
            normal_view,
            &params_buffer,
        );

        Self {
            ssao_texture,
            ssao_view,
            ssao_blurred_texture,
            ssao_blurred_view,
            kernel_buffer,
            params_buffer,
            noise_texture,
            noise_view,
            noise_sampler,
            ssao_sampler,
            ssao_pipeline,
            ssao_bind_group_layout,
            ssao_bind_group,
            blur_pipeline,
            blur_bind_group_layout,
            blur_bind_group,
            width,
            height,
            radius,
            bias,
            power,
        }
    }

    fn create_ssao_texture(context: &RenderContext, width: u32, height: u32) -> wgpu::Texture {
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    fn generate_kernel() -> [[f32; 4]; KERNEL_SIZE] {
        let mut rng = rand::rng();
        let mut kernel = [[0.0f32; 4]; KERNEL_SIZE];

        for i in 0..KERNEL_SIZE {
            // Random point in hemisphere (positive Z)
            let mut sample = [
                rng.random::<f32>() * 2.0 - 1.0,
                rng.random::<f32>() * 2.0 - 1.0,
                rng.random::<f32>(),
                0.0,
            ];

            // Normalize
            let len = (sample[0] * sample[0] + sample[1] * sample[1] + sample[2] * sample[2]).sqrt();
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

            kernel[i] = sample;
        }

        kernel
    }

    fn create_noise_texture(context: &RenderContext) -> (wgpu::Texture, wgpu::TextureView) {
        let mut rng = rand::rng();
        let mut noise_data = vec![0u8; (NOISE_SIZE * NOISE_SIZE * 4) as usize];

        for i in 0..(NOISE_SIZE * NOISE_SIZE) as usize {
            let x = rng.random::<f32>() * 2.0 - 1.0;
            let y = rng.random::<f32>() * 2.0 - 1.0;
            let len = (x * x + y * y).sqrt();
            let (nx, ny) = if len > 0.0 { (x / len, y / len) } else { (1.0, 0.0) };

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
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
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
        depth_view: &wgpu::TextureView,
        noise_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        noise_sampler: &wgpu::Sampler,
        kernel_buffer: &wgpu::Buffer,
        params_buffer: &wgpu::Buffer,
        normal_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSAO Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(noise_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(noise_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: kernel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
            ],
        })
    }

    fn create_blur_bind_group(
        context: &RenderContext,
        layout: &wgpu::BindGroupLayout,
        ssao_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        depth_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        params_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSAO Blur Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(ssao_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        })
    }

    /// Update projection and view matrices (call before render_ssao)
    pub fn update_matrices(&self, queue: &wgpu::Queue, proj: Mat4, view: Mat4, near: f32, far: f32) {
        let inv_proj = proj.inverse();
        let params = SsaoParams {
            inv_proj: inv_proj.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            screen_size: [self.width as f32, self.height as f32],
            near,
            far,
            radius: self.radius,
            bias: self.bias,
            power: self.power,
            _pad: 0.0,
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&[params]));
    }

    pub fn resize(&mut self, context: &RenderContext, depth_view: &wgpu::TextureView, normal_view: &wgpu::TextureView) {
        if context.render_width() == self.width && context.render_height() == self.height {
            return;
        }

        self.width = context.render_width();
        self.height = context.render_height();

        self.ssao_texture = Self::create_ssao_texture(context, self.width, self.height);
        self.ssao_view = self.ssao_texture.create_view(&Default::default());
        self.ssao_blurred_texture = Self::create_ssao_texture(context, self.width, self.height);
        self.ssao_blurred_view = self.ssao_blurred_texture.create_view(&Default::default());

        self.ssao_bind_group = Self::create_ssao_bind_group(
            context,
            &self.ssao_bind_group_layout,
            depth_view,
            &self.noise_view,
            &self.ssao_sampler,
            &self.noise_sampler,
            &self.kernel_buffer,
            &self.params_buffer,
            normal_view,
        );

        self.blur_bind_group = Self::create_blur_bind_group(
            context,
            &self.blur_bind_group_layout,
            &self.ssao_view,
            &self.ssao_sampler,
            depth_view,
            normal_view,
            &self.params_buffer,
        );
    }

    /// Render SSAO (call after geometry pass)
    pub fn render_ssao(&self, encoder: &mut wgpu::CommandEncoder) {
        // SSAO pass
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSAO Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ssao_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_pipeline(&self.ssao_pipeline);
            pass.set_bind_group(0, &self.ssao_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Blur pass
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSAO Blur Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.ssao_blurred_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_pipeline(&self.blur_pipeline);
            pass.set_bind_group(0, &self.blur_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }

    pub fn get_ssao_view(&self) -> &wgpu::TextureView {
        &self.ssao_blurred_view
    }
}
