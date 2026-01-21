use crate::render_context::RenderContext;
use glam::Vec3;

const COUNT: u32 = 100;

pub struct AtomRenderer {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    pub depth_view: wgpu::TextureView,
    pub positions: Vec<Vec3>,
}

impl AtomRenderer {
    fn create_render_pipeline(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = context
            .device
            .create_shader_module(wgpu::include_wgsl!("../assets/shaders/camera_spheres.wgsl"));

        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..3)
            .map(|ii| wgpu::BindGroupLayoutEntry {
                binding: ii,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();

        let atom_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Atom Layout"),
                    entries: &entries,
                });

        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Atom Renderer Pipeline Layout"),
                    bind_group_layouts: &[&atom_bind_group_layout, camera_layout],
                    immediate_size: 0,
                });

        context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
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
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
    }

    fn generate_positions() -> Vec<Vec3> {
        let mut positions = Vec::with_capacity(COUNT as usize);
        for _ in 0..COUNT {
            positions.push(Vec3::new(
                rand::random::<f32>() * 100.0,
                rand::random::<f32>() * 100.0,
                rand::random::<f32>() * 100.0,
            ));
        }
        positions
    }

    fn random_xyz_bind_group(
        context: &RenderContext,
        pipeline: &wgpu::RenderPipeline,
        positions: &[Vec3],
    ) -> wgpu::BindGroup {
        let mut xs = Vec::with_capacity(positions.len());
        let mut ys = Vec::with_capacity(positions.len());
        let mut zs = Vec::with_capacity(positions.len());

        for pos in positions {
            xs.push(pos.x);
            ys.push(pos.y);
            zs.push(pos.z);
        }

        use wgpu::util::DeviceExt;
        let x_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("X Offset Buffer"),
                contents: bytemuck::cast_slice(&xs),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let y_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Y Offset Buffer"),
                contents: bytemuck::cast_slice(&ys),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let z_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Z Offset Buffer"),
                contents: bytemuck::cast_slice(&zs),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: x_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: y_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: z_buffer.as_entire_binding(),
                    },
                ],
                label: Some("Random XYZ Bind Group"),
            });

        bind_group
    }

    pub fn create_depth_view(context: &RenderContext) -> wgpu::TextureView {
        let size = wgpu::Extent3d {
            width: context.config.width,
            height: context.config.height,
            depth_or_array_layers: 1,
        };

        let desc = wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        };

        context
            .device
            .create_texture(&desc)
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    pub async fn new(context: &RenderContext, camera_layout: &wgpu::BindGroupLayout) -> Self {
        let pipeline = Self::create_render_pipeline(&context, camera_layout);
        let positions = Self::generate_positions();
        let bind_group = Self::random_xyz_bind_group(&context, &pipeline, &positions);
        let depth_view = Self::create_depth_view(&context);

        Self {
            pipeline,
            bind_group,
            depth_view,
            positions,
        }
    }

    pub fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
    ) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_bind_group(1, camera_bind_group, &[]);
        render_pass.draw(0..6, 0..COUNT);
    }
}
