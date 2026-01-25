//! Atom renderer using ray-marched spheres
//!
//! Renders atoms as spheres with hydrophobicity-based coloring.
//! Uses a storage buffer for positions that can grow dynamically.

use crate::dynamic_buffer::TypedBuffer;
use crate::render_context::RenderContext;
use glam::Vec3;

pub struct AtomRenderer {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    positions_buffer: TypedBuffer<[f32; 4]>,
    pub positions: Vec<Vec3>,
    pub atom_count: u32,
}

impl AtomRenderer {
    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Atom Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    fn create_render_pipeline(
        context: &RenderContext,
        atom_bind_group_layout: &wgpu::BindGroupLayout,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader = context
            .device
            .create_shader_module(wgpu::include_wgsl!("../assets/shaders/camera_spheres.wgsl"));

        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Atom Renderer Pipeline Layout"),
                    bind_group_layouts: &[atom_bind_group_layout, camera_layout, lighting_layout],
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

    /// Create a new AtomRenderer with positions and hydrophobicity flags
    /// hydrophobicity: true = hydrophobic (blue), false = hydrophilic (orange)
    pub fn new(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        positions: Vec<Vec3>,
        hydrophobicity: Vec<bool>,
    ) -> Self {
        let bind_group_layout = Self::create_bind_group_layout(&context.device);
        let pipeline =
            Self::create_render_pipeline(context, &bind_group_layout, camera_layout, lighting_layout);

        // Convert to GPU format [x, y, z, w] where w encodes hydrophobicity
        // w = 1.0 for hydrophobic, w = 0.0 for hydrophilic
        let positions_data: Vec<[f32; 4]> = positions
            .iter()
            .zip(hydrophobicity.iter())
            .map(|(v, &hydro)| [v.x, v.y, v.z, if hydro { 1.0 } else { 0.0 }])
            .collect();

        let positions_buffer = TypedBuffer::new_with_data(
            &context.device,
            "Atom Positions Buffer",
            &positions_data,
            wgpu::BufferUsages::STORAGE,
        );

        let bind_group = context.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: positions_buffer.buffer().as_entire_binding(),
            }],
            label: Some("Atom XYZ Bind Group"),
        });

        let atom_count = positions.len() as u32;

        Self {
            pipeline,
            bind_group,
            bind_group_layout,
            positions_buffer,
            positions,
            atom_count,
        }
    }

    pub fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        lighting_bind_group: &'a wgpu::BindGroup,
    ) {
        if self.atom_count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_bind_group(1, camera_bind_group, &[]);
        render_pass.set_bind_group(2, lighting_bind_group, &[]);
        render_pass.draw(0..6, 0..self.atom_count);
    }

    /// Update atom positions dynamically (for animation)
    /// If the new positions exceed buffer capacity, the buffer and bind group are recreated.
    pub fn update_positions(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &[Vec3],
        hydrophobicity: &[bool],
    ) {
        let positions_data: Vec<[f32; 4]> = positions
            .iter()
            .zip(hydrophobicity.iter())
            .map(|(v, &hydro)| [v.x, v.y, v.z, if hydro { 1.0 } else { 0.0 }])
            .collect();

        // TypedBuffer handles resizing automatically
        let reallocated = self.positions_buffer.write(device, queue, &positions_data);

        if reallocated {
            // Buffer was reallocated, recreate bind group
            self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.positions_buffer.buffer().as_entire_binding(),
                }],
                label: Some("Atom XYZ Bind Group"),
            });
        }

        self.positions = positions.to_vec();
        self.atom_count = positions.len() as u32;
    }
}
