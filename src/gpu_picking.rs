//! GPU-based picking using a dedicated picking render pass
//!
//! Renders residue indices to an offscreen buffer, then reads back
//! the pixel at the mouse position to determine which residue is under the cursor.
//! This is exact - it matches exactly what's rendered on screen.

use crate::render_context::RenderContext;
use crate::tube_renderer::tube_vertex_buffer_layout;

/// GPU picking buffer that stores residue indices
pub struct GpuPicking {
    /// Picking texture (R32Uint format for residue indices)
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    /// Depth texture for the picking pass
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    /// Staging buffer for reading back pixel data
    staging_buffer: wgpu::Buffer,
    /// Pipeline for rendering tubes to picking buffer
    tube_pipeline: wgpu::RenderPipeline,
    /// Pipeline for rendering capsules to picking buffer
    capsule_pipeline: wgpu::RenderPipeline,
    /// Bind group layout for capsule storage buffer
    capsule_bind_group_layout: wgpu::BindGroupLayout,
    /// Current dimensions
    width: u32,
    height: u32,
    /// Currently hovered residue (-1 if none)
    pub hovered_residue: i32,
    /// Currently selected residue indices
    pub selected_residues: Vec<i32>,
    /// Pending mouse position for async readback
    pending_mouse_pos: Option<(u32, u32)>,
}

impl GpuPicking {
    pub fn new(
        context: &RenderContext,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let width = context.config.width;
        let height = context.config.height;

        let (texture, texture_view) = Self::create_picking_texture(&context.device, width, height);
        let (depth_texture, depth_view) = Self::create_depth_texture(&context.device, width, height);

        // Staging buffer for single pixel readback (256 bytes minimum, we only need 4)
        let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Picking Staging Buffer"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Load picking shader for tubes
        let tube_shader = context
            .device
            .create_shader_module(wgpu::include_wgsl!("../assets/shaders/picking.wgsl"));

        // Tube picking pipeline
        let tube_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Picking Tube Pipeline Layout"),
                    bind_group_layouts: &[camera_bind_group_layout],
                    immediate_size: 0,
                });

        let tube_pipeline =
            context
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Picking Tube Pipeline"),
                    layout: Some(&tube_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &tube_shader,
                        entry_point: Some("vs_main"),
                        buffers: &[tube_vertex_buffer_layout()],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &tube_shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::R32Uint,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        cull_mode: Some(wgpu::Face::Back),
                        ..Default::default()
                    },
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
                });

        // Capsule picking pipeline
        let capsule_shader = context
            .device
            .create_shader_module(wgpu::include_wgsl!("../assets/shaders/picking_capsule.wgsl"));

        let capsule_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Picking Capsule Bind Group Layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let capsule_pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Picking Capsule Pipeline Layout"),
                    bind_group_layouts: &[&capsule_bind_group_layout, camera_bind_group_layout],
                    immediate_size: 0,
                });

        let capsule_pipeline =
            context
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Picking Capsule Pipeline"),
                    layout: Some(&capsule_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &capsule_shader,
                        entry_point: Some("vs_main"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &capsule_shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::R32Uint,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        cull_mode: None,
                        ..Default::default()
                    },
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
                });

        Self {
            texture,
            texture_view,
            depth_texture,
            depth_view,
            staging_buffer,
            tube_pipeline,
            capsule_pipeline,
            capsule_bind_group_layout,
            width,
            height,
            hovered_residue: -1,
            selected_residues: Vec::new(),
            pending_mouse_pos: None,
        }
    }

    fn create_picking_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Picking Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Picking Depth Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;
        let (texture, texture_view) = Self::create_picking_texture(device, width, height);
        self.texture = texture;
        self.texture_view = texture_view;
        let (depth_texture, depth_view) = Self::create_depth_texture(device, width, height);
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
    }

    /// Create a bind group for capsule storage buffer
    pub fn create_capsule_bind_group(
        &self,
        device: &wgpu::Device,
        capsule_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Picking Capsule Bind Group"),
            layout: &self.capsule_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: capsule_buffer.as_entire_binding(),
            }],
        })
    }

    /// Render the picking pass and request readback at mouse position
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        camera_bind_group: &wgpu::BindGroup,
        tube_vertex_buffer: &wgpu::Buffer,
        tube_index_buffer: &wgpu::Buffer,
        tube_index_count: u32,
        capsule_bind_group: Option<&wgpu::BindGroup>,
        capsule_count: u32,
        mouse_x: u32,
        mouse_y: u32,
    ) {
        // Render picking pass
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Picking Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            // Draw tubes
            if tube_index_count > 0 {
                render_pass.set_pipeline(&self.tube_pipeline);
                render_pass.set_bind_group(0, camera_bind_group, &[]);
                render_pass.set_vertex_buffer(0, tube_vertex_buffer.slice(..));
                render_pass.set_index_buffer(tube_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..tube_index_count, 0, 0..1);
            }

            // Draw capsules
            if let Some(capsule_bg) = capsule_bind_group {
                if capsule_count > 0 {
                    render_pass.set_pipeline(&self.capsule_pipeline);
                    render_pass.set_bind_group(0, capsule_bg, &[]);
                    render_pass.set_bind_group(1, camera_bind_group, &[]);
                    render_pass.draw(0..6, 0..capsule_count);
                }
            }
        }

        // Copy pixel at mouse position to staging buffer
        if mouse_x < self.width && mouse_y < self.height {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: mouse_x,
                        y: mouse_y,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.staging_buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            self.pending_mouse_pos = Some((mouse_x, mouse_y));
        }
    }

    /// Complete the readback (call after queue.submit)
    pub fn complete_readback(&mut self, device: &wgpu::Device) {
        if self.pending_mouse_pos.is_none() {
            return;
        }

        let buffer_slice = self.staging_buffer.slice(..4);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        if let Ok(Ok(())) = receiver.recv() {
            let data = buffer_slice.get_mapped_range();
            let residue_id = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            drop(data);
            self.staging_buffer.unmap();

            // residue_id is residue_idx + 1, so 0 means no hit
            self.hovered_residue = if residue_id == 0 {
                -1
            } else {
                (residue_id - 1) as i32
            };
        }

        self.pending_mouse_pos = None;
    }

    /// Handle click for selection
    pub fn handle_click(&mut self, shift_held: bool) -> bool {
        let hit = self.hovered_residue;

        if hit < 0 {
            if !self.selected_residues.is_empty() {
                self.selected_residues.clear();
                return true;
            }
            return false;
        }

        if shift_held {
            if let Some(pos) = self.selected_residues.iter().position(|&r| r == hit) {
                self.selected_residues.remove(pos);
            } else {
                self.selected_residues.push(hit);
            }
        } else {
            self.selected_residues.clear();
            self.selected_residues.push(hit);
        }

        true
    }

    /// Clear all selection
    pub fn clear_selection(&mut self) {
        self.selected_residues.clear();
        self.hovered_residue = -1;
    }
}
