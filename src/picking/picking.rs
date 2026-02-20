//! GPU-based picking using a dedicated picking render pass
//!
//! Renders residue indices to an offscreen buffer, then reads back
//! the pixel at the mouse position to determine which residue is under the
//! cursor. This is exact - it matches exactly what's rendered on screen.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use wgpu::util::DeviceExt;

use crate::{
    gpu::{render_context::RenderContext, shader_composer::ShaderComposer},
    renderer::molecular::tube::tube_vertex_buffer_layout,
};

/// Selection buffer for GPU - stores selection state as a bit array
pub struct SelectionBuffer {
    buffer: wgpu::Buffer,
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    /// Number of residues (for sizing)
    capacity: usize,
}

impl SelectionBuffer {
    /// Create a selection buffer sized for up to `max_residues` residues.
    pub fn new(device: &wgpu::Device, max_residues: usize) -> Self {
        // Round up to multiple of 32 bits
        let num_words = max_residues.div_ceil(32);
        let data = vec![0u32; num_words.max(1)];

        let buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Selection Buffer"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST,
            });

        let layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Selection Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: true,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Selection Bind Group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        Self {
            buffer,
            layout,
            bind_group,
            capacity: max_residues,
        }
    }

    /// Update selection state from a list of selected residue indices
    pub fn update(&self, queue: &wgpu::Queue, selected_residues: &[i32]) {
        let num_words = self.capacity.div_ceil(32);
        let mut data = vec![0u32; num_words.max(1)];

        for &idx in selected_residues {
            if idx >= 0 && (idx as usize) < self.capacity {
                let word_idx = idx as usize / 32;
                let bit_idx = idx as usize % 32;
                data[word_idx] |= 1u32 << bit_idx;
            }
        }

        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&data));
    }

    /// Ensure the buffer has capacity for at least `required` residues.
    /// Recreates the buffer and bind_group if current capacity is insufficient.
    pub fn ensure_capacity(&mut self, device: &wgpu::Device, required: usize) {
        if required <= self.capacity {
            return;
        }

        // Need to grow - recreate buffer with new capacity
        let new_capacity = required;
        let num_words = new_capacity.div_ceil(32);
        let data = vec![0u32; num_words.max(1)];

        self.buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Selection Buffer"),
                contents: bytemuck::cast_slice(&data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST,
            });

        self.bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Selection Bind Group"),
                layout: &self.layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                }],
            });

        self.capacity = new_capacity;
    }
}

/// Geometry buffers needed for the picking render pass.
pub struct PickingGeometry<'a> {
    pub tube_vertex_buffer: &'a wgpu::Buffer,
    pub tube_index_buffer: &'a wgpu::Buffer,
    pub tube_index_count: u32,
    pub ribbon_vertex_buffer: Option<&'a wgpu::Buffer>,
    pub ribbon_index_buffer: Option<&'a wgpu::Buffer>,
    pub ribbon_index_count: u32,
    pub capsule_bind_group: Option<&'a wgpu::BindGroup>,
    pub capsule_count: u32,
    pub bns_capsule_bind_group: Option<&'a wgpu::BindGroup>,
    pub bns_capsule_count: u32,
}

/// Manages GPU-based residue picking via an offscreen R32Uint render pass.
pub struct Picking {
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
    /// Whether a readback is in flight (buffer mapping requested)
    readback_in_flight: bool,
    /// Flag set by callback when buffer mapping is complete
    map_complete: Arc<AtomicBool>,
}

impl Picking {
    /// Create a new picking system with pipelines and textures sized to the current context.
    pub fn new(
        context: &RenderContext,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        shader_composer: &mut ShaderComposer,
    ) -> Self {
        let width = context.config.width;
        let height = context.config.height;

        let (texture, texture_view) =
            Self::create_picking_texture(&context.device, width, height);
        let (depth_texture, depth_view) =
            Self::create_depth_texture(&context.device, width, height);

        // Staging buffer for single pixel readback (256 bytes minimum, we only
        // need 4)
        let staging_buffer =
            context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Picking Staging Buffer"),
                size: 256,
                usage: wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

        // Load picking shader for tubes
        let tube_shader = shader_composer.compose(
            &context.device,
            "Picking Tube Shader",
            "utility/picking_mesh.wgsl",
        );

        // Tube picking pipeline
        let tube_pipeline_layout = context.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Picking Tube Pipeline Layout"),
                bind_group_layouts: &[camera_bind_group_layout],
                immediate_size: 0,
            },
        );

        let tube_pipeline = context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
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
            },
        );

        // Capsule picking pipeline
        let capsule_shader = shader_composer.compose(
            &context.device,
            "Picking Capsule Shader",
            "utility/picking_capsule.wgsl",
        );

        let capsule_bind_group_layout = context
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Picking Capsule Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: true,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let capsule_pipeline_layout = context.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Picking Capsule Pipeline Layout"),
                bind_group_layouts: &[
                    &capsule_bind_group_layout,
                    camera_bind_group_layout,
                ],
                immediate_size: 0,
            },
        );

        let capsule_pipeline = context.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
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
            },
        );

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
            readback_in_flight: false,
            map_complete: Arc::new(AtomicBool::new(false)),
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC,
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

    /// Resize the picking and depth textures to match the new dimensions.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;
        let (texture, texture_view) =
            Self::create_picking_texture(device, width, height);
        self.texture = texture;
        self.texture_view = texture_view;
        let (depth_texture, depth_view) =
            Self::create_depth_texture(device, width, height);
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
    ///
    /// In Ribbon mode, pass ribbon buffers to render helices/sheets for
    /// picking. Tubes are still rendered (for coils in ribbon mode, or
    /// everything in tube mode).
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        camera_bind_group: &wgpu::BindGroup,
        geometry: &PickingGeometry,
        mouse_x: u32,
        mouse_y: u32,
    ) {
        let PickingGeometry {
            tube_vertex_buffer,
            tube_index_buffer,
            tube_index_count,
            ribbon_vertex_buffer,
            ribbon_index_buffer,
            ribbon_index_count,
            capsule_bind_group,
            capsule_count,
            bns_capsule_bind_group,
            bns_capsule_count,
        } = *geometry;
        // Render picking pass
        {
            let mut render_pass =
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Picking Render Pass"),
                    color_attachments: &[Some(
                        wgpu::RenderPassColorAttachment {
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
                        },
                    )],
                    depth_stencil_attachment: Some(
                        wgpu::RenderPassDepthStencilAttachment {
                            view: &self.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        },
                    ),
                    ..Default::default()
                });

            // Draw tubes (coils in ribbon mode, everything in tube mode)
            if tube_index_count > 0 {
                render_pass.set_pipeline(&self.tube_pipeline);
                render_pass.set_bind_group(0, camera_bind_group, &[]);
                render_pass.set_vertex_buffer(0, tube_vertex_buffer.slice(..));
                render_pass.set_index_buffer(
                    tube_index_buffer.slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                render_pass.draw_indexed(0..tube_index_count, 0, 0..1);
            }

            // Draw ribbons (helices/sheets in ribbon mode)
            // Uses the same pipeline as tubes - same vertex layout and picking
            // shader
            if let (Some(ribbon_vb), Some(ribbon_ib)) =
                (ribbon_vertex_buffer, ribbon_index_buffer)
            {
                if ribbon_index_count > 0 {
                    render_pass.set_pipeline(&self.tube_pipeline);
                    render_pass.set_bind_group(0, camera_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, ribbon_vb.slice(..));
                    render_pass.set_index_buffer(
                        ribbon_ib.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    render_pass.draw_indexed(0..ribbon_index_count, 0, 0..1);
                }
            }

            // Draw capsules (sidechains)
            if let Some(capsule_bg) = capsule_bind_group {
                if capsule_count > 0 {
                    render_pass.set_pipeline(&self.capsule_pipeline);
                    render_pass.set_bind_group(0, capsule_bg, &[]);
                    render_pass.set_bind_group(1, camera_bind_group, &[]);
                    render_pass.draw(0..6, 0..capsule_count);
                }
            }

            // Draw ball-and-stick picking (degenerate capsules for spheres +
            // bonds)
            if let Some(bns_bg) = bns_capsule_bind_group {
                if bns_capsule_count > 0 {
                    render_pass.set_pipeline(&self.capsule_pipeline);
                    render_pass.set_bind_group(0, bns_bg, &[]);
                    render_pass.set_bind_group(1, camera_bind_group, &[]);
                    render_pass.draw(0..6, 0..bns_capsule_count);
                }
            }
        }

        // Copy pixel at mouse position to staging buffer (only if not already
        // in flight)
        if mouse_x < self.width
            && mouse_y < self.height
            && !self.readback_in_flight
        {
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
        }
    }

    /// Start async readback (call after queue.submit)
    /// This initiates the buffer mapping without blocking
    pub fn start_readback(&mut self) {
        if self.readback_in_flight {
            return;
        }
        self.readback_in_flight = true;
        self.map_complete.store(false, Ordering::SeqCst);
        let map_complete = self.map_complete.clone();
        let buffer_slice = self.staging_buffer.slice(..4);
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            if result.is_ok() {
                map_complete.store(true, Ordering::SeqCst);
            }
        });
    }

    /// Try to complete the readback without blocking.
    /// Returns true if new data was read, false if still pending.
    /// Uses previous frame's hover result until new data is ready.
    pub fn complete_readback(&mut self, device: &wgpu::Device) -> bool {
        if !self.readback_in_flight {
            return false;
        }

        // Poll without waiting - process callbacks
        let _ = device.poll(wgpu::PollType::Poll);

        // Check if the callback has signaled completion
        if !self.map_complete.load(Ordering::SeqCst) {
            // Not ready yet - keep using cached hovered_residue
            return false;
        }

        // Buffer is mapped - read the data
        let buffer_slice = self.staging_buffer.slice(..4);
        let data = buffer_slice.get_mapped_range();
        let residue_id =
            u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        self.staging_buffer.unmap();
        self.readback_in_flight = false;

        // residue_id is residue_idx + 1, so 0 means no hit
        self.hovered_residue = if residue_id == 0 {
            -1
        } else {
            (residue_id - 1) as i32
        };
        true
    }

    /// Process a click event, updating the selection based on the hovered residue.
    ///
    /// Returns `true` if the selection changed. Shift-click toggles individual residues.
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
            if let Some(pos) =
                self.selected_residues.iter().position(|&r| r == hit)
            {
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
