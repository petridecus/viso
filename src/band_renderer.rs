//! Band renderer
//!
//! Renders constraint bands between atoms as capsules (cylinders with hemispherical caps).
//! Bands are used to pull atoms toward each other during minimization.
//!
//! Color coding:
//! - Green: Pull mode (attract atoms)
//! - Red: Push mode (repel atoms)
//! - Gray: Disabled
//!
//! Uses the same capsule_impostor.wgsl shader as the sidechain renderer.

use crate::dynamic_buffer::TypedBuffer;
use crate::render_context::RenderContext;
use glam::Vec3;

/// Per-instance data for capsule impostor
/// Must match the WGSL CapsuleInstance struct layout exactly
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CapsuleInstance {
    /// Endpoint A position (xyz), radius (w)
    endpoint_a: [f32; 4],
    /// Endpoint B position (xyz), residue_idx (w) - packed as float
    endpoint_b: [f32; 4],
    /// Color at endpoint A (RGB), w unused
    color_a: [f32; 4],
    /// Color at endpoint B (RGB), w unused
    color_b: [f32; 4],
}

// Color constants for bands
const PULL_COLOR: [f32; 3] = [0.2, 0.8, 0.3]; // Green for pull mode
const PUSH_COLOR: [f32; 3] = [0.9, 0.2, 0.2]; // Red for push mode
const DISABLED_COLOR: [f32; 3] = [0.5, 0.5, 0.5]; // Gray for disabled
const NEUTRAL_COLOR: [f32; 3] = [0.3, 0.6, 0.9]; // Blue for neutral (neither push nor pull)
const BAND_RADIUS: f32 = 0.15; // Thinner than sidechains for visual distinction

/// Information about a band to be rendered
#[derive(Debug, Clone)]
pub struct BandRenderInfo {
    /// World-space position of first endpoint
    pub endpoint_a: Vec3,
    /// World-space position of second endpoint
    pub endpoint_b: Vec3,
    /// Whether the band is in pull mode
    pub is_pull: bool,
    /// Whether the band is in push mode
    pub is_push: bool,
    /// Whether the band is disabled
    pub is_disabled: bool,
    /// Residue index for picking (typically the first residue)
    pub residue_idx: u32,
}

pub struct BandRenderer {
    pipeline: wgpu::RenderPipeline,
    instance_buffer: TypedBuffer<CapsuleInstance>,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    instance_count: u32,
}

impl BandRenderer {
    pub fn new(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        selection_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Start with empty buffer
        let instances: Vec<CapsuleInstance> = Vec::new();

        let instance_buffer = TypedBuffer::with_capacity(
            &context.device,
            "Band Instance Buffer",
            64, // Initial capacity for ~64 bands
            wgpu::BufferUsages::STORAGE,
        );

        let bind_group_layout = Self::create_bind_group_layout(&context.device);
        let bind_group = Self::create_bind_group(&context.device, &bind_group_layout, &instance_buffer);
        let pipeline = Self::create_pipeline(context, &bind_group_layout, camera_layout, lighting_layout, selection_layout);

        Self {
            pipeline,
            instance_buffer,
            bind_group_layout,
            bind_group,
            instance_count: instances.len() as u32,
        }
    }

    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Band Renderer Layout"),
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
        })
    }

    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        instance_buffer: &TypedBuffer<CapsuleInstance>,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: instance_buffer.buffer().as_entire_binding(),
            }],
            label: Some("Band Renderer Bind Group"),
        })
    }

    fn create_pipeline(
        context: &RenderContext,
        bind_group_layout: &wgpu::BindGroupLayout,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        selection_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        // Reuse the same capsule impostor shader
        let shader = context
            .device
            .create_shader_module(wgpu::include_wgsl!("../assets/shaders/capsule_impostor.wgsl"));

        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Band Renderer Pipeline Layout"),
                    bind_group_layouts: &[bind_group_layout, camera_layout, lighting_layout, selection_layout],
                    immediate_size: 0,
                });

        context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Band Renderer Pipeline"),
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

    /// Generate capsule instances from band data
    fn generate_instances(bands: &[BandRenderInfo]) -> Vec<CapsuleInstance> {
        bands
            .iter()
            .map(|band| {
                let color = if band.is_disabled {
                    DISABLED_COLOR
                } else if band.is_push {
                    PUSH_COLOR
                } else if band.is_pull {
                    PULL_COLOR
                } else {
                    NEUTRAL_COLOR
                };

                CapsuleInstance {
                    endpoint_a: [band.endpoint_a.x, band.endpoint_a.y, band.endpoint_a.z, BAND_RADIUS],
                    endpoint_b: [band.endpoint_b.x, band.endpoint_b.y, band.endpoint_b.z, band.residue_idx as f32],
                    color_a: [color[0], color[1], color[2], 0.0],
                    color_b: [color[0], color[1], color[2], 0.0],
                }
            })
            .collect()
    }

    /// Update band geometry
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bands: &[BandRenderInfo],
    ) {
        let instances = Self::generate_instances(bands);

        let reallocated = self.instance_buffer.write(device, queue, &instances);

        if reallocated {
            self.bind_group = Self::create_bind_group(device, &self.bind_group_layout, &self.instance_buffer);
        }

        self.instance_count = instances.len() as u32;
    }

    /// Clear all bands
    pub fn clear(&mut self) {
        self.instance_count = 0;
    }

    pub fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        camera_bind_group: &'a wgpu::BindGroup,
        lighting_bind_group: &'a wgpu::BindGroup,
        selection_bind_group: &'a wgpu::BindGroup,
    ) {
        if self.instance_count == 0 {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_bind_group(1, camera_bind_group, &[]);
        render_pass.set_bind_group(2, lighting_bind_group, &[]);
        render_pass.set_bind_group(3, selection_bind_group, &[]);

        // 6 vertices per quad, one quad per capsule
        render_pass.draw(0..6, 0..self.instance_count);
    }

    /// Get band count for debugging
    pub fn band_count(&self) -> u32 {
        self.instance_count
    }
}
