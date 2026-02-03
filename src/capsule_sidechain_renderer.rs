//! Capsule sidechain renderer
//!
//! Renders sidechains as capsule chains (cylinders with hemispherical caps).
//! This replaces both AtomRenderer and CylinderImpostorRenderer:
//! - Bonds are rendered as capsules
//! - "Atoms" are simply the hemispherical caps at capsule endpoints
//! - No separate sphere pass needed, no junction artifacts
//!
//! Uses the same capsule_impostor.wgsl shader as the tube renderer.

use crate::dynamic_buffer::TypedBuffer;
use crate::protein_data::BackboneSidechainBond;
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

// Color constants
const HYDROPHOBIC_COLOR: [f32; 3] = [0.3, 0.5, 0.9]; // Blue
const HYDROPHILIC_COLOR: [f32; 3] = [0.95, 0.6, 0.2]; // Orange
const CAPSULE_RADIUS: f32 = 0.3;

pub struct CapsuleSidechainRenderer {
    pipeline: wgpu::RenderPipeline,
    instance_buffer: TypedBuffer<CapsuleInstance>,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    pub instance_count: u32,
}

impl CapsuleSidechainRenderer {
    pub fn new(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        selection_layout: &wgpu::BindGroupLayout,
        sidechain_positions: &[Vec3],
        sidechain_bonds: &[(u32, u32)],
        backbone_sidechain_bonds: &[BackboneSidechainBond],
        hydrophobicity: &[bool],
        residue_indices: &[u32],
    ) -> Self {
        let instances = Self::generate_instances(
            sidechain_positions,
            sidechain_bonds,
            backbone_sidechain_bonds,
            hydrophobicity,
            residue_indices,
        );

        let instance_count = instances.len() as u32;

        let instance_buffer = TypedBuffer::new_with_data(
            &context.device,
            "Capsule Sidechain Instance Buffer",
            &instances,
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
            instance_count,
        }
    }

    fn create_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Capsule Sidechain Layout"),
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
            label: Some("Capsule Sidechain Bind Group"),
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
                    label: Some("Capsule Sidechain Pipeline Layout"),
                    bind_group_layouts: &[bind_group_layout, camera_layout, lighting_layout, selection_layout],
                    immediate_size: 0,
                });

        context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Capsule Sidechain Pipeline"),
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

    /// Generate capsule instances from sidechain data
    fn generate_instances(
        sidechain_positions: &[Vec3],
        sidechain_bonds: &[(u32, u32)],
        backbone_sidechain_bonds: &[BackboneSidechainBond],
        hydrophobicity: &[bool],
        residue_indices: &[u32],
    ) -> Vec<CapsuleInstance> {
        let mut instances = Vec::with_capacity(sidechain_bonds.len() + backbone_sidechain_bonds.len());

        // Helper to get color for an atom index
        let get_color = |idx: usize| -> [f32; 3] {
            if hydrophobicity.get(idx).copied().unwrap_or(false) {
                HYDROPHOBIC_COLOR
            } else {
                HYDROPHILIC_COLOR
            }
        };
        
        // Helper to get residue index for an atom
        let get_residue_idx = |idx: usize| -> f32 {
            residue_indices.get(idx).copied().unwrap_or(0) as f32
        };

        // Sidechain-sidechain bonds
        for &(a, b) in sidechain_bonds {
            let a_idx = a as usize;
            let b_idx = b as usize;
            if a_idx >= sidechain_positions.len() || b_idx >= sidechain_positions.len() {
                continue;
            }

            let pos_a = sidechain_positions[a_idx];
            let pos_b = sidechain_positions[b_idx];
            let color_a = get_color(a_idx);
            let color_b = get_color(b_idx);
            // Use residue index from first atom (both should be same residue for internal bonds)
            let res_idx = get_residue_idx(a_idx);

            instances.push(CapsuleInstance {
                endpoint_a: [pos_a.x, pos_a.y, pos_a.z, CAPSULE_RADIUS],
                endpoint_b: [pos_b.x, pos_b.y, pos_b.z, res_idx],
                color_a: [color_a[0], color_a[1], color_a[2], 0.0],
                color_b: [color_b[0], color_b[1], color_b[2], 0.0],
            });
        }

        // Backbone-sidechain bonds (CA to CB)
        for bond in backbone_sidechain_bonds {
            let cb_idx = bond.cb_index as usize;
            if cb_idx >= sidechain_positions.len() {
                continue;
            }

            let ca_pos = bond.ca_position;
            let cb_pos = sidechain_positions[cb_idx];
            let cb_color = get_color(cb_idx);
            let res_idx = get_residue_idx(cb_idx);

            // CA end uses same color as CB for visual continuity
            instances.push(CapsuleInstance {
                endpoint_a: [ca_pos.x, ca_pos.y, ca_pos.z, CAPSULE_RADIUS],
                endpoint_b: [cb_pos.x, cb_pos.y, cb_pos.z, res_idx],
                color_a: [cb_color[0], cb_color[1], cb_color[2], 0.0],
                color_b: [cb_color[0], cb_color[1], cb_color[2], 0.0],
            });
        }

        instances
    }

    /// Update sidechain geometry
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sidechain_positions: &[Vec3],
        sidechain_bonds: &[(u32, u32)],
        backbone_sidechain_bonds: &[BackboneSidechainBond],
        hydrophobicity: &[bool],
        residue_indices: &[u32],
    ) {
        let instances = Self::generate_instances(
            sidechain_positions,
            sidechain_bonds,
            backbone_sidechain_bonds,
            hydrophobicity,
            residue_indices,
        );

        let reallocated = self.instance_buffer.write(device, queue, &instances);

        if reallocated {
            self.bind_group = Self::create_bind_group(device, &self.bind_group_layout, &self.instance_buffer);
        }

        self.instance_count = instances.len() as u32;
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

    /// Get the capsule instance buffer for picking
    pub fn capsule_buffer(&self) -> &wgpu::Buffer {
        self.instance_buffer.buffer()
    }
}
