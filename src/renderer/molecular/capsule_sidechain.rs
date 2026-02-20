//! Capsule sidechain renderer
//!
//! Renders sidechains as capsule chains (cylinders with hemispherical caps).
//! This replaces both AtomRenderer and CylinderImpostorRenderer:
//! - Bonds are rendered as capsules
//! - "Atoms" are simply the hemispherical caps at capsule endpoints
//! - No separate sphere pass needed, no junction artifacts
//!
//! Uses the same capsule_impostor.wgsl shader as the tube renderer.

use bytemuck::Zeroable;
use glam::Vec3;

use super::capsule_instance::CapsuleInstance;
use crate::{
    camera::frustum::Frustum,
    gpu::{
        dynamic_buffer::TypedBuffer, render_context::RenderContext,
        shader_composer::ShaderComposer,
    },
    renderer::pipeline_util,
};

/// Radius used for frustum culling (capsule bounding sphere)
const CULL_RADIUS: f32 = 5.0;

// Color constants
const HYDROPHOBIC_COLOR: [f32; 3] = [0.3, 0.5, 0.9]; // Blue
const HYDROPHILIC_COLOR: [f32; 3] = [0.95, 0.6, 0.2]; // Orange
const CAPSULE_RADIUS: f32 = 0.3;

/// Sidechain geometry data that always travels together.
pub struct SidechainData<'a> {
    /// World-space atom positions.
    pub positions: &'a [Vec3],
    /// Intra-sidechain bonds as `(atom_a, atom_b)` index pairs.
    pub bonds: &'a [(u32, u32)],
    /// CA-CB backbone bonds as `(ca_position, cb_sidechain_index)`.
    pub backbone_bonds: &'a [(Vec3, u32)],
    /// Per-atom hydrophobicity flags.
    pub hydrophobicity: &'a [bool],
    /// Residue index for each sidechain atom.
    pub residue_indices: &'a [u32],
}

/// Renders sidechains as capsule chains (cylinders with hemispherical caps).
pub struct CapsuleSidechainRenderer {
    pipeline: wgpu::RenderPipeline,
    instance_buffer: TypedBuffer<CapsuleInstance>,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    /// Number of capsule instances to draw.
    pub instance_count: u32,
}

impl CapsuleSidechainRenderer {
    /// `sidechain.backbone_bonds`: CA-CB bonds as (ca_position,
    /// cb_sidechain_index) tuples.
    pub fn new(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        selection_layout: &wgpu::BindGroupLayout,
        sidechain: &SidechainData,
        shader_composer: &mut ShaderComposer,
    ) -> Self {
        // No frustum culling on initial creation
        let instances = Self::generate_instances(
            sidechain.positions,
            sidechain.bonds,
            sidechain.backbone_bonds,
            sidechain.hydrophobicity,
            sidechain.residue_indices,
            None,
            None,
        );

        let instance_count = instances.len() as u32;

        // wgpu requires non-zero buffer for bind group; use a dummy element if
        // empty
        let instance_buffer = if instances.is_empty() {
            TypedBuffer::new_with_data(
                &context.device,
                "Capsule Sidechain Instance Buffer",
                &[CapsuleInstance::zeroed()],
                wgpu::BufferUsages::STORAGE,
            )
        } else {
            TypedBuffer::new_with_data(
                &context.device,
                "Capsule Sidechain Instance Buffer",
                &instances,
                wgpu::BufferUsages::STORAGE,
            )
        };

        let bind_group_layout = Self::create_bind_group_layout(&context.device);
        let bind_group = Self::create_bind_group(
            &context.device,
            &bind_group_layout,
            &instance_buffer,
        );
        let pipeline = Self::create_pipeline(
            context,
            &bind_group_layout,
            camera_layout,
            lighting_layout,
            selection_layout,
            shader_composer,
        );

        Self {
            pipeline,
            instance_buffer,
            bind_group_layout,
            bind_group,
            instance_count,
        }
    }

    fn create_bind_group_layout(
        device: &wgpu::Device,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Capsule Sidechain Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX
                    | wgpu::ShaderStages::FRAGMENT,
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
        shader_composer: &mut ShaderComposer,
    ) -> wgpu::RenderPipeline {
        // Reuse the same capsule impostor shader
        let shader = shader_composer.compose(
            &context.device,
            "Capsule Sidechain Shader",
            "raster/impostor/capsule.wgsl",
        );

        let pipeline_layout = context.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Capsule Sidechain Pipeline Layout"),
                bind_group_layouts: &[
                    bind_group_layout,
                    camera_layout,
                    lighting_layout,
                    selection_layout,
                ],
                immediate_size: 0,
            },
        );

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
                    targets: &pipeline_util::hdr_fragment_targets(),
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(pipeline_util::depth_stencil_state()),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
    }

    /// Generate capsule instances from sidechain data.
    /// - `backbone_sidechain_bonds`: CA-CB bonds as (ca_position,
    ///   cb_sidechain_index) tuples
    /// - `frustum`: Optional frustum for culling (None = no culling)
    /// - `sidechain_colors`: Optional (hydrophobic, hydrophilic) color override
    pub(crate) fn generate_instances(
        sidechain_positions: &[Vec3],
        sidechain_bonds: &[(u32, u32)],
        backbone_sidechain_bonds: &[(Vec3, u32)],
        hydrophobicity: &[bool],
        residue_indices: &[u32],
        frustum: Option<&Frustum>,
        sidechain_colors: Option<([f32; 3], [f32; 3])>,
    ) -> Vec<CapsuleInstance> {
        let mut instances = Vec::with_capacity(
            sidechain_bonds.len() + backbone_sidechain_bonds.len(),
        );

        let (hydrophobic_color, hydrophilic_color) =
            sidechain_colors.unwrap_or((HYDROPHOBIC_COLOR, HYDROPHILIC_COLOR));

        // Helper to get color for an atom index
        let get_color = |idx: usize| -> [f32; 3] {
            if hydrophobicity.get(idx).copied().unwrap_or(false) {
                hydrophobic_color
            } else {
                hydrophilic_color
            }
        };

        // Helper to get residue index for an atom
        let get_residue_idx = |idx: usize| -> f32 {
            residue_indices.get(idx).copied().unwrap_or(0) as f32
        };

        // Helper to check if a capsule is visible (either endpoint in frustum)
        let is_visible = |pos_a: Vec3, pos_b: Vec3| -> bool {
            match frustum {
                Some(f) => {
                    let center = (pos_a + pos_b) * 0.5;
                    let half_len = (pos_a - pos_b).length() * 0.5;
                    f.intersects_sphere(center, half_len + CULL_RADIUS)
                }
                None => true,
            }
        };

        // Sidechain-sidechain bonds
        for &(a, b) in sidechain_bonds {
            let a_idx = a as usize;
            let b_idx = b as usize;
            if a_idx >= sidechain_positions.len()
                || b_idx >= sidechain_positions.len()
            {
                continue;
            }

            let pos_a = sidechain_positions[a_idx];
            let pos_b = sidechain_positions[b_idx];

            // Frustum cull
            if !is_visible(pos_a, pos_b) {
                continue;
            }

            let color_a = get_color(a_idx);
            let color_b = get_color(b_idx);
            // Use residue index from first atom (both should be same residue
            // for internal bonds)
            let res_idx = get_residue_idx(a_idx);

            instances.push(CapsuleInstance {
                endpoint_a: [pos_a.x, pos_a.y, pos_a.z, CAPSULE_RADIUS],
                endpoint_b: [pos_b.x, pos_b.y, pos_b.z, res_idx],
                color_a: [color_a[0], color_a[1], color_a[2], 0.0],
                color_b: [color_b[0], color_b[1], color_b[2], 0.0],
            });
        }

        // Backbone-sidechain bonds (CA to CB)
        for &(ca_pos, cb_idx) in backbone_sidechain_bonds {
            let cb_idx = cb_idx as usize;
            if cb_idx >= sidechain_positions.len() {
                continue;
            }

            let cb_pos = sidechain_positions[cb_idx];

            // Frustum cull
            if !is_visible(ca_pos, cb_pos) {
                continue;
            }

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

    /// Update sidechain geometry (no frustum culling).
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sidechain: &SidechainData,
    ) {
        self.update_with_frustum(device, queue, sidechain, None);
    }

    /// Update sidechain geometry with frustum culling.
    pub fn update_with_frustum(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sidechain: &SidechainData,
        frustum: Option<&Frustum>,
    ) {
        let instances = Self::generate_instances(
            sidechain.positions,
            sidechain.bonds,
            sidechain.backbone_bonds,
            sidechain.hydrophobicity,
            sidechain.residue_indices,
            frustum,
            None,
        );

        let reallocated = self.instance_buffer.write(device, queue, &instances);

        if reallocated {
            self.bind_group = Self::create_bind_group(
                device,
                &self.bind_group_layout,
                &self.instance_buffer,
            );
        }

        self.instance_count = instances.len() as u32;
    }

    /// Draw sidechain capsules into the given render pass.
    pub fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &super::draw_context::DrawBindGroups<'a>,
    ) {
        if self.instance_count == 0 {
            return;
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_bind_group(1, bind_groups.camera, &[]);
        render_pass.set_bind_group(2, bind_groups.lighting, &[]);
        render_pass.set_bind_group(3, bind_groups.selection, &[]);

        // 6 vertices per quad, one quad per capsule
        render_pass.draw(0..6, 0..self.instance_count);
    }

    /// Apply pre-computed instance data (GPU upload only, no CPU generation).
    ///
    /// Returns `true` if the buffer was reallocated (picking bind groups need
    /// recreation).
    pub fn apply_prepared(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[u8],
        instance_count: u32,
    ) -> bool {
        let reallocated =
            self.instance_buffer.write_bytes(device, queue, instances);
        if reallocated {
            self.bind_group = Self::create_bind_group(
                device,
                &self.bind_group_layout,
                &self.instance_buffer,
            );
        }
        self.instance_count = instance_count;
        reallocated
    }

    /// Get the capsule instance buffer for picking
    pub fn capsule_buffer(&self) -> &wgpu::Buffer {
        self.instance_buffer.buffer()
    }
}

impl super::MolecularRenderer for CapsuleSidechainRenderer {
    fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &super::draw_context::DrawBindGroups<'a>,
    ) {
        self.draw(render_pass, bind_groups);
    }
}
