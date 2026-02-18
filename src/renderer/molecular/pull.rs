//! Pull renderer
//!
//! Renders the active pull constraint during drag operations.
//! A pull is a temporary constraint used to drag an atom toward the mouse.
//!
//! Visual style matches original Foldit:
//! - Cylinder from atom to near the mouse position
//! - Cone/arrow at the mouse end pointing toward the target
//! - Purple color
//!
//! Only one pull can be active at a time.

use crate::gpu::dynamic_buffer::TypedBuffer;
use crate::gpu::render_context::RenderContext;
use crate::gpu::shader_composer::ShaderComposer;
use glam::Vec3;

use super::capsule_instance::CapsuleInstance;
use crate::renderer::pipeline_util;

/// Per-instance data for cone impostor (arrow tip)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ConeInstance {
    base: [f32; 4],  // xyz = base position, w = base radius
    tip: [f32; 4],   // xyz = tip position, w = residue_idx
    color: [f32; 4], // xyz = RGB, w = unused
    _pad: [f32; 4],  // padding for alignment
}

// Pull visual constants - match band defaults
const PULL_COLOR: [f32; 3] = [0.5, 0.0, 0.5]; // Purple - same as BAND_COLOR
const PULL_CYLINDER_RADIUS: f32 = 0.25; // Same as BAND_MID_RADIUS (default strength)
const PULL_CONE_RADIUS: f32 = 0.6; // Larger than cylinder for visible arrow
const PULL_CONE_LENGTH: f32 = 2.0;

/// Information about the active pull
#[derive(Debug, Clone)]
pub struct PullRenderInfo {
    /// Position of the atom being pulled
    pub atom_pos: Vec3,
    /// Target position (mouse position in world space)
    pub target_pos: Vec3,
    /// Residue index for picking
    pub residue_idx: u32,
}

pub struct PullRenderer {
    // Capsule pipeline for cylinder
    capsule_pipeline: wgpu::RenderPipeline,
    capsule_buffer: TypedBuffer<CapsuleInstance>,
    capsule_bind_group_layout: wgpu::BindGroupLayout,
    capsule_bind_group: wgpu::BindGroup,
    capsule_count: u32,

    // Cone pipeline for arrow tip
    cone_pipeline: wgpu::RenderPipeline,
    cone_buffer: TypedBuffer<ConeInstance>,
    cone_bind_group_layout: wgpu::BindGroupLayout,
    cone_bind_group: wgpu::BindGroup,
    cone_count: u32,
}

impl PullRenderer {
    pub fn new(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        selection_layout: &wgpu::BindGroupLayout,
        shader_composer: &mut ShaderComposer,
    ) -> Self {
        // Capsule (cylinder) setup
        let capsule_buffer = TypedBuffer::with_capacity(
            &context.device,
            "Pull Capsule Buffer",
            1,
            wgpu::BufferUsages::STORAGE,
        );
        let capsule_bind_group_layout =
            Self::create_bind_group_layout(&context.device, "Pull Capsule");
        let capsule_bind_group = Self::create_capsule_bind_group(
            &context.device,
            &capsule_bind_group_layout,
            &capsule_buffer,
        );
        let capsule_pipeline = Self::create_capsule_pipeline(
            context,
            &capsule_bind_group_layout,
            camera_layout,
            lighting_layout,
            selection_layout,
            shader_composer,
        );

        // Cone (arrow tip) setup
        let cone_buffer = TypedBuffer::with_capacity(
            &context.device,
            "Pull Cone Buffer",
            1,
            wgpu::BufferUsages::STORAGE,
        );
        let cone_bind_group_layout = Self::create_bind_group_layout(&context.device, "Pull Cone");
        let cone_bind_group =
            Self::create_cone_bind_group(&context.device, &cone_bind_group_layout, &cone_buffer);
        let cone_pipeline = Self::create_cone_pipeline(
            context,
            &cone_bind_group_layout,
            camera_layout,
            lighting_layout,
            selection_layout,
            shader_composer,
        );

        Self {
            capsule_pipeline,
            capsule_buffer,
            capsule_bind_group_layout,
            capsule_bind_group,
            capsule_count: 0,
            cone_pipeline,
            cone_buffer,
            cone_bind_group_layout,
            cone_bind_group,
            cone_count: 0,
        }
    }

    fn create_bind_group_layout(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{} Layout", label)),
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

    fn create_capsule_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        buffer: &TypedBuffer<CapsuleInstance>,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.buffer().as_entire_binding(),
            }],
            label: Some("Pull Capsule Bind Group"),
        })
    }

    fn create_cone_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        buffer: &TypedBuffer<ConeInstance>,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.buffer().as_entire_binding(),
            }],
            label: Some("Pull Cone Bind Group"),
        })
    }

    fn create_capsule_pipeline(
        context: &RenderContext,
        bind_group_layout: &wgpu::BindGroupLayout,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        selection_layout: &wgpu::BindGroupLayout,
        shader_composer: &mut ShaderComposer,
    ) -> wgpu::RenderPipeline {
        let shader = shader_composer.compose(
            &context.device,
            "Pull Capsule Shader",
            include_str!("../../../assets/shaders/raster/impostor/capsule.wgsl"),
            "capsule_impostor.wgsl",
        );

        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Pull Capsule Pipeline Layout"),
                    bind_group_layouts: &[
                        bind_group_layout,
                        camera_layout,
                        lighting_layout,
                        selection_layout,
                    ],
                    immediate_size: 0,
                });

        context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Pull Capsule Pipeline"),
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

    fn create_cone_pipeline(
        context: &RenderContext,
        bind_group_layout: &wgpu::BindGroupLayout,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        selection_layout: &wgpu::BindGroupLayout,
        shader_composer: &mut ShaderComposer,
    ) -> wgpu::RenderPipeline {
        let shader = shader_composer.compose(
            &context.device,
            "Pull Cone Shader",
            include_str!("../../../assets/shaders/raster/impostor/cone.wgsl"),
            "cone_impostor.wgsl",
        );

        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Pull Cone Pipeline Layout"),
                    bind_group_layouts: &[
                        bind_group_layout,
                        camera_layout,
                        lighting_layout,
                        selection_layout,
                    ],
                    immediate_size: 0,
                });

        context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Pull Cone Pipeline"),
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

    /// Update with the active pull, or clear if None
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pull: Option<&PullRenderInfo>,
    ) {
        match pull {
            Some(p) => {
                let (capsules, cones) = Self::generate_instances(p);

                let capsule_reallocated = self.capsule_buffer.write(device, queue, &capsules);
                if capsule_reallocated {
                    self.capsule_bind_group = Self::create_capsule_bind_group(
                        device,
                        &self.capsule_bind_group_layout,
                        &self.capsule_buffer,
                    );
                }
                self.capsule_count = capsules.len() as u32;

                let cone_reallocated = self.cone_buffer.write(device, queue, &cones);
                if cone_reallocated {
                    self.cone_bind_group = Self::create_cone_bind_group(
                        device,
                        &self.cone_bind_group_layout,
                        &self.cone_buffer,
                    );
                }
                self.cone_count = cones.len() as u32;
            }
            None => {
                self.capsule_count = 0;
                self.cone_count = 0;
            }
        }
    }

    /// Clear the pull visualization
    pub fn clear(&mut self) {
        self.capsule_count = 0;
        self.cone_count = 0;
    }

    fn generate_instances(pull: &PullRenderInfo) -> (Vec<CapsuleInstance>, Vec<ConeInstance>) {
        let mut capsules = Vec::with_capacity(1);
        let mut cones = Vec::with_capacity(1);

        let atom_pos = pull.atom_pos;
        let target_pos = pull.target_pos;
        let to_target = target_pos - atom_pos;
        let distance = to_target.length();

        if distance < 0.001 {
            return (capsules, cones);
        }

        let direction = to_target / distance;

        // Cone base is PULL_CONE_LENGTH from the target, pointing toward target
        let cone_base = if distance > PULL_CONE_LENGTH {
            target_pos - direction * PULL_CONE_LENGTH
        } else {
            atom_pos
        };

        // Cylinder from atom to cone base
        if atom_pos.distance_squared(cone_base) > 0.001 {
            capsules.push(CapsuleInstance {
                endpoint_a: [atom_pos.x, atom_pos.y, atom_pos.z, PULL_CYLINDER_RADIUS],
                endpoint_b: [
                    cone_base.x,
                    cone_base.y,
                    cone_base.z,
                    pull.residue_idx as f32,
                ],
                color_a: [PULL_COLOR[0], PULL_COLOR[1], PULL_COLOR[2], 0.0],
                color_b: [PULL_COLOR[0], PULL_COLOR[1], PULL_COLOR[2], 0.0],
            });
        }

        // Cone from cone_base to target (arrow pointing toward mouse)
        if cone_base.distance_squared(target_pos) > 0.001 {
            cones.push(ConeInstance {
                base: [cone_base.x, cone_base.y, cone_base.z, PULL_CONE_RADIUS],
                tip: [
                    target_pos.x,
                    target_pos.y,
                    target_pos.z,
                    pull.residue_idx as f32,
                ],
                color: [PULL_COLOR[0], PULL_COLOR[1], PULL_COLOR[2], 0.0],
                _pad: [0.0, 0.0, 0.0, 0.0],
            });
        }

        (capsules, cones)
    }

    pub fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &super::draw_context::DrawBindGroups<'a>,
    ) {
        // Draw cylinder(s)
        if self.capsule_count > 0 {
            render_pass.set_pipeline(&self.capsule_pipeline);
            render_pass.set_bind_group(0, &self.capsule_bind_group, &[]);
            render_pass.set_bind_group(1, bind_groups.camera, &[]);
            render_pass.set_bind_group(2, bind_groups.lighting, &[]);
            render_pass.set_bind_group(3, bind_groups.selection, &[]);
            render_pass.draw(0..6, 0..self.capsule_count);
        }

        // Draw cone(s)
        if self.cone_count > 0 {
            render_pass.set_pipeline(&self.cone_pipeline);
            render_pass.set_bind_group(0, &self.cone_bind_group, &[]);
            render_pass.set_bind_group(1, bind_groups.camera, &[]);
            render_pass.set_bind_group(2, bind_groups.lighting, &[]);
            render_pass.set_bind_group(3, bind_groups.selection, &[]);
            render_pass.draw(0..6, 0..self.cone_count);
        }
    }

    pub fn is_active(&self) -> bool {
        self.capsule_count > 0 || self.cone_count > 0
    }
}

impl super::MolecularRenderer for PullRenderer {
    fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &super::draw_context::DrawBindGroups<'a>,
    ) {
        self.draw(render_pass, bind_groups);
    }
}
