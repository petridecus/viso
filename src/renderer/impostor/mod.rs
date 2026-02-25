//! Reusable impostor-pass primitives.
//!
//! Every impostor renderer (capsule, sphere, cone, polygon) follows the same
//! pattern: one storage buffer, one bind group, one pipeline, and a
//! `draw(0..N, 0..instance_count)` call.  `ImpostorPass<T>` extracts that
//! boilerplate so each renderer just composes from shared primitives.

pub mod capsule;
pub mod cone;
pub mod polygon;
pub mod sphere;

use bytemuck::{Pod, Zeroable};

use crate::{
    gpu::{
        dynamic_buffer::TypedBuffer, render_context::RenderContext,
        shader_composer::ShaderComposer,
    },
    renderer::{pipeline_util, PipelineLayouts},
};

/// Shader identity: label + path, always passed together.
pub struct ShaderDef<'a> {
    pub label: &'a str,
    pub path: &'a str,
}

/// A single impostor draw pass: pipeline + typed storage buffer + bind group.
///
/// All impostor shaders use the same bind group layout convention:
/// - group(0): storage buffer (instances)
/// - group(1): camera uniform
/// - group(2): lighting uniform + textures
/// - group(3): selection storage
pub struct ImpostorPass<T: Pod + Zeroable> {
    pipeline: wgpu::RenderPipeline,
    instance_buffer: TypedBuffer<T>,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    pub instance_count: u32,
    vertices_per_instance: u32,
}

impl<T: Pod + Zeroable> ImpostorPass<T> {
    /// Create a new impostor pass with the given shader.
    pub fn new(
        context: &RenderContext,
        shader: &ShaderDef,
        layouts: &PipelineLayouts,
        vertices_per_instance: u32,
        shader_composer: &mut ShaderComposer,
    ) -> Self {
        let label = shader.label;
        let instance_buffer = TypedBuffer::new_with_data(
            &context.device,
            &format!("{label} Buffer"),
            &[T::zeroed()],
            wgpu::BufferUsages::STORAGE,
        );

        let bind_group_layout =
            Self::create_bind_group_layout(&context.device, label);
        let bind_group = Self::create_bind_group(
            &context.device,
            &bind_group_layout,
            &instance_buffer,
            label,
        );
        let pipeline = Self::create_pipeline(
            context,
            shader,
            &bind_group_layout,
            layouts,
            shader_composer,
        );

        Self {
            pipeline,
            instance_buffer,
            bind_group_layout,
            bind_group,
            instance_count: 0,
            vertices_per_instance,
        }
    }

    fn create_bind_group_layout(
        device: &wgpu::Device,
        label: &str,
    ) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label} Layout")),
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
        buffer: &TypedBuffer<T>,
        label: &str,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.buffer().as_entire_binding(),
            }],
            label: Some(&format!("{label} Bind Group")),
        })
    }

    fn create_pipeline(
        context: &RenderContext,
        shader_def: &ShaderDef,
        bind_group_layout: &wgpu::BindGroupLayout,
        layouts: &PipelineLayouts,
        shader_composer: &mut ShaderComposer,
    ) -> wgpu::RenderPipeline {
        let label = shader_def.label;
        let shader = shader_composer.compose(
            &context.device,
            &format!("{label} Shader"),
            shader_def.path,
        );

        let pipeline_layout = context.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{label} Pipeline Layout")),
                bind_group_layouts: &[
                    bind_group_layout,
                    &layouts.camera,
                    &layouts.lighting,
                    &layouts.selection,
                ],
                push_constant_ranges: &[],
            },
        );

        context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("{label} Pipeline")),
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
                multiview: None,
                cache: None,
            })
    }

    /// Write typed instances to the GPU buffer. Recreates the bind group if
    /// the buffer was reallocated.
    ///
    /// Returns `true` if the underlying buffer was reallocated (callers that
    /// hold external bind groups — e.g. picking — need to recreate them).
    pub fn write_instances(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[T],
    ) -> bool {
        let data = if instances.is_empty() {
            &[T::zeroed()]
        } else {
            instances
        };
        let reallocated = self.instance_buffer.write(device, queue, data);
        if reallocated {
            self.bind_group = Self::create_bind_group(
                device,
                &self.bind_group_layout,
                &self.instance_buffer,
                "reallocated",
            );
        }
        self.instance_count = instances.len() as u32;
        reallocated
    }

    /// Write raw instance bytes to the GPU buffer. Recreates the bind group if
    /// the buffer was reallocated.
    ///
    /// Returns `true` if the underlying buffer was reallocated.
    pub fn write_bytes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        count: u32,
    ) -> bool {
        let zeroed = T::zeroed();
        let data = if bytes.is_empty() {
            bytemuck::bytes_of(&zeroed)
        } else {
            bytes
        };
        let reallocated = self.instance_buffer.write_bytes(device, queue, data);
        if reallocated {
            self.bind_group = Self::create_bind_group(
                device,
                &self.bind_group_layout,
                &self.instance_buffer,
                "reallocated",
            );
        }
        self.instance_count = count;
        reallocated
    }

    /// Issue the draw call for this pass.
    ///
    /// Sets the pipeline and bind groups 0–3, then draws.
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
        render_pass.draw(0..self.vertices_per_instance, 0..self.instance_count);
    }

    /// The underlying `wgpu::Buffer` (for picking bind groups, etc.).
    pub fn buffer(&self) -> &wgpu::Buffer {
        self.instance_buffer.buffer()
    }

    /// `(label, used_bytes, allocated_bytes)` for debug overlay.
    pub fn buffer_info(
        &self,
        label: &'static str,
    ) -> (&'static str, usize, usize) {
        (
            label,
            self.instance_buffer.len_bytes(),
            self.instance_buffer.capacity_bytes(),
        )
    }
}
