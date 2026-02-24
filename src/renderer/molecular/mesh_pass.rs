//! Shared indexed-mesh draw-pass abstraction.
//!
//! `MeshPass` owns a render pipeline and an index buffer, providing a
//! reusable draw/write interface for indexed-mesh renderers. The vertex
//! buffer is external so multiple passes can share one (e.g. backbone
//! tube + ribbon passes reference the same vertices).
//!
//! Used by `BackboneRenderer` today; designed for future use by isosurface
//! renderers and any other indexed-mesh geometry.

use crate::{
    gpu::{
        dynamic_buffer::DynamicBuffer, render_context::RenderContext,
        shader_composer::ShaderComposer,
    },
    renderer::pipeline_util,
};

/// Create a standard indexed-mesh render pipeline.
pub(crate) fn create_mesh_pipeline(
    context: &RenderContext,
    label: &str,
    shader_path: &str,
    cull_mode: Option<wgpu::Face>,
    bind_group_layouts: &[&wgpu::BindGroupLayout],
    vertex_layout: wgpu::VertexBufferLayout<'static>,
    shader_composer: &mut ShaderComposer,
) -> wgpu::RenderPipeline {
    let shader = shader_composer.compose(&context.device, label, shader_path);

    let pipeline_layout = context.device.create_pipeline_layout(
        &wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label} Layout")),
            bind_group_layouts,
            push_constant_ranges: &[],
        },
    );

    context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &pipeline_util::hdr_fragment_targets(),
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode,
                ..Default::default()
            },
            depth_stencil: Some(pipeline_util::depth_stencil_state()),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        })
}

/// An indexed-mesh draw pass: pipeline + index buffer.
///
/// Vertex buffer is external so passes can share vertices.
pub(crate) struct MeshPass {
    pipeline: wgpu::RenderPipeline,
    index_buffer: DynamicBuffer,
    pub index_count: u32,
}

impl MeshPass {
    /// Create a pass with initial index data.
    pub fn new(
        device: &wgpu::Device,
        label: &str,
        pipeline: wgpu::RenderPipeline,
        indices: &[u32],
    ) -> Self {
        let index_buffer = if indices.is_empty() {
            DynamicBuffer::new(
                device,
                label,
                size_of::<u32>() * 3000,
                wgpu::BufferUsages::INDEX,
            )
        } else {
            DynamicBuffer::new_with_data(
                device,
                label,
                indices,
                wgpu::BufferUsages::INDEX,
            )
        };
        Self {
            pipeline,
            index_buffer,
            index_count: indices.len() as u32,
        }
    }

    /// Set pipeline, vertex buffer, index buffer, and draw.
    ///
    /// Caller must set bind groups before calling this.
    pub fn draw_indexed<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        vertex_buffer: &'a wgpu::Buffer,
    ) {
        if self.index_count == 0 {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.set_index_buffer(
            self.index_buffer.buffer().slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(0..self.index_count, 0, 0..1);
    }

    /// Draw a sub-range of the index buffer (for frustum culling).
    ///
    /// Caller must set bind groups before calling this.
    pub fn draw_indexed_range<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        vertex_buffer: &'a wgpu::Buffer,
        index_range: std::ops::Range<u32>,
    ) {
        if index_range.is_empty() {
            return;
        }
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        render_pass.set_index_buffer(
            self.index_buffer.buffer().slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(index_range, 0, 0..1);
    }

    /// Write typed index data.
    pub fn write_indices(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        indices: &[u32],
    ) {
        if !indices.is_empty() {
            let _ = self.index_buffer.write(device, queue, indices);
        }
        self.index_count = indices.len() as u32;
    }

    /// Write raw index bytes (from scene processor).
    pub fn write_indices_bytes(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
        count: u32,
    ) {
        if !data.is_empty() {
            let _ = self.index_buffer.write_bytes(device, queue, data);
        }
        self.index_count = count;
    }

    /// Get the underlying index buffer (for picking).
    pub fn index_buffer(&self) -> &wgpu::Buffer {
        self.index_buffer.buffer()
    }
}
