//! Isosurface mesh renderer for electron density maps and molecular
//! surfaces.
//!
//! Renders isosurfaces as triangle meshes via marching cubes on the
//! CPU, uploaded through a `MeshPass` + `DynamicBuffer` vertex buffer.
//! Integrates with depth, normals, SSAO, and bloom through the standard
//! dual render target (color + normal).

pub mod cavity;
pub(crate) mod cpu_marching_cubes;
pub mod density;
pub mod gaussian_surface;
pub(crate) mod mesh_smooth;
pub(crate) mod sdf_grid;
pub mod ses;
pub(crate) mod tables;

use crate::error::VisoError;
use crate::gpu::dynamic_buffer::DynamicBuffer;
use crate::gpu::{RenderContext, Shader, ShaderComposer};
use crate::renderer::draw_context::DrawBindGroups;
use crate::renderer::mesh::{create_mesh_pipeline, MeshPass, MeshPipelineDef};
use crate::renderer::PipelineLayouts;

/// A vertex on the extracted isosurface.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IsosurfaceVertex {
    /// World-space position.
    pub position: [f32; 3],
    /// Surface normal (central-difference gradient).
    pub normal: [f32; 3],
    /// Vertex color RGBA (alpha controls transparency).
    pub color: [f32; 4],
}

/// Isosurface vertex buffer layout for wgpu.
pub fn isosurface_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: size_of::<IsosurfaceVertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 24,
                shader_location: 2,
            },
        ],
    }
}

/// Renderer for isosurface meshes (electron density, etc.).
///
/// Owns a vertex buffer and a `MeshPass` (index buffer + pipeline).
/// Uses only camera + lighting bind groups (groups 0-1) — no selection
/// or per-residue color, since density maps have no residue concept.
pub(crate) struct IsosurfaceRenderer {
    mesh_pass: MeshPass,
    vertex_buffer: DynamicBuffer,
    vertex_count: u32,
}

impl IsosurfaceRenderer {
    /// Create a new isosurface renderer with empty buffers.
    pub fn new(
        context: &RenderContext,
        layouts: &PipelineLayouts,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, VisoError> {
        let pipeline = create_mesh_pipeline(
            context,
            &MeshPipelineDef {
                label: "Isosurface",
                shader: Shader::Isosurface,
                cull_mode: Some(wgpu::Face::Back),
                vertex_layout: isosurface_vertex_layout(),
            },
            &[&layouts.camera, &layouts.lighting],
            shader_composer,
        )?;

        let mesh_pass =
            MeshPass::new(&context.device, "Isosurface Indices", pipeline, &[]);

        let vertex_buffer = DynamicBuffer::new(
            &context.device,
            "Isosurface Vertices",
            size_of::<IsosurfaceVertex>() * 1000,
            wgpu::BufferUsages::VERTEX,
        );

        Ok(Self {
            mesh_pass,
            vertex_buffer,
            vertex_count: 0,
        })
    }

    /// Upload new isosurface mesh data to the GPU.
    pub fn apply_prepared(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[IsosurfaceVertex],
        indices: &[u32],
    ) {
        if vertices.is_empty() || indices.is_empty() {
            self.vertex_count = 0;
            self.mesh_pass.write_indices(device, queue, &[]);
            return;
        }
        let _ = self.vertex_buffer.write(device, queue, vertices);
        self.vertex_count = vertices.len() as u32;
        self.mesh_pass.write_indices(device, queue, indices);
    }

    /// Draw the isosurface mesh into the given render pass.
    ///
    /// Only binds camera (group 0) and lighting (group 1) — no selection
    /// or color groups needed.
    pub fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &DrawBindGroups<'a>,
    ) {
        if self.vertex_count == 0 {
            return;
        }
        render_pass.set_bind_group(0, bind_groups.camera, &[]);
        render_pass.set_bind_group(1, bind_groups.lighting, &[]);
        self.mesh_pass
            .draw_indexed(render_pass, self.vertex_buffer.buffer());
    }

    /// GPU buffer sizes: `(label, used_bytes, allocated_bytes)`.
    pub fn buffer_info(&self) -> Vec<(&'static str, usize, usize)> {
        vec![
            (
                "Isosurface Vertices",
                self.vertex_buffer.len(),
                self.vertex_buffer.capacity(),
            ),
            (
                "Isosurface Indices",
                self.mesh_pass.index_buffer_len(),
                self.mesh_pass.index_buffer_capacity(),
            ),
        ]
    }
}
