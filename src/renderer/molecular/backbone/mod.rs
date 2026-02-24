//! Unified backbone renderer for protein and nucleic acid chains.
//!
//! Two GPU pipelines share one vertex buffer:
//! - **Tube pass** (back-face culling): round cross-sections
//! - **Ribbon pass** (no culling): flat cross-sections

pub(crate) mod mesh;
pub(crate) mod profile;
pub(crate) mod sheet;
pub(crate) mod spline;

use foldit_conv::secondary_structure::SSType;
use glam::Vec3;
pub(crate) use mesh::ChainRange;

use super::{
    draw_context::DrawBindGroups,
    mesh_pass::{create_mesh_pipeline, MeshPass},
};
use crate::{
    camera::frustum::Frustum,
    gpu::{
        dynamic_buffer::DynamicBuffer, render_context::RenderContext,
        shader_composer::ShaderComposer,
    },
    options::GeometryOptions,
    util::hash::hash_vec3_slices,
};

// ==================== VERTEX FORMAT ====================

/// 52-byte backbone vertex, shared by tube and ribbon pipelines.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BackboneVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    pub residue_idx: u32,
    pub center_pos: [f32; 3],
}

pub fn backbone_vertex_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: size_of::<BackboneVertex>() as wgpu::BufferAddress,
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
                format: wgpu::VertexFormat::Float32x3,
                offset: 24,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32,
                offset: 36,
                shader_location: 3,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 40,
                shader_location: 4,
            },
        ],
    }
}

// ==================== PREPARED DATA ====================

/// Pre-computed backbone mesh for GPU upload (from scene processor).
pub struct PreparedBackboneData<'a> {
    pub vertices: &'a [u8],
    pub tube_indices: &'a [u8],
    pub ribbon_indices: &'a [u8],
    pub tube_index_count: u32,
    pub ribbon_index_count: u32,
    pub sheet_offsets: Vec<(u32, Vec3)>,
    pub chain_ranges: Vec<ChainRange>,
    pub cached_chains: Vec<Vec<Vec3>>,
    pub cached_na_chains: Vec<Vec<Vec3>>,
    pub ss_override: Option<Vec<SSType>>,
}

// ==================== RENDERER ====================

/// Unified backbone renderer with tube + ribbon passes.
pub struct BackboneRenderer {
    tube_pass: MeshPass,
    ribbon_pass: MeshPass,
    vertex_buffer: DynamicBuffer,
    last_hash: u64,
    cached_chains: Vec<Vec<Vec3>>,
    cached_na_chains: Vec<Vec<Vec3>>,
    ss_override: Option<Vec<SSType>>,
    sheet_offsets: Vec<(u32, Vec3)>,
    chain_ranges: Vec<ChainRange>,
    cached_lod_tiers: Vec<u8>,
}

impl BackboneRenderer {
    pub fn new(
        context: &RenderContext,
        camera_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
        selection_layout: &wgpu::BindGroupLayout,
        color_layout: &wgpu::BindGroupLayout,
        protein_chains: &[Vec<Vec3>],
        na_chains: &[Vec<Vec3>],
        shader_composer: &mut ShaderComposer,
    ) -> Self {
        // No mesh generation here — the scene processor background thread
        // handles all mesh generation. We just create empty GPU buffers
        // and pipelines. The first frame will be blank until
        // apply_prepared() uploads the background thread's result.

        let device = &context.device;
        let vertex_buffer = new_buffer::<BackboneVertex>(
            device,
            "Backbone Vertex",
            &[],
            wgpu::BufferUsages::VERTEX,
        );

        let layouts = &[
            camera_layout,
            lighting_layout,
            selection_layout,
            color_layout,
        ];
        let vl = backbone_vertex_buffer_layout();

        let tube_pipeline = create_mesh_pipeline(
            context,
            "Backbone Tube",
            "raster/mesh/backbone_tube.wgsl",
            Some(wgpu::Face::Back),
            layouts,
            vl.clone(),
            shader_composer,
        );
        let ribbon_pipeline = create_mesh_pipeline(
            context,
            "Backbone Ribbon",
            "raster/mesh/backbone_tube.wgsl",
            None,
            layouts,
            vl,
            shader_composer,
        );

        let tube_pass =
            MeshPass::new(device, "Backbone Tube Index", tube_pipeline, &[]);
        let ribbon_pass = MeshPass::new(
            device,
            "Backbone Ribbon Index",
            ribbon_pipeline,
            &[],
        );

        Self {
            tube_pass,
            ribbon_pass,
            vertex_buffer,
            last_hash: combined_hash(protein_chains, na_chains),
            cached_chains: protein_chains.to_vec(),
            cached_na_chains: na_chains.to_vec(),
            ss_override: None,
            sheet_offsets: Vec::new(),
            chain_ranges: Vec::new(),
            cached_lod_tiers: Vec::new(),
        }
    }

    // ── Draw ──

    pub fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &DrawBindGroups<'a>,
    ) {
        let Some(color) = bind_groups.color else {
            return;
        };
        render_pass.set_bind_group(0, bind_groups.camera, &[]);
        render_pass.set_bind_group(1, bind_groups.lighting, &[]);
        render_pass.set_bind_group(2, bind_groups.selection, &[]);
        render_pass.set_bind_group(3, color, &[]);

        let vb = self.vertex_buffer.buffer();
        self.tube_pass.draw_indexed(render_pass, vb);
        self.ribbon_pass.draw_indexed(render_pass, vb);
    }

    /// Frustum-culled draw: skip chains whose bounding sphere is off-screen.
    pub fn draw_culled<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &DrawBindGroups<'a>,
        frustum: &Frustum,
    ) {
        let Some(color) = bind_groups.color else {
            return;
        };
        render_pass.set_bind_group(0, bind_groups.camera, &[]);
        render_pass.set_bind_group(1, bind_groups.lighting, &[]);
        render_pass.set_bind_group(2, bind_groups.selection, &[]);
        render_pass.set_bind_group(3, color, &[]);

        let vb = self.vertex_buffer.buffer();

        if self.chain_ranges.is_empty() {
            // No chain range data — fall back to full draw
            self.tube_pass.draw_indexed(render_pass, vb);
            self.ribbon_pass.draw_indexed(render_pass, vb);
            return;
        }

        for range in &self.chain_ranges {
            if !frustum
                .intersects_sphere(range.bounding_center, range.bounding_radius)
            {
                continue;
            }

            self.tube_pass.draw_indexed_range(
                render_pass,
                vb,
                range.tube_index_start..range.tube_index_end,
            );
            self.ribbon_pass.draw_indexed_range(
                render_pass,
                vb,
                range.ribbon_index_start..range.ribbon_index_end,
            );
        }
    }

    // ── Live update ──

    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        protein_chains: &[Vec<Vec3>],
        na_chains: &[Vec<Vec3>],
        ss_types: Option<&[SSType]>,
        geo: &GeometryOptions,
    ) {
        let new_hash = combined_hash(protein_chains, na_chains);
        if new_hash == self.last_hash && ss_types.is_none() {
            return;
        }
        self.last_hash = new_hash;
        self.cached_chains = protein_chains.to_vec();
        self.cached_na_chains = na_chains.to_vec();
        if let Some(ss) = ss_types {
            self.ss_override = Some(ss.to_vec());
        }
        self.write_mesh(device, queue, geo);
    }

    pub fn set_ss_override(&mut self, ss_types: Option<Vec<SSType>>) {
        self.ss_override = ss_types;
    }

    pub fn regenerate(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        geo: &GeometryOptions,
    ) {
        self.write_mesh(device, queue, geo);
    }

    // ── Scene-processor path ──

    pub fn apply_prepared(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: PreparedBackboneData,
    ) {
        if !data.vertices.is_empty() {
            let _ =
                self.vertex_buffer.write_bytes(device, queue, data.vertices);
            self.tube_pass.write_indices_bytes(
                device,
                queue,
                data.tube_indices,
                data.tube_index_count,
            );
            self.ribbon_pass.write_indices_bytes(
                device,
                queue,
                data.ribbon_indices,
                data.ribbon_index_count,
            );
        }
        self.sheet_offsets = data.sheet_offsets;
        self.chain_ranges = data.chain_ranges;
        self.cached_chains = data.cached_chains;
        self.cached_na_chains = data.cached_na_chains;
        self.last_hash =
            combined_hash(&self.cached_chains, &self.cached_na_chains);
        if let Some(ss) = data.ss_override {
            self.ss_override = Some(ss);
        }
    }

    pub fn update_metadata(
        &mut self,
        cached_chains: Vec<Vec<Vec3>>,
        cached_na_chains: Vec<Vec<Vec3>>,
        ss_override: Option<Vec<SSType>>,
    ) {
        self.cached_chains = cached_chains;
        self.cached_na_chains = cached_na_chains;
        self.last_hash =
            combined_hash(&self.cached_chains, &self.cached_na_chains);
        if let Some(ss) = ss_override {
            self.ss_override = Some(ss);
        }
    }

    pub fn apply_mesh(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vertices: &[u8],
        tube_indices: &[u8],
        ribbon_indices: &[u8],
        tube_index_count: u32,
        ribbon_index_count: u32,
        sheet_offsets: Vec<(u32, Vec3)>,
        chain_ranges: Vec<ChainRange>,
    ) {
        if !vertices.is_empty() {
            let _ = self.vertex_buffer.write_bytes(device, queue, vertices);
            self.tube_pass.write_indices_bytes(
                device,
                queue,
                tube_indices,
                tube_index_count,
            );
            self.ribbon_pass.write_indices_bytes(
                device,
                queue,
                ribbon_indices,
                ribbon_index_count,
            );
        }
        self.sheet_offsets = sheet_offsets;
        self.chain_ranges = chain_ranges;
    }

    // ── Accessors ──

    pub fn chain_ranges(&self) -> &[ChainRange] {
        &self.chain_ranges
    }
    pub fn cached_lod_tiers(&self) -> &[u8] {
        &self.cached_lod_tiers
    }
    pub fn set_cached_lod_tiers(&mut self, tiers: Vec<u8>) {
        self.cached_lod_tiers = tiers;
    }
    pub fn sheet_offsets(&self) -> &[(u32, Vec3)] {
        &self.sheet_offsets
    }
    pub fn cached_chains(&self) -> &[Vec<Vec3>] {
        &self.cached_chains
    }
    pub fn cached_na_chains(&self) -> &[Vec<Vec3>] {
        &self.cached_na_chains
    }
    pub fn vertex_buffer(&self) -> &wgpu::Buffer {
        self.vertex_buffer.buffer()
    }
    pub fn tube_index_buffer(&self) -> &wgpu::Buffer {
        self.tube_pass.index_buffer()
    }
    pub fn ribbon_index_buffer(&self) -> &wgpu::Buffer {
        self.ribbon_pass.index_buffer()
    }
    pub fn tube_index_count(&self) -> u32 {
        self.tube_pass.index_count
    }
    pub fn ribbon_index_count(&self) -> u32 {
        self.ribbon_pass.index_count
    }

    /// GPU buffer sizes: `(label, used_bytes, allocated_bytes)`.
    pub fn buffer_info(&self) -> Vec<(&'static str, usize, usize)> {
        vec![
            (
                "Backbone Vertex",
                self.vertex_buffer.len(),
                self.vertex_buffer.capacity(),
            ),
            (
                "Backbone Tube Idx",
                self.tube_pass.index_buffer_len(),
                self.tube_pass.index_buffer_capacity(),
            ),
            (
                "Backbone Ribbon Idx",
                self.ribbon_pass.index_buffer_len(),
                self.ribbon_pass.index_buffer_capacity(),
            ),
        ]
    }

    // ── Static mesh generation ──

    pub(crate) fn generate_mesh(
        protein_chains: &[Vec<Vec3>],
        na_chains: &[Vec<Vec3>],
        ss_override: Option<&[SSType]>,
        geo: &GeometryOptions,
    ) -> (
        Vec<BackboneVertex>,
        Vec<u32>,
        Vec<u32>,
        Vec<(u32, Vec3)>,
        Vec<ChainRange>,
    ) {
        mesh::generate_mesh_colored(
            protein_chains,
            na_chains,
            ss_override,
            None,
            geo,
            None,
        )
    }

    pub(crate) fn generate_mesh_colored(
        protein_chains: &[Vec<Vec3>],
        na_chains: &[Vec<Vec3>],
        ss_override: Option<&[SSType]>,
        per_residue_colors: Option<&[[f32; 3]]>,
        geo: &GeometryOptions,
        per_chain_lod: Option<&[(usize, usize)]>,
    ) -> (
        Vec<BackboneVertex>,
        Vec<u32>,
        Vec<u32>,
        Vec<(u32, Vec3)>,
        Vec<ChainRange>,
    ) {
        mesh::generate_mesh_colored(
            protein_chains,
            na_chains,
            ss_override,
            per_residue_colors,
            geo,
            per_chain_lod,
        )
    }

    // ── Internal ──

    fn write_mesh(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        geo: &GeometryOptions,
    ) {
        let (verts, t_idx, r_idx, offsets, ranges) = Self::generate_mesh(
            &self.cached_chains,
            &self.cached_na_chains,
            self.ss_override.as_deref(),
            geo,
        );
        if verts.is_empty() {
            self.tube_pass.index_count = 0;
            self.ribbon_pass.index_count = 0;
            self.chain_ranges.clear();
            return;
        }
        let _ = self.vertex_buffer.write(device, queue, &verts);
        self.tube_pass.write_indices(device, queue, &t_idx);
        self.ribbon_pass.write_indices(device, queue, &r_idx);
        self.sheet_offsets = offsets;
        self.chain_ranges = ranges;
    }
}

impl super::MolecularRenderer for BackboneRenderer {
    fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &DrawBindGroups<'a>,
    ) {
        self.draw(render_pass, bind_groups);
    }
}

// ── Helpers ──

fn new_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &str,
    data: &[T],
    usage: wgpu::BufferUsages,
) -> DynamicBuffer {
    if data.is_empty() {
        DynamicBuffer::new(device, label, size_of::<T>() * 1000, usage)
    } else {
        DynamicBuffer::new_with_data(device, label, data, usage)
    }
}

fn combined_hash(protein_chains: &[Vec<Vec3>], na_chains: &[Vec<Vec3>]) -> u64 {
    use std::{
        collections::hash_map::DefaultHasher,
        hash::{Hash, Hasher},
    };
    let mut h = DefaultHasher::new();
    hash_vec3_slices(protein_chains).hash(&mut h);
    hash_vec3_slices(na_chains).hash(&mut h);
    h.finish()
}
