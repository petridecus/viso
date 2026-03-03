//! Rendering subsystems for molecular visualization.
//!
//! Contains molecular renderers (tubes, ribbons, sidechains, ball-and-stick,
//! nucleic acids) and post-processing effects (SSAO, bloom, FXAA).

/// Bind groups shared across all molecular draw calls.
pub mod draw_context;
/// All GPU infrastructure grouped together (device, renderers, picking, etc.).
pub(crate) mod gpu_pipeline;
pub(crate) use gpu_pipeline::GpuPipeline;
/// Molecular geometry assemblers (backbone, sidechain, ball-and-stick, etc.).
pub mod geometry;
/// Reusable impostor-pass primitives and instance types.
pub mod impostor;
/// Shared indexed-mesh draw-pass abstraction.
pub(crate) mod mesh;
/// GPU-based object picking and selection management.
pub mod picking;
/// Background mesh generation pipeline (scene → GPU-ready buffers).
pub(crate) mod pipeline;
pub(crate) mod pipeline_util;
/// Post-processing effects (SSAO, bloom, FXAA).
pub mod postprocess;

use foldit_conv::render::RenderCoords;
use foldit_conv::types::entity::MoleculeEntity;
use glam::Vec3;

use self::draw_context::DrawBindGroups;
use self::geometry::{
    BackboneRenderer, BallAndStickRenderer, BandRenderer, ChainPair,
    NucleicAcidRenderer, PullRenderer, SidechainRenderer, SidechainView,
};
use crate::camera::frustum::Frustum;
use crate::engine::entity_store::EntityStore;
use crate::gpu::{RenderContext, ShaderComposer};
use crate::options::VisoOptions;

/// Bind group layouts shared by all molecular geometry pipelines.
///
/// Every molecular renderer (backbone, sidechain, ball-and-stick, band, pull,
/// nucleic acid) binds the same camera, lighting, and selection groups.
pub struct PipelineLayouts {
    pub camera: wgpu::BindGroupLayout,
    pub lighting: wgpu::BindGroupLayout,
    pub selection: wgpu::BindGroupLayout,
    pub color: wgpu::BindGroupLayout,
}

// ---------------------------------------------------------------------------
// GeometryPassInput
// ---------------------------------------------------------------------------

/// Input for the main geometry render pass.
pub(crate) struct GeometryPassInput<'a> {
    /// Color attachment (post-process input).
    pub color: &'a wgpu::TextureView,
    /// Normal attachment (post-process input).
    pub normal: &'a wgpu::TextureView,
    /// Depth attachment.
    pub depth: &'a wgpu::TextureView,
    /// Whether sidechain capsules should be drawn.
    pub show_sidechains: bool,
}

// ---------------------------------------------------------------------------
// Renderers
// ---------------------------------------------------------------------------

/// All geometry renderers grouped together.
pub(crate) struct Renderers {
    pub backbone: BackboneRenderer,
    pub sidechain: SidechainRenderer,
    pub band: BandRenderer,
    pub pull: PullRenderer,
    pub ball_and_stick: BallAndStickRenderer,
    pub nucleic_acid: NucleicAcidRenderer,
}

impl Renderers {
    /// Create all geometry renderers from the loaded entity data.
    pub fn new(
        context: &RenderContext,
        layouts: &PipelineLayouts,
        render_coords: &RenderCoords,
        store: &EntityStore,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, crate::error::VisoError> {
        let na_chains: Vec<Vec<Vec3>> = store
            .nucleic_acid_entities()
            .flat_map(|se| se.entity.extract_p_atom_chains())
            .collect();
        let backbone = BackboneRenderer::new(
            context,
            layouts,
            &ChainPair {
                protein: &render_coords.backbone_chains,
                na: &na_chains,
            },
            shader_composer,
        )?;
        let sidechain_positions = render_coords.sidechain_positions();
        let sidechain_hydrophobicity = render_coords.sidechain_hydrophobicity();
        let sidechain_residue_indices =
            render_coords.sidechain_residue_indices();
        let sidechain = SidechainRenderer::new(
            context,
            layouts,
            &SidechainView {
                positions: &sidechain_positions,
                bonds: &render_coords.sidechain_bonds,
                backbone_bonds: &render_coords.backbone_sidechain_bonds,
                hydrophobicity: &sidechain_hydrophobicity,
                residue_indices: &sidechain_residue_indices,
            },
            shader_composer,
        )?;
        let band = BandRenderer::new(context, layouts, shader_composer)?;
        let pull = PullRenderer::new(context, layouts, shader_composer)?;
        let ball_and_stick =
            BallAndStickRenderer::new(context, layouts, shader_composer)?;
        let na_rings: Vec<foldit_conv::types::entity::NucleotideRing> = store
            .nucleic_acid_entities()
            .flat_map(|se| se.entity.extract_base_rings())
            .collect();
        let nucleic_acid = NucleicAcidRenderer::new(
            context,
            layouts,
            &na_chains,
            &na_rings,
            shader_composer,
        )?;
        Ok(Self {
            backbone,
            sidechain,
            band,
            pull,
            ball_and_stick,
            nucleic_acid,
        })
    }

    /// Populate ball-and-stick renderer with non-protein entities.
    pub fn init_ball_and_stick_entities(
        &mut self,
        context: &RenderContext,
        store: &EntityStore,
        options: &VisoOptions,
    ) {
        let non_protein_refs: Vec<MoleculeEntity> = store
            .entities()
            .iter()
            .filter(|se| !se.is_protein())
            .map(|se| se.entity.clone())
            .collect();
        self.ball_and_stick.update_from_entities(
            context,
            &non_protein_refs,
            &options.display,
            Some(&options.colors),
        );
    }

    /// Encode the main geometry render pass.
    pub fn encode_geometry_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GeometryPassInput<'_>,
        bind_groups: &DrawBindGroups<'_>,
        frustum: &Frustum,
    ) {
        let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main render pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: input.color,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: input.normal,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                }),
            ],
            depth_stencil_attachment: Some(
                wgpu::RenderPassDepthStencilAttachment {
                    view: input.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                },
            ),
            ..Default::default()
        });

        self.backbone.draw_culled(&mut rp, bind_groups, frustum);

        if input.show_sidechains {
            self.sidechain.draw(&mut rp, bind_groups);
        }

        self.ball_and_stick.draw(&mut rp, bind_groups);
        self.nucleic_acid.draw(&mut rp, bind_groups);
        self.band.draw(&mut rp, bind_groups);
        self.pull.draw(&mut rp, bind_groups);
    }

    /// GPU buffer sizes across all renderers.
    ///
    /// Each entry is `(label, used_bytes, allocated_bytes)`.
    pub fn buffer_info(&self) -> Vec<(&str, usize, usize)> {
        let mut stats = Vec::new();
        stats.extend(self.backbone.buffer_info());
        stats.extend(self.sidechain.buffer_info());
        stats.extend(self.ball_and_stick.buffer_info());
        stats.extend(self.band.buffer_info());
        stats.extend(self.pull.buffer_info());
        stats.extend(self.nucleic_acid.buffer_info());
        stats
    }
}
