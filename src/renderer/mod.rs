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

use glam::Vec3;
use molex::entity::molecule::nucleic_acid::NAEntity;
use molex::entity::molecule::protein::ProteinEntity;
use molex::MoleculeEntity;

use self::draw_context::DrawBindGroups;
use self::geometry::isosurface::IsosurfaceRenderer;
use self::geometry::{
    BackboneRenderer, BallAndStickRenderer, BandRenderer, BondRenderer,
    ChainPair, NucleicAcidRenderer, PullRenderer, SidechainRenderer,
    SidechainView,
};
use crate::camera::frustum::Frustum;
use crate::engine::entity_store::EntityStore;
use crate::gpu::{RenderContext, ShaderComposer};
use crate::options::VisoOptions;

/// Bind group layouts shared by all molecular geometry pipelines.
///
/// Every molecular renderer (backbone, sidechain, ball-and-stick, band, pull,
/// nucleic acid) binds the same camera, lighting, and selection groups.
pub(crate) struct PipelineLayouts {
    pub(crate) camera: wgpu::BindGroupLayout,
    pub(crate) lighting: wgpu::BindGroupLayout,
    pub(crate) selection: wgpu::BindGroupLayout,
    pub(crate) color: wgpu::BindGroupLayout,
}

// ---------------------------------------------------------------------------
// GeometryPassInput
// ---------------------------------------------------------------------------

/// Input for the main geometry render pass.
pub(crate) struct GeometryPassInput<'a> {
    /// Color attachment (post-process input).
    pub(crate) color: &'a wgpu::TextureView,
    /// Normal attachment (post-process input).
    pub(crate) normal: &'a wgpu::TextureView,
    /// Depth attachment.
    pub(crate) depth: &'a wgpu::TextureView,
    /// Whether sidechain capsules should be drawn.
    pub(crate) show_sidechains: bool,
}

// ---------------------------------------------------------------------------
// Renderers
// ---------------------------------------------------------------------------

/// All geometry renderers grouped together.
pub(crate) struct Renderers {
    pub(crate) backbone: BackboneRenderer,
    pub(crate) sidechain: SidechainRenderer,
    pub(crate) bond: BondRenderer,
    pub(crate) band: BandRenderer,
    pub(crate) pull: PullRenderer,
    pub(crate) ball_and_stick: BallAndStickRenderer,
    pub(crate) nucleic_acid: NucleicAcidRenderer,
    pub(crate) isosurface: IsosurfaceRenderer,
}

impl Renderers {
    /// Create all geometry renderers from the loaded entity data.
    pub fn new(
        context: &RenderContext,
        layouts: &PipelineLayouts,
        store: &EntityStore,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, crate::error::VisoError> {
        let na_chains: Vec<Vec<Vec3>> = store
            .nucleic_acid_entities()
            .filter_map(|se| se.entity.as_nucleic_acid())
            .flat_map(NAEntity::extract_p_atom_segments)
            .collect();
        // Extract protein backbone chains from all protein entities.
        let protein_chains: Vec<Vec<Vec3>> = store
            .entities()
            .iter()
            .filter_map(|se| se.entity.as_protein())
            .flat_map(ProteinEntity::to_interleaved_segments)
            .collect();
        let backbone = BackboneRenderer::new(
            context,
            layouts,
            &ChainPair {
                protein: &protein_chains,
                na: &na_chains,
            },
            shader_composer,
        )?;
        // Start with empty sidechain data — it will be populated on first
        // scene sync when per-entity data is computed.
        let sidechain = SidechainRenderer::new(
            context,
            layouts,
            &SidechainView {
                positions: &[],
                bonds: &[],
                backbone_bonds: &[],
                hydrophobicity: &[],
                residue_indices: &[],
            },
            shader_composer,
        )?;
        let bond = BondRenderer::new(context, layouts, shader_composer)?;
        let band = BandRenderer::new(context, layouts, shader_composer)?;
        let pull = PullRenderer::new(context, layouts, shader_composer)?;
        let ball_and_stick =
            BallAndStickRenderer::new(context, layouts, shader_composer)?;
        let na_rings: Vec<molex::NucleotideRing> = store
            .nucleic_acid_entities()
            .filter_map(|se| se.entity.as_nucleic_acid())
            .flat_map(NAEntity::extract_base_rings)
            .collect();
        let nucleic_acid = NucleicAcidRenderer::new(
            context,
            layouts,
            &na_chains,
            &na_rings,
            shader_composer,
        )?;
        let isosurface =
            IsosurfaceRenderer::new(context, layouts, shader_composer)?;
        Ok(Self {
            backbone,
            sidechain,
            bond,
            band,
            pull,
            ball_and_stick,
            nucleic_acid,
            isosurface,
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
        self.bond.draw(&mut rp, bind_groups);
        self.band.draw(&mut rp, bind_groups);
        self.pull.draw(&mut rp, bind_groups);
        self.isosurface.draw(&mut rp, bind_groups);
    }

    /// GPU buffer sizes across all renderers.
    ///
    /// Each entry is `(label, used_bytes, allocated_bytes)`.
    pub fn buffer_info(&self) -> Vec<(&str, usize, usize)> {
        let mut stats = Vec::new();
        stats.extend(self.backbone.buffer_info());
        stats.extend(self.sidechain.buffer_info());
        stats.extend(self.ball_and_stick.buffer_info());
        stats.extend(self.bond.buffer_info());
        stats.extend(self.band.buffer_info());
        stats.extend(self.pull.buffer_info());
        stats.extend(self.nucleic_acid.buffer_info());
        stats.extend(self.isosurface.buffer_info());
        stats
    }
}
