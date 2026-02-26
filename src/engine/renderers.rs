use foldit_conv::render::RenderCoords;
use foldit_conv::types::entity::MoleculeEntity;
use glam::Vec3;

use crate::camera::frustum::Frustum;
use crate::gpu::render_context::RenderContext;
use crate::gpu::shader_composer::ShaderComposer;
use crate::options::VisoOptions;
use crate::renderer::draw_context::DrawBindGroups;
use crate::renderer::geometry::backbone::{BackboneRenderer, ChainPair};
use crate::renderer::geometry::ball_and_stick::BallAndStickRenderer;
use crate::renderer::geometry::band::BandRenderer;
use crate::renderer::geometry::nucleic_acid::NucleicAcidRenderer;
use crate::renderer::geometry::pull::PullRenderer;
use crate::renderer::geometry::sidechain::{SidechainRenderer, SidechainView};
use crate::renderer::PipelineLayouts;
use crate::scene::Scene;

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
    /// Create all geometry renderers from the loaded scene data.
    pub fn new(
        context: &RenderContext,
        layouts: &PipelineLayouts,
        render_coords: &RenderCoords,
        scene: &Scene,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, crate::error::VisoError> {
        let na_chains: Vec<Vec<Vec3>> = scene
            .nucleic_acid_entities()
            .iter()
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
        let na_rings: Vec<foldit_conv::types::entity::NucleotideRing> = scene
            .nucleic_acid_entities()
            .iter()
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
        scene: &Scene,
        options: &VisoOptions,
    ) {
        let non_protein_refs: Vec<MoleculeEntity> = scene
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
