use std::collections::HashMap;

use foldit_conv::render::sidechain::SidechainAtoms;
use foldit_conv::secondary_structure::SSType;
use glam::Vec3;

use super::prepared::{
    BackboneMeshData, BallAndStickInstances, CachedBackbone, CachedEntityMesh,
    NucleicAcidInstances, PreparedAnimationFrame,
};
use super::PerEntityData;
use crate::options::{ColorOptions, DisplayOptions, GeometryOptions};
use crate::renderer::geometry::backbone::{BackboneRenderer, ChainPair};
use crate::renderer::geometry::ball_and_stick::BallAndStickRenderer;
use crate::renderer::geometry::nucleic_acid::NucleicAcidRenderer;
use crate::renderer::geometry::sidechain::{SidechainRenderer, SidechainView};
use crate::util::score_color;
use crate::util::sheet_adjust::{
    adjust_bonds_for_sheet, adjust_sidechains_for_sheet,
};

/// Generate sidechain capsule instances with sheet-surface adjustment.
fn generate_sidechain_bytes(
    g: &PerEntityData,
    sheet_offsets: &[(u32, Vec3)],
    colors: &ColorOptions,
) -> (Vec<u8>, u32) {
    let sidechain_positions = g.sidechains.positions();
    let sidechain_hydrophobicity = g.sidechains.hydrophobicity();
    let sidechain_residue_indices = g.sidechains.residue_indices();

    let offset_map: HashMap<u32, Vec3> =
        sheet_offsets.iter().copied().collect();
    let adjusted_positions = adjust_sidechains_for_sheet(
        &sidechain_positions,
        &sidechain_residue_indices,
        &offset_map,
    );
    let adjusted_bonds = adjust_bonds_for_sheet(
        &g.sidechains.backbone_bonds,
        &sidechain_residue_indices,
        &offset_map,
    );
    let sd = SidechainView {
        positions: &adjusted_positions,
        bonds: &g.sidechains.bonds,
        backbone_bonds: &adjusted_bonds,
        hydrophobicity: &sidechain_hydrophobicity,
        residue_indices: &sidechain_residue_indices,
    };
    let insts = SidechainRenderer::generate_instances(
        &sd,
        None,
        Some((colors.hydrophobic_sidechain, colors.hydrophilic_sidechain)),
    );
    let count = insts.len() as u32;
    (bytemuck::cast_slice(&insts).to_vec(), count)
}

/// Generate ball-and-stick + nucleic acid instance bytes.
fn generate_non_backbone_bytes(
    g: &PerEntityData,
    display: &DisplayOptions,
    colors: &ColorOptions,
) -> (BallAndStickInstances, NucleicAcidInstances) {
    let (bns_spheres, bns_capsules) =
        BallAndStickRenderer::generate_all_instances(
            &g.non_protein_entities,
            display,
            Some(colors),
            0,
        );
    let (na_stems, na_rings) = NucleicAcidRenderer::generate_instances(
        &g.nucleic_acid_chains,
        &g.nucleic_acid_rings,
        Some(colors.nucleic_acid),
    );
    (
        BallAndStickInstances {
            sphere_instances: bytemuck::cast_slice(&bns_spheres).to_vec(),
            sphere_count: bns_spheres.len() as u32,
            capsule_instances: bytemuck::cast_slice(&bns_capsules).to_vec(),
            capsule_count: bns_capsules.len() as u32,
        },
        NucleicAcidInstances {
            stem_instances: bytemuck::cast_slice(&na_stems).to_vec(),
            stem_count: na_stems.len() as u32,
            ring_instances: bytemuck::cast_slice(&na_rings).to_vec(),
            ring_count: na_rings.len() as u32,
        },
    )
}

/// Generate mesh for a single entity.
pub fn generate_entity_mesh(
    g: &PerEntityData,
    display: &DisplayOptions,
    colors: &ColorOptions,
    geometry: &GeometryOptions,
) -> CachedEntityMesh {
    use crate::options::BackboneColorMode;
    let per_residue_colors = match display.backbone_color_mode {
        BackboneColorMode::Score => g
            .per_residue_scores
            .as_ref()
            .map(|s| score_color::per_residue_score_colors(s)),
        BackboneColorMode::ScoreRelative => g
            .per_residue_scores
            .as_ref()
            .map(|s| score_color::per_residue_score_colors_relative(s)),
        BackboneColorMode::SecondaryStructure | BackboneColorMode::Chain => {
            None
        }
    };

    let backbone_mesh = BackboneRenderer::generate_mesh_colored(
        &ChainPair {
            protein: &g.backbone_chains,
            na: &g.nucleic_acid_chains,
        },
        g.ss_override.as_deref(),
        per_residue_colors.as_deref(),
        geometry,
        None,
    );

    let (sidechain_instances, sidechain_instance_count) =
        generate_sidechain_bytes(g, &backbone_mesh.sheet_offsets, colors);
    let (bns, na) = generate_non_backbone_bytes(g, display, colors);

    CachedEntityMesh {
        backbone: CachedBackbone {
            verts: bytemuck::cast_slice(&backbone_mesh.vertices).to_vec(),
            tube_inds: backbone_mesh.tube_indices,
            ribbon_inds: backbone_mesh.ribbon_indices,
            vert_count: backbone_mesh.vertices.len() as u32,
            sheet_offsets: backbone_mesh.sheet_offsets,
            chain_ranges: backbone_mesh.chain_ranges,
        },
        sidechain_instances,
        sidechain_instance_count,
        bns,
        na,
        residue_count: g.residue_count,
        backbone_chains: g.backbone_chains.clone(),
        nucleic_acid_chains: g.nucleic_acid_chains.clone(),
        sidechain: g.sidechains.clone(),
        ss_override: g.ss_override.clone(),
        per_residue_colors,
        non_protein_entities: g.non_protein_entities.clone(),
        nucleic_acid_rings: g.nucleic_acid_rings.clone(),
    }
}

/// Generate sidechain instances for an animation frame.
fn generate_animation_sidechains(
    sc: &SidechainAtoms,
    sheet_offsets: &[(u32, Vec3)],
) -> (Option<Vec<u8>>, u32) {
    let positions = sc.positions();
    let hydrophobicity = sc.hydrophobicity();
    let residue_indices = sc.residue_indices();
    let offset_map: HashMap<u32, Vec3> =
        sheet_offsets.iter().copied().collect();
    let adjusted_positions =
        adjust_sidechains_for_sheet(&positions, &residue_indices, &offset_map);
    let adjusted_bonds = adjust_bonds_for_sheet(
        &sc.backbone_bonds,
        &residue_indices,
        &offset_map,
    );
    let sd = SidechainView {
        positions: &adjusted_positions,
        bonds: &sc.bonds,
        backbone_bonds: &adjusted_bonds,
        hydrophobicity: &hydrophobicity,
        residue_indices: &residue_indices,
    };
    let insts = SidechainRenderer::generate_instances(&sd, None, None);
    let count = insts.len() as u32;
    (Some(bytemuck::cast_slice(&insts).to_vec()), count)
}

/// Input data for [`process_animation_frame`].
pub struct AnimationFrameInput<'a> {
    pub backbone_chains: &'a [Vec<Vec3>],
    pub na_chains: &'a [Vec<Vec3>],
    pub sidechains: Option<&'a SidechainAtoms>,
    pub ss_types: Option<&'a [SSType]>,
    pub per_residue_colors: Option<&'a [[f32; 3]]>,
    pub geometry: &'a GeometryOptions,
    pub per_chain_lod: Option<&'a [(usize, usize)]>,
}

/// Generate backbone + optional sidechain mesh for an animation frame.
pub fn process_animation_frame(
    input: &AnimationFrameInput,
) -> PreparedAnimationFrame {
    // --- Backbone mesh (protein + nucleic acid, unified) ---
    let total_residues: usize = input
        .backbone_chains
        .iter()
        .map(|c| c.len() / 3)
        .sum::<usize>()
        + input.na_chains.iter().map(Vec::len).sum::<usize>();
    let safe_geo = input.geometry.clamped_for_residues(total_residues);
    let backbone_mesh = BackboneRenderer::generate_mesh_colored(
        &ChainPair {
            protein: input.backbone_chains,
            na: input.na_chains,
        },
        input.ss_types,
        input.per_residue_colors,
        &safe_geo,
        input.per_chain_lod,
    );
    let backbone_tube_index_count = backbone_mesh.tube_indices.len() as u32;
    let backbone_ribbon_index_count = backbone_mesh.ribbon_indices.len() as u32;
    let backbone_vertices =
        bytemuck::cast_slice(&backbone_mesh.vertices).to_vec();
    let backbone_tube_indices =
        bytemuck::cast_slice(&backbone_mesh.tube_indices).to_vec();
    let backbone_ribbon_indices =
        bytemuck::cast_slice(&backbone_mesh.ribbon_indices).to_vec();

    // --- Optional sidechain capsules ---
    let (sidechain_instances, sidechain_instance_count) =
        input.sidechains.map_or((None, 0), |sc| {
            generate_animation_sidechains(sc, &backbone_mesh.sheet_offsets)
        });

    PreparedAnimationFrame {
        backbone: BackboneMeshData {
            vertices: backbone_vertices,
            tube_indices: backbone_tube_indices,
            tube_index_count: backbone_tube_index_count,
            ribbon_indices: backbone_ribbon_indices,
            ribbon_index_count: backbone_ribbon_index_count,
            sheet_offsets: backbone_mesh.sheet_offsets,
            chain_ranges: backbone_mesh.chain_ranges,
        },
        sidechain_instances,
        sidechain_instance_count,
    }
}
