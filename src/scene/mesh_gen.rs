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

/// Generate mesh for a single entity.
pub fn generate_entity_mesh(
    g: &PerEntityData,
    display: &DisplayOptions,
    colors: &ColorOptions,
    geometry: &GeometryOptions,
) -> CachedEntityMesh {
    // Derive per-residue colors from scores when in score coloring mode
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

    // --- Backbone mesh (protein + nucleic acid, unified) ---
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
    let backbone_vert_count = backbone_mesh.vertices.len() as u32;
    let backbone_verts = bytemuck::cast_slice(&backbone_mesh.vertices).to_vec();

    // --- Sidechain capsules ---
    let sidechain_positions = g.sidechains.positions();
    let sidechain_hydrophobicity = g.sidechains.hydrophobicity();
    let sidechain_residue_indices = g.sidechains.residue_indices();

    let offset_map: HashMap<u32, Vec3> =
        backbone_mesh.sheet_offsets.iter().copied().collect();
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
    let sidechain_insts = SidechainRenderer::generate_instances(
        &sd,
        None,
        Some((colors.hydrophobic_sidechain, colors.hydrophilic_sidechain)),
    );
    let sidechain_instance_count = sidechain_insts.len() as u32;
    let sidechain_instances = bytemuck::cast_slice(&sidechain_insts).to_vec();

    // --- Ball-and-stick instances ---
    let (bns_spheres, bns_capsules) =
        BallAndStickRenderer::generate_all_instances(
            &g.non_protein_entities,
            display,
            Some(colors),
            0, // pick IDs are 0-based; concatenation applies global offset
        );
    let bns_sphere_count = bns_spheres.len() as u32;
    let bns_capsule_count = bns_capsules.len() as u32;
    let bns_sphere_instances = bytemuck::cast_slice(&bns_spheres).to_vec();
    let bns_capsule_instances = bytemuck::cast_slice(&bns_capsules).to_vec();

    // --- Nucleic acid instances ---
    let (na_stems, na_rings) = NucleicAcidRenderer::generate_instances(
        &g.nucleic_acid_chains,
        &g.nucleic_acid_rings,
        Some(colors.nucleic_acid),
    );
    let na_stem_count = na_stems.len() as u32;
    let na_stem_instances = bytemuck::cast_slice(&na_stems).to_vec();
    let na_ring_count = na_rings.len() as u32;
    let na_ring_instances = bytemuck::cast_slice(&na_rings).to_vec();

    CachedEntityMesh {
        backbone: CachedBackbone {
            verts: backbone_verts,
            tube_inds: backbone_mesh.tube_indices,
            ribbon_inds: backbone_mesh.ribbon_indices,
            vert_count: backbone_vert_count,
            sheet_offsets: backbone_mesh.sheet_offsets,
            chain_ranges: backbone_mesh.chain_ranges,
        },
        sidechain_instances,
        sidechain_instance_count,
        bns: BallAndStickInstances {
            sphere_instances: bns_sphere_instances,
            sphere_count: bns_sphere_count,
            capsule_instances: bns_capsule_instances,
            capsule_count: bns_capsule_count,
        },
        na: NucleicAcidInstances {
            stem_instances: na_stem_instances,
            stem_count: na_stem_count,
            ring_instances: na_ring_instances,
            ring_count: na_ring_count,
        },
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

/// Generate backbone + optional sidechain mesh for an animation frame.
pub fn process_animation_frame(
    backbone_chains: &[Vec<Vec3>],
    na_chains: &[Vec<Vec3>],
    sidechains: Option<&SidechainAtoms>,
    ss_types: Option<&[SSType]>,
    per_residue_colors: Option<&[[f32; 3]]>,
    geometry: &GeometryOptions,
    per_chain_lod: Option<&[(usize, usize)]>,
) -> PreparedAnimationFrame {
    // --- Backbone mesh (protein + nucleic acid, unified) ---
    let total_residues: usize =
        backbone_chains.iter().map(|c| c.len() / 3).sum::<usize>()
            + na_chains.iter().map(Vec::len).sum::<usize>();
    let safe_geo = geometry.clamped_for_residues(total_residues);
    let backbone_mesh = BackboneRenderer::generate_mesh_colored(
        &ChainPair {
            protein: backbone_chains,
            na: na_chains,
        },
        ss_types,
        per_residue_colors,
        &safe_geo,
        per_chain_lod,
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
        sidechains.map_or((None, 0), |sc| {
            let positions = sc.positions();
            let hydrophobicity = sc.hydrophobicity();
            let residue_indices = sc.residue_indices();
            let offset_map: HashMap<u32, Vec3> =
                backbone_mesh.sheet_offsets.iter().copied().collect();
            let adjusted_positions = adjust_sidechains_for_sheet(
                &positions,
                &residue_indices,
                &offset_map,
            );
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
