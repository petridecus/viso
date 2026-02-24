use std::collections::HashMap;

use foldit_conv::secondary_structure::SSType;
use glam::Vec3;

use super::{
    prepared::{
        AnimationSidechainData, BackboneMeshData, BallAndStickInstances,
        CachedBackbone, CachedEntityMesh, NucleicAcidInstances,
        PreparedAnimationFrame, SidechainCpuData,
    },
    PerEntityData,
};
use crate::{
    options::{ColorOptions, DisplayOptions, GeometryOptions},
    renderer::molecular::{
        backbone::BackboneRenderer, ball_and_stick::BallAndStickRenderer,
        capsule_sidechain::CapsuleSidechainRenderer,
        nucleic_acid::NucleicAcidRenderer,
    },
    util::{
        score_color,
        sheet_adjust::{adjust_bonds_for_sheet, adjust_sidechains_for_sheet},
    },
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
    let (
        backbone_verts_typed,
        backbone_tube_inds,
        backbone_ribbon_inds,
        sheet_offsets,
        backbone_chain_ranges,
    ) = BackboneRenderer::generate_mesh_colored(
        &g.backbone_chains,
        &g.nucleic_acid_chains,
        g.ss_override.as_deref(),
        per_residue_colors.as_deref(),
        geometry,
        None,
    );
    let backbone_vert_count = backbone_verts_typed.len() as u32;
    let backbone_verts = bytemuck::cast_slice(&backbone_verts_typed).to_vec();

    // --- Sidechain capsules ---
    let sidechain_positions: Vec<Vec3> =
        g.sidechain_atoms.iter().map(|a| a.position).collect();
    let sidechain_hydrophobicity: Vec<bool> =
        g.sidechain_atoms.iter().map(|a| a.is_hydrophobic).collect();
    let sidechain_residue_indices: Vec<u32> =
        g.sidechain_atoms.iter().map(|a| a.residue_idx).collect();

    let offset_map: HashMap<u32, Vec3> =
        sheet_offsets.iter().copied().collect();
    let adjusted_positions = adjust_sidechains_for_sheet(
        &sidechain_positions,
        &sidechain_residue_indices,
        &offset_map,
    );
    let adjusted_bonds = adjust_bonds_for_sheet(
        &g.backbone_sidechain_bonds,
        &sidechain_residue_indices,
        &offset_map,
    );
    let sidechain_insts = CapsuleSidechainRenderer::generate_instances(
        &adjusted_positions,
        &g.sidechain_bonds,
        &adjusted_bonds,
        &sidechain_hydrophobicity,
        &sidechain_residue_indices,
        None,
        Some((colors.hydrophobic_sidechain, colors.hydrophilic_sidechain)),
    );
    let sidechain_instance_count = sidechain_insts.len() as u32;
    let sidechain_instances = bytemuck::cast_slice(&sidechain_insts).to_vec();

    // --- Ball-and-stick instances ---
    let (bns_spheres, bns_capsules, bns_picking) =
        BallAndStickRenderer::generate_all_instances(
            &g.non_protein_entities,
            display,
            Some(colors),
        );
    let bns_sphere_count = bns_spheres.len() as u32;
    let bns_capsule_count = bns_capsules.len() as u32;
    let bns_picking_count = bns_picking.len() as u32;
    let bns_sphere_instances = bytemuck::cast_slice(&bns_spheres).to_vec();
    let bns_capsule_instances = bytemuck::cast_slice(&bns_capsules).to_vec();
    let bns_picking_capsules = bytemuck::cast_slice(&bns_picking).to_vec();

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

    // --- Passthrough data ---
    let sidechain_atom_names: Vec<String> = g
        .sidechain_atoms
        .iter()
        .map(|a| a.atom_name.clone())
        .collect();

    CachedEntityMesh {
        backbone: CachedBackbone {
            verts: backbone_verts,
            tube_inds: backbone_tube_inds,
            ribbon_inds: backbone_ribbon_inds,
            vert_count: backbone_vert_count,
            sheet_offsets,
            chain_ranges: backbone_chain_ranges,
        },
        sidechain_instances,
        sidechain_instance_count,
        bns: BallAndStickInstances {
            sphere_instances: bns_sphere_instances,
            sphere_count: bns_sphere_count,
            capsule_instances: bns_capsule_instances,
            capsule_count: bns_capsule_count,
            picking_capsules: bns_picking_capsules,
            picking_count: bns_picking_count,
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
        sidechain: SidechainCpuData {
            positions: sidechain_positions,
            bonds: g.sidechain_bonds.clone(),
            backbone_bonds: g.backbone_sidechain_bonds.clone(),
            hydrophobicity: sidechain_hydrophobicity,
            residue_indices: sidechain_residue_indices,
            atom_names: sidechain_atom_names,
        },
        ss_override: g.ss_override.clone(),
        per_residue_colors,
        non_protein_entities: g.non_protein_entities.clone(),
        nucleic_acid_rings: g.nucleic_acid_rings.clone(),
    }
}

/// Generate backbone + optional sidechain mesh for an animation frame.
pub fn process_animation_frame(
    backbone_chains: Vec<Vec<Vec3>>,
    na_chains: Vec<Vec<Vec3>>,
    sidechains: Option<AnimationSidechainData>,
    ss_types: Option<Vec<SSType>>,
    per_residue_colors: Option<Vec<[f32; 3]>>,
    geometry: &GeometryOptions,
    per_chain_lod: Option<Vec<(usize, usize)>>,
) -> PreparedAnimationFrame {
    // --- Backbone mesh (protein + nucleic acid, unified) ---
    let total_residues: usize =
        backbone_chains.iter().map(|c| c.len() / 3).sum::<usize>()
            + na_chains.iter().map(Vec::len).sum::<usize>();
    let safe_geo = geometry.clamped_for_residues(total_residues);
    let (verts, tube_inds, ribbon_inds, sheet_offsets, chain_ranges) =
        BackboneRenderer::generate_mesh_colored(
            &backbone_chains,
            &na_chains,
            ss_types.as_deref(),
            per_residue_colors.as_deref(),
            &safe_geo,
            per_chain_lod.as_deref(),
        );
    let backbone_tube_index_count = tube_inds.len() as u32;
    let backbone_ribbon_index_count = ribbon_inds.len() as u32;
    let backbone_vertices = bytemuck::cast_slice(&verts).to_vec();
    let backbone_tube_indices = bytemuck::cast_slice(&tube_inds).to_vec();
    let backbone_ribbon_indices = bytemuck::cast_slice(&ribbon_inds).to_vec();

    // --- Optional sidechain capsules ---
    let (sidechain_instances, sidechain_instance_count) =
        if let Some(sc) = sidechains {
            let offset_map: HashMap<u32, Vec3> =
                sheet_offsets.iter().copied().collect();
            let adjusted_positions = adjust_sidechains_for_sheet(
                &sc.sidechain_positions,
                &sc.sidechain_residue_indices,
                &offset_map,
            );
            let adjusted_bonds = adjust_bonds_for_sheet(
                &sc.backbone_sidechain_bonds,
                &sc.sidechain_residue_indices,
                &offset_map,
            );
            let insts = CapsuleSidechainRenderer::generate_instances(
                &adjusted_positions,
                &sc.sidechain_bonds,
                &adjusted_bonds,
                &sc.sidechain_hydrophobicity,
                &sc.sidechain_residue_indices,
                None,
                None,
            );
            let count = insts.len() as u32;
            (Some(bytemuck::cast_slice(&insts).to_vec()), count)
        } else {
            (None, 0)
        };

    PreparedAnimationFrame {
        backbone: BackboneMeshData {
            vertices: backbone_vertices,
            tube_indices: backbone_tube_indices,
            tube_index_count: backbone_tube_index_count,
            ribbon_indices: backbone_ribbon_indices,
            ribbon_index_count: backbone_ribbon_index_count,
            sheet_offsets,
            chain_ranges,
        },
        sidechain_instances,
        sidechain_instance_count,
    }
}
