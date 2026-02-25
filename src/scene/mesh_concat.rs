use std::collections::HashMap;

use foldit_conv::{
    coords::{entity::NucleotideRing, MoleculeEntity},
    secondary_structure::SSType,
};
use glam::Vec3;

use super::prepared::{
    BackboneMeshData, BallAndStickInstances, CachedEntityMesh,
    NucleicAcidInstances, PreparedScene, SidechainCpuData,
    FALLBACK_RESIDUE_COLOR,
};
use crate::{
    animation::transition::Transition, renderer::geometry::backbone::ChainRange,
};

/// Offset the `residue_idx` field embedded in raw vertex bytes.
///
/// `BackboneVertex` and `NaVertex` share the same 52-byte
/// layout with a `u32 residue_idx` at byte offset 36. When concatenating
/// vertices from multiple entities, each entity's local residue indices
/// must be shifted by the global offset so the GPU's per-residue color
/// buffer is indexed correctly.
pub fn offset_vertex_residue_idx(dst: &mut Vec<u8>, src: &[u8], offset: u32) {
    const VERTEX_SIZE: usize = 52;
    const RESIDUE_IDX_OFFSET: usize = 36;

    if offset == 0 {
        dst.extend_from_slice(src);
        return;
    }

    let start = dst.len();
    dst.extend_from_slice(src);

    // Patch each vertex's residue_idx in-place
    let mut pos = start + RESIDUE_IDX_OFFSET;
    while pos + 4 <= dst.len() {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&dst[pos..pos + 4]);
        let patched = u32::from_ne_bytes(bytes) + offset;
        dst[pos..pos + 4].copy_from_slice(&patched.to_ne_bytes());
        pos += VERTEX_SIZE;
    }
}

/// Concatenate per-entity cached meshes into a single PreparedScene.
pub fn concatenate_meshes(
    entity_meshes: &[(u32, &CachedEntityMesh)],
    entity_transitions: HashMap<u32, Transition>,
) -> PreparedScene {
    // --- Backbone (unified vertex buffer, partitioned index buffers) ---
    let mut all_backbone_verts: Vec<u8> = Vec::new();
    let mut all_backbone_tube_inds: Vec<u32> = Vec::new();
    let mut all_backbone_ribbon_inds: Vec<u32> = Vec::new();
    let mut backbone_vert_offset: u32 = 0;
    let mut all_sheet_offsets: Vec<(u32, Vec3)> = Vec::new();
    let mut all_chain_ranges: Vec<ChainRange> = Vec::new();

    // --- Sidechain ---
    let mut all_sidechain: Vec<u8> = Vec::new();
    let mut total_sidechain_count: u32 = 0;

    // --- BNS ---
    let mut all_bns_spheres: Vec<u8> = Vec::new();
    let mut total_bns_sphere_count: u32 = 0;
    let mut all_bns_capsules: Vec<u8> = Vec::new();
    let mut total_bns_capsule_count: u32 = 0;
    let mut all_bns_picking: Vec<u8> = Vec::new();
    let mut total_bns_picking_count: u32 = 0;

    // --- Nucleic acid ---
    let mut all_na_stem_bytes: Vec<u8> = Vec::new();
    let mut total_na_stem_count: u32 = 0;
    let mut all_na_ring_bytes: Vec<u8> = Vec::new();
    let mut total_na_ring_count: u32 = 0;

    // --- Passthrough ---
    let mut all_backbone_chains: Vec<Vec<Vec3>> = Vec::new();
    let mut all_sidechain_positions: Vec<Vec3> = Vec::new();
    let mut all_sidechain_bonds: Vec<(u32, u32)> = Vec::new();
    let mut all_sidechain_hydrophobicity: Vec<bool> = Vec::new();
    let mut all_sidechain_residue_indices: Vec<u32> = Vec::new();
    let mut all_sidechain_atom_names: Vec<String> = Vec::new();
    let mut all_backbone_sidechain_bonds: Vec<(Vec3, u32)> = Vec::new();
    let mut all_non_protein: Vec<MoleculeEntity> = Vec::new();
    let mut all_na_chains: Vec<Vec<Vec3>> = Vec::new();
    let mut all_na_rings: Vec<NucleotideRing> = Vec::new();
    let mut all_positions: Vec<Vec3> = Vec::new();

    // SS types: built from per-group overrides
    let mut has_any_ss = false;
    let mut ss_parts: Vec<(u32, Option<Vec<SSType>>, u32)> = Vec::new();
    let mut global_residue_offset: u32 = 0;

    // Per-residue colors: concatenated from cached group colors
    let mut has_any_colors = false;
    let mut all_per_residue_colors: Vec<[f32; 3]> = Vec::new();

    // Track where each entity's residues land in the flat arrays
    let mut entity_residue_ranges: Vec<(u32, u32, u32)> = Vec::new();

    for (entity_id, mesh) in entity_meshes {
        let sc_atom_offset = all_sidechain_positions.len() as u32;

        // Backbone: offset vertex residue_idx and indices
        offset_vertex_residue_idx(
            &mut all_backbone_verts,
            &mesh.backbone.verts,
            global_residue_offset,
        );
        for &idx in &mesh.backbone.tube_inds {
            all_backbone_tube_inds.push(idx + backbone_vert_offset);
        }
        for &idx in &mesh.backbone.ribbon_inds {
            all_backbone_ribbon_inds.push(idx + backbone_vert_offset);
        }
        // Sheet offsets: offset residue indices
        for &(res_idx, offset) in &mesh.backbone.sheet_offsets {
            all_sheet_offsets.push((res_idx + global_residue_offset, offset));
        }
        // Chain ranges: offset index ranges into the global buffers
        let tube_idx_offset = all_backbone_tube_inds.len() as u32
            - mesh.backbone.tube_inds.len() as u32;
        let ribbon_idx_offset = all_backbone_ribbon_inds.len() as u32
            - mesh.backbone.ribbon_inds.len() as u32;
        for r in &mesh.backbone.chain_ranges {
            all_chain_ranges.push(ChainRange {
                tube_index_start: r.tube_index_start + tube_idx_offset,
                tube_index_end: r.tube_index_end + tube_idx_offset,
                ribbon_index_start: r.ribbon_index_start + ribbon_idx_offset,
                ribbon_index_end: r.ribbon_index_end + ribbon_idx_offset,
                bounding_center: r.bounding_center,
                bounding_radius: r.bounding_radius,
            });
        }
        backbone_vert_offset += mesh.backbone.vert_count;

        // Sidechain: concatenate directly (instances are self-contained)
        all_sidechain.extend_from_slice(&mesh.sidechain_instances);
        total_sidechain_count += mesh.sidechain_instance_count;

        // BNS: concatenate directly
        all_bns_spheres.extend_from_slice(&mesh.bns.sphere_instances);
        total_bns_sphere_count += mesh.bns.sphere_count;
        all_bns_capsules.extend_from_slice(&mesh.bns.capsule_instances);
        total_bns_capsule_count += mesh.bns.capsule_count;
        all_bns_picking.extend_from_slice(&mesh.bns.picking_capsules);
        total_bns_picking_count += mesh.bns.picking_count;

        // NA: concatenate instances directly (self-contained)
        all_na_stem_bytes.extend_from_slice(&mesh.na.stem_instances);
        total_na_stem_count += mesh.na.stem_count;
        all_na_ring_bytes.extend_from_slice(&mesh.na.ring_instances);
        total_na_ring_count += mesh.na.ring_count;

        // Passthrough
        for chain in &mesh.backbone_chains {
            all_backbone_chains.push(chain.clone());
            all_positions.extend(chain);
        }
        all_sidechain_positions.extend(&mesh.sidechain.positions);
        for &(a, b) in &mesh.sidechain.bonds {
            all_sidechain_bonds.push((a + sc_atom_offset, b + sc_atom_offset));
        }
        all_sidechain_hydrophobicity.extend(&mesh.sidechain.hydrophobicity);
        for &ri in &mesh.sidechain.residue_indices {
            all_sidechain_residue_indices.push(ri + global_residue_offset);
        }
        all_sidechain_atom_names
            .extend(mesh.sidechain.atom_names.iter().cloned());
        for &(ca_pos, cb_idx) in &mesh.sidechain.backbone_bonds {
            all_backbone_sidechain_bonds
                .push((ca_pos, cb_idx + sc_atom_offset));
        }
        all_positions.extend(&mesh.sidechain.positions);
        all_non_protein.extend(mesh.non_protein_entities.iter().cloned());
        for chain in &mesh.nucleic_acid_chains {
            all_na_chains.push(chain.clone());
            all_positions.extend(chain);
        }
        all_na_rings.extend(mesh.nucleic_acid_rings.iter().cloned());

        // SS override tracking
        if mesh.ss_override.is_some() {
            has_any_ss = true;
        }
        ss_parts.push((
            global_residue_offset,
            mesh.ss_override.clone(),
            mesh.residue_count,
        ));

        // Per-residue color tracking
        if let Some(ref colors) = mesh.per_residue_colors {
            has_any_colors = true;
            all_per_residue_colors.extend_from_slice(colors);
        } else {
            // Pad with default so indices stay aligned
            all_per_residue_colors.extend(std::iter::repeat_n(
                FALLBACK_RESIDUE_COLOR,
                mesh.residue_count as usize,
            ));
        }

        // Track entity residue range
        entity_residue_ranges.push((
            *entity_id,
            global_residue_offset,
            mesh.residue_count,
        ));

        global_residue_offset += mesh.residue_count;
    }

    // Build flat ss_types
    let ss_types = if has_any_ss {
        let total = global_residue_offset as usize;
        let mut ss = vec![SSType::Coil; total];
        for (offset, ss_override, count) in &ss_parts {
            if let Some(overrides) = ss_override {
                let start = *offset as usize;
                let end = (start + *count as usize).min(total);
                for (i, &s) in overrides.iter().enumerate() {
                    if start + i < end {
                        ss[start + i] = s;
                    }
                }
            }
        }
        Some(ss)
    } else {
        None
    };

    PreparedScene {
        backbone: BackboneMeshData {
            vertices: all_backbone_verts,
            tube_indices: bytemuck::cast_slice(&all_backbone_tube_inds)
                .to_vec(),
            tube_index_count: all_backbone_tube_inds.len() as u32,
            ribbon_indices: bytemuck::cast_slice(&all_backbone_ribbon_inds)
                .to_vec(),
            ribbon_index_count: all_backbone_ribbon_inds.len() as u32,
            sheet_offsets: all_sheet_offsets,
            chain_ranges: all_chain_ranges,
        },
        sidechain_instances: all_sidechain,
        sidechain_instance_count: total_sidechain_count,
        bns: BallAndStickInstances {
            sphere_instances: all_bns_spheres,
            sphere_count: total_bns_sphere_count,
            capsule_instances: all_bns_capsules,
            capsule_count: total_bns_capsule_count,
            picking_capsules: all_bns_picking,
            picking_count: total_bns_picking_count,
        },
        na: NucleicAcidInstances {
            stem_instances: all_na_stem_bytes,
            stem_count: total_na_stem_count,
            ring_instances: all_na_ring_bytes,
            ring_count: total_na_ring_count,
        },
        backbone_chains: all_backbone_chains,
        na_chains: all_na_chains,
        sidechain: SidechainCpuData {
            positions: all_sidechain_positions,
            bonds: all_sidechain_bonds,
            backbone_bonds: all_backbone_sidechain_bonds,
            hydrophobicity: all_sidechain_hydrophobicity,
            residue_indices: all_sidechain_residue_indices,
            atom_names: all_sidechain_atom_names,
        },
        ss_types,
        per_residue_colors: if has_any_colors {
            Some(all_per_residue_colors)
        } else {
            None
        },
        all_positions,
        entity_transitions,
        entity_residue_ranges,
        non_protein_entities: all_non_protein,
        nucleic_acid_rings: all_na_rings,
    }
}
