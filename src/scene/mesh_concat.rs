use std::collections::HashMap;

use foldit_conv::render::sidechain::{SidechainAtomData, SidechainAtoms};
use foldit_conv::secondary_structure::SSType;
use foldit_conv::types::entity::{MoleculeEntity, NucleotideRing};
use glam::Vec3;

use super::prepared::{
    BackboneMeshData, BallAndStickInstances, CachedEntityMesh,
    NucleicAcidInstances, PreparedScene, FALLBACK_RESIDUE_COLOR,
};
use super::EntityResidueRange;
use crate::animation::transition::Transition;
use crate::renderer::geometry::backbone::ChainRange;
use crate::renderer::picking::PickMap;

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

/// Patch pick IDs in BnS instance byte buffers.
///
/// Each instance has a pick ID stored as `f32` at `pick_id_byte_offset`
/// within each `instance_stride`-sized block. Adds `(new_offset - old_offset)`
/// to each pick ID so that 0-based local atom indices become globally unique.
fn offset_bns_pick_ids(
    dst: &mut Vec<u8>,
    src: &[u8],
    old_offset: u32,
    new_offset: u32,
    instance_stride: usize,
    pick_id_byte_offset: usize,
) {
    if old_offset == new_offset {
        dst.extend_from_slice(src);
        return;
    }

    let start = dst.len();
    dst.extend_from_slice(src);

    let delta = new_offset as f32 - old_offset as f32;
    let mut pos = start + pick_id_byte_offset;
    while pos + 4 <= dst.len() {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&dst[pos..pos + 4]);
        let old_val = f32::from_ne_bytes(bytes);
        let patched = old_val + delta;
        dst[pos..pos + 4].copy_from_slice(&patched.to_ne_bytes());
        pos += instance_stride;
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
    // Running atom offset for making BnS pick IDs unique across mesh groups.
    // Starts at 0; final `global_residue_offset` is added after the loop.
    let mut bns_pick_offset: u32 = 0;

    // --- Nucleic acid ---
    let mut all_na_stem_bytes: Vec<u8> = Vec::new();
    let mut total_na_stem_count: u32 = 0;
    let mut all_na_ring_bytes: Vec<u8> = Vec::new();
    let mut total_na_ring_count: u32 = 0;

    // --- Passthrough ---
    let mut all_backbone_chains: Vec<Vec<Vec3>> = Vec::new();
    let mut all_sidechain_atoms: Vec<SidechainAtomData> = Vec::new();
    let mut all_sidechain_bonds: Vec<(u32, u32)> = Vec::new();
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
    let mut entity_residue_ranges: Vec<EntityResidueRange> = Vec::new();

    for (entity_id, mesh) in entity_meshes {
        let sc_atom_offset = all_sidechain_atoms.len() as u32;

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

        // BNS: offset 0-based pick IDs to contiguous global IDs
        // SphereInstance: 32 bytes, pick_id at byte 28 (color.w)
        // CapsuleInstance: 64 bytes, pick_id at byte 28 (endpoint_b.w)
        offset_bns_pick_ids(
            &mut all_bns_spheres,
            &mesh.bns.sphere_instances,
            0,
            bns_pick_offset,
            32,
            28,
        );
        total_bns_sphere_count += mesh.bns.sphere_count;
        offset_bns_pick_ids(
            &mut all_bns_capsules,
            &mesh.bns.capsule_instances,
            0,
            bns_pick_offset,
            64,
            28,
        );
        total_bns_capsule_count += mesh.bns.capsule_count;
        // Advance offset by total atoms across this entity group's
        // non-protein entities
        for npe in &mesh.non_protein_entities {
            bns_pick_offset += npe.coords.num_atoms as u32;
        }

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
        // Sidechain atoms: offset residue indices during concatenation
        for atom in &mesh.sidechain.atoms {
            all_positions.push(atom.position);
            all_sidechain_atoms.push(SidechainAtomData {
                position: atom.position,
                residue_idx: atom.residue_idx + global_residue_offset,
                atom_name: atom.atom_name.clone(),
                is_hydrophobic: atom.is_hydrophobic,
            });
        }
        for &(a, b) in &mesh.sidechain.bonds {
            all_sidechain_bonds.push((a + sc_atom_offset, b + sc_atom_offset));
        }
        for &(ca_pos, cb_idx) in &mesh.sidechain.backbone_bonds {
            all_backbone_sidechain_bonds
                .push((ca_pos, cb_idx + sc_atom_offset));
        }
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
        entity_residue_ranges.push(EntityResidueRange {
            entity_id: *entity_id,
            start: global_residue_offset,
            count: mesh.residue_count,
        });

        global_residue_offset += mesh.residue_count;
    }

    // Add global_residue_offset to all BnS pick IDs so they're contiguous
    // after backbone/sidechain residue IDs.
    if global_residue_offset > 0 {
        let delta = global_residue_offset as f32;
        // Patch sphere pick IDs (32-byte stride, offset 28)
        let mut pos = 28;
        while pos + 4 <= all_bns_spheres.len() {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&all_bns_spheres[pos..pos + 4]);
            let patched = f32::from_ne_bytes(bytes) + delta;
            all_bns_spheres[pos..pos + 4]
                .copy_from_slice(&patched.to_ne_bytes());
            pos += 32;
        }
        // Patch capsule pick IDs (64-byte stride, offset 28)
        let mut pos = 28;
        while pos + 4 <= all_bns_capsules.len() {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&all_bns_capsules[pos..pos + 4]);
            let patched = f32::from_ne_bytes(bytes) + delta;
            all_bns_capsules[pos..pos + 4]
                .copy_from_slice(&patched.to_ne_bytes());
            pos += 64;
        }
    }

    // Build flat ss_types
    let ss_types =
        build_ss_types(has_any_ss, &ss_parts, global_residue_offset as usize);

    // Build PickMap from non-protein entities
    let mut atom_entries: Vec<(u32, u32)> = Vec::new();
    for entity in &all_non_protein {
        for atom_idx in 0..entity.coords.num_atoms as u32 {
            atom_entries.push((entity.entity_id, atom_idx));
        }
    }
    let pick_map = PickMap::new(global_residue_offset, atom_entries);

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
        },
        na: NucleicAcidInstances {
            stem_instances: all_na_stem_bytes,
            stem_count: total_na_stem_count,
            ring_instances: all_na_ring_bytes,
            ring_count: total_na_ring_count,
        },
        backbone_chains: all_backbone_chains,
        na_chains: all_na_chains,
        sidechain: SidechainAtoms {
            atoms: all_sidechain_atoms,
            bonds: all_sidechain_bonds,
            backbone_bonds: all_backbone_sidechain_bonds,
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
        pick_map,
    }
}

/// Build a flat `SSType` array from per-entity overrides, or `None` if no
/// entity provided secondary-structure data.
fn build_ss_types(
    has_any_ss: bool,
    ss_parts: &[(u32, Option<Vec<SSType>>, u32)],
    total: usize,
) -> Option<Vec<SSType>> {
    if !has_any_ss {
        return None;
    }
    let mut ss = vec![SSType::Coil; total];
    for (offset, ss_override, count) in ss_parts {
        let Some(overrides) = ss_override else {
            continue;
        };
        let start = *offset as usize;
        let end = (start + *count as usize).min(total);
        let n = end.saturating_sub(start);
        for (i, &s) in overrides.iter().enumerate().take(n) {
            ss[start + i] = s;
        }
    }
    Some(ss)
}
