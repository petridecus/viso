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

/// Layout descriptor for patching pick IDs in BnS byte buffers.
struct BnsPickPatch {
    old_offset: u32,
    new_offset: u32,
    instance_stride: usize,
    pick_id_byte_offset: usize,
}

/// Patch pick IDs in BnS instance byte buffers.
///
/// Each instance has a pick ID stored as `f32` at `pick_id_byte_offset`
/// within each `instance_stride`-sized block. Adds `(new_offset - old_offset)`
/// to each pick ID so that 0-based local atom indices become globally unique.
fn offset_bns_pick_ids(dst: &mut Vec<u8>, src: &[u8], patch: &BnsPickPatch) {
    if patch.old_offset == patch.new_offset {
        dst.extend_from_slice(src);
        return;
    }

    let start = dst.len();
    dst.extend_from_slice(src);

    let delta = patch.new_offset as f32 - patch.old_offset as f32;
    let mut pos = start + patch.pick_id_byte_offset;
    while pos + 4 <= dst.len() {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&dst[pos..pos + 4]);
        let old_val = f32::from_ne_bytes(bytes);
        let patched = old_val + delta;
        dst[pos..pos + 4].copy_from_slice(&patched.to_ne_bytes());
        pos += patch.instance_stride;
    }
}

/// Accumulator that merges per-entity cached meshes into combined buffers.
#[derive(Default)]
struct MeshAccumulator {
    // Backbone
    backbone_verts: Vec<u8>,
    backbone_tube_inds: Vec<u32>,
    backbone_ribbon_inds: Vec<u32>,
    backbone_vert_offset: u32,
    sheet_offsets: Vec<(u32, Vec3)>,
    chain_ranges: Vec<ChainRange>,
    // Sidechain instances
    sidechain_bytes: Vec<u8>,
    sidechain_count: u32,
    // Ball-and-stick
    bns_spheres: Vec<u8>,
    bns_sphere_count: u32,
    bns_capsules: Vec<u8>,
    bns_capsule_count: u32,
    bns_pick_offset: u32,
    // Nucleic acid instances
    na_stem_bytes: Vec<u8>,
    na_stem_count: u32,
    na_ring_bytes: Vec<u8>,
    na_ring_count: u32,
    // Passthrough data
    backbone_chains: Vec<Vec<Vec3>>,
    sidechain_atoms: Vec<SidechainAtomData>,
    sidechain_bonds: Vec<(u32, u32)>,
    backbone_sidechain_bonds: Vec<(Vec3, u32)>,
    non_protein: Vec<MoleculeEntity>,
    na_chains: Vec<Vec<Vec3>>,
    na_rings: Vec<NucleotideRing>,
    all_positions: Vec<Vec3>,
    // SS / color / residue tracking
    has_any_ss: bool,
    ss_parts: Vec<(u32, Option<Vec<SSType>>, u32)>,
    residue_offset: u32,
    has_any_colors: bool,
    per_residue_colors: Vec<[f32; 3]>,
    entity_residue_ranges: Vec<EntityResidueRange>,
}

impl MeshAccumulator {
    fn push_entity(&mut self, entity_id: u32, mesh: &CachedEntityMesh) {
        let sc_atom_offset = self.sidechain_atoms.len() as u32;

        self.push_backbone(mesh);

        // Sidechain instances (self-contained)
        self.sidechain_bytes
            .extend_from_slice(&mesh.sidechain_instances);
        self.sidechain_count += mesh.sidechain_instance_count;

        self.push_bns(mesh);

        // NA instances (self-contained)
        self.na_stem_bytes
            .extend_from_slice(&mesh.na.stem_instances);
        self.na_stem_count += mesh.na.stem_count;
        self.na_ring_bytes
            .extend_from_slice(&mesh.na.ring_instances);
        self.na_ring_count += mesh.na.ring_count;

        self.push_passthrough(mesh, sc_atom_offset);

        // SS override tracking
        if mesh.ss_override.is_some() {
            self.has_any_ss = true;
        }
        self.ss_parts.push((
            self.residue_offset,
            mesh.ss_override.clone(),
            mesh.residue_count,
        ));

        // Per-residue colors
        if let Some(ref colors) = mesh.per_residue_colors {
            self.has_any_colors = true;
            self.per_residue_colors.extend_from_slice(colors);
        } else {
            self.per_residue_colors.extend(std::iter::repeat_n(
                FALLBACK_RESIDUE_COLOR,
                mesh.residue_count as usize,
            ));
        }

        // Entity residue range
        self.entity_residue_ranges.push(EntityResidueRange {
            entity_id,
            start: self.residue_offset,
            count: mesh.residue_count,
        });

        self.residue_offset += mesh.residue_count;
    }

    fn push_backbone(&mut self, mesh: &CachedEntityMesh) {
        offset_vertex_residue_idx(
            &mut self.backbone_verts,
            &mesh.backbone.verts,
            self.residue_offset,
        );
        for &idx in &mesh.backbone.tube_inds {
            self.backbone_tube_inds
                .push(idx + self.backbone_vert_offset);
        }
        for &idx in &mesh.backbone.ribbon_inds {
            self.backbone_ribbon_inds
                .push(idx + self.backbone_vert_offset);
        }
        for &(res_idx, offset) in &mesh.backbone.sheet_offsets {
            self.sheet_offsets
                .push((res_idx + self.residue_offset, offset));
        }
        let tube_idx_offset = self.backbone_tube_inds.len() as u32
            - mesh.backbone.tube_inds.len() as u32;
        let ribbon_idx_offset = self.backbone_ribbon_inds.len() as u32
            - mesh.backbone.ribbon_inds.len() as u32;
        for r in &mesh.backbone.chain_ranges {
            self.chain_ranges.push(ChainRange {
                tube_index_start: r.tube_index_start + tube_idx_offset,
                tube_index_end: r.tube_index_end + tube_idx_offset,
                ribbon_index_start: r.ribbon_index_start + ribbon_idx_offset,
                ribbon_index_end: r.ribbon_index_end + ribbon_idx_offset,
                bounding_center: r.bounding_center,
                bounding_radius: r.bounding_radius,
            });
        }
        self.backbone_vert_offset += mesh.backbone.vert_count;
    }

    fn push_bns(&mut self, mesh: &CachedEntityMesh) {
        // SphereInstance: 32 bytes, pick_id at byte 28 (color.w)
        offset_bns_pick_ids(
            &mut self.bns_spheres,
            &mesh.bns.sphere_instances,
            &BnsPickPatch {
                old_offset: 0,
                new_offset: self.bns_pick_offset,
                instance_stride: 32,
                pick_id_byte_offset: 28,
            },
        );
        self.bns_sphere_count += mesh.bns.sphere_count;
        // CapsuleInstance: 64 bytes, pick_id at byte 28 (endpoint_b.w)
        offset_bns_pick_ids(
            &mut self.bns_capsules,
            &mesh.bns.capsule_instances,
            &BnsPickPatch {
                old_offset: 0,
                new_offset: self.bns_pick_offset,
                instance_stride: 64,
                pick_id_byte_offset: 28,
            },
        );
        self.bns_capsule_count += mesh.bns.capsule_count;
        for npe in &mesh.non_protein_entities {
            self.bns_pick_offset += npe.coords.num_atoms as u32;
        }
    }

    fn push_passthrough(
        &mut self,
        mesh: &CachedEntityMesh,
        sc_atom_offset: u32,
    ) {
        for chain in &mesh.backbone_chains {
            self.backbone_chains.push(chain.clone());
            self.all_positions.extend(chain);
        }
        for atom in &mesh.sidechain.atoms {
            self.all_positions.push(atom.position);
            self.sidechain_atoms.push(SidechainAtomData {
                position: atom.position,
                residue_idx: atom.residue_idx + self.residue_offset,
                atom_name: atom.atom_name.clone(),
                is_hydrophobic: atom.is_hydrophobic,
            });
        }
        for &(a, b) in &mesh.sidechain.bonds {
            self.sidechain_bonds
                .push((a + sc_atom_offset, b + sc_atom_offset));
        }
        for &(ca_pos, cb_idx) in &mesh.sidechain.backbone_bonds {
            self.backbone_sidechain_bonds
                .push((ca_pos, cb_idx + sc_atom_offset));
        }
        self.non_protein
            .extend(mesh.non_protein_entities.iter().cloned());
        for chain in &mesh.nucleic_acid_chains {
            self.na_chains.push(chain.clone());
            self.all_positions.extend(chain);
        }
        self.na_rings
            .extend(mesh.nucleic_acid_rings.iter().cloned());
    }

    /// Shift all BnS pick IDs by the global residue offset so they sit
    /// after backbone/sidechain residue IDs in the pick map.
    fn finalize_bns_pick_ids(&mut self) {
        if self.residue_offset == 0 {
            return;
        }
        let delta = self.residue_offset as f32;
        patch_pick_id_buffer(&mut self.bns_spheres, delta, 32, 28);
        patch_pick_id_buffer(&mut self.bns_capsules, delta, 64, 28);
    }

    fn build_pick_map(&self) -> PickMap {
        let mut atom_entries: Vec<(u32, u32)> = Vec::new();
        for entity in &self.non_protein {
            for atom_idx in 0..entity.coords.num_atoms as u32 {
                atom_entries.push((entity.entity_id, atom_idx));
            }
        }
        PickMap::new(self.residue_offset, atom_entries)
    }

    fn into_prepared_scene(
        mut self,
        entity_transitions: HashMap<u32, Transition>,
    ) -> PreparedScene {
        self.finalize_bns_pick_ids();
        let ss_types = build_ss_types(
            self.has_any_ss,
            &self.ss_parts,
            self.residue_offset as usize,
        );
        let pick_map = self.build_pick_map();
        let per_residue_colors =
            self.has_any_colors.then_some(self.per_residue_colors);
        let tube_index_count = self.backbone_tube_inds.len() as u32;
        let ribbon_index_count = self.backbone_ribbon_inds.len() as u32;
        PreparedScene {
            backbone: BackboneMeshData {
                vertices: self.backbone_verts,
                tube_indices: bytemuck::cast_slice(&self.backbone_tube_inds)
                    .to_vec(),
                tube_index_count,
                ribbon_indices: bytemuck::cast_slice(
                    &self.backbone_ribbon_inds,
                )
                .to_vec(),
                ribbon_index_count,
                sheet_offsets: self.sheet_offsets,
                chain_ranges: self.chain_ranges,
            },
            sidechain_instances: self.sidechain_bytes,
            sidechain_instance_count: self.sidechain_count,
            bns: BallAndStickInstances {
                sphere_instances: self.bns_spheres,
                sphere_count: self.bns_sphere_count,
                capsule_instances: self.bns_capsules,
                capsule_count: self.bns_capsule_count,
            },
            na: NucleicAcidInstances {
                stem_instances: self.na_stem_bytes,
                stem_count: self.na_stem_count,
                ring_instances: self.na_ring_bytes,
                ring_count: self.na_ring_count,
            },
            backbone_chains: self.backbone_chains,
            na_chains: self.na_chains,
            sidechain: SidechainAtoms {
                atoms: self.sidechain_atoms,
                bonds: self.sidechain_bonds,
                backbone_bonds: self.backbone_sidechain_bonds,
            },
            ss_types,
            per_residue_colors,
            all_positions: self.all_positions,
            entity_transitions,
            entity_residue_ranges: self.entity_residue_ranges,
            non_protein_entities: self.non_protein,
            nucleic_acid_rings: self.na_rings,
            pick_map,
        }
    }
}

/// Patch f32 pick IDs in a raw byte buffer by adding `delta`.
fn patch_pick_id_buffer(
    buf: &mut [u8],
    delta: f32,
    stride: usize,
    offset: usize,
) {
    let mut pos = offset;
    while pos + 4 <= buf.len() {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&buf[pos..pos + 4]);
        let patched = f32::from_ne_bytes(bytes) + delta;
        buf[pos..pos + 4].copy_from_slice(&patched.to_ne_bytes());
        pos += stride;
    }
}

/// Concatenate per-entity cached meshes into a single `PreparedScene`.
pub fn concatenate_meshes(
    entity_meshes: &[(u32, &CachedEntityMesh)],
    entity_transitions: HashMap<u32, Transition>,
) -> PreparedScene {
    let mut acc = MeshAccumulator::default();
    for &(entity_id, mesh) in entity_meshes {
        acc.push_entity(entity_id, mesh);
    }
    acc.into_prepared_scene(entity_transitions)
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
