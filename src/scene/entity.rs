use foldit_conv::{
    coords::{
        entity::{merge_entities, MoleculeEntity, MoleculeType},
        protein_only,
        render::extract_sequences,
        types::Coords,
        RenderCoords,
    },
    secondary_structure::SSType,
    types::assembly::protein_coords as assembly_protein_coords,
};

use super::{PerEntityData, SidechainAtom};
use crate::util::bond_topology::{get_residue_bonds, is_hydrophobic};

// ---------------------------------------------------------------------------
// Cached per-entity rendering data
// ---------------------------------------------------------------------------

/// Cached rendering data for a single entity's protein content.
#[derive(Debug, Clone)]
pub struct EntityRenderData {
    /// Extracted backbone and sidechain coordinates for rendering.
    pub render_coords: RenderCoords,
    /// Full amino acid sequence string.
    pub sequence: String,
    /// Per-chain sequences as (chain_id, sequence) pairs.
    pub chain_sequences: Vec<(u8, String)>,
}

// ---------------------------------------------------------------------------
// SceneEntity
// ---------------------------------------------------------------------------

/// A scene entity wrapping a single [`MoleculeEntity`] with rendering
/// metadata.
#[derive(Debug, Clone)]
pub struct SceneEntity {
    /// Core molecule data from foldit-conv.
    pub entity: MoleculeEntity,
    /// Whether this entity is visible in the scene.
    pub visible: bool,
    /// Human-readable name (defaults to entity label).
    pub name: String,
    /// Pre-computed secondary structure assignments.
    pub ss_override: Option<Vec<SSType>>,
    /// Cached per-residue energy scores from Rosetta (raw data for viz).
    pub per_residue_scores: Option<Vec<f64>>,
    pub(super) mesh_version: u64,
    pub(super) render_cache: Option<EntityRenderData>,
}

impl SceneEntity {
    /// Entity identifier (delegates to `entity.entity_id`).
    #[must_use]
    pub fn id(&self) -> u32 {
        self.entity.entity_id
    }

    /// Immutable access to the underlying molecule entity.
    #[must_use]
    pub fn entity(&self) -> &MoleculeEntity {
        &self.entity
    }

    /// Replace the underlying entity data. Preserves the scene-assigned
    /// entity ID. Bumps mesh version and invalidates the render cache.
    pub fn set_entity(&mut self, entity: MoleculeEntity) {
        let id = self.entity.entity_id; // preserve scene-assigned ID
        self.entity = entity;
        self.entity.entity_id = id;
        self.mesh_version += 1;
        self.render_cache = None;
    }

    /// Current mesh version (cache key for scene processor).
    #[must_use]
    pub fn mesh_version(&self) -> u64 {
        self.mesh_version
    }

    /// Derive (or return cached) render data from protein content.
    pub fn render_data(&mut self) -> Option<&EntityRenderData> {
        if self.render_cache.is_none() {
            self.render_cache = self.compute_render_data();
        }
        self.render_cache.as_ref()
    }

    /// Invalidate cached render data (call after coord changes).
    pub fn invalidate_render_cache(&mut self) {
        self.render_cache = None;
        self.mesh_version += 1;
    }

    /// Set per-residue energy scores (raw data cached for future viz).
    /// Does NOT bump mesh_version — the scene processor derives and caches
    /// colors from these scores, and the animation path reuses cached colors.
    pub fn set_per_residue_scores(&mut self, scores: Option<Vec<f64>>) {
        self.per_residue_scores = scores;
    }

    /// Get protein-only Coords for this entity.
    #[must_use]
    pub fn protein_coords(&self) -> Option<Coords> {
        if self.entity.molecule_type != MoleculeType::Protein {
            return None;
        }
        let coords =
            assembly_protein_coords(std::slice::from_ref(&self.entity));
        if coords.num_atoms == 0 {
            None
        } else {
            Some(coords)
        }
    }

    /// Get merged Coords for this entity.
    #[must_use]
    pub fn all_coords(&self) -> Option<Coords> {
        if self.entity.coords.num_atoms == 0 {
            return None;
        }
        let coords = merge_entities(std::slice::from_ref(&self.entity));
        if coords.num_atoms == 0 {
            None
        } else {
            Some(coords)
        }
    }

    /// Build [`PerEntityData`] from this entity's current state.
    pub fn to_per_entity_data(&mut self) -> Option<PerEntityData> {
        match self.entity.molecule_type {
            MoleculeType::Protein => self.per_entity_data_protein(),
            MoleculeType::DNA | MoleculeType::RNA => {
                self.per_entity_data_nucleic_acid()
            }
            _ => self.per_entity_data_non_protein(),
        }
    }

    /// Render data for a protein entity.
    fn per_entity_data_protein(&mut self) -> Option<PerEntityData> {
        let id = self.entity.entity_id;
        let mesh_version = self.mesh_version;
        let ss_override = self.ss_override.clone();
        let per_residue_scores = self.per_residue_scores.clone();

        let render_data = self.render_data()?;
        let rc = &render_data.render_coords;
        let res_count: u32 = rc
            .backbone_chains
            .iter()
            .map(|c| (c.len() / 3) as u32)
            .sum();

        let sc_atoms: Vec<SidechainAtom> = rc
            .sidechain_atoms
            .iter()
            .map(|a| SidechainAtom {
                position: a.position,
                is_hydrophobic: a.is_hydrophobic,
                residue_idx: a.residue_idx,
                atom_name: a.atom_name.clone(),
            })
            .collect();

        Some(PerEntityData {
            id,
            mesh_version,
            backbone_chains: rc.backbone_chains.clone(),
            backbone_chain_ids: rc.backbone_chain_ids.clone(),
            backbone_residue_chains: rc.backbone_residue_chains.clone(),
            sidechain_atoms: sc_atoms,
            sidechain_bonds: rc.sidechain_bonds.clone(),
            backbone_sidechain_bonds: rc.backbone_sidechain_bonds.clone(),
            ss_override,
            per_residue_scores,
            non_protein_entities: vec![],
            nucleic_acid_chains: vec![],
            nucleic_acid_rings: vec![],
            residue_count: res_count,
        })
    }

    /// Render data for a DNA/RNA entity.
    fn per_entity_data_nucleic_acid(&self) -> Option<PerEntityData> {
        let chains = self.entity.extract_p_atom_chains();
        let rings = self.entity.extract_base_rings();
        if chains.is_empty() && rings.is_empty() {
            return None;
        }
        Some(PerEntityData {
            id: self.entity.entity_id,
            mesh_version: self.mesh_version,
            backbone_chains: vec![],
            backbone_chain_ids: vec![],
            backbone_residue_chains: vec![],
            sidechain_atoms: vec![],
            sidechain_bonds: vec![],
            backbone_sidechain_bonds: vec![],
            ss_override: None,
            per_residue_scores: None,
            non_protein_entities: vec![self.entity.clone()],
            nucleic_acid_chains: chains,
            nucleic_acid_rings: rings,
            residue_count: 0,
        })
    }

    /// Render data for a non-protein, non-nucleic-acid entity.
    fn per_entity_data_non_protein(&self) -> Option<PerEntityData> {
        if self.entity.coords.num_atoms == 0 {
            return None;
        }
        Some(PerEntityData {
            id: self.entity.entity_id,
            mesh_version: self.mesh_version,
            backbone_chains: vec![],
            backbone_chain_ids: vec![],
            backbone_residue_chains: vec![],
            sidechain_atoms: vec![],
            sidechain_bonds: vec![],
            backbone_sidechain_bonds: vec![],
            ss_override: None,
            per_residue_scores: None,
            non_protein_entities: vec![self.entity.clone()],
            nucleic_acid_chains: vec![],
            nucleic_acid_rings: vec![],
            residue_count: 0,
        })
    }

    fn compute_render_data(&self) -> Option<EntityRenderData> {
        let protein_coords = self.protein_coords()?;
        let coords = protein_only(&protein_coords);
        if coords.num_atoms == 0 {
            return None;
        }

        let render_coords = RenderCoords::from_coords_with_topology(
            &coords,
            is_hydrophobic,
            |name| get_residue_bonds(name).map(<[(&str, &str)]>::to_vec),
        );

        let (sequence, chain_sequences) = extract_sequences(&coords);

        Some(EntityRenderData {
            render_coords,
            sequence,
            chain_sequences,
        })
    }
}
