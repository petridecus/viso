use foldit_conv::render::sidechain::SidechainAtoms;
use foldit_conv::secondary_structure::SSType;
use foldit_conv::types::assembly::protein_coords as assembly_protein_coords;
use foldit_conv::types::coords::Coords;
use foldit_conv::types::entity::{
    merge_entities, MoleculeEntity, MoleculeType,
};

use super::PerEntityData;
use crate::util::bond_topology::{get_residue_bonds, is_hydrophobic};

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

    /// Whether this entity is a protein.
    #[must_use]
    pub fn is_protein(&self) -> bool {
        self.entity.molecule_type == MoleculeType::Protein
    }

    /// Whether this entity is a nucleic acid (DNA or RNA).
    #[must_use]
    pub fn is_nucleic_acid(&self) -> bool {
        matches!(
            self.entity.molecule_type,
            MoleculeType::DNA | MoleculeType::RNA
        )
    }

    /// Whether this entity is a ligand (not protein, DNA, or RNA).
    #[must_use]
    pub fn is_ligand(&self) -> bool {
        !self.is_protein() && !self.is_nucleic_acid()
    }

    /// Replace the underlying entity data. Preserves the scene-assigned
    /// entity ID. Bumps mesh version.
    pub fn set_entity(&mut self, entity: MoleculeEntity) {
        let id = self.entity.entity_id; // preserve scene-assigned ID
        self.entity = entity;
        self.entity.entity_id = id;
        self.mesh_version += 1;
    }

    /// Current mesh version (cache key for scene processor).
    #[must_use]
    pub fn mesh_version(&self) -> u64 {
        self.mesh_version
    }

    /// Invalidate (bump mesh version after coord changes).
    pub fn invalidate_render_cache(&mut self) {
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
    #[must_use]
    pub fn to_per_entity_data(&self) -> Option<PerEntityData> {
        match self.entity.molecule_type {
            MoleculeType::Protein => self.per_entity_data_protein(),
            MoleculeType::DNA | MoleculeType::RNA => {
                self.per_entity_data_nucleic_acid()
            }
            _ => self.per_entity_data_non_protein(),
        }
    }

    /// Render data for a protein entity.
    fn per_entity_data_protein(&self) -> Option<PerEntityData> {
        let backbone = self.entity.extract_backbone();
        if backbone.chains.is_empty() {
            return None;
        }
        let res_count = backbone.residue_count() as u32;
        let sidechains =
            self.entity.extract_sidechains(is_hydrophobic, |name| {
                get_residue_bonds(name).map(<[(&str, &str)]>::to_vec)
            });

        Some(PerEntityData {
            id: self.entity.entity_id,
            mesh_version: self.mesh_version,
            backbone_chain_ids: backbone.chain_ids.clone(),
            backbone_chains: backbone.into_chain_vecs(),
            sidechains,
            ss_override: self.ss_override.clone(),
            per_residue_scores: self.per_residue_scores.clone(),
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
            sidechains: SidechainAtoms::default(),
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
            sidechains: SidechainAtoms::default(),
            ss_override: None,
            per_residue_scores: None,
            non_protein_entities: vec![self.entity.clone()],
            nucleic_acid_chains: vec![],
            nucleic_acid_rings: vec![],
            residue_count: 0,
        })
    }
}
