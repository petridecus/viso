use foldit_conv::render::sidechain::SidechainAtoms;
use foldit_conv::secondary_structure::SSType;
use foldit_conv::types::entity::{MoleculeEntity, NucleotideRing};
use glam::Vec3;

/// All render data for a single entity, ready for the scene processor.
/// Indices are LOCAL (0-based within the entity).
#[derive(Debug, Clone)]
pub struct PerEntityData {
    /// Entity identifier.
    pub id: u32,
    /// Monotonic version counter for cache invalidation.
    pub mesh_version: u64,
    /// Protein backbone atom chains (N, CA, C triplets).
    pub backbone_chains: Vec<Vec<Vec3>>,
    /// Chain IDs for each backbone chain.
    pub backbone_chain_ids: Vec<u8>,
    /// Sidechain atom data with topology (positions, bonds, backbone bonds).
    pub sidechains: SidechainAtoms,
    /// Pre-computed secondary structure assignments.
    pub ss_override: Option<Vec<SSType>>,
    /// Per-residue energy scores for color derivation.
    pub per_residue_scores: Option<Vec<f64>>,
    /// Non-protein entities (ligands, ions, etc.).
    pub non_protein_entities: Vec<MoleculeEntity>,
    /// P-atom chains from DNA/RNA entities.
    pub nucleic_acid_chains: Vec<Vec<Vec3>>,
    /// Base ring geometry from DNA/RNA entities.
    pub nucleic_acid_rings: Vec<NucleotideRing>,
    /// Total residue count in this entity.
    pub residue_count: u32,
}

/// A contiguous range of residues belonging to a single entity.
///
/// Used to track which global residue indices map back to which entity
/// during animation and scene processing.
#[derive(Debug, Clone, Copy)]
pub struct EntityResidueRange {
    /// Entity identifier.
    pub entity_id: u32,
    /// First global residue index owned by this entity.
    pub start: u32,
    /// Number of residues in this entity.
    pub count: u32,
}

impl EntityResidueRange {
    /// One-past-the-end global residue index.
    #[must_use]
    pub fn end(&self) -> u32 {
        self.start + self.count
    }

    /// Whether the given global residue index falls within this range.
    #[must_use]
    pub fn contains(&self, residue_idx: u32) -> bool {
        residue_idx >= self.start && residue_idx < self.end()
    }
}
