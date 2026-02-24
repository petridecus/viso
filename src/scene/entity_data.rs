use foldit_conv::{
    coords::{
        entity::{MoleculeEntity, NucleotideRing},
        RenderBackboneResidue,
    },
    secondary_structure::SSType,
};
use glam::Vec3;

/// Sidechain atom data for a single entity (local indices).
#[derive(Debug, Clone)]
pub struct SidechainAtom {
    /// Atom position in world space.
    pub position: Vec3,
    /// Whether this atom is hydrophobic.
    pub is_hydrophobic: bool,
    /// Local residue index within the entity.
    pub residue_idx: u32,
    /// PDB atom name (e.g. "CB", "CG").
    pub atom_name: String,
}

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
    /// Per-chain backbone residue data.
    pub backbone_residue_chains: Vec<Vec<RenderBackboneResidue>>,
    /// Sidechain atom data (local residue indices).
    pub sidechain_atoms: Vec<SidechainAtom>,
    /// Sidechain intra-residue bonds as (atom_idx, atom_idx) pairs.
    pub sidechain_bonds: Vec<(u32, u32)>,
    /// Backbone-to-sidechain bonds as (CA position, sidechain atom idx).
    pub backbone_sidechain_bonds: Vec<(Vec3, u32)>,
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
