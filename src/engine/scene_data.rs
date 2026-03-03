//! Entity data types, bond topology tables, and scene aggregation functions.
//!
//! These are pure data definitions and static lookup tables used by the
//! engine, entity store, and renderer pipeline. They don't belong to any
//! single subsystem.

use foldit_conv::render::sidechain::{SidechainAtomData, SidechainAtoms};
use foldit_conv::secondary_structure::SSType;
use foldit_conv::types::assembly::protein_coords as assembly_protein_coords;
use foldit_conv::types::coords::Coords;
use foldit_conv::types::entity::{
    MoleculeEntity, MoleculeType, NucleotideRing,
};
use glam::Vec3;

use crate::animation::transition::Transition;
use crate::animation::SidechainAnimPositions;

// ---------------------------------------------------------------------------
// Bounding sphere
// ---------------------------------------------------------------------------

/// Compute (centroid, radius) for a molecule entity's atom positions.
/// Returns `(Vec3::ZERO, 0.0)` when the entity has no atoms.
fn compute_bounding_sphere(entity: &MoleculeEntity) -> (Vec3, f32) {
    let atoms = &entity.coords.atoms;
    if atoms.is_empty() {
        return (Vec3::ZERO, 0.0);
    }
    let n = atoms.len() as f32;
    let centroid = atoms
        .iter()
        .fold(Vec3::ZERO, |acc, a| acc + Vec3::new(a.x, a.y, a.z))
        / n;
    let radius = atoms
        .iter()
        .map(|a| (Vec3::new(a.x, a.y, a.z) - centroid).length())
        .fold(0.0f32, f32::max);
    (centroid, radius)
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
    /// Pre-computed secondary structure assignments.
    pub ss_override: Option<Vec<SSType>>,
    /// Cached per-residue energy scores from Rosetta (raw data for viz).
    pub per_residue_scores: Option<Vec<f64>>,
    pub(crate) mesh_version: u64,
    /// Cached centroid of all atom positions.
    pub(crate) cached_centroid: Vec3,
    /// Cached bounding sphere radius around `cached_centroid`.
    pub(crate) cached_bounding_radius: f32,
}

impl SceneEntity {
    /// Create a new `SceneEntity` with pre-computed bounding sphere.
    pub(crate) fn new(entity: MoleculeEntity) -> Self {
        let (centroid, radius) = compute_bounding_sphere(&entity);
        Self {
            entity,
            visible: true,
            ss_override: None,
            per_residue_scores: None,
            mesh_version: 0,
            cached_centroid: centroid,
            cached_bounding_radius: radius,
        }
    }

    /// Recompute the cached bounding sphere from current atom positions.
    pub(crate) fn recompute_bounds(&mut self) {
        let (centroid, radius) = compute_bounding_sphere(&self.entity);
        self.cached_centroid = centroid;
        self.cached_bounding_radius = radius;
    }

    /// Entity identifier (delegates to `entity.entity_id`).
    #[must_use]
    pub fn id(&self) -> u32 {
        self.entity.entity_id
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

    /// Invalidate (bump mesh version after coord changes).
    pub fn invalidate_render_cache(&mut self) {
        self.mesh_version += 1;
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
            backbone_chains: backbone.into_chain_vecs(),
            sidechains,
            ss_override: self.ss_override.clone(),
            per_residue_colors: None,
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
            sidechains: SidechainAtoms::default(),
            ss_override: None,
            per_residue_colors: None,
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
            sidechains: SidechainAtoms::default(),
            ss_override: None,
            per_residue_colors: None,
            non_protein_entities: vec![self.entity.clone()],
            nucleic_acid_chains: vec![],
            nucleic_acid_rings: vec![],
            residue_count: 0,
        })
    }
}

// ---------------------------------------------------------------------------
// Per-entity data types
// ---------------------------------------------------------------------------

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
    /// Sidechain atom data with topology (positions, bonds, backbone bonds).
    pub sidechains: SidechainAtoms,
    /// Pre-computed secondary structure assignments.
    pub ss_override: Option<Vec<SSType>>,
    /// Pre-computed per-residue colors (derived from scores on main thread).
    pub per_residue_colors: Option<Vec<[f32; 3]>>,
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
    #[allow(dead_code)]
    pub fn contains(&self, residue_idx: u32) -> bool {
        residue_idx >= self.start && residue_idx < self.end()
    }
}

// ---------------------------------------------------------------------------
// Entity aggregation functions
// ---------------------------------------------------------------------------

/// Concatenate per-entity sidechain data into a single [`SidechainAtoms`].
///
/// Replicates the index-offsetting logic from the background thread's
/// `MeshAccumulator::push_passthrough`, allowing the main thread to
/// compute sidechain topology before sending work to the background.
#[must_use]
pub fn concatenate_sidechain_atoms(
    entities: &[PerEntityData],
    ranges: &[EntityResidueRange],
) -> SidechainAtoms {
    let mut atoms = Vec::new();
    let mut bonds = Vec::new();
    let mut backbone_bonds = Vec::new();

    for (e, range) in entities.iter().zip(ranges) {
        let sc_atom_offset = atoms.len() as u32;
        for atom in &e.sidechains.atoms {
            atoms.push(SidechainAtomData {
                position: atom.position,
                residue_idx: atom.residue_idx + range.start,
                atom_name: atom.atom_name.clone(),
                is_hydrophobic: atom.is_hydrophobic,
            });
        }
        for &(a, b) in &e.sidechains.bonds {
            bonds.push((a + sc_atom_offset, b + sc_atom_offset));
        }
        for &(ca_pos, cb_idx) in &e.sidechains.backbone_bonds {
            backbone_bonds.push((ca_pos, cb_idx + sc_atom_offset));
        }
    }

    SidechainAtoms {
        atoms,
        bonds,
        backbone_bonds,
    }
}

/// Compute concatenated secondary structure types from per-entity data.
///
/// Uses each entity's `ss_override` where available, falling back to DSSP
/// detection from backbone chains. Returns `None` if no entity has
/// backbone chains (i.e., zero residues).
#[must_use]
pub fn concatenate_ss_types(
    entities: &[PerEntityData],
    ranges: &[EntityResidueRange],
) -> Vec<SSType> {
    use foldit_conv::secondary_structure::auto::detect as detect_ss;

    let total: usize = ranges.iter().map(|r| r.count as usize).sum();
    if total == 0 {
        return Vec::new();
    }

    let mut ss = vec![SSType::Coil; total];

    for (e, range) in entities.iter().zip(ranges) {
        let start = range.start as usize;
        let end = (start + range.count as usize).min(total);

        if let Some(ref overrides) = e.ss_override {
            let n = (end - start).min(overrides.len());
            ss[start..start + n].copy_from_slice(&overrides[..n]);
        } else {
            // DSSP fallback per-chain
            let mut offset = start;
            for chain in &e.backbone_chains {
                let ca_positions =
                    foldit_conv::render::backbone::ca_positions_from_chains(
                        std::slice::from_ref(chain),
                    );
                let chain_ss = detect_ss(&ca_positions);
                let n = chain_ss.len().min(end.saturating_sub(offset));
                ss[offset..offset + n].copy_from_slice(&chain_ss[..n]);
                offset += n;
            }
        }
    }

    ss
}

/// Compute entity residue ranges from per-entity data.
///
/// Produces the same ranges that `MeshAccumulator` would when concatenating
/// entities, but without requiring a background thread round-trip.
#[must_use]
pub fn compute_entity_residue_ranges(
    entities: &[PerEntityData],
) -> Vec<EntityResidueRange> {
    let mut ranges = Vec::with_capacity(entities.len());
    let mut offset = 0u32;
    for e in entities {
        ranges.push(EntityResidueRange {
            entity_id: e.id,
            start: offset,
            count: e.residue_count,
        });
        offset += e.residue_count;
    }
    ranges
}

/// Extract sidechain animation positions for a single entity.
///
/// Returns `None` if the entity has no sidechain atoms in its range,
/// or if `transition` is `None` (entity has no active transition).
#[must_use]
pub fn extract_entity_sidechain(
    all_positions: &[Vec3],
    all_residue_indices: &[u32],
    all_ca: &[Vec3],
    range: &EntityResidueRange,
    transition: Option<&Transition>,
) -> Option<SidechainAnimPositions> {
    let transition = transition?;
    let res_start = range.start as usize;
    let res_end = range.end() as usize;
    let collapse_to_ca = transition.allows_size_change;

    let mut start = Vec::new();
    let mut target = Vec::new();

    for (i, &res_idx) in all_residue_indices.iter().enumerate() {
        let r = res_idx as usize;
        if !(res_start..res_end).contains(&r) {
            continue;
        }
        let Some(&pos) = all_positions.get(i) else {
            continue;
        };
        target.push(pos);
        if collapse_to_ca {
            start.push(all_ca.get(r).copied().unwrap_or(Vec3::ZERO));
        } else {
            start.push(pos);
        }
    }

    if target.is_empty() {
        return None;
    }

    Some(SidechainAnimPositions { start, target })
}

/// Extract backbone-sidechain bonds that belong to a single entity.
///
/// Filters `all_backbone_bonds` by looking up each bond's CB atom
/// index in `sidechain_residue_indices` and keeping only bonds whose
/// residue falls within the entity's range.
#[must_use]
pub fn extract_entity_backbone_bonds(
    all_backbone_bonds: &[(Vec3, u32)],
    sidechain_residue_indices: &[u32],
    range: &EntityResidueRange,
) -> Vec<(Vec3, u32)> {
    let res_start = range.start as usize;
    let res_end = range.end() as usize;
    all_backbone_bonds
        .iter()
        .filter(|(_, cb_idx)| {
            let r = sidechain_residue_indices
                .get(*cb_idx as usize)
                .copied()
                .unwrap_or(0) as usize;
            (res_start..res_end).contains(&r)
        })
        .copied()
        .collect()
}

// ---------------------------------------------------------------------------
// Bond topology (static tables)
// ---------------------------------------------------------------------------

/// Get the bond pairs for a residue type (sidechain internal bonds only).
/// Returns None for unknown residue types.
pub fn get_residue_bonds(
    residue_name: &str,
) -> Option<&'static [(&'static str, &'static str)]> {
    match residue_name.to_uppercase().as_str() {
        "ALA" => Some(ALANINE_BONDS),
        "ARG" => Some(ARGININE_BONDS),
        "ASN" => Some(ASPARAGINE_BONDS),
        "ASP" => Some(ASPARTATE_BONDS),
        "CYS" => Some(CYSTEINE_BONDS),
        "GLN" => Some(GLUTAMINE_BONDS),
        "GLU" => Some(GLUTAMATE_BONDS),
        "GLY" => Some(GLYCINE_BONDS),
        "HIS" => Some(HISTIDINE_BONDS),
        "ILE" => Some(ISOLEUCINE_BONDS),
        "LEU" => Some(LEUCINE_BONDS),
        "LYS" => Some(LYSINE_BONDS),
        "MET" => Some(METHIONINE_BONDS),
        "PHE" => Some(PHENYLALANINE_BONDS),
        "PRO" => Some(PROLINE_BONDS),
        "SER" => Some(SERINE_BONDS),
        "THR" => Some(THREONINE_BONDS),
        "TRP" => Some(TRYPTOPHAN_BONDS),
        "TYR" => Some(TYROSINE_BONDS),
        "VAL" => Some(VALINE_BONDS),
        _ => None,
    }
}

/// Check if an amino acid is hydrophobic.
pub fn is_hydrophobic(residue_name: &str) -> bool {
    matches!(
        residue_name.to_uppercase().as_str(),
        "ALA" | "VAL" | "ILE" | "LEU" | "MET" | "PHE" | "TRP" | "PRO" | "GLY"
    )
}

// Glycine has no sidechain atoms (only backbone N, CA, C, O)
const GLYCINE_BONDS: &[(&str, &str)] = &[];

// Alanine: CA-CB (CB is the only sidechain atom)
const ALANINE_BONDS: &[(&str, &str)] = &[];

// Valine: CA-CB-CG1, CB-CG2
const VALINE_BONDS: &[(&str, &str)] = &[("CB", "CG1"), ("CB", "CG2")];

// Leucine: CA-CB-CG-CD1, CG-CD2
const LEUCINE_BONDS: &[(&str, &str)] =
    &[("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")];

// Isoleucine: CA-CB-CG1-CD1, CB-CG2
const ISOLEUCINE_BONDS: &[(&str, &str)] =
    &[("CB", "CG1"), ("CG1", "CD1"), ("CB", "CG2")];

// Proline: CA-CB-CG-CD-N (ring)
const PROLINE_BONDS: &[(&str, &str)] = &[("CB", "CG"), ("CG", "CD")];

// Serine: CA-CB-OG
const SERINE_BONDS: &[(&str, &str)] = &[("CB", "OG")];

// Threonine: CA-CB-OG1, CB-CG2
const THREONINE_BONDS: &[(&str, &str)] = &[("CB", "OG1"), ("CB", "CG2")];

// Cysteine: CA-CB-SG
const CYSTEINE_BONDS: &[(&str, &str)] = &[("CB", "SG")];

// Methionine: CA-CB-CG-SD-CE
const METHIONINE_BONDS: &[(&str, &str)] =
    &[("CB", "CG"), ("CG", "SD"), ("SD", "CE")];

// Asparagine: CA-CB-CG-OD1, CG-ND2
const ASPARAGINE_BONDS: &[(&str, &str)] =
    &[("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")];

// Aspartate: CA-CB-CG-OD1, CG-OD2
const ASPARTATE_BONDS: &[(&str, &str)] =
    &[("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")];

// Glutamine: CA-CB-CG-CD-OE1, CD-NE2
const GLUTAMINE_BONDS: &[(&str, &str)] =
    &[("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")];

// Glutamate: CA-CB-CG-CD-OE1, CD-OE2
const GLUTAMATE_BONDS: &[(&str, &str)] =
    &[("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")];

// Lysine: CA-CB-CG-CD-CE-NZ
const LYSINE_BONDS: &[(&str, &str)] =
    &[("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")];

// Arginine: CA-CB-CG-CD-NE-CZ-NH1, CZ-NH2
const ARGININE_BONDS: &[(&str, &str)] = &[
    ("CB", "CG"),
    ("CG", "CD"),
    ("CD", "NE"),
    ("NE", "CZ"),
    ("CZ", "NH1"),
    ("CZ", "NH2"),
];

// Histidine: CA-CB-CG-ND1-CE1-NE2-CD2-CG (imidazole ring)
const HISTIDINE_BONDS: &[(&str, &str)] = &[
    ("CB", "CG"),
    ("CG", "ND1"),
    ("ND1", "CE1"),
    ("CE1", "NE2"),
    ("NE2", "CD2"),
    ("CD2", "CG"),
];

// Phenylalanine: CA-CB-CG benzene ring
const PHENYLALANINE_BONDS: &[(&str, &str)] = &[
    ("CB", "CG"),
    ("CG", "CD1"),
    ("CD1", "CE1"),
    ("CE1", "CZ"),
    ("CZ", "CE2"),
    ("CE2", "CD2"),
    ("CD2", "CG"),
];

// Tyrosine: Like Phe + OH on CZ
const TYROSINE_BONDS: &[(&str, &str)] = &[
    ("CB", "CG"),
    ("CG", "CD1"),
    ("CD1", "CE1"),
    ("CE1", "CZ"),
    ("CZ", "OH"),
    ("CZ", "CE2"),
    ("CE2", "CD2"),
    ("CD2", "CG"),
];

// Tryptophan: indole ring system
const TRYPTOPHAN_BONDS: &[(&str, &str)] = &[
    ("CB", "CG"),
    ("CG", "CD1"),
    ("CD1", "NE1"),
    ("NE1", "CE2"),
    ("CE2", "CD2"),
    ("CD2", "CG"),
    // Benzene ring of indole
    ("CE2", "CZ2"),
    ("CZ2", "CH2"),
    ("CH2", "CZ3"),
    ("CZ3", "CE3"),
    ("CE3", "CD2"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_residue_bonds() {
        assert!(get_residue_bonds("ALA").is_some());
        assert!(get_residue_bonds("ala").is_some()); // case insensitive
        assert!(get_residue_bonds("GLY").is_some());
        assert!(get_residue_bonds("TRP").is_some());
        assert!(get_residue_bonds("XXX").is_none());
    }

    #[test]
    fn test_phenylalanine_ring() {
        assert!(matches!(get_residue_bonds("PHE"), Some(b) if b.len() == 7));
    }
}
