//! Entity data types, bond topology tables, and scene aggregation functions.
//!
//! These are pure data definitions and static lookup tables used by the
//! engine, entity store, and renderer pipeline. They don't belong to any
//! single subsystem.

use glam::Vec3;
use molex::{
    MoleculeEntity, MoleculeType, NucleotideRing, ProteinResidue, SSType,
};

use crate::animation::transition::Transition;
use crate::animation::SidechainAnimPositions;
use crate::options::DrawingMode;
use crate::renderer::geometry::backbone::sheet_fit;

// ---------------------------------------------------------------------------
// Flat sidechain types (scene-level, concatenated across residues/entities)
// ---------------------------------------------------------------------------

/// Data for a single sidechain atom in the flat scene-level layout.
#[derive(Debug, Clone)]
pub(crate) struct SidechainAtomData {
    /// 3D position of this atom.
    pub(crate) position: Vec3,
    /// Global residue index (across the entire scene).
    pub(crate) residue_idx: u32,
    /// PDB atom name (e.g. "CB", "CG").
    pub(crate) atom_name: String,
    /// Whether the parent residue is hydrophobic.
    pub(crate) is_hydrophobic: bool,
}

/// Flat sidechain data concatenated across all residues in an entity or scene.
///
/// Molex provides per-residue [`molex::Sidechain`] with local indices; this
/// type flattens those into global indices for the GPU pipeline.
#[derive(Debug, Clone, Default)]
pub(crate) struct SidechainAtoms {
    /// Per-atom data (all residues concatenated).
    pub(crate) atoms: Vec<SidechainAtomData>,
    /// Intra-sidechain bonds as `(atom_idx_a, atom_idx_b)` with global
    /// indices.
    pub(crate) bonds: Vec<(u32, u32)>,
    /// Backbone-to-sidechain bonds as `(CA_position, sidechain_atom_idx)`.
    pub(crate) backbone_bonds: Vec<(Vec3, u32)>,
}

impl SidechainAtoms {
    /// All sidechain atom positions.
    #[must_use]
    pub(crate) fn positions(&self) -> Vec<Vec3> {
        self.atoms.iter().map(|a| a.position).collect()
    }

    /// Per-atom hydrophobicity flags.
    #[must_use]
    pub(crate) fn hydrophobicity(&self) -> Vec<bool> {
        self.atoms.iter().map(|a| a.is_hydrophobic).collect()
    }

    /// Per-atom residue indices.
    #[must_use]
    pub(crate) fn residue_indices(&self) -> Vec<u32> {
        self.atoms.iter().map(|a| a.residue_idx).collect()
    }

    /// Per-atom PDB names.
    #[must_use]
    pub(crate) fn atom_names(&self) -> Vec<String> {
        self.atoms.iter().map(|a| a.atom_name.clone()).collect()
    }

    /// Build flat sidechain data from per-residue protein residues.
    ///
    /// `residue_offset` is the global residue index of the first residue
    /// (used when concatenating multiple entities).
    #[must_use]
    pub(crate) fn from_protein_residues(
        residues: &[ProteinResidue],
        residue_offset: u32,
    ) -> Self {
        let mut atoms = Vec::new();
        let mut bonds = Vec::new();
        let mut backbone_bonds = Vec::new();

        for (i, res) in residues.iter().enumerate() {
            let global_res_idx = residue_offset + i as u32;
            let atom_offset = atoms.len() as u32;

            // Flatten sidechain atoms with global residue index
            for atom in &res.sidechain.atoms {
                let name = std::str::from_utf8(&atom.name)
                    .unwrap_or("")
                    .trim()
                    .to_owned();
                atoms.push(SidechainAtomData {
                    position: atom.position,
                    residue_idx: global_res_idx,
                    atom_name: name,
                    is_hydrophobic: res.sidechain.is_hydrophobic,
                });
            }

            // Remap local bond indices to global
            for &(a, b) in &res.sidechain.bonds {
                bonds.push((atom_offset + a as u32, atom_offset + b as u32));
            }

            // CA → first sidechain atom (CB) bond
            if !res.sidechain.atoms.is_empty() {
                backbone_bonds.push((res.backbone.ca, atom_offset));
            }
        }

        Self {
            atoms,
            bonds,
            backbone_bonds,
        }
    }
}

// ---------------------------------------------------------------------------
// Bounding sphere
// ---------------------------------------------------------------------------

/// Compute (centroid, radius) for a molecule entity's atom positions.
/// Returns `(Vec3::ZERO, 0.0)` when the entity has no atoms.
fn compute_bounding_sphere(entity: &MoleculeEntity) -> (Vec3, f32) {
    let atoms = entity.atom_set();
    if atoms.is_empty() {
        return (Vec3::ZERO, 0.0);
    }
    let n = atoms.len() as f32;
    let centroid = atoms.iter().fold(Vec3::ZERO, |acc, a| acc + a.position) / n;
    let radius = atoms
        .iter()
        .map(|a| (a.position - centroid).length())
        .fold(0.0f32, f32::max);
    (centroid, radius)
}

// ---------------------------------------------------------------------------
// SceneEntity
// ---------------------------------------------------------------------------

/// A scene entity wrapping a single [`MoleculeEntity`] with rendering
/// metadata.
#[derive(Debug, Clone)]
pub(crate) struct SceneEntity {
    /// Core molecule data from molex.
    pub(crate) entity: MoleculeEntity,
    /// Whether this entity is visible in the scene.
    pub(crate) visible: bool,
    /// Pre-computed secondary structure assignments.
    pub(crate) ss_override: Option<Vec<SSType>>,
    /// Cached per-residue energy scores from Rosetta (raw data for viz).
    pub(crate) per_residue_scores: Option<Vec<f64>>,
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

    /// Entity identifier.
    #[must_use]
    pub fn id(&self) -> u32 {
        *self.entity.id()
    }

    /// Whether this entity is a protein.
    #[must_use]
    pub fn is_protein(&self) -> bool {
        self.entity.molecule_type() == MoleculeType::Protein
    }

    /// Whether this entity is a nucleic acid (DNA or RNA).
    #[must_use]
    pub fn is_nucleic_acid(&self) -> bool {
        matches!(
            self.entity.molecule_type(),
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

    /// Build [`PerEntityData`] from this entity's current state.
    ///
    /// `mode` controls the drawing mode: Cartoon uses the normal backbone
    /// mesh pipeline, while Stick/BallAndStick routes the entity through
    /// the ball-and-stick renderer instead.
    #[must_use]
    pub fn to_per_entity_data(
        &self,
        mode: DrawingMode,
    ) -> Option<PerEntityData> {
        // For Stick/BnS on a normally-cartoon entity, skip backbone and
        // route through the BnS renderer instead.
        if mode != DrawingMode::Cartoon {
            return self.per_entity_data_as_bns(mode);
        }
        match self.entity.molecule_type() {
            MoleculeType::Protein => self.per_entity_data_protein(),
            MoleculeType::DNA | MoleculeType::RNA => {
                self.per_entity_data_nucleic_acid()
            }
            _ => self.per_entity_data_non_protein(),
        }
    }

    /// Render data for a protein entity.
    fn per_entity_data_protein(&self) -> Option<PerEntityData> {
        let protein = self.entity.as_protein()?;
        let backbone_chains = protein.to_interleaved_segments();
        if backbone_chains.is_empty() {
            return None;
        }
        let res_count =
            backbone_chains.iter().map(|c| c.len() / 3).sum::<usize>() as u32;
        let protein_residues = protein
            .to_protein_residues(is_hydrophobic, |name| {
                get_residue_bonds(name).map(<[(&str, &str)]>::to_vec)
            });
        let sidechains =
            SidechainAtoms::from_protein_residues(&protein_residues, 0);

        let ss = self
            .ss_override
            .clone()
            .unwrap_or_else(|| protein.detect_ss());

        // Fit one plane per β-sheet from the H-bond topology so every
        // strand in a given sheet shares the same face orientation.
        // `to_backbone()` and `to_interleaved_segments()` apply the
        // same "has complete backbone atoms" filter, so residue
        // indices align with `ss` and with the backbone_chains
        // counting used downstream.
        let residue_backbones = protein.to_backbone();
        let hbonds = molex::analysis::detect_hbonds(&residue_backbones);
        let ca_positions: Vec<Vec3> =
            residue_backbones.iter().map(|r| r.ca).collect();
        let sheet_plane_normals =
            sheet_fit::compute_sheet_plane_normals(&hbonds, &ss, &ca_positions);

        Some(PerEntityData {
            id: *self.entity.id(),
            mesh_version: self.mesh_version,
            drawing_mode: DrawingMode::Cartoon,
            backbone_chains,

            sidechains,
            ss_override: Some(ss),
            sheet_plane_normals,
            per_residue_colors: None,
            non_protein_entities: vec![],
            nucleic_acid_chains: vec![],
            nucleic_acid_rings: vec![],
            residue_count: res_count,
        })
    }

    /// Render data for a DNA/RNA entity.
    fn per_entity_data_nucleic_acid(&self) -> Option<PerEntityData> {
        let na = self.entity.as_nucleic_acid()?;
        let chains = na.extract_p_atom_segments();
        let rings = na.extract_base_rings();
        if chains.is_empty() && rings.is_empty() {
            return None;
        }
        Some(PerEntityData {
            id: *self.entity.id(),
            mesh_version: self.mesh_version,
            drawing_mode: DrawingMode::Cartoon,
            backbone_chains: vec![],

            sidechains: SidechainAtoms::default(),
            ss_override: None,
            sheet_plane_normals: Vec::new(),
            per_residue_colors: None,
            non_protein_entities: vec![self.entity.clone()],
            nucleic_acid_chains: chains,
            nucleic_acid_rings: rings,
            residue_count: 0,
        })
    }

    /// Render data for a non-protein, non-nucleic-acid entity.
    fn per_entity_data_non_protein(&self) -> Option<PerEntityData> {
        if self.entity.atom_count() == 0 {
            return None;
        }
        Some(PerEntityData {
            id: *self.entity.id(),
            mesh_version: self.mesh_version,
            drawing_mode: DrawingMode::BallAndStick,
            backbone_chains: vec![],

            sidechains: SidechainAtoms::default(),
            ss_override: None,
            sheet_plane_normals: Vec::new(),
            per_residue_colors: None,
            non_protein_entities: vec![self.entity.clone()],
            nucleic_acid_chains: vec![],
            nucleic_acid_rings: vec![],
            residue_count: 0,
        })
    }

    /// Render data for a polymer entity in Stick or `BallAndStick` mode.
    ///
    /// For proteins: keeps `backbone_chains` populated so that
    /// `prepare_scene_metadata` can compute per-residue colors (chain,
    /// SS, score, etc.) exactly as it does for Cartoon mode. The
    /// backbone mesh generation is skipped later by
    /// `generate_entity_mesh` when it sees a non-Cartoon drawing mode.
    /// The entity is also placed in `non_protein_entities` so the BnS
    /// renderer processes it.
    fn per_entity_data_as_bns(
        &self,
        mode: DrawingMode,
    ) -> Option<PerEntityData> {
        if self.entity.atom_count() == 0 {
            return None;
        }
        // For proteins, extract backbone so colors can be computed.
        if let Some(protein) = self.entity.as_protein() {
            let backbone_chains = protein.to_interleaved_segments();
            let res_count = if backbone_chains.is_empty() {
                0
            } else {
                backbone_chains.iter().map(|c| c.len() / 3).sum::<usize>()
                    as u32
            };
            return Some(PerEntityData {
                id: *self.entity.id(),
                mesh_version: self.mesh_version,
                drawing_mode: mode,
                backbone_chains,

                sidechains: SidechainAtoms::default(),
                ss_override: self.ss_override.clone(),
                sheet_plane_normals: Vec::new(),
                per_residue_colors: None,
                non_protein_entities: vec![self.entity.clone()],
                nucleic_acid_chains: vec![],
                nucleic_acid_rings: vec![],
                residue_count: res_count,
            });
        }
        Some(PerEntityData {
            id: *self.entity.id(),
            mesh_version: self.mesh_version,
            drawing_mode: mode,
            backbone_chains: vec![],

            sidechains: SidechainAtoms::default(),
            ss_override: None,
            sheet_plane_normals: Vec::new(),
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
pub(crate) struct PerEntityData {
    /// Entity identifier.
    pub(crate) id: u32,
    /// Monotonic version counter for cache invalidation.
    pub(crate) mesh_version: u64,
    /// Resolved drawing mode for this entity.
    pub(crate) drawing_mode: DrawingMode,
    /// Protein backbone atom chains (N, CA, C triplets).
    pub(crate) backbone_chains: Vec<Vec<Vec3>>,
    /// Sidechain atom data with topology (positions, bonds, backbone bonds).
    pub(crate) sidechains: SidechainAtoms,
    /// Pre-computed secondary structure assignments.
    pub(crate) ss_override: Option<Vec<SSType>>,
    /// Sparse per-residue β-sheet plane normals, fitted from the
    /// backbone H-bond topology. One `(residue_idx, normal)` pair per
    /// residue that belongs to a multi-strand sheet whose plane was
    /// successfully fitted; residues not in the list fall back to the
    /// local peptide-plane computation. Empty when no sheets were
    /// found or during animation frames (see `compute_sheet_geometry`).
    pub(crate) sheet_plane_normals: Vec<(u32, Vec3)>,
    /// Pre-computed per-residue colors (derived from scores on main thread).
    pub(crate) per_residue_colors: Option<Vec<[f32; 3]>>,
    /// Non-protein entities (ligands, ions, etc.).
    pub(crate) non_protein_entities: Vec<MoleculeEntity>,
    /// P-atom chains from DNA/RNA entities.
    pub(crate) nucleic_acid_chains: Vec<Vec<Vec3>>,
    /// Base ring geometry from DNA/RNA entities.
    pub(crate) nucleic_acid_rings: Vec<NucleotideRing>,
    /// Total residue count in this entity.
    pub(crate) residue_count: u32,
}

/// A contiguous range of residues belonging to a single entity.
///
/// Used to track which global residue indices map back to which entity
/// during animation and scene processing.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EntityResidueRange {
    /// Entity identifier.
    pub(crate) entity_id: u32,
    /// First global residue index owned by this entity.
    pub(crate) start: u32,
    /// Number of residues in this entity.
    pub(crate) count: u32,
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
    let total: usize = ranges.iter().map(|r| r.count as usize).sum();
    if total == 0 {
        return Vec::new();
    }

    let mut ss = vec![SSType::Coil; total];

    for (e, range) in entities.iter().zip(ranges) {
        let start = range.start as usize;
        let end = (start + range.count as usize).min(total);

        if let Some(ref types) = e.ss_override {
            let n = (end - start).min(types.len());
            ss[start..start + n].copy_from_slice(&types[..n]);
        }
    }

    ss
}

/// Estimate carbonyl O position from CA(i), C(i), N(i+1).
pub(crate) fn estimate_carbonyl_o(next_n: Vec3, ca: Vec3, c: Vec3) -> Vec3 {
    let c_to_n = (next_n - c).normalize_or_zero();
    let c_to_ca = (ca - c).normalize_or_zero();
    let plane_normal = c_to_n.cross(c_to_ca);
    if plane_normal.length_squared() < 1e-6 {
        // Degenerate — place O arbitrarily
        return c + c_to_ca * 1.231;
    }
    let rot = glam::Quat::from_axis_angle(
        plane_normal.normalize(),
        -2.1031, // ~120.5° C=O angle
    );
    let c_to_o = rot * c_to_n;
    c + c_to_o * 1.231
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
