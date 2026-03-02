//! Authoritative scene: flat entity storage, focus cycling, per-entity
//! render data.
//!
//! Everything is a
//! [`MoleculeEntity`]. Each entity
//! is wrapped in a [`SceneEntity`] that pairs the core data with rendering
//! metadata (visibility, name, SS override, score cache, mesh version).

use foldit_conv::render::sidechain::{SidechainAtomData, SidechainAtoms};
use foldit_conv::secondary_structure::SSType;
use foldit_conv::types::assembly::{
    protein_coords as assembly_protein_coords, update_protein_entities,
};
use foldit_conv::types::coords::Coords;
use foldit_conv::types::entity::{
    MoleculeEntity, MoleculeType, NucleotideRing,
};
use glam::Vec3;

use crate::animation::transition::Transition;
use crate::animation::SidechainAnimPositions;

// ---------------------------------------------------------------------------
// Focus
// ---------------------------------------------------------------------------

/// Focus state for tab cycling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Focus {
    /// All entities.
    #[default]
    Session,
    /// A specific entity by ID.
    Entity(u32),
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

/// The authoritative scene. Owns all entities in a flat list, plus the
/// current visual state (which may be interpolated during animation).
pub struct Scene {
    /// Entities in insertion order.
    entities: Vec<SceneEntity>,
    focus: Focus,
    next_entity_id: u32,
    /// Monotonically increasing generation; bumped on any structural mutation
    /// (add/remove entity, coords update).
    generation: u64,
    /// Generation that was last consumed by the renderer.
    rendered_generation: u64,
    /// Position generation; bumped each animation frame.
    position_generation: u64,
    /// Position generation last consumed by the renderer.
    rendered_position_generation: u64,

    // -- Current visual state (may be interpolated during animation) --
    /// Current visual backbone chains (interpolated or at-rest).
    pub(crate) visual_backbone_chains: Vec<Vec<Vec3>>,
    /// Current visual sidechain atom positions (interpolated or at-rest).
    pub(crate) visual_sidechain_positions: Vec<Vec3>,
    /// Current visual backbone-sidechain bonds (interpolated or at-rest).
    pub(crate) visual_backbone_sidechain_bonds: Vec<(Vec3, u32)>,
    /// Nucleic acid P-atom chains (stable between animation frames).
    pub(crate) na_chains: Vec<Vec<Vec3>>,
    /// Per-entity residue ranges in the flat concatenated arrays.
    pub(crate) entity_residue_ranges: Vec<EntityResidueRange>,

    // -- Sidechain topology (stable between animation frames) --
    /// Bond pairs (atom index A, atom index B) within sidechains.
    pub(crate) sidechain_bonds: Vec<(u32, u32)>,
    /// Per-atom hydrophobicity flag.
    pub(crate) sidechain_hydrophobicity: Vec<bool>,
    /// Residue index for each sidechain atom.
    pub(crate) sidechain_residue_indices: Vec<u32>,
    /// Atom names for each sidechain atom (used for by-name lookup).
    pub(crate) sidechain_atom_names: Vec<String>,
    /// Target sidechain atom positions (the "at-rest" goal).
    pub(crate) target_sidechain_positions: Vec<Vec3>,
    /// Target backbone-sidechain bond endpoints (CA pos, CB atom index).
    pub(crate) target_backbone_sidechain_bonds: Vec<(Vec3, u32)>,

    // -- Render-derived state --
    /// Per-residue secondary structure types (flat across all chains).
    pub(crate) ss_types: Vec<SSType>,
    /// Per-residue colors derived from scores or color mode.
    pub(crate) per_residue_colors: Option<Vec<[f32; 3]>>,
}

impl Scene {
    /// Create an empty scene with no entities and session-level focus.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            focus: Focus::Session,
            next_entity_id: 0,
            generation: 0,
            rendered_generation: 0,
            position_generation: 0,
            rendered_position_generation: 0,
            visual_backbone_chains: Vec::new(),
            visual_sidechain_positions: Vec::new(),
            visual_backbone_sidechain_bonds: Vec::new(),
            na_chains: Vec::new(),
            entity_residue_ranges: Vec::new(),
            sidechain_bonds: Vec::new(),
            sidechain_hydrophobicity: Vec::new(),
            sidechain_residue_indices: Vec::new(),
            sidechain_atom_names: Vec::new(),
            target_sidechain_positions: Vec::new(),
            target_backbone_sidechain_bonds: Vec::new(),
            ss_types: Vec::new(),
            per_residue_colors: None,
        }
    }

    // -- Mutation helpers --

    fn invalidate(&mut self) {
        self.generation += 1;
    }

    /// Whether scene data changed since last `mark_rendered()`.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.generation != self.rendered_generation
    }

    /// Force the scene dirty (e.g. when display options change but scene data
    /// hasn't).
    pub fn force_dirty(&mut self) {
        self.invalidate();
    }

    /// Mark current generation as rendered (call after updating renderers).
    pub fn mark_rendered(&mut self) {
        self.rendered_generation = self.generation;
    }

    /// Whether visual positions changed since last `mark_position_rendered()`.
    #[must_use]
    pub fn is_position_dirty(&self) -> bool {
        self.position_generation != self.rendered_position_generation
    }

    /// Mark current position generation as rendered.
    pub fn mark_position_rendered(&mut self) {
        self.rendered_position_generation = self.position_generation;
    }

    /// Update full visual state: backbone, sidechain, and bonds.
    ///
    /// Called each animation frame with the complete
    /// [`AnimationFrame`](crate::animation::AnimationFrame) output.
    /// Bumps position generation.
    pub fn update_visual_state(
        &mut self,
        backbone_chains: Vec<Vec<Vec3>>,
        sidechain_positions: Vec<Vec3>,
        backbone_sidechain_bonds: Vec<(Vec3, u32)>,
    ) {
        self.visual_backbone_chains = backbone_chains;
        self.visual_sidechain_positions = sidechain_positions;
        self.visual_backbone_sidechain_bonds = backbone_sidechain_bonds;
        self.position_generation += 1;
    }

    /// Update sidechain topology and target positions from prepared data.
    ///
    /// Called when the scene changes (new entities, coord updates).
    /// Populates bond topology, target positions, and per-atom metadata.
    pub fn update_sidechain_topology(&mut self, sidechain: &SidechainAtoms) {
        self.target_sidechain_positions = sidechain.positions();
        self.target_backbone_sidechain_bonds
            .clone_from(&sidechain.backbone_bonds);
        self.sidechain_bonds.clone_from(&sidechain.bonds);
        self.sidechain_hydrophobicity = sidechain.hydrophobicity();
        self.sidechain_residue_indices = sidechain.residue_indices();
        self.sidechain_atom_names = sidechain.atom_names();
    }

    /// Build a [`SidechainAtoms`] from interpolated positions and bonds,
    /// using this scene's topology metadata.
    ///
    /// Used when submitting animation frames to the background mesh
    /// generator.
    #[must_use]
    pub fn to_interpolated_sidechain_atoms(
        &self,
        positions: &[Vec3],
        backbone_bonds: &[(Vec3, u32)],
    ) -> SidechainAtoms {
        SidechainAtoms {
            atoms: positions
                .iter()
                .enumerate()
                .map(|(i, &pos)| SidechainAtomData {
                    position: pos,
                    residue_idx: self
                        .sidechain_residue_indices
                        .get(i)
                        .copied()
                        .unwrap_or(0),
                    atom_name: String::new(),
                    is_hydrophobic: self
                        .sidechain_hydrophobicity
                        .get(i)
                        .copied()
                        .unwrap_or(false),
                })
                .collect(),
            bonds: self.sidechain_bonds.clone(),
            backbone_bonds: backbone_bonds.to_vec(),
        }
    }

    /// Set entity residue ranges (populated from prepared scene data).
    pub fn set_entity_residue_ranges(
        &mut self,
        ranges: Vec<EntityResidueRange>,
    ) {
        self.entity_residue_ranges = ranges;
    }

    // -- Entity management --

    /// Add entities to the scene. Entity IDs are reassigned to be globally
    /// unique. Returns the assigned entity IDs.
    pub fn add_entities(&mut self, entities: Vec<MoleculeEntity>) -> Vec<u32> {
        let mut ids = Vec::with_capacity(entities.len());
        for mut entity in entities {
            let id = self.next_entity_id;
            self.next_entity_id += 1;
            entity.entity_id = id;
            self.entities.push(SceneEntity {
                entity,
                visible: true,
                ss_override: None,
                per_residue_scores: None,
                mesh_version: 0,
            });
            ids.push(id);
        }
        self.invalidate();
        ids
    }

    /// Read access to an entity.
    #[must_use]
    pub fn entity(&self, id: u32) -> Option<&SceneEntity> {
        self.entities.iter().find(|e| e.id() == id)
    }

    /// Mutable access to an entity.
    pub fn entity_mut(&mut self, id: u32) -> Option<&mut SceneEntity> {
        self.entities.iter_mut().find(|e| e.id() == id)
    }

    /// Remove an entity by ID. Returns true if the entity existed.
    pub fn remove_entity(&mut self, id: u32) -> bool {
        let before = self.entities.len();
        self.entities.retain(|e| e.id() != id);
        let removed = self.entities.len() < before;
        if removed {
            self.invalidate();
        }
        removed
    }

    /// Read access to all entities (insertion order).
    #[must_use]
    pub fn entities(&self) -> &[SceneEntity] {
        &self.entities
    }

    // -- Focus / tab cycling --

    /// Current focus state.
    #[must_use]
    pub fn focus(&self) -> &Focus {
        &self.focus
    }

    /// Set the focus state directly.
    pub fn set_focus(&mut self, focus: Focus) {
        self.focus = focus;
    }

    /// Cycle: Session -> Entity1 -> ... -> EntityN -> Session.
    pub fn cycle_focus(&mut self) -> Focus {
        let focusable: Vec<u32> = self
            .entities
            .iter()
            .filter(|e| e.visible)
            .map(SceneEntity::id)
            .collect();

        self.focus = match self.focus {
            Focus::Session => focusable
                .first()
                .map_or(Focus::Session, |&id| Focus::Entity(id)),
            Focus::Entity(current_id) => {
                let idx = focusable.iter().position(|&id| id == current_id);
                match idx {
                    Some(i) if i + 1 < focusable.len() => {
                        Focus::Entity(focusable[i + 1])
                    }
                    _ => Focus::Session,
                }
            }
        };
        self.focus
    }

    // -- Filtered entity access --

    /// Visible nucleic acid (DNA/RNA) entities.
    #[must_use]
    pub fn nucleic_acid_entities(&self) -> Vec<&SceneEntity> {
        self.entities
            .iter()
            .filter(|e| e.visible && e.is_nucleic_acid())
            .collect()
    }

    /// Visible ligand entities (not protein, not nucleic acid).
    #[must_use]
    pub fn ligand_entities(&self) -> Vec<&SceneEntity> {
        self.entities
            .iter()
            .filter(|e| e.visible && e.is_ligand())
            .collect()
    }

    // -- Per-entity data for scene processor --

    /// Collect per-entity render data for all visible entities.
    pub fn per_entity_data(&self) -> Vec<PerEntityData> {
        self.entities
            .iter()
            .filter(|e| e.visible)
            .filter_map(SceneEntity::to_per_entity_data)
            .collect()
    }

    /// All atom positions across all visible entities (for camera fitting).
    #[must_use]
    pub fn all_positions(&self) -> Vec<Vec3> {
        self.entities
            .iter()
            .filter(|e| e.visible)
            .flat_map(|e| {
                e.entity
                    .coords
                    .atoms
                    .iter()
                    .map(|a| Vec3::new(a.x, a.y, a.z))
            })
            .collect()
    }

    /// Replace the `MoleculeEntity` for an existing entity by id.
    ///
    /// Bumps mesh version and invalidates the scene generation counter.
    pub fn replace_entity(&mut self, entity: MoleculeEntity) {
        if let Some(se) = self
            .entities
            .iter_mut()
            .find(|e| e.entity.entity_id == entity.entity_id)
        {
            se.entity = entity;
            se.invalidate_render_cache();
        }
        self.invalidate();
    }

    /// Update protein entity coords in a specific entity.
    pub fn update_entity_protein_coords(&mut self, id: u32, coords: Coords) {
        if let Some(se) =
            self.entities.iter_mut().find(|e| e.entity.entity_id == id)
        {
            // Wrap in a temporary Vec for the assembly update function
            let mut entities = vec![se.entity.clone()];
            update_protein_entities(&mut entities, coords);
            if let Some(updated) = entities.into_iter().next() {
                se.entity = updated;
            }
            se.invalidate_render_cache();
        }
        self.invalidate();
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
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
}

impl SceneEntity {
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
