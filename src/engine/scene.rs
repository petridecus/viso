//! Scene topology: derived structural metadata computed from entities.
//!
//! Does NOT own entities (those live in
//! [`super::entity_store::EntityStore`]). Holds sidechain topology,
//! SS types, per-residue colors, residue ranges, and NA chains — all
//! recomputed when entities change.

use foldit_conv::render::sidechain::{SidechainAtomData, SidechainAtoms};
use foldit_conv::secondary_structure::SSType;
use glam::Vec3;
use rustc_hash::FxHashMap;

use super::scene_data::EntityResidueRange;

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
// SidechainTopology
// ---------------------------------------------------------------------------

/// Consolidated sidechain structural data: bond topology, per-atom metadata,
/// and at-rest target positions. Stable between animation frames; recomputed
/// when entities change.
pub(crate) struct SidechainTopology {
    /// Bond pairs (atom index A, atom index B) within sidechains.
    pub(crate) bonds: Vec<(u32, u32)>,
    /// Per-atom hydrophobicity flag.
    pub(crate) hydrophobicity: Vec<bool>,
    /// Residue index for each sidechain atom.
    pub(crate) residue_indices: Vec<u32>,
    /// Atom names for each sidechain atom (used for by-name lookup).
    pub(crate) atom_names: Vec<String>,
    /// Target sidechain atom positions (the "at-rest" goal).
    pub(crate) target_positions: Vec<Vec3>,
    /// Target backbone-sidechain bond endpoints (CA pos, CB atom index).
    pub(crate) target_backbone_bonds: Vec<(Vec3, u32)>,
    /// (residue_idx, atom_name) → flat index for O(1) sidechain lookup.
    pub(crate) atom_index: FxHashMap<(u32, String), usize>,
}

impl SidechainTopology {
    /// Create an empty sidechain topology.
    fn new() -> Self {
        Self {
            bonds: Vec::new(),
            hydrophobicity: Vec::new(),
            residue_indices: Vec::new(),
            atom_names: Vec::new(),
            target_positions: Vec::new(),
            target_backbone_bonds: Vec::new(),
            atom_index: FxHashMap::default(),
        }
    }

    /// Build a [`SidechainAtoms`] from interpolated positions and bonds,
    /// using this topology's metadata.
    #[must_use]
    pub fn to_interpolated_atoms(
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
                        .residue_indices
                        .get(i)
                        .copied()
                        .unwrap_or(0),
                    atom_name: String::new(),
                    is_hydrophobic: self
                        .hydrophobicity
                        .get(i)
                        .copied()
                        .unwrap_or(false),
                })
                .collect(),
            bonds: self.bonds.clone(),
            backbone_bonds: backbone_bonds.to_vec(),
        }
    }
}

// ---------------------------------------------------------------------------
// VisualState
// ---------------------------------------------------------------------------

/// Animation output buffer: interpolated positions written by the animator
/// each frame, read by renderers and constraint resolution.
///
/// Lives on [`super::VisoEngine`], separate from [`SceneTopology`], making
/// the data flow explicit: animator writes to `visual`, renderer reads
/// `visual`.
pub(crate) struct VisualState {
    /// Current visual backbone chains (interpolated or at-rest).
    pub(crate) backbone_chains: Vec<Vec<Vec3>>,
    /// Current visual sidechain atom positions (interpolated or at-rest).
    pub(crate) sidechain_positions: Vec<Vec3>,
    /// Current visual backbone-sidechain bonds (interpolated or at-rest).
    pub(crate) backbone_sidechain_bonds: Vec<(Vec3, u32)>,
    /// Position generation; bumped each animation frame.
    position_generation: u64,
    /// Position generation last consumed by the renderer.
    rendered_position_generation: u64,
}

impl VisualState {
    /// Create an empty visual state.
    pub fn new() -> Self {
        Self {
            backbone_chains: Vec::new(),
            sidechain_positions: Vec::new(),
            backbone_sidechain_bonds: Vec::new(),
            position_generation: 0,
            rendered_position_generation: 0,
        }
    }

    /// Update full visual state: backbone, sidechain, and bonds.
    ///
    /// Called each animation frame with the complete
    /// [`AnimationFrame`](crate::animation::AnimationFrame) output.
    /// Bumps position generation.
    pub fn update(
        &mut self,
        backbone_chains: Vec<Vec<Vec3>>,
        sidechain_positions: Vec<Vec3>,
        backbone_sidechain_bonds: Vec<(Vec3, u32)>,
    ) {
        self.backbone_chains = backbone_chains;
        self.sidechain_positions = sidechain_positions;
        self.backbone_sidechain_bonds = backbone_sidechain_bonds;
        self.position_generation += 1;
    }

    /// Whether visual positions changed since last `mark_rendered()`.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.position_generation != self.rendered_position_generation
    }

    /// Mark current position generation as rendered.
    pub fn mark_rendered(&mut self) {
        self.rendered_position_generation = self.position_generation;
    }
}

// ---------------------------------------------------------------------------
// SceneTopology
// ---------------------------------------------------------------------------

/// Derived scene topology: structural metadata computed from entities.
///
/// Does NOT own entities (those live in [`super::entity_store::EntityStore`]).
/// Holds sidechain topology, SS types, per-residue colors, residue ranges,
/// and NA chains — all recomputed when entities change.
pub struct SceneTopology {
    /// Nucleic acid P-atom chains (stable between animation frames).
    pub(crate) na_chains: Vec<Vec<Vec3>>,
    /// Per-entity residue ranges in the flat concatenated arrays.
    pub(crate) entity_residue_ranges: Vec<EntityResidueRange>,

    /// Sidechain bond topology, per-atom metadata, and target positions.
    pub(crate) sidechain_topology: SidechainTopology,

    /// Cumulative residue offset per backbone chain (for O(log n) lookup).
    /// Element `i` is the total number of residues in chains `0..i`.
    pub(crate) backbone_chain_offsets: Vec<u32>,

    // -- Render-derived state --
    /// Per-residue secondary structure types (flat across all chains).
    pub(crate) ss_types: Vec<SSType>,
    /// Per-residue colors derived from scores or color mode.
    pub(crate) per_residue_colors: Option<Vec<[f32; 3]>>,
}

impl SceneTopology {
    /// Create an empty scene topology.
    #[must_use]
    pub fn new() -> Self {
        Self {
            na_chains: Vec::new(),
            entity_residue_ranges: Vec::new(),
            sidechain_topology: SidechainTopology::new(),
            backbone_chain_offsets: Vec::new(),
            ss_types: Vec::new(),
            per_residue_colors: None,
        }
    }

    /// Rebuild structural metadata from visible entity data.
    ///
    /// Recomputes entity residue ranges, sidechain topology, secondary
    /// structure types, and nucleic acid chains. Call when entities change.
    pub fn rebuild(&mut self, entities: &[super::scene_data::PerEntityData]) {
        let ranges = super::scene_data::compute_entity_residue_ranges(entities);
        let sidechain =
            super::scene_data::concatenate_sidechain_atoms(entities, &ranges);
        self.update_sidechain_topology(&sidechain);
        self.ss_types =
            super::scene_data::concatenate_ss_types(entities, &ranges);
        self.na_chains = entities
            .iter()
            .flat_map(|e| e.nucleic_acid_chains.iter().cloned())
            .collect();
        self.entity_residue_ranges = ranges;

        // Precompute backbone chain offsets for O(log n) constraint lookup
        let mut offset = 0u32;
        self.backbone_chain_offsets = entities
            .iter()
            .flat_map(|e| &e.backbone_chains)
            .map(|chain| {
                let start = offset;
                offset += (chain.len() / 3) as u32;
                start
            })
            .collect();
    }

    /// Update sidechain topology and target positions from prepared data.
    ///
    /// Called when the scene changes (new entities, coord updates).
    /// Populates bond topology, target positions, and per-atom metadata.
    fn update_sidechain_topology(&mut self, sidechain: &SidechainAtoms) {
        self.sidechain_topology.target_positions = sidechain.positions();
        self.sidechain_topology
            .target_backbone_bonds
            .clone_from(&sidechain.backbone_bonds);
        self.sidechain_topology.bonds.clone_from(&sidechain.bonds);
        self.sidechain_topology.hydrophobicity = sidechain.hydrophobicity();
        self.sidechain_topology.residue_indices = sidechain.residue_indices();
        self.sidechain_topology.atom_names = sidechain.atom_names();

        // Build O(1) sidechain atom lookup
        self.sidechain_topology.atom_index = self
            .sidechain_topology
            .residue_indices
            .iter()
            .zip(self.sidechain_topology.atom_names.iter())
            .enumerate()
            .map(|(i, (&res_idx, name))| ((res_idx, name.clone()), i))
            .collect();
    }
}

// ---------------------------------------------------------------------------
// Scene lifecycle (impl VisoEngine)
// ---------------------------------------------------------------------------

use foldit_conv::types::entity::MoleculeEntity;

impl super::VisoEngine {
    /// Replace the current scene with new entities.
    ///
    /// Clears all existing entities and loads the new ones via
    /// [`load_entities`](Self::load_entities) with camera fit.
    /// Returns the assigned entity IDs.
    pub fn replace_scene(&mut self, entities: Vec<MoleculeEntity>) -> Vec<u32> {
        self.entities.clear_all();
        self.visual = VisualState::new();
        self.animation = crate::animation::AnimationState::new();
        self.load_entities(entities, true)
    }

    /// Remove all entities from the scene.
    ///
    /// Clears the entity store and triggers a full renderer sync so the
    /// viewport becomes empty.
    pub fn clear_scene(&mut self) {
        self.entities.clear_all();
        self.visual = VisualState::new();
        self.animation = crate::animation::AnimationState::new();
        self.sync_scene_to_renderers(std::collections::HashMap::new());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visual_new_is_clean() {
        let vs = VisualState::new();
        assert!(!vs.is_dirty());
    }

    #[test]
    fn visual_update_marks_dirty() {
        let mut vs = VisualState::new();
        vs.update(vec![vec![Vec3::ZERO]], vec![Vec3::X], vec![(Vec3::Y, 0)]);
        assert!(vs.is_dirty());
    }

    #[test]
    fn visual_mark_rendered_clears() {
        let mut vs = VisualState::new();
        vs.update(vec![], vec![], vec![]);
        assert!(vs.is_dirty());
        vs.mark_rendered();
        assert!(!vs.is_dirty());
    }
}
