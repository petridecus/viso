//! Cached scene-derived sidechain data used by renderers and queries.
//!
//! This is NOT animation state — it holds topology and target positions
//! that are updated when the scene changes (new entities, coord updates)
//! and remain stable between animation frames.

use foldit_conv::render::sidechain::{SidechainAtomData, SidechainAtoms};
use glam::Vec3;

use crate::animation::StructureAnimator;

/// Cached scene-derived sidechain data used by renderers and queries.
///
/// Updated when the scene changes (new entities, coords updates) and
/// remains stable between animation frames.
pub(crate) struct SidechainCache {
    /// Bond pairs (atom index A, atom index B) within sidechains.
    pub sidechain_bonds: Vec<(u32, u32)>,
    /// Per-atom hydrophobicity flag.
    pub sidechain_hydrophobicity: Vec<bool>,
    /// Residue index for each sidechain atom.
    pub sidechain_residue_indices: Vec<u32>,
    /// Atom names for each sidechain atom (used for by-name lookup).
    pub sidechain_atom_names: Vec<String>,
    /// Per-residue secondary structure types (flat across all chains).
    pub ss_types: Vec<foldit_conv::secondary_structure::SSType>,
    /// Per-residue colors derived from scores or color mode.
    pub per_residue_colors: Option<Vec<[f32; 3]>>,
    /// Target sidechain atom positions (the "at-rest" goal).
    pub target_sidechain_positions: Vec<Vec3>,
    /// Target backbone-sidechain bond endpoints (CA pos, CB atom index).
    pub target_backbone_sidechain_bonds: Vec<(Vec3, u32)>,
}

impl SidechainCache {
    /// Create empty cache.
    pub fn new() -> Self {
        Self {
            sidechain_bonds: Vec::new(),
            sidechain_hydrophobicity: Vec::new(),
            sidechain_residue_indices: Vec::new(),
            sidechain_atom_names: Vec::new(),
            ss_types: Vec::new(),
            per_residue_colors: None,
            target_sidechain_positions: Vec::new(),
            target_backbone_sidechain_bonds: Vec::new(),
        }
    }

    /// Update cached topology and target positions from a
    /// [`SidechainAtoms`].
    ///
    /// Updates only the cache's own fields. The caller is responsible for
    /// updating any animation start/target data separately.
    pub fn update_from_sidechain_atoms(&mut self, sidechain: &SidechainAtoms) {
        self.target_sidechain_positions = sidechain.positions();
        self.target_backbone_sidechain_bonds
            .clone_from(&sidechain.backbone_bonds);
        self.sidechain_bonds.clone_from(&sidechain.bonds);
        self.sidechain_hydrophobicity = sidechain.hydrophobicity();
        self.sidechain_residue_indices = sidechain.residue_indices();
        self.sidechain_atom_names = sidechain.atom_names();
    }

    /// Build a [`SidechainAtoms`] from interpolated positions and bonds,
    /// using this cache's topology metadata.
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

    /// Compute interpolated backbone-sidechain bonds from animator's
    /// current CA positions.
    ///
    /// During animation, CA positions are interpolated between start and
    /// target. This maps each bond's CA endpoint to the animator's current
    /// interpolated CA, producing smooth bond movement.
    #[must_use]
    pub fn interpolated_backbone_bonds(
        &self,
        animator: &StructureAnimator,
    ) -> Vec<(Vec3, u32)> {
        self.target_backbone_sidechain_bonds
            .iter()
            .map(|(target_ca_pos, cb_idx)| {
                let res_idx = self
                    .sidechain_residue_indices
                    .get(*cb_idx as usize)
                    .copied()
                    .unwrap_or(0) as usize;
                let ca_pos =
                    animator.get_ca_position(res_idx).unwrap_or(*target_ca_pos);
                (ca_pos, *cb_idx)
            })
            .collect()
    }
}
