use foldit_conv::render::sidechain::{SidechainAtomData, SidechainAtoms};
use glam::Vec3;

use crate::animation::animator::StructureAnimator;

/// Animation start/target pairs for sidechain positions and
/// backbone-sidechain bonds.
///
/// These fields change every time a new animation target is set, and are
/// interpolated each frame by the animator.
pub(crate) struct SidechainAnimData {
    /// Start sidechain atom positions (animation begin state).
    pub start_sidechain_positions: Vec<Vec3>,
    /// Target sidechain atom positions (animation end state).
    pub target_sidechain_positions: Vec<Vec3>,
    /// Start backbone-sidechain bond endpoints (CA pos, CB atom index).
    pub start_backbone_sidechain_bonds: Vec<(Vec3, u32)>,
    /// Target backbone-sidechain bond endpoints (CA pos, CB atom index).
    pub target_backbone_sidechain_bonds: Vec<(Vec3, u32)>,
}

impl SidechainAnimData {
    /// Create empty animation data.
    pub fn new() -> Self {
        Self {
            start_sidechain_positions: Vec::new(),
            target_sidechain_positions: Vec::new(),
            start_backbone_sidechain_bonds: Vec::new(),
            target_backbone_sidechain_bonds: Vec::new(),
        }
    }

    /// Set start positions equal to target (instant snap, no interpolation).
    pub fn snap_positions(&mut self) {
        self.start_sidechain_positions
            .clone_from(&self.target_sidechain_positions);
        self.start_backbone_sidechain_bonds
            .clone_from(&self.target_backbone_sidechain_bonds);
    }

    /// Compute interpolated backbone-sidechain bonds using the animator's
    /// current CA positions.
    ///
    /// During animation, CA positions are interpolated between start and
    /// target. This maps each bond's CA endpoint to the animator's current
    /// interpolated CA, producing smooth bond movement.
    #[must_use]
    pub fn interpolated_backbone_bonds(
        &self,
        animator: &StructureAnimator,
        cached_residue_indices: &[u32],
    ) -> Vec<(Vec3, u32)> {
        self.target_backbone_sidechain_bonds
            .iter()
            .map(|(target_ca_pos, cb_idx)| {
                let res_idx = cached_residue_indices
                    .get(*cb_idx as usize)
                    .copied()
                    .unwrap_or(0) as usize;
                let ca_pos =
                    animator.get_ca_position(res_idx).unwrap_or(*target_ca_pos);
                (ca_pos, *cb_idx)
            })
            .collect()
    }

    /// Build a [`SidechainAtoms`] from interpolated positions and bonds.
    ///
    /// Used when submitting animation frames to the background mesh
    /// generator.
    #[must_use]
    pub fn to_interpolated_sidechain_atoms(
        positions: &[Vec3],
        backbone_bonds: &[(Vec3, u32)],
        cache: &SidechainCache,
    ) -> SidechainAtoms {
        SidechainAtoms {
            atoms: positions
                .iter()
                .enumerate()
                .map(|(i, &pos)| SidechainAtomData {
                    position: pos,
                    residue_idx: cache
                        .sidechain_residue_indices
                        .get(i)
                        .copied()
                        .unwrap_or(0),
                    atom_name: String::new(),
                    is_hydrophobic: cache
                        .sidechain_hydrophobicity
                        .get(i)
                        .copied()
                        .unwrap_or(false),
                })
                .collect(),
            bonds: cache.sidechain_bonds.clone(),
            backbone_bonds: backbone_bonds.to_vec(),
        }
    }
}

/// Cached scene-derived sidechain data used by renderers and queries.
///
/// These fields are updated when the scene changes (new entities, coords
/// updates) and remain stable between animation frames.
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
        }
    }

    /// Update cached fields from a [`SidechainAtoms`] and set new target
    /// positions on the animation data.
    ///
    /// Used by both snap and animation-target setup paths.
    pub fn update_from_sidechain_atoms(
        &mut self,
        sidechain: &SidechainAtoms,
        anim: &mut SidechainAnimData,
    ) {
        anim.target_sidechain_positions = sidechain.positions();
        anim.target_backbone_sidechain_bonds
            .clone_from(&sidechain.backbone_bonds);
        self.sidechain_bonds.clone_from(&sidechain.bonds);
        self.sidechain_hydrophobicity = sidechain.hydrophobicity();
        self.sidechain_residue_indices = sidechain.residue_indices();
        self.sidechain_atom_names = sidechain.atom_names();
    }
}
