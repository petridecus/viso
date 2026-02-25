use foldit_conv::render::sidechain::{SidechainAtomData, SidechainAtoms};
use foldit_conv::secondary_structure::SSType;
use glam::Vec3;

use crate::animation::animator::StructureAnimator;

/// Owns sidechain animation state, cached secondary structure types,
/// per-residue colors, and the frustum culling camera eye.
///
/// These fields are always read/written together during animation
/// and scene application.
pub(crate) struct SidechainAnimationState {
    pub start_sidechain_positions: Vec<Vec3>,
    pub target_sidechain_positions: Vec<Vec3>,
    pub start_backbone_sidechain_bonds: Vec<(Vec3, u32)>,
    pub target_backbone_sidechain_bonds: Vec<(Vec3, u32)>,
    pub cached_sidechain_bonds: Vec<(u32, u32)>,
    pub cached_sidechain_hydrophobicity: Vec<bool>,
    pub cached_sidechain_residue_indices: Vec<u32>,
    pub cached_sidechain_atom_names: Vec<String>,
    pub cached_ss_types: Vec<SSType>,
    pub cached_per_residue_colors: Option<Vec<[f32; 3]>>,
    pub last_cull_camera_eye: Vec3,
}

impl SidechainAnimationState {
    pub fn new() -> Self {
        Self {
            start_sidechain_positions: Vec::new(),
            target_sidechain_positions: Vec::new(),
            start_backbone_sidechain_bonds: Vec::new(),
            target_backbone_sidechain_bonds: Vec::new(),
            cached_sidechain_bonds: Vec::new(),
            cached_sidechain_hydrophobicity: Vec::new(),
            cached_sidechain_residue_indices: Vec::new(),
            cached_sidechain_atom_names: Vec::new(),
            cached_ss_types: Vec::new(),
            cached_per_residue_colors: None,
            last_cull_camera_eye: Vec3::ZERO,
        }
    }

    /// Set all cached and target fields from a [`SidechainAtoms`].
    ///
    /// Used by both snap and animation-target setup paths.
    pub fn update_cached_from_sidechain_atoms(
        &mut self,
        sidechain: &SidechainAtoms,
    ) {
        self.target_sidechain_positions = sidechain.positions();
        self.target_backbone_sidechain_bonds
            .clone_from(&sidechain.backbone_bonds);
        self.cached_sidechain_bonds.clone_from(&sidechain.bonds);
        self.cached_sidechain_hydrophobicity = sidechain.hydrophobicity();
        self.cached_sidechain_residue_indices = sidechain.residue_indices();
        self.cached_sidechain_atom_names = sidechain.atom_names();
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
    ) -> Vec<(Vec3, u32)> {
        self.target_backbone_sidechain_bonds
            .iter()
            .map(|(target_ca_pos, cb_idx)| {
                let res_idx = self
                    .cached_sidechain_residue_indices
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
                        .cached_sidechain_residue_indices
                        .get(i)
                        .copied()
                        .unwrap_or(0),
                    atom_name: String::new(),
                    is_hydrophobic: self
                        .cached_sidechain_hydrophobicity
                        .get(i)
                        .copied()
                        .unwrap_or(false),
                })
                .collect(),
            bonds: self.cached_sidechain_bonds.clone(),
            backbone_bonds: backbone_bonds.to_vec(),
        }
    }
}
