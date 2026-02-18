use foldit_conv::secondary_structure::SSType;
use glam::Vec3;

/// Owns sidechain animation state, cached secondary structure types,
/// per-residue colors, and the frustum culling camera eye.
///
/// These 11 fields are always read/written together during animation
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
}
