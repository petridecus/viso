//! Constraint resolution: resolve band/pull specs to world-space positions.

use glam::{UVec2, Vec2, Vec3};

use super::command::{
    AtomRef, BandInfo, BandTarget, PullInfo, ResolvedBand, ResolvedPull,
};
use super::scene::{SceneTopology, VisualState};
use crate::camera::controller::CameraController;

/// Borrowed scene state needed for atom position lookups.
pub(super) struct ScenePositions<'a> {
    /// Interpolated animation output (backbone chains, sidechain positions).
    pub visual: &'a VisualState,
    /// Derived metadata (sidechain topology with target positions/names).
    pub topology: &'a SceneTopology,
    /// Backbone renderer's cached chain positions (pre-animation fallback).
    pub cached_chains: &'a [Vec<Vec3>],
}

/// Resolve a single band spec to world-space endpoint positions.
pub(super) fn resolve_band(
    scene: &ScenePositions<'_>,
    band: &BandInfo,
) -> Option<ResolvedBand> {
    let endpoint_a = resolve_atom_ref(scene, &band.anchor_a)?;
    let endpoint_b = match &band.anchor_b {
        BandTarget::Atom(atom) => resolve_atom_ref(scene, atom)?,
        BandTarget::Position(pos) => *pos,
    };
    let is_space_pull = matches!(band.anchor_b, BandTarget::Position(_));

    Some(ResolvedBand {
        endpoint_a,
        endpoint_b,
        is_disabled: band.is_disabled,
        strength: band.strength,
        target_length: band.target_length,
        residue_idx: band.anchor_a.residue,
        is_space_pull,
        band_type: band.band_type,
        from_script: band.from_script,
    })
}

/// Resolve a pull spec to world-space atom and target positions.
pub(super) fn resolve_pull(
    scene: &ScenePositions<'_>,
    camera: &CameraController,
    viewport: (u32, u32),
    pull: &PullInfo,
) -> Option<ResolvedPull> {
    let atom_pos = resolve_atom_ref(scene, &pull.atom)?;
    let target_pos = camera.screen_to_world_at_depth(
        Vec2::new(pull.screen_target.0, pull.screen_target.1),
        UVec2::new(viewport.0, viewport.1),
        atom_pos,
    );

    Some(ResolvedPull {
        atom_pos,
        target_pos,
        residue_idx: pull.atom.residue,
    })
}

/// Resolve an [`AtomRef`] to a world-space position from scene data.
///
/// Uses interpolated visual positions during animation so constraints
/// track animated atoms. Falls back to `cached_chains` (from the backbone
/// renderer) before the first animation frame is available.
fn resolve_atom_ref(
    scene: &ScenePositions<'_>,
    atom: &AtomRef,
) -> Option<Vec3> {
    let name = atom.atom_name.as_str();

    // Backbone atoms: N, CA, C — look up in visual_backbone_chains
    if name == "N" || name == "CA" || name == "C" {
        let offset = match name {
            "N" => 0,
            "CA" => 1,
            "C" => 2,
            _ => return None,
        };
        let chains = if scene.visual.backbone_chains.is_empty() {
            // Before first animation, fall back to renderer cache
            scene.cached_chains
        } else {
            &scene.visual.backbone_chains
        };
        let mut current_idx: u32 = 0;
        for chain in chains {
            let residues_in_chain = (chain.len() / 3) as u32;
            if atom.residue < current_idx + residues_in_chain {
                let local = (atom.residue - current_idx) as usize;
                return chain.get(local * 3 + offset).copied();
            }
            current_idx += residues_in_chain;
        }
        return None;
    }

    // Sidechain atoms — look up in visual_sidechain_positions
    let positions = if scene.visual.sidechain_positions.is_empty() {
        &scene.topology.sidechain_topology.target_positions
    } else {
        &scene.visual.sidechain_positions
    };
    for (i, (res_idx, sc_name)) in scene
        .topology
        .sidechain_topology
        .residue_indices
        .iter()
        .zip(scene.topology.sidechain_topology.atom_names.iter())
        .enumerate()
    {
        if *res_idx == atom.residue && sc_name == name {
            return positions.get(i).copied();
        }
    }

    None
}
