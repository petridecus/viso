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

/// Public entry point for [`resolve_atom_ref`] (used by
/// [`super::VisoEngine::resolve_atom_position`]).
pub(super) fn resolve_atom_ref_pub(
    scene: &ScenePositions<'_>,
    atom: &AtomRef,
) -> Option<Vec3> {
    resolve_atom_ref(scene, atom)
}

/// Resolve an [`AtomRef`] to a world-space position from scene data.
///
/// Uses interpolated visual positions during animation so constraints
/// track animated atoms. Falls back to `cached_chains` (from the backbone
/// renderer) before the first animation frame is available.
///
/// Backbone atoms (N, CA, C) use precomputed chain offsets for O(log n)
/// lookup. Sidechain atoms use a hash map for O(1) lookup.
fn resolve_atom_ref(
    scene: &ScenePositions<'_>,
    atom: &AtomRef,
) -> Option<Vec3> {
    let name = atom.atom_name.as_str();

    // Backbone atoms: N, CA, C — use chain offset index
    if name == "N" || name == "CA" || name == "C" {
        let offset = match name {
            "N" => 0,
            "CA" => 1,
            "C" => 2,
            _ => return None,
        };
        let chains = if scene.visual.backbone_chains.is_empty() {
            scene.cached_chains
        } else {
            &scene.visual.backbone_chains
        };
        let offsets = &scene.topology.backbone_chain_offsets;
        // Binary search: find the chain containing this residue
        let chain_idx = match offsets.binary_search(&atom.residue) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        let chain = chains.get(chain_idx)?;
        let chain_start = offsets.get(chain_idx).copied().unwrap_or(0);
        let local = (atom.residue - chain_start) as usize;
        return chain.get(local * 3 + offset).copied();
    }

    // Sidechain atoms — O(1) hash lookup
    let positions = if scene.visual.sidechain_positions.is_empty() {
        &scene.topology.sidechain_topology.target_positions
    } else {
        &scene.visual.sidechain_positions
    };
    let idx = scene
        .topology
        .sidechain_topology
        .atom_index
        .get(&(atom.residue, name.to_owned()))?;
    positions.get(*idx).copied()
}

#[cfg(test)]
mod tests {
    use glam::Vec3;
    use rustc_hash::FxHashMap;

    use super::*;
    use crate::engine::scene::{SceneTopology, SidechainTopology, VisualState};

    /// Build a minimal scene with 1 chain, 2 residues (6 backbone atoms),
    /// and 1 sidechain atom ("CB" at residue 0).
    fn test_scene() -> (VisualState, SceneTopology, Vec<Vec<Vec3>>) {
        // Backbone: 2 residues × 3 atoms (N, CA, C)
        let chain = vec![
            Vec3::new(0.0, 0.0, 0.0), // res 0 N
            Vec3::new(1.5, 0.0, 0.0), // res 0 CA
            Vec3::new(3.0, 0.0, 0.0), // res 0 C
            Vec3::new(3.8, 0.0, 0.0), // res 1 N
            Vec3::new(5.3, 0.0, 0.0), // res 1 CA
            Vec3::new(6.8, 0.0, 0.0), // res 1 C
        ];

        let mut visual = VisualState::new();
        visual.backbone_chains = vec![chain.clone()];
        visual.sidechain_positions = vec![Vec3::new(1.5, 2.0, 0.0)]; // CB

        let mut atom_index = FxHashMap::default();
        let _ = atom_index.insert((0_u32, "CB".to_owned()), 0_usize);

        let mut topology = SceneTopology::new();
        topology.backbone_chain_offsets = vec![0];
        topology.sidechain_topology = SidechainTopology {
            bonds: vec![],
            hydrophobicity: vec![false],
            residue_indices: vec![0],
            atom_names: vec!["CB".to_owned()],
            target_positions: vec![Vec3::new(1.5, 2.0, 0.0)],
            target_backbone_bonds: vec![],
            atom_index,
        };

        let cached = vec![chain];
        (visual, topology, cached)
    }

    fn make_ref(residue: u32, name: &str) -> AtomRef {
        AtomRef {
            residue,
            atom_name: name.to_owned(),
        }
    }

    #[test]
    fn resolve_ca_from_visual() {
        let (visual, topology, cached) = test_scene();
        let scene = ScenePositions {
            visual: &visual,
            topology: &topology,
            cached_chains: &cached,
        };
        let pos = resolve_atom_ref(&scene, &make_ref(0, "CA"));
        assert_eq!(pos, Some(Vec3::new(1.5, 0.0, 0.0)));
    }

    #[test]
    fn resolve_n_and_c() {
        let (visual, topology, cached) = test_scene();
        let scene = ScenePositions {
            visual: &visual,
            topology: &topology,
            cached_chains: &cached,
        };
        assert_eq!(
            resolve_atom_ref(&scene, &make_ref(0, "N")),
            Some(Vec3::new(0.0, 0.0, 0.0))
        );
        assert_eq!(
            resolve_atom_ref(&scene, &make_ref(0, "C")),
            Some(Vec3::new(3.0, 0.0, 0.0))
        );
    }

    #[test]
    fn resolve_falls_back_to_cached() {
        let (_, topology, cached) = test_scene();
        let empty_visual = VisualState::new(); // empty backbone_chains
        let scene = ScenePositions {
            visual: &empty_visual,
            topology: &topology,
            cached_chains: &cached,
        };
        // Should fall back to cached_chains
        let pos = resolve_atom_ref(&scene, &make_ref(1, "CA"));
        assert_eq!(pos, Some(Vec3::new(5.3, 0.0, 0.0)));
    }

    #[test]
    fn resolve_sidechain_atom() {
        let (visual, topology, cached) = test_scene();
        let scene = ScenePositions {
            visual: &visual,
            topology: &topology,
            cached_chains: &cached,
        };
        let pos = resolve_atom_ref(&scene, &make_ref(0, "CB"));
        assert_eq!(pos, Some(Vec3::new(1.5, 2.0, 0.0)));
    }

    #[test]
    fn resolve_unknown_returns_none() {
        let (visual, topology, cached) = test_scene();
        let scene = ScenePositions {
            visual: &visual,
            topology: &topology,
            cached_chains: &cached,
        };
        assert_eq!(resolve_atom_ref(&scene, &make_ref(0, "XYZ")), None);
    }

    #[test]
    fn resolve_out_of_range_returns_none() {
        let (visual, topology, cached) = test_scene();
        let scene = ScenePositions {
            visual: &visual,
            topology: &topology,
            cached_chains: &cached,
        };
        assert_eq!(resolve_atom_ref(&scene, &make_ref(99, "CA")), None);
    }
}
