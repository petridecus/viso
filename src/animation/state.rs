//! Grouped animation state: structural animator, trajectory player, and
//! pending per-entity transitions.

use std::collections::HashMap;
use web_time::Instant;

use glam::Vec3;
use molex::adapters::dcd::DcdFrame;

use super::{AnimationFrame, EntitySidechainData, StructureAnimator};
use crate::animation::transition::Transition;
use crate::engine::scene::SidechainTopology;
use crate::engine::scene_data::{self, EntityResidueRange};
use crate::engine::trajectory::TrajectoryPlayer;

/// Grouped animation fields: structural animator, trajectory player, and
/// pending per-entity transitions.
pub(crate) struct AnimationState {
    /// Structural animation driver (per-entity interpolation).
    pub animator: StructureAnimator,
    /// Multi-frame trajectory player, if loaded.
    pub trajectory_player: Option<TrajectoryPlayer>,
    /// Transitions pending from the last sync (avoids round-trip through
    /// the background thread). Empty when no sync is pending.
    pub pending_transitions: HashMap<u32, Transition>,
}

impl AnimationState {
    /// Create a new `AnimationState` with default values.
    pub(crate) fn new() -> Self {
        Self {
            animator: StructureAnimator::new(),
            trajectory_player: None,
            pending_transitions: HashMap::new(),
        }
    }

    /// Tick the structural animator and return the current frame if it
    /// produced any interpolated output.
    pub(crate) fn tick(&mut self, now: Instant) -> Option<AnimationFrame> {
        if !self.animator.update(now) {
            return None;
        }
        Some(self.animator.get_frame())
    }

    /// Feed the current trajectory frame (if any) through per-entity
    /// animation with `Transition::snap()`.
    pub(crate) fn advance_trajectory(
        &mut self,
        now: Instant,
        entity_residue_ranges: &[EntityResidueRange],
    ) {
        let Some(ref mut player) = self.trajectory_player else {
            return;
        };
        let Some(backbone_chains) = player.tick(now) else {
            return;
        };

        let snap = Transition::snap();
        for range in entity_residue_ranges {
            self.animator.animate_entity(
                range,
                &backbone_chains,
                &snap,
                EntitySidechainData {
                    positions: None,
                    backbone_bonds: Vec::new(),
                },
            );
        }
    }

    /// Set up per-entity animation from prepared scene data.
    ///
    /// For each entity with a transition, dispatches to
    /// `animator.animate_entity()` so each entity gets its own runner.
    /// Returns the resulting animation frame.
    pub(crate) fn setup_per_entity(
        &mut self,
        entity_transitions: &HashMap<u32, Transition>,
        backbone_chains: &[Vec<Vec3>],
        sidechain_topology: &SidechainTopology,
        entity_residue_ranges: &[EntityResidueRange],
    ) -> AnimationFrame {
        let ca_positions =
            molex::render::backbone::ca_positions_from_chains(backbone_chains);
        let sidechain_positions = sidechain_topology.target_positions.clone();
        let sidechain_residue_indices =
            sidechain_topology.residue_indices.clone();
        let sidechain_backbone_bonds =
            sidechain_topology.target_backbone_bonds.clone();

        self.animator
            .set_sidechain_residue_indices(sidechain_residue_indices.clone());

        for range in entity_residue_ranges {
            let transition = entity_transitions
                .get(&range.entity_id)
                .cloned()
                .unwrap_or_default();

            let positions = scene_data::extract_entity_sidechain(
                &sidechain_positions,
                &sidechain_residue_indices,
                &ca_positions,
                range,
                entity_transitions.get(&range.entity_id),
            );

            let backbone_bonds = scene_data::extract_entity_backbone_bonds(
                &sidechain_backbone_bonds,
                &sidechain_residue_indices,
                range,
            );

            self.animator.animate_entity(
                range,
                backbone_chains,
                &transition,
                EntitySidechainData {
                    positions,
                    backbone_bonds,
                },
            );
        }

        self.animator.get_frame()
    }

    /// Create a trajectory player from parsed DCD frames.
    pub(crate) fn load_trajectory(
        &mut self,
        frames: Vec<DcdFrame>,
        num_atoms: usize,
        backbone_chains: &[Vec<Vec3>],
        backbone_indices: Vec<usize>,
    ) {
        let num_frames = frames.len();
        let duration_secs = num_frames as f64 / 30.0;

        let player = TrajectoryPlayer::new(
            frames,
            num_atoms,
            backbone_chains,
            backbone_indices,
        );
        self.trajectory_player = Some(player);

        log::info!(
            "Trajectory loaded: {num_frames} frames, {num_atoms} atoms, \
             ~{duration_secs:.1}s at 30fps",
        );
    }

    /// Toggle trajectory playback (play/pause). No-op if no trajectory loaded.
    pub(crate) fn toggle_trajectory(&mut self) {
        if let Some(ref mut player) = self.trajectory_player {
            player.toggle_playback();
            let state = if player.is_playing() {
                "playing"
            } else {
                "paused"
            };
            log::info!(
                "Trajectory {state} (frame {}/{})",
                player.current_frame(),
                player.total_frames()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_has_no_trajectory() {
        let state = AnimationState::new();
        assert!(state.trajectory_player.is_none());
    }

    #[test]
    fn pending_transitions_lifecycle() {
        let mut state = AnimationState::new();
        assert!(state.pending_transitions.is_empty());
        let _ = state.pending_transitions.insert(0, Transition::snap());
        assert!(!state.pending_transitions.is_empty());
        state.pending_transitions.clear();
        assert!(state.pending_transitions.is_empty());
    }
}
