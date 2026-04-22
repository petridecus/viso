//! Grouped animation state: structural animator, trajectory player,
//! and pending per-entity transitions.

use std::collections::HashMap;

use web_time::Instant;

use super::StructureAnimator;
use crate::animation::transition::Transition;
use crate::engine::positions::EntityPositions;
use crate::engine::trajectory::{TrajectoryFrame, TrajectoryPlayer};

/// Grouped animation fields.
pub(crate) struct AnimationState {
    /// Per-entity interpolation runner.
    pub animator: StructureAnimator,
    /// Multi-frame trajectory player, if loaded.
    pub trajectory_player: Option<TrajectoryPlayer>,
    /// Transitions pending from the last shim-driven mutation, keyed on
    /// raw entity id. Consumed by the engine's sync pipeline when the
    /// new snapshot arrives.
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

    /// Tick the structural animator. Returns `true` if any positions
    /// were written.
    pub(crate) fn tick(
        &mut self,
        now: Instant,
        positions: &mut EntityPositions,
    ) -> bool {
        self.animator.update(now, positions)
    }

    /// Advance the trajectory player, returning the per-entity frame
    /// update (if any).
    pub(crate) fn advance_trajectory(
        &mut self,
        now: Instant,
    ) -> Option<TrajectoryFrame> {
        let player = self.trajectory_player.as_mut()?;
        player.tick(now)
    }

    /// Create a trajectory player from parsed DCD frames.
    pub(crate) fn load_trajectory(
        &mut self,
        player: TrajectoryPlayer,
        num_frames: usize,
        num_atoms: usize,
    ) {
        let duration_secs = num_frames as f64 / 30.0;
        self.trajectory_player = Some(player);
        log::info!(
            "Trajectory loaded: {num_frames} frames, {num_atoms} atoms, \
             ~{duration_secs:.1}s at 30fps",
        );
    }

    /// Toggle trajectory playback (play/pause). No-op if no trajectory
    /// loaded.
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
