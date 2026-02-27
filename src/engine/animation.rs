//! Animation methods for VisoEngine

use std::path::Path;
use std::time::Instant;

use super::trajectory::TrajectoryPlayer;
use super::VisoEngine;
use crate::animation::transition::Transition;
use crate::scene::SceneEntity;

impl VisoEngine {
    /// Tick animation (both trajectory and structural), submitting any
    /// interpolated frame to the background thread.
    ///
    /// Trajectory frames are fed through `animate_entity()` with
    /// `Transition::snap()`, so both paths converge through the
    /// animator's update loop.
    pub(crate) fn tick_animation(&mut self) {
        let now = Instant::now();

        // If a trajectory is active, feed its frame through per-entity
        // animation so it converges with the standard path.
        self.advance_trajectory(now);

        if !self.animator.update(now) {
            return;
        }

        let backbone = self.animator.get_backbone();
        let include_sidechains =
            !self.sc_cache.target_sidechain_positions.is_empty()
                && self.animator.should_include_sidechains();

        self.scene.update_visual_positions(backbone.clone());
        self.submit_animation_frame_with_backbone(backbone, include_sidechains);
    }

    /// Feed the current trajectory frame (if any) through per-entity
    /// animation with `Transition::snap()`.
    fn advance_trajectory(&mut self, now: Instant) {
        let Some(ref mut player) = self.trajectory_player else {
            return;
        };
        let Some(backbone_chains) = player.tick(now) else {
            return;
        };

        let snap = Transition::snap();
        for range in &self.entity_ranges {
            self.animator
                .animate_entity(range, &backbone_chains, &snap, None);
        }
    }

    /// Load a DCD trajectory file and begin playback.
    pub fn load_trajectory(&mut self, path: &Path) {
        use foldit_conv::adapters::dcd::dcd_file_to_frames;
        use foldit_conv::ops::transform::protein_only;

        use super::trajectory::build_backbone_atom_indices;

        let (header, frames) = match dcd_file_to_frames(path) {
            Ok(r) => r,
            Err(e) => {
                log::error!("Failed to load DCD trajectory: {e}");
                return;
            }
        };

        // Get protein coords from the first visible entity to build backbone
        // mapping
        let protein_coords = self
            .scene
            .entities()
            .iter()
            .filter(|e| e.visible)
            .find_map(SceneEntity::protein_coords);

        let protein_coords = if let Some(c) = protein_coords {
            protein_only(&c)
        } else {
            log::error!("No protein structure loaded — cannot play trajectory");
            return;
        };

        // Validate atom count
        if (header.num_atoms as usize) < protein_coords.num_atoms {
            log::error!(
                "DCD atom count ({}) is less than protein atom count ({})",
                header.num_atoms,
                protein_coords.num_atoms,
            );
            return;
        }

        // Build backbone atom index mapping
        let backbone_indices = build_backbone_atom_indices(&protein_coords);

        // Get current backbone chains for topology
        let backbone_chains =
            foldit_conv::ops::transform::extract_backbone_chains(
                &protein_coords,
            );

        let num_atoms = header.num_atoms as usize;
        let num_frames = frames.len();
        let duration_secs = num_frames as f64 / 30.0;

        let player = TrajectoryPlayer::new(
            frames,
            num_atoms,
            &backbone_chains,
            backbone_indices,
        );
        self.trajectory_player = Some(player);

        log::info!(
            "Trajectory loaded: {num_frames} frames, {num_atoms} atoms, \
             ~{duration_secs:.1}s at 30fps",
        );
    }

    /// Toggle trajectory playback (play/pause). No-op if no trajectory loaded.
    pub fn toggle_trajectory(&mut self) {
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
