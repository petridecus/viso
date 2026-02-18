//! Animation methods for ProteinRenderEngine

use super::ProteinRenderEngine;
use crate::animation::AnimationAction;
use crate::renderer::molecular::capsule_sidechain::SidechainData;
use crate::util::trajectory::TrajectoryPlayer;
use glam::Vec3;
use std::path::Path;

impl ProteinRenderEngine {
    /// Load a DCD trajectory file and begin playback.
    pub fn load_trajectory(&mut self, path: &Path) {
        use crate::util::trajectory::build_backbone_atom_indices;
        use foldit_conv::coords::{dcd_file_to_frames, protein_only};

        let (header, frames) = match dcd_file_to_frames(path) {
            Ok(r) => r,
            Err(e) => {
                log::error!("Failed to load DCD trajectory: {e}");
                return;
            }
        };

        // Get protein coords from the first visible group to build backbone mapping
        let protein_coords = self
            .scene
            .iter()
            .filter(|g| g.visible)
            .find_map(|g| g.protein_coords());

        let protein_coords = match protein_coords {
            Some(c) => protein_only(&c),
            None => {
                log::error!(
                    "No protein structure loaded — cannot play trajectory"
                );
                return;
            }
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
            foldit_conv::coords::extract_backbone_chains(&protein_coords);

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
            "Trajectory loaded: {} frames, {} atoms, ~{:.1}s at 30fps",
            num_frames,
            num_atoms,
            duration_secs,
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

    /// Animate backbone to new pose with specified action.
    pub fn animate_to_pose(
        &mut self,
        new_backbone: &[Vec<Vec3>],
        action: AnimationAction,
    ) {
        self.animator.set_target(new_backbone, action);

        // If animator has visual state, update renderers
        if self.animator.residue_count() > 0 {
            let visual_backbone = self.animator.get_backbone();
            self.update_backbone(&visual_backbone);
        }
    }

    /// Animate to new pose with sidechain data.
    ///
    /// Uses AnimationAction::Wiggle by default for backwards compatibility.
    pub fn animate_to_full_pose(
        &mut self,
        new_backbone: &[Vec<Vec3>],
        sidechain: &SidechainData,
        sidechain_atom_names: &[String],
    ) {
        self.animate_to_full_pose_with_action(
            new_backbone,
            sidechain,
            sidechain_atom_names,
            AnimationAction::Wiggle,
        );
    }

    /// Animate to new pose with sidechain data and explicit action.
    pub fn animate_to_full_pose_with_action(
        &mut self,
        new_backbone: &[Vec<Vec3>],
        sidechain: &SidechainData,
        sidechain_atom_names: &[String],
        action: AnimationAction,
    ) {
        // Capture current VISUAL positions as start (for smooth preemption)
        // If animation is in progress, use interpolated positions, not old targets
        if self.sc.target_sidechain_positions.len() == sidechain.positions.len()
        {
            if self.animator.is_animating()
                && self.animator.has_sidechain_data()
            {
                // Animation in progress - sync to current visual state (like backbone does)
                self.sc.start_sidechain_positions =
                    self.animator.get_sidechain_positions();
                // Also interpolate backbone-sidechain bonds
                let ctx = self.animator.interpolation_context();
                self.sc.start_backbone_sidechain_bonds = self
                    .sc
                    .start_backbone_sidechain_bonds
                    .iter()
                    .zip(self.sc.target_backbone_sidechain_bonds.iter())
                    .map(|((start_pos, idx), (target_pos, _))| {
                        let pos = *start_pos
                            + (*target_pos - *start_pos) * ctx.eased_t;
                        (pos, *idx)
                    })
                    .collect();
            } else {
                // No animation - use previous target as new start
                self.sc.start_sidechain_positions =
                    self.sc.target_sidechain_positions.clone();
                self.sc.start_backbone_sidechain_bonds =
                    self.sc.target_backbone_sidechain_bonds.clone();
            }
        } else {
            // Size changed - snap to new positions
            self.sc.start_sidechain_positions = sidechain.positions.to_vec();
            self.sc.start_backbone_sidechain_bonds =
                sidechain.backbone_bonds.to_vec();
        }

        // Set new targets and cached data
        self.sc.target_sidechain_positions = sidechain.positions.to_vec();
        self.sc.target_backbone_sidechain_bonds =
            sidechain.backbone_bonds.to_vec();
        self.sc.cached_sidechain_bonds = sidechain.bonds.to_vec();
        self.sc.cached_sidechain_hydrophobicity =
            sidechain.hydrophobicity.to_vec();
        self.sc.cached_sidechain_residue_indices =
            sidechain.residue_indices.to_vec();
        self.sc.cached_sidechain_atom_names = sidechain_atom_names.to_vec();

        // Extract CA positions from backbone for sidechain collapse animation
        // CA is the second atom (index 1) in each group of 3 (N, CA, C) per residue
        let ca_positions: Vec<Vec3> = new_backbone
            .iter()
            .flat_map(|chain| {
                chain.chunks(3).filter_map(|chunk| chunk.get(1).copied())
            })
            .collect();

        // Pass sidechain data to animator FIRST (before set_target)
        // This allows set_target to detect sidechain changes and force animation
        // even when backbone is unchanged (for Shake/MPNN animations)
        self.animator.set_sidechain_target_with_action(
            sidechain.positions,
            sidechain.residue_indices,
            &ca_positions,
            Some(action),
        );

        // Set backbone target (this starts the animation, checking sidechain changes)
        self.animator.set_target(new_backbone, action);

        // Update renderers with START visual state (animation will interpolate from here)
        if self.animator.residue_count() > 0 {
            let visual_backbone = self.animator.get_backbone();
            self.update_backbone(&visual_backbone);
        }

        // Update sidechain renderer with start positions (adjusted for sheet surface)
        let offset_map = self.sheet_offset_map();
        let adjusted_positions =
            crate::util::sheet_adjust::adjust_sidechains_for_sheet(
                &self.sc.start_sidechain_positions,
                sidechain.residue_indices,
                &offset_map,
            );
        let adjusted_bonds = crate::util::sheet_adjust::adjust_bonds_for_sheet(
            &self.sc.start_backbone_sidechain_bonds,
            sidechain.residue_indices,
            &offset_map,
        );
        self.sidechain_renderer.update(
            &self.context.device,
            &self.context.queue,
            &SidechainData {
                positions: &adjusted_positions,
                bonds: sidechain.bonds,
                backbone_bonds: &adjusted_bonds,
                hydrophobicity: sidechain.hydrophobicity,
                residue_indices: sidechain.residue_indices,
            },
        );
    }

    /// Skip all animations to final state.
    pub fn skip_animations(&mut self) {
        self.animator.skip();
    }

    /// Cancel all animations.
    pub fn cancel_animations(&mut self) {
        self.animator.cancel();
    }

    /// Set animation enabled/disabled.
    pub fn set_animation_enabled(&mut self, enabled: bool) {
        self.animator.set_enabled(enabled);
    }
}
