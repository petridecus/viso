//! Animation methods for ProteinRenderEngine

use std::path::Path;

use glam::Vec3;

use super::ProteinRenderEngine;
use crate::animation::transition::Transition;
use crate::renderer::geometry::sidechain::SidechainView;
use crate::scene::SceneEntity;
use crate::util::trajectory::TrajectoryPlayer;

impl ProteinRenderEngine {
    /// Load a DCD trajectory file and begin playback.
    pub fn load_trajectory(&mut self, path: &Path) {
        use foldit_conv::adapters::dcd::dcd_file_to_frames;
        use foldit_conv::ops::transform::protein_only;

        use crate::util::trajectory::build_backbone_atom_indices;

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

    /// Animate backbone to new pose with specified transition.
    pub fn animate_to_pose(
        &mut self,
        new_backbone: &[Vec<Vec3>],
        transition: &Transition,
    ) {
        self.animator.set_target(new_backbone, transition);

        // If animator has visual state, update renderers
        if self.animator.residue_count() > 0 {
            let visual_backbone = self.animator.get_backbone();
            self.update_backbone(&visual_backbone);
        }
    }

    /// Capture sidechain start positions for animation preemption.
    fn capture_sidechain_start(&mut self, sidechain: &SidechainView) {
        if self.sc.target_sidechain_positions.len() == sidechain.positions.len()
        {
            if self.animator.is_animating()
                && self.animator.has_sidechain_data()
            {
                self.sc.start_sidechain_positions =
                    self.animator.get_sidechain_positions();
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
                self.sc.start_sidechain_positions =
                    self.sc.target_sidechain_positions.clone();
                self.sc.start_backbone_sidechain_bonds =
                    self.sc.target_backbone_sidechain_bonds.clone();
            }
        } else {
            self.sc.start_sidechain_positions = sidechain.positions.to_vec();
            self.sc.start_backbone_sidechain_bonds =
                sidechain.backbone_bonds.to_vec();
        }
    }

    /// Animate to new pose with sidechain data and explicit transition.
    pub fn animate_to_full_pose(
        &mut self,
        new_backbone: &[Vec<Vec3>],
        sidechain: &SidechainView,
        sidechain_atom_names: &[String],
        transition: &Transition,
    ) {
        self.capture_sidechain_start(sidechain);

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
        let ca_positions =
            foldit_conv::render::backbone::ca_positions_from_chains(
                new_backbone,
            );

        // Pass sidechain data to animator FIRST (before set_target)
        // This allows set_target to detect sidechain changes and force
        // animation even when backbone is unchanged (for Shake/MPNN
        // animations)
        self.animator.set_sidechain_target_with_transition(
            sidechain.positions,
            sidechain.residue_indices,
            &ca_positions,
            Some(transition),
        );

        // Set backbone target (this starts the animation, checking sidechain
        // changes)
        self.animator.set_target(new_backbone, transition);

        // Update renderers with START visual state (animation will interpolate
        // from here)
        if self.animator.residue_count() > 0 {
            let visual_backbone = self.animator.get_backbone();
            self.update_backbone(&visual_backbone);
        }

        // Update sidechain renderer with start positions (adjusted for sheet
        // surface)
        let offset_map = self.sheet_offset_map();
        let raw_view = SidechainView {
            positions: &self.sc.start_sidechain_positions,
            bonds: sidechain.bonds,
            backbone_bonds: &self.sc.start_backbone_sidechain_bonds,
            hydrophobicity: sidechain.hydrophobicity,
            residue_indices: sidechain.residue_indices,
        };
        let adjusted = crate::util::sheet_adjust::sheet_adjusted_view(
            &raw_view,
            &offset_map,
        );
        self.renderers.sidechain.update(
            &self.context.device,
            &self.context.queue,
            &adjusted.as_view(),
        );
    }
}
