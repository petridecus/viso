//! DCD trajectory playback.
//!
//! A [`TrajectoryPlayer`] is scoped to a single entity. Its DCD stream
//! is mapped to that entity's atom indices at load time, and
//! [`TrajectoryPlayer::tick`] returns a [`TrajectoryFrame`] containing
//! the per-atom position updates the engine applies to
//! [`crate::engine::positions::EntityPositions`] for that entity.

use std::time::Duration;

use glam::Vec3;
use molex::adapters::dcd::{dcd_file_to_frames, DcdFrame};
use molex::entity::molecule::id::EntityId;
use molex::entity::molecule::protein::ProteinEntity;
use molex::MoleculeType;
use web_time::Instant;

/// Per-entity trajectory frame update: which entity-local atom
/// indices to overwrite and the new [`Vec3`] at each.
pub(crate) struct TrajectoryFrame {
    /// The entity this frame belongs to.
    pub entity: EntityId,
    /// Replacement positions — parallel to `atom_indices`.
    pub positions: Vec<Vec3>,
    /// Entity-local atom indices to overwrite (parallel to
    /// `positions`). An atom not in this list keeps its previous
    /// position.
    pub atom_indices: Vec<u32>,
}

/// Auto-advancing DCD frame sequencer for a single entity.
pub struct TrajectoryPlayer {
    frames: Vec<DcdFrame>,
    num_atoms: usize,
    current_frame: usize,
    last_advance: Instant,
    frame_duration: Duration,
    playing: bool,
    looping: bool,
    entity: EntityId,
    /// For each entity-local atom the trajectory drives, the
    /// corresponding DCD atom index.
    atom_index_map: Vec<usize>,
}

impl TrajectoryPlayer {
    /// New player over pre-loaded DCD frames.
    pub fn new(
        frames: Vec<DcdFrame>,
        num_atoms: usize,
        entity: EntityId,
        atom_index_map: Vec<usize>,
    ) -> Self {
        Self {
            frames,
            num_atoms,
            current_frame: 0,
            last_advance: Instant::now(),
            frame_duration: Duration::from_secs_f64(1.0 / 30.0),
            playing: true,
            looping: true,
            entity,
            atom_index_map,
        }
    }

    /// Advance time and return the next frame's per-entity update.
    pub fn tick(&mut self, now: Instant) -> Option<TrajectoryFrame> {
        if !self.playing || self.frames.is_empty() {
            return None;
        }
        if now.duration_since(self.last_advance) < self.frame_duration {
            return None;
        }
        self.last_advance = now;

        let next = self.current_frame + 1;
        if next >= self.frames.len() {
            if self.looping {
                self.current_frame = 0;
            } else {
                self.playing = false;
                return None;
            }
        } else {
            self.current_frame = next;
        }

        let frame = &self.frames[self.current_frame];
        let positions = self
            .atom_index_map
            .iter()
            .map(|&dcd_idx| {
                if dcd_idx < self.num_atoms {
                    Vec3::new(
                        frame.x[dcd_idx],
                        frame.y[dcd_idx],
                        frame.z[dcd_idx],
                    )
                } else {
                    Vec3::ZERO
                }
            })
            .collect();
        let atom_indices: Vec<u32> =
            (0..self.atom_index_map.len() as u32).collect();

        Some(TrajectoryFrame {
            entity: self.entity,
            positions,
            atom_indices,
        })
    }

    /// Toggle between playing and paused states.
    pub fn toggle_playback(&mut self) {
        self.playing = !self.playing;
        if self.playing {
            self.last_advance = Instant::now();
        }
    }

    /// Index of the current frame.
    pub fn current_frame(&self) -> usize {
        self.current_frame
    }

    /// Total number of frames in the trajectory.
    pub fn total_frames(&self) -> usize {
        self.frames.len()
    }

    /// Whether the player is currently advancing frames.
    pub fn is_playing(&self) -> bool {
        self.playing
    }
}

// ---------------------------------------------------------------------------
// Engine-level loader
// ---------------------------------------------------------------------------

impl super::VisoEngine {
    /// Load a DCD trajectory file and begin playback against the first
    /// visible protein entity.
    pub fn load_trajectory(&mut self, path: &std::path::Path) {
        let (header, frames) = match dcd_file_to_frames(path) {
            Ok(r) => r,
            Err(e) => {
                log::error!("Failed to load DCD trajectory: {e}");
                return;
            }
        };

        let Some((entity_id, atom_count, atom_index_map)) =
            self.pick_trajectory_target()
        else {
            log::error!("No protein structure loaded — cannot play trajectory");
            return;
        };

        if (header.num_atoms as usize) < atom_count {
            log::error!(
                "DCD atom count ({}) is less than protein atom count ({})",
                header.num_atoms,
                atom_count,
            );
            return;
        }

        let num_atoms = header.num_atoms as usize;
        let num_frames = frames.len();
        let player =
            TrajectoryPlayer::new(frames, num_atoms, entity_id, atom_index_map);
        self.animation
            .load_trajectory(player, num_frames, num_atoms);
    }

    /// Pick the first visible protein entity as the trajectory target.
    fn pick_trajectory_target(&self) -> Option<(EntityId, usize, Vec<usize>)> {
        let entity = self.scene.current.entities().iter().find(|e| {
            self.is_entity_visible(e.id().raw())
                && e.molecule_type() == MoleculeType::Protein
        })?;
        let protein = entity.as_protein()?;
        let indices = build_backbone_atom_indices(protein);
        Some((entity.id(), protein.atoms.len(), indices))
    }
}

/// Build the backbone atom index mapping from a [`ProteinEntity`].
///
/// For each backbone atom (N, CA, C) that `extract_backbone_segments`
/// produces, returns the corresponding index into the flat atom array
/// (= DCD index, since DCD matches atom-array order for this entity).
pub(crate) fn build_backbone_atom_indices(
    protein: &ProteinEntity,
) -> Vec<usize> {
    let mut indices = Vec::new();
    let mut last_res_num: Option<i32> = None;
    let mut current_chain_indices: Vec<usize> = Vec::new();

    for res in &protein.residues {
        let atoms_in_res = &protein.atoms[res.atom_range.clone()];
        let mut n_idx = None;
        let mut ca_idx = None;
        let mut c_idx = None;

        for (local_offset, atom) in atoms_in_res.iter().enumerate() {
            let global_idx = res.atom_range.start + local_offset;
            let name = std::str::from_utf8(&atom.name).unwrap_or("").trim();
            match name {
                "N" => n_idx = Some(global_idx),
                "CA" => ca_idx = Some(global_idx),
                "C" => c_idx = Some(global_idx),
                _ => {}
            }
        }

        let (Some(n), Some(ca), Some(c)) = (n_idx, ca_idx, c_idx) else {
            continue;
        };

        let is_sequence_gap =
            last_res_num.is_some_and(|r| (res.number - r).abs() > 1);
        if is_sequence_gap && !current_chain_indices.is_empty() {
            indices.append(&mut current_chain_indices);
        }

        current_chain_indices.push(n);
        current_chain_indices.push(ca);
        current_chain_indices.push(c);
        last_res_num = Some(res.number);
    }

    if !current_chain_indices.is_empty() {
        indices.append(&mut current_chain_indices);
    }

    indices
}
