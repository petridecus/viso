//! Trajectory playback sequencer for DCD molecular dynamics files.
//!
//! Feeds pre-computed frames to the renderer's backbone update path,
//! bypassing the animation interpolation system.

use std::time::Duration;

use glam::Vec3;
use molex::adapters::dcd::DcdFrame;
use molex::entity::molecule::protein::ProteinEntity;
use web_time::Instant;

/// Minimal frame sequencer — auto-advances with configurable speed.
pub struct TrajectoryPlayer {
    frames: Vec<DcdFrame>,
    num_atoms: usize,
    current_frame: usize,
    last_advance: Instant,
    frame_duration: Duration,
    playing: bool,
    looping: bool,
    /// Maps each backbone position (flat index across all chains) to a DCD
    /// atom index.
    backbone_map: Vec<usize>,
    /// Length of each chain in the backbone_chains layout.
    chain_lengths: Vec<usize>,
}

impl TrajectoryPlayer {
    /// Trajectory player over pre-loaded DCD frames.
    ///
    /// - `frames`: all DCD frames (loaded into memory)
    /// - `num_atoms`: atom count per frame (from DCD header)
    /// - `backbone_chains`: current backbone chain layout (used for chain
    ///   topology)
    /// - `backbone_atom_indices`: for each position in the flattened
    ///   backbone_chains, the corresponding atom index in the DCD's flat
    ///   coordinate arrays. Built by the caller from the entity's residue
    ///   metadata.
    pub fn new(
        frames: Vec<DcdFrame>,
        num_atoms: usize,
        backbone_chains: &[Vec<Vec3>],
        backbone_atom_indices: Vec<usize>,
    ) -> Self {
        let chain_lengths: Vec<usize> =
            backbone_chains.iter().map(Vec::len).collect();
        Self {
            frames,
            num_atoms,
            current_frame: 0,
            last_advance: Instant::now(),
            frame_duration: Duration::from_secs_f64(1.0 / 30.0),
            playing: true,
            looping: true,
            backbone_map: backbone_atom_indices,
            chain_lengths,
        }
    }

    /// Advance time and return new backbone chains if a frame step occurred.
    pub fn tick(&mut self, now: Instant) -> Option<Vec<Vec<Vec3>>> {
        if !self.playing || self.frames.is_empty() {
            return None;
        }

        if now.duration_since(self.last_advance) < self.frame_duration {
            return None;
        }

        self.last_advance = now;

        // Advance frame
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

        Some(self.frame_to_backbone(&self.frames[self.current_frame]))
    }

    /// Convert a DCD frame to backbone chains using the prebuilt mapping.
    fn frame_to_backbone(&self, frame: &DcdFrame) -> Vec<Vec<Vec3>> {
        let mut chains: Vec<Vec<Vec3>> = self
            .chain_lengths
            .iter()
            .map(|&len| Vec::with_capacity(len))
            .collect();

        let mut flat_idx = 0;
        for (chain_idx, &chain_len) in self.chain_lengths.iter().enumerate() {
            for _ in 0..chain_len {
                let dcd_idx = self.backbone_map[flat_idx];
                let pos = if dcd_idx < self.num_atoms {
                    Vec3::new(
                        frame.x[dcd_idx],
                        frame.y[dcd_idx],
                        frame.z[dcd_idx],
                    )
                } else {
                    Vec3::ZERO
                };
                chains[chain_idx].push(pos);
                flat_idx += 1;
            }
        }

        chains
    }

    /// Toggle between playing and paused states.
    pub fn toggle_playback(&mut self) {
        self.playing = !self.playing;
        if self.playing {
            // Reset advance timer so we don't immediately skip frames
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
// VisoEngine trajectory loading
// ---------------------------------------------------------------------------

impl super::VisoEngine {
    /// Load a DCD trajectory file and begin playback.
    pub fn load_trajectory(&mut self, path: &std::path::Path) {
        use molex::adapters::dcd::dcd_file_to_frames;
        use molex::MoleculeType;

        let (header, frames) = match dcd_file_to_frames(path) {
            Ok(r) => r,
            Err(e) => {
                log::error!("Failed to load DCD trajectory: {e}");
                return;
            }
        };

        // Get the first visible protein entity directly
        let protein = self
            .entities
            .entities()
            .iter()
            .filter(|e| e.visible)
            .find_map(|e| {
                if e.entity.molecule_type() == MoleculeType::Protein {
                    e.entity.as_protein()
                } else {
                    None
                }
            });

        let Some(protein) = protein else {
            log::error!("No protein structure loaded — cannot play trajectory");
            return;
        };

        // Validate atom count
        let protein_atom_count = protein.atoms.len();
        if (header.num_atoms as usize) < protein_atom_count {
            log::error!(
                "DCD atom count ({}) is less than protein atom count ({})",
                header.num_atoms,
                protein_atom_count,
            );
            return;
        }

        // Build backbone atom index mapping from protein residues
        let backbone_indices = build_backbone_atom_indices(protein);

        // Get current backbone chains for topology
        let entities_slice: Vec<molex::MoleculeEntity> = self
            .entities
            .entities()
            .iter()
            .filter(|e| e.visible && e.is_protein())
            .map(|e| e.entity.clone())
            .collect();
        let backbone_chains =
            molex::ops::transform::extract_backbone_segments(&entities_slice);

        let num_atoms = header.num_atoms as usize;
        self.animation.load_trajectory(
            frames,
            num_atoms,
            &backbone_chains,
            backbone_indices,
        );
    }
}

/// Build the backbone atom index mapping from a [`ProteinEntity`].
///
/// For each backbone atom (N, CA, C) that would appear in the backbone_chains
/// produced by `extract_backbone_segments`, returns the corresponding index
/// into the full atom array (i.e. the DCD's flat coordinate index).
///
/// This mirrors the logic in `extract_backbone_segments` from molex so
/// the mapping is consistent with the chain topology the renderer uses.
pub fn build_backbone_atom_indices(protein: &ProteinEntity) -> Vec<usize> {
    let mut indices = Vec::new();
    let mut last_res_num: Option<i32> = None;
    let mut current_chain_indices: Vec<usize> = Vec::new();

    for res in &protein.residues {
        let atoms_in_res = &protein.atoms[res.atom_range.clone()];

        // Find N, CA, C atom indices (global, into the entity atom array)
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

        // Skip residues missing backbone atoms
        let (Some(n), Some(ca), Some(c)) = (n_idx, ca_idx, c_idx) else {
            continue;
        };

        // Detect sequence gaps (chain breaks are handled by segment_breaks
        // in the protein entity, but for DCD mapping we just check residue
        // number continuity)
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
