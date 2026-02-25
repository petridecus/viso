//! Trajectory playback sequencer for DCD molecular dynamics files.
//!
//! Feeds pre-computed frames to the renderer's backbone update path,
//! bypassing the animation interpolation system.

use std::time::{Duration, Instant};

use foldit_conv::adapters::dcd::DcdFrame;
use glam::Vec3;

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
    ///   coordinate arrays. Built by the caller from the structure's `Coords`
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

    /// Set playback speed in frames per second (clamped to >= 0.1).
    pub fn set_fps(&mut self, fps: f32) {
        self.frame_duration =
            Duration::from_secs_f64(1.0 / fps.max(0.1) as f64);
    }

    /// Enable or disable looping at the end of the trajectory.
    pub fn set_looping(&mut self, looping: bool) {
        self.looping = looping;
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

/// Build the backbone atom index mapping from a `Coords` struct.
///
/// For each backbone atom (N, CA, C) that would appear in the backbone_chains
/// produced by `extract_backbone_chains`, returns the corresponding index into
/// the full atom array (i.e. the DCD's flat coordinate index).
///
/// This mirrors the logic in `extract_backbone_chains` from foldit-conv so
/// the mapping is consistent with the chain topology the renderer uses.
pub fn build_backbone_atom_indices(
    coords: &foldit_conv::types::coords::Coords,
) -> Vec<usize> {
    let mut indices = Vec::new();
    let mut last_chain_id: Option<u8> = None;
    let mut last_res_num: Option<i32> = None;
    let mut current_chain_indices: Vec<usize> = Vec::new();

    for i in 0..coords.num_atoms {
        let atom_name = std::str::from_utf8(&coords.atom_names[i])
            .unwrap_or("")
            .trim();

        if atom_name != "N" && atom_name != "CA" && atom_name != "C" {
            continue;
        }

        let chain_id = coords.chain_ids[i];
        let res_num = coords.res_nums[i];

        let is_chain_break = last_chain_id.is_some_and(|c| c != chain_id);
        let is_sequence_gap =
            last_res_num.is_some_and(|r| (res_num - r).abs() > 1);

        if (is_chain_break || is_sequence_gap)
            && !current_chain_indices.is_empty()
        {
            indices.append(&mut current_chain_indices);
        }

        current_chain_indices.push(i);
        last_chain_id = Some(chain_id);

        if atom_name == "CA" {
            last_res_num = Some(res_num);
        }
    }

    if !current_chain_indices.is_empty() {
        indices.append(&mut current_chain_indices);
    }

    indices
}
