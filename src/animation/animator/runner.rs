//! Animation runner executes a single animation.

use std::time::{Duration, Instant};

use super::state::StructureState;
use crate::animation::behaviors::{ResidueVisualState, SharedBehavior};

/// Data for animating a single residue.
#[derive(Debug, Clone)]
pub struct ResidueAnimationData {
    /// Global residue index.
    pub residue_idx: usize,
    /// Start state for this animation.
    pub start: ResidueVisualState,
    /// Target state for this animation.
    pub target: ResidueVisualState,
}

/// Executes a single animation from start to target states.
///
/// The runner holds:
/// - The behavior being used
/// - Per-residue start/target states
/// - Timing information
pub struct AnimationRunner {
    /// When the animation started.
    start_time: Instant,
    /// Behavior being used.
    behavior: SharedBehavior,
    /// Per-residue animation data.
    residues: Vec<ResidueAnimationData>,
}

impl AnimationRunner {
    /// Start a new animation with the given behavior and residue data.
    pub fn new(
        behavior: SharedBehavior,
        residues: Vec<ResidueAnimationData>,
    ) -> Self {
        Self {
            start_time: Instant::now(),
            behavior,
            residues,
        }
    }

    /// Create with explicit start time (for testing).
    #[allow(dead_code)]
    pub fn with_start_time(
        start_time: Instant,
        behavior: SharedBehavior,
        residues: Vec<ResidueAnimationData>,
    ) -> Self {
        Self {
            start_time,
            behavior,
            residues,
        }
    }

    /// Get the animation behavior.
    pub fn behavior(&self) -> &SharedBehavior {
        &self.behavior
    }

    /// Get the total animation duration.
    #[allow(dead_code)] // public API, not yet called internally
    pub fn duration(&self) -> Duration {
        self.behavior.duration()
    }

    /// Calculate normalized progress (0.0 to 1.0).
    pub fn progress(&self, now: Instant) -> f32 {
        let elapsed = now.saturating_duration_since(self.start_time);
        let duration = self.behavior.duration();

        if duration.is_zero() {
            1.0
        } else {
            (elapsed.as_secs_f32() / duration.as_secs_f32()).min(1.0)
        }
    }

    /// Whether the animation has reached completion.
    #[allow(dead_code)] // public API, not yet called internally
    pub fn is_complete(&self, now: Instant) -> bool {
        self.progress(now) >= 1.0
    }

    /// Compute visual state for a specific residue at given progress.
    pub fn compute_residue_state(
        &self,
        data: &ResidueAnimationData,
        t: f32,
    ) -> ResidueVisualState {
        self.behavior.compute_state(t, &data.start, &data.target)
    }

    /// Apply interpolated states to a StructureState using pre-computed
    /// progress.
    pub fn apply_to_state(&self, state: &mut StructureState, t: f32) {
        for data in &self.residues {
            let visual = self.compute_residue_state(data, t);
            state.set_current(data.residue_idx, visual);
        }
    }

    /// Remove residues whose global index falls within any of the given ranges.
    /// Used to exclude non-targeted entity residues from animation.
    pub fn remove_residue_ranges(&mut self, ranges: &[(usize, usize)]) {
        self.residues.retain(|data| {
            !ranges.iter().any(|&(start, end)| {
                data.residue_idx >= start && data.residue_idx < end
            })
        });
    }
}

impl std::fmt::Debug for AnimationRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnimationRunner")
            .field("behavior", &self.behavior.name())
            .field("residue_count", &self.residues.len())
            .field("duration", &self.behavior.duration())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::animation::behaviors::{shared, SmoothInterpolation, Snap};

    fn make_residue_data(
        idx: usize,
        start_y: f32,
        end_y: f32,
    ) -> ResidueAnimationData {
        ResidueAnimationData {
            residue_idx: idx,
            start: ResidueVisualState::backbone_only([
                Vec3::new(0.0, start_y, 0.0),
                Vec3::new(1.0, start_y, 0.0),
                Vec3::new(2.0, start_y, 0.0),
            ]),
            target: ResidueVisualState::backbone_only([
                Vec3::new(0.0, end_y, 0.0),
                Vec3::new(1.0, end_y, 0.0),
                Vec3::new(2.0, end_y, 0.0),
            ]),
        }
    }

    #[test]
    fn test_runner_progress() {
        let behavior =
            shared(SmoothInterpolation::linear(Duration::from_millis(100)));
        let residues = vec![make_residue_data(0, 0.0, 10.0)];
        let start = Instant::now();
        let runner =
            AnimationRunner::with_start_time(start, behavior, residues);

        // At start
        assert!((runner.progress(start) - 0.0).abs() < 0.01);

        // At midpoint
        let mid = start + Duration::from_millis(50);
        assert!((runner.progress(mid) - 0.5).abs() < 0.01);

        // At end
        let end = start + Duration::from_millis(100);
        assert!((runner.progress(end) - 1.0).abs() < 0.01);

        // Past end (clamped)
        let past = start + Duration::from_millis(200);
        assert!((runner.progress(past) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_runner_compute_state() {
        let behavior =
            shared(SmoothInterpolation::linear(Duration::from_millis(100)));
        let residues = vec![make_residue_data(0, 0.0, 10.0)];
        let start = Instant::now();
        let runner =
            AnimationRunner::with_start_time(start, behavior, residues);

        // At t=0.5 (50ms of 100ms total), should be at midpoint
        let state = runner.compute_residue_state(&runner.residues[0], 0.5);

        // Should be at midpoint (y=5)
        assert!((state.backbone[0].y - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_runner_snap_behavior() {
        let behavior = shared(Snap);
        let residues = vec![make_residue_data(0, 0.0, 10.0)];
        let runner = AnimationRunner::new(behavior, residues);

        // Snap should be instant
        assert!(runner.is_complete(Instant::now()));
        assert_eq!(runner.duration(), Duration::ZERO);
    }

    #[test]
    fn test_runner_is_complete() {
        let behavior =
            shared(SmoothInterpolation::linear(Duration::from_millis(100)));
        let residues = vec![make_residue_data(0, 0.0, 10.0)];
        let start = Instant::now();
        let runner =
            AnimationRunner::with_start_time(start, behavior, residues);

        assert!(!runner.is_complete(start));
        assert!(!runner.is_complete(start + Duration::from_millis(50)));
        assert!(runner.is_complete(start + Duration::from_millis(100)));
        assert!(runner.is_complete(start + Duration::from_millis(200)));
    }
}
