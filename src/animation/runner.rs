//! Animation runner executes a single animation.

use std::time::{Duration, Instant};

use glam::Vec3;

use super::transition::{AnimationPhase, Transition};

/// Per-entity sidechain animation positions.
///
/// Start and target positions for a single entity's sidechain atoms,
/// lerped with the same `eased_t` as the backbone.
pub struct SidechainAnimPositions {
    /// Start sidechain atom positions.
    pub(crate) start: Vec<Vec3>,
    /// Target sidechain atom positions.
    pub(crate) target: Vec<Vec3>,
}

/// The visual state of a residue at a point in time.
#[derive(Debug, Clone, Copy)]
pub struct ResidueVisualState {
    /// Backbone atom positions: N, CA, C
    pub backbone: [Vec3; 3],
}

impl ResidueVisualState {
    /// Residue state from backbone atom positions.
    pub fn new(backbone: [Vec3; 3]) -> Self {
        Self { backbone }
    }

    /// Linear interpolation between two states.
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let backbone = [
            lerp_vec3(t, self.backbone[0], other.backbone[0]),
            lerp_vec3(t, self.backbone[1], other.backbone[1]),
            lerp_vec3(t, self.backbone[2], other.backbone[2]),
        ];
        Self { backbone }
    }
}

/// Linear interpolation between two Vec3 positions.
#[inline]
pub fn lerp_vec3(t: f32, start: Vec3, end: Vec3) -> Vec3 {
    start + (end - start) * t
}

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
/// - Animation phases (easing, duration, lerp range per phase)
/// - Per-residue start/target backbone states
/// - Optional sidechain atom positions (lerped with the same eased_t)
/// - Timing information
pub struct AnimationRunner {
    /// When the animation started.
    start_time: Instant,
    /// Animation phases.
    phases: Vec<AnimationPhase>,
    /// Total duration across all phases.
    total_duration: Duration,
    /// Debug name.
    name: &'static str,
    /// Per-residue animation data (backbone).
    residues: Vec<ResidueAnimationData>,
    /// Optional sidechain atom start/target positions.
    sidechain: Option<SidechainAnimPositions>,
}

impl AnimationRunner {
    /// Start a new animation from the given transition.
    pub fn new(
        transition: &Transition,
        residues: Vec<ResidueAnimationData>,
        sidechain: Option<SidechainAnimPositions>,
    ) -> Self {
        Self {
            start_time: Instant::now(),
            phases: transition.phases.clone(),
            total_duration: transition.total_duration(),
            name: transition.name,
            residues,
            sidechain,
        }
    }

    /// Create with explicit start time (for testing).
    #[cfg(test)]
    pub fn with_start_time(
        start_time: Instant,
        transition: &Transition,
        residues: Vec<ResidueAnimationData>,
        sidechain: Option<SidechainAnimPositions>,
    ) -> Self {
        Self {
            start_time,
            phases: transition.phases.clone(),
            total_duration: transition.total_duration(),
            name: transition.name,
            residues,
            sidechain,
        }
    }

    /// Get the total animation duration.
    #[allow(dead_code)] // public API, not yet called internally
    pub fn duration(&self) -> Duration {
        self.total_duration
    }

    /// Calculate normalized progress (0.0 to 1.0).
    pub fn progress(&self, now: Instant) -> f32 {
        let elapsed = now.saturating_duration_since(self.start_time);

        if self.total_duration.is_zero() {
            1.0
        } else {
            (elapsed.as_secs_f32() / self.total_duration.as_secs_f32()).min(1.0)
        }
    }

    /// Whether the animation has reached completion.
    #[cfg(test)]
    pub fn is_complete(&self, now: Instant) -> bool {
        self.progress(now) >= 1.0
    }

    /// Compute the eased interpolation value for the given progress.
    ///
    /// Maps `raw_t` (0→1 over total duration) through the phase
    /// sequence. Each phase applies its own easing within its lerp range.
    pub fn eased_t(&self, raw_t: f32) -> f32 {
        if raw_t >= 1.0 {
            return 1.0;
        }
        match self.current_phase(raw_t) {
            Some((phase, local_t)) => {
                let local_eased = phase.easing.evaluate(local_t);
                phase.lerp_start
                    + local_eased * (phase.lerp_end - phase.lerp_start)
            }
            None => 1.0,
        }
    }

    /// Whether sidechains should be visible at the given progress.
    pub fn should_include_sidechains(&self, raw_t: f32) -> bool {
        if raw_t >= 1.0 {
            return true;
        }
        self.current_phase(raw_t)
            .is_none_or(|(phase, _)| phase.include_sidechains)
    }

    /// Compute interpolated backbone states for this runner's residues.
    ///
    /// Returns an iterator of `(residue_idx, lerped_visual)` pairs using
    /// the same eased_t as sidechain interpolation.
    pub fn interpolate_residues(
        &self,
        t: f32,
    ) -> impl Iterator<Item = (usize, ResidueVisualState)> + '_ {
        let eased = self.eased_t(t);
        self.residues.iter().map(move |data| {
            (data.residue_idx, data.start.lerp(&data.target, eased))
        })
    }

    /// Compute interpolated sidechain positions at the given progress.
    ///
    /// Uses the same `eased_t` as backbone interpolation. Returns `None`
    /// if this runner has no sidechain data.
    pub fn interpolate_sidechain(&self, t: f32) -> Option<Vec<Vec3>> {
        let sc = self.sidechain.as_ref()?;
        if t >= 1.0 {
            return Some(sc.target.clone());
        }
        let eased = self.eased_t(t);
        Some(
            sc.start
                .iter()
                .zip(sc.target.iter())
                .map(|(s, e)| lerp_vec3(eased, *s, *e))
                .collect(),
        )
    }

    /// Find the current phase and local progress for a given `raw_t`.
    ///
    /// Returns the active phase and the local progress (0→1) within it.
    fn current_phase(&self, raw_t: f32) -> Option<(&AnimationPhase, f32)> {
        if self.phases.is_empty() {
            return None;
        }

        let total_secs = self.total_duration.as_secs_f32();
        let mut cumulative = 0.0;

        for (i, phase) in self.phases.iter().enumerate() {
            let phase_frac = if total_secs == 0.0 {
                1.0 / self.phases.len() as f32
            } else {
                phase.duration.as_secs_f32() / total_secs
            };

            let is_last = i == self.phases.len() - 1;

            if raw_t < cumulative + phase_frac || is_last {
                let local_t = if phase_frac > 0.0 {
                    ((raw_t - cumulative) / phase_frac).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                return Some((phase, local_t));
            }

            cumulative += phase_frac;
        }

        self.phases.last().map(|p| (p, 1.0f32))
    }
}

impl std::fmt::Debug for AnimationRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnimationRunner")
            .field("name", &self.name)
            .field("residue_count", &self.residues.len())
            .field("duration", &self.total_duration)
            .field("phases", &self.phases.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;

    fn make_residue_data(
        idx: usize,
        start_y: f32,
        end_y: f32,
    ) -> ResidueAnimationData {
        ResidueAnimationData {
            residue_idx: idx,
            start: ResidueVisualState::new([
                Vec3::new(0.0, start_y, 0.0),
                Vec3::new(1.0, start_y, 0.0),
                Vec3::new(2.0, start_y, 0.0),
            ]),
            target: ResidueVisualState::new([
                Vec3::new(0.0, end_y, 0.0),
                Vec3::new(1.0, end_y, 0.0),
                Vec3::new(2.0, end_y, 0.0),
            ]),
        }
    }

    #[test]
    fn test_runner_progress() {
        let transition = Transition::linear(Duration::from_millis(100));
        let residues = vec![make_residue_data(0, 0.0, 10.0)];
        let start = Instant::now();
        let runner = AnimationRunner::with_start_time(
            start,
            &transition,
            residues,
            None,
        );

        assert!((runner.progress(start) - 0.0).abs() < 0.01);

        let mid = start + Duration::from_millis(50);
        assert!((runner.progress(mid) - 0.5).abs() < 0.01);

        let end = start + Duration::from_millis(100);
        assert!((runner.progress(end) - 1.0).abs() < 0.01);

        let past = start + Duration::from_millis(200);
        assert!((runner.progress(past) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_runner_eased_t_linear() {
        let transition = Transition::linear(Duration::from_millis(100));
        let residues = vec![make_residue_data(0, 0.0, 10.0)];
        let start = Instant::now();
        let runner = AnimationRunner::with_start_time(
            start,
            &transition,
            residues,
            None,
        );

        assert!((runner.eased_t(0.0) - 0.0).abs() < 0.01);
        assert!((runner.eased_t(0.5) - 0.5).abs() < 0.01);
        assert!((runner.eased_t(1.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_runner_snap() {
        let transition = Transition::snap();
        let residues = vec![make_residue_data(0, 0.0, 10.0)];
        let runner = AnimationRunner::new(&transition, residues, None);

        assert!(runner.is_complete(Instant::now()));
        assert_eq!(runner.duration(), Duration::ZERO);
    }

    #[test]
    fn test_runner_is_complete() {
        let transition = Transition::linear(Duration::from_millis(100));
        let residues = vec![make_residue_data(0, 0.0, 10.0)];
        let start = Instant::now();
        let runner = AnimationRunner::with_start_time(
            start,
            &transition,
            residues,
            None,
        );

        assert!(!runner.is_complete(start));
        assert!(!runner.is_complete(start + Duration::from_millis(50)));
        assert!(runner.is_complete(start + Duration::from_millis(100)));
        assert!(runner.is_complete(start + Duration::from_millis(200)));
    }

    #[test]
    fn test_two_phase_eased_t() {
        let transition = Transition::collapse_expand(
            Duration::from_millis(200),
            Duration::from_millis(300),
        );
        let residues = vec![make_residue_data(0, 0.0, 10.0)];
        let runner = AnimationRunner::new(&transition, residues, None);

        // At start
        assert!((runner.eased_t(0.0) - 0.0).abs() < 0.01);
        // At end
        assert!((runner.eased_t(1.0) - 1.0).abs() < 0.01);
        // Phase boundary (t=0.4 = end of phase 1)
        // QuadraticIn at local_t=1.0 gives 1.0
        // eased = 0.0 + 1.0 * 0.4 = 0.4
        assert!((runner.eased_t(0.4) - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_should_include_sidechains() {
        let transition = Transition::backbone_then_expand(
            Duration::from_millis(300),
            Duration::from_millis(200),
        );
        let residues = vec![make_residue_data(0, 0.0, 10.0)];
        let runner = AnimationRunner::new(&transition, residues, None);

        // Phase 1 (0→0.6): sidechains hidden
        assert!(!runner.should_include_sidechains(0.0));
        assert!(!runner.should_include_sidechains(0.3));
        // Phase 2 (0.6→1.0): sidechains visible
        assert!(runner.should_include_sidechains(0.7));
        // Completed: always visible
        assert!(runner.should_include_sidechains(1.0));
    }
}
