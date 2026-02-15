//! Cascade animation with staggered per-residue delays.

use std::time::Duration;

use crate::util::easing::EasingFunction;

use super::super::interpolation::InterpolationContext;
use super::state::ResidueVisualState;
use super::traits::{AnimationBehavior, PreemptionStrategy};

/// Cascade animation where residues animate with staggered delays.
///
/// Creates a "wave" effect across the structure. Useful for reveals
/// where you want the structure to appear progressively.
///
/// # Timing
///
/// Each residue starts its animation after a delay based on its index:
/// - Residue 0: starts at t=0
/// - Residue 1: starts at t=delay_per_residue
/// - Residue N: starts at t=N*delay_per_residue
///
/// Each residue's individual animation lasts `base_duration`.
#[derive(Debug, Clone)]
pub struct Cascade {
    /// Base duration for each residue's animation.
    pub base_duration: Duration,
    /// Delay between each residue starting.
    pub delay_per_residue: Duration,
    /// Easing for individual residue animations.
    pub easing: EasingFunction,
}

impl Cascade {
    /// Create a new cascade animation.
    pub fn new(base_duration: Duration, delay_per_residue: Duration) -> Self {
        Self {
            base_duration,
            delay_per_residue,
            easing: EasingFunction::QuadraticOut,
        }
    }

    /// Set custom easing.
    pub fn with_easing(mut self, easing: EasingFunction) -> Self {
        self.easing = easing;
        self
    }

    /// Compute the local t value for a specific residue index.
    ///
    /// This maps the global animation progress to a per-residue progress,
    /// accounting for staggered start times.
    ///
    /// # Arguments
    /// * `global_t` - Overall animation progress (0.0 to 1.0)
    /// * `residue_idx` - Index of the residue
    /// * `total_residues` - Total number of residues being animated
    ///
    /// # Returns
    /// Local progress for this residue (0.0 to 1.0)
    pub fn residue_t(&self, global_t: f32, residue_idx: usize, total_residues: usize) -> f32 {
        if total_residues == 0 {
            return global_t;
        }

        let total_duration = self.total_duration_for(total_residues);
        let total_secs = total_duration.as_secs_f32();

        if total_secs == 0.0 {
            return global_t;
        }

        // Convert global t to absolute time
        let global_time = global_t * total_secs;

        // Start and end time for this residue
        let start_time = residue_idx as f32 * self.delay_per_residue.as_secs_f32();
        let base_secs = self.base_duration.as_secs_f32();

        if base_secs == 0.0 {
            return if global_time >= start_time { 1.0 } else { 0.0 };
        }

        // Compute local t for this residue
        if global_time < start_time {
            0.0
        } else {
            let local_time = global_time - start_time;
            (local_time / base_secs).min(1.0)
        }
    }

    /// Total duration for a given number of residues.
    pub fn total_duration_for(&self, num_residues: usize) -> Duration {
        if num_residues == 0 {
            return Duration::ZERO;
        }
        let last_start = self.delay_per_residue * (num_residues - 1) as u32;
        last_start + self.base_duration
    }
}

impl Default for Cascade {
    fn default() -> Self {
        Self::new(Duration::from_millis(200), Duration::from_millis(10))
    }
}

impl AnimationBehavior for Cascade {
    fn eased_t(&self, t: f32) -> f32 {
        self.easing.evaluate(t)
    }

    fn compute_context(&self, raw_t: f32) -> InterpolationContext {
        InterpolationContext::simple(raw_t, self.easing.evaluate(raw_t))
    }

    fn compute_state(
        &self,
        t: f32,
        start: &ResidueVisualState,
        end: &ResidueVisualState,
    ) -> ResidueVisualState {
        // Note: Cascade is special - it needs residue index context for proper behavior.
        // When used through the animator, residue_t() should be called first to get
        // the per-residue progress. This basic implementation applies easing to global t.
        let eased_t = self.eased_t(t);
        start.lerp(end, eased_t)
    }

    fn duration(&self) -> Duration {
        // Base duration; actual duration depends on residue count.
        // The animator will use total_duration_for() for accurate timing.
        self.base_duration
    }

    fn preemption(&self) -> PreemptionStrategy {
        // Cascade animations shouldn't be interrupted mid-reveal
        PreemptionStrategy::Ignore
    }

    fn name(&self) -> &'static str {
        "cascade"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cascade_residue_t_first_residue() {
        let cascade = Cascade::new(
            Duration::from_millis(100),
            Duration::from_millis(50),
        );

        // First residue starts immediately
        let t = cascade.residue_t(0.0, 0, 10);
        assert!((t - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_cascade_residue_t_later_residue() {
        let cascade = Cascade::new(
            Duration::from_millis(100),
            Duration::from_millis(50),
        );

        // Total duration for 10 residues: 50*9 + 100 = 550ms
        // Residue 2 starts at 100ms (2 * 50ms)
        // At global_t = 100/550 ≈ 0.18, residue 2 should be at local_t = 0
        let total_dur = cascade.total_duration_for(10).as_secs_f32();
        let global_t = 100.0 / 1000.0 / total_dur; // 100ms in terms of total

        let t = cascade.residue_t(global_t, 0, 10);
        // Residue 0 at 100ms should be complete (its animation is 100ms)
        assert!(t >= 0.99, "Residue 0 should be nearly complete at 100ms");
    }

    #[test]
    fn test_cascade_total_duration() {
        let cascade = Cascade::new(
            Duration::from_millis(100),
            Duration::from_millis(10),
        );

        // 5 residues: delay for residues 0-4 is 0, 10, 20, 30, 40ms
        // Last residue finishes at 40 + 100 = 140ms
        let total = cascade.total_duration_for(5);
        assert_eq!(total, Duration::from_millis(140));
    }

    #[test]
    fn test_cascade_total_duration_single_residue() {
        let cascade = Cascade::new(
            Duration::from_millis(200),
            Duration::from_millis(50),
        );

        // Single residue: just base duration
        let total = cascade.total_duration_for(1);
        assert_eq!(total, Duration::from_millis(200));
    }

    #[test]
    fn test_cascade_total_duration_zero_residues() {
        let cascade = Cascade::default();
        let total = cascade.total_duration_for(0);
        assert_eq!(total, Duration::ZERO);
    }
}
