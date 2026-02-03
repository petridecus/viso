//! Animation timeline for managing active animations.
//!
//! The timeline tracks active animations, performs updates, and provides
//! interpolated state for rendering. Designed for minimal allocations
//! during the update loop.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use super::state::{InterpolatedResidue, ResidueAnimationState};
use crate::easing::EasingFunction;

/// Configuration for animations.
#[derive(Debug, Clone)]
pub struct AnimationConfig {
    /// Duration of animations. Default: 300ms
    pub duration: Duration,
    /// Easing function to use. Default: CubicHermite(0.33, 1.0)
    pub easing: EasingFunction,
    /// Whether animations are enabled.
    pub enabled: bool,
}

impl Default for AnimationConfig {
    fn default() -> Self {
        Self {
            duration: Duration::from_millis(300),
            easing: EasingFunction::DEFAULT,
            enabled: true,
        }
    }
}

/// An active animation being played.
#[derive(Debug)]
pub struct ActiveAnimation {
    /// When the animation started.
    pub start_time: Instant,
    /// Total duration of the animation.
    pub duration: Duration,
    /// Easing function for this animation.
    pub easing: EasingFunction,
    /// States for each residue being animated.
    pub residue_states: Vec<ResidueAnimationState>,
    /// Whether the animation has completed.
    pub is_done: bool,
}

impl ActiveAnimation {
    /// Create a new active animation.
    pub fn new(
        start_time: Instant,
        duration: Duration,
        easing: EasingFunction,
        residue_states: Vec<ResidueAnimationState>,
    ) -> Self {
        Self {
            start_time,
            duration,
            easing,
            residue_states,
            is_done: false,
        }
    }

    /// Calculate the progress of this animation (0.0 to 1.0).
    #[inline]
    pub fn progress(&self, now: Instant) -> f32 {
        let elapsed = now.saturating_duration_since(self.start_time);
        if self.duration.is_zero() {
            return 1.0;
        }
        (elapsed.as_secs_f32() / self.duration.as_secs_f32()).min(1.0)
    }

    /// Get the eased progress value.
    #[inline]
    pub fn eased_progress(&self, now: Instant) -> f32 {
        let t = self.progress(now);
        self.easing.evaluate(t)
    }
}

/// Timeline for managing multiple active animations.
///
/// The timeline pre-allocates buffers to avoid allocations during updates.
/// Target performance: <100us for 100 animations.
pub struct AnimationTimeline {
    /// Currently active animations.
    active_animations: Vec<ActiveAnimation>,
    /// Global animation configuration.
    pub config: AnimationConfig,
    /// Chain indices that have been modified.
    dirty_chains: HashSet<usize>,
    /// Residue indices that have been modified (those with sidechains).
    dirty_residues: HashSet<usize>,
    /// Pre-allocated buffer for interpolated results.
    interpolation_buffer: Vec<InterpolatedResidue>,
    /// Maximum number of residues (for buffer sizing).
    max_residues: usize,
}

impl AnimationTimeline {
    /// Create a new animation timeline with pre-allocated buffers.
    ///
    /// # Arguments
    /// * `max_residues` - Maximum number of residues to pre-allocate for.
    pub fn new(max_residues: usize) -> Self {
        Self {
            active_animations: Vec::with_capacity(16), // Typical case: few concurrent animations
            config: AnimationConfig::default(),
            dirty_chains: HashSet::with_capacity(16),
            dirty_residues: HashSet::with_capacity(max_residues),
            interpolation_buffer: Vec::with_capacity(max_residues),
            max_residues,
        }
    }

    /// Add a new animation to the timeline.
    ///
    /// This preempts any existing animations for the same residues by marking
    /// them as done and starting a new animation from the current interpolated state.
    ///
    /// # Arguments
    /// * `residue_states` - The states for each residue to animate.
    /// * `duration` - Animation duration (uses config default if None).
    /// * `easing` - Easing function (uses config default if None).
    pub fn add(
        &mut self,
        residue_states: Vec<ResidueAnimationState>,
        duration: Option<Duration>,
        easing: Option<EasingFunction>,
    ) {
        if !self.config.enabled || residue_states.is_empty() {
            return;
        }

        let duration = duration.unwrap_or(self.config.duration);
        let easing = easing.unwrap_or(self.config.easing);

        // Mark all existing animations as done (preemption)
        // In a more sophisticated system, we would only preempt overlapping residues
        for anim in &mut self.active_animations {
            anim.is_done = true;
        }

        // Create the new animation
        let animation = ActiveAnimation::new(Instant::now(), duration, easing, residue_states);

        self.active_animations.push(animation);
    }

    /// Update all active animations.
    ///
    /// Returns `true` if any animations are still active.
    ///
    /// Target performance: <100us for 100 animations.
    pub fn update(&mut self, now: Instant) -> bool {
        // Clear previous frame's dirty tracking
        self.dirty_chains.clear();
        self.dirty_residues.clear();
        self.interpolation_buffer.clear();

        // Remove completed animations
        self.active_animations.retain(|a| !a.is_done);

        if self.active_animations.is_empty() {
            return false;
        }

        // Process each active animation
        for anim in &mut self.active_animations {
            let t = anim.progress(now);
            let eased_t = anim.easing.evaluate(t);

            // Check if animation is complete
            if t >= 1.0 {
                anim.is_done = true;
            }

            // Interpolate each residue
            for state in &anim.residue_states {
                if !state.needs_animation {
                    continue;
                }

                let interpolated = state.interpolate(eased_t);

                // Track dirty state
                // Note: We'd need a residue-to-chain map for accurate chain tracking,
                // but for now we mark the residue as dirty and let the caller handle it
                self.dirty_residues.insert(state.residue_idx);

                self.interpolation_buffer.push(interpolated);
            }
        }

        // Return true if any animations are still running
        self.active_animations.iter().any(|a| !a.is_done)
    }

    /// Get the interpolated residue states from the last update.
    #[inline]
    pub fn get_interpolated(&self) -> &[InterpolatedResidue] {
        &self.interpolation_buffer
    }

    /// Skip all animations to their end state (t=1.0).
    pub fn skip(&mut self) {
        self.interpolation_buffer.clear();
        self.dirty_chains.clear();
        self.dirty_residues.clear();

        for anim in &mut self.active_animations {
            // Interpolate to final state (t=1.0)
            for state in &anim.residue_states {
                if !state.needs_animation {
                    continue;
                }

                let interpolated = state.interpolate(1.0);
                self.dirty_residues.insert(state.residue_idx);
                self.interpolation_buffer.push(interpolated);
            }
            anim.is_done = true;
        }

        // Clean up completed animations
        self.active_animations.clear();
    }

    /// Cancel all active animations without applying final state.
    pub fn cancel(&mut self) {
        self.active_animations.clear();
        self.interpolation_buffer.clear();
        self.dirty_chains.clear();
        self.dirty_residues.clear();
    }

    /// Check if any animations are currently active.
    #[inline]
    pub fn is_animating(&self) -> bool {
        !self.active_animations.is_empty()
    }

    /// Get the set of chain indices that have been modified.
    #[inline]
    pub fn dirty_chains(&self) -> &HashSet<usize> {
        &self.dirty_chains
    }

    /// Get the set of residue indices that have been modified.
    #[inline]
    pub fn dirty_residues(&self) -> &HashSet<usize> {
        &self.dirty_residues
    }

    /// Clear the dirty tracking sets.
    pub fn clear_dirty(&mut self) {
        self.dirty_chains.clear();
        self.dirty_residues.clear();
    }

    /// Reserve capacity for a new maximum number of residues.
    ///
    /// This resizes the interpolation buffer if needed.
    pub fn reserve(&mut self, max_residues: usize) {
        if max_residues > self.max_residues {
            self.interpolation_buffer.reserve(max_residues - self.max_residues);
            self.dirty_residues.reserve(max_residues - self.max_residues);
            self.max_residues = max_residues;
        }
    }

    /// Get the number of active animations.
    #[inline]
    pub fn active_count(&self) -> usize {
        self.active_animations.iter().filter(|a| !a.is_done).count()
    }
}

impl std::fmt::Debug for AnimationTimeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnimationTimeline")
            .field("active_animations", &self.active_animations.len())
            .field("config", &self.config)
            .field("max_residues", &self.max_residues)
            .field("interpolation_buffer_len", &self.interpolation_buffer.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// Helper to create a test residue animation state.
    fn make_test_state(residue_idx: usize) -> ResidueAnimationState {
        ResidueAnimationState {
            residue_idx,
            start_backbone: [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
            ],
            end_backbone: [
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(2.0, 1.0, 0.0),
            ],
            start_chis: [0.0, 0.0, 0.0, 0.0],
            end_chis: [90.0, 0.0, 0.0, 0.0],
            num_chis: 1,
            needs_animation: true,
        }
    }

    #[test]
    fn test_timeline_new() {
        let timeline = AnimationTimeline::new(100);
        assert!(!timeline.is_animating());
        assert!(timeline.dirty_chains().is_empty());
        assert!(timeline.dirty_residues().is_empty());
        assert!(timeline.get_interpolated().is_empty());
    }

    #[test]
    fn test_timeline_add_animation() {
        let mut timeline = AnimationTimeline::new(100);

        let states = vec![make_test_state(0), make_test_state(1)];
        timeline.add(states, None, None);

        assert!(timeline.is_animating());
        assert_eq!(timeline.active_count(), 1);
    }

    #[test]
    fn test_timeline_update_lifecycle() {
        let mut timeline = AnimationTimeline::new(100);

        let states = vec![make_test_state(0)];
        timeline.add(states, Some(Duration::from_millis(50)), None);

        // First update - animation should be active
        let now = Instant::now();
        let still_animating = timeline.update(now);
        assert!(still_animating);
        assert!(!timeline.get_interpolated().is_empty());

        // Wait for animation to complete
        std::thread::sleep(Duration::from_millis(100));

        // Update after animation completes
        let still_animating = timeline.update(Instant::now());
        // Animation should be done after this update
        assert!(!still_animating || timeline.active_count() == 0);
    }

    #[test]
    fn test_timeline_skip() {
        let mut timeline = AnimationTimeline::new(100);

        let states = vec![make_test_state(0), make_test_state(1)];
        timeline.add(states, Some(Duration::from_secs(10)), None);

        assert!(timeline.is_animating());

        timeline.skip();

        // After skip, animations should be cleared
        assert!(!timeline.is_animating());

        // Interpolated buffer should contain final states
        let interpolated = timeline.get_interpolated();
        assert_eq!(interpolated.len(), 2);

        // Values should be at final state (t=1.0)
        for interp in interpolated {
            // End backbone Y was 1.0
            assert!((interp.backbone[0].y - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_timeline_cancel() {
        let mut timeline = AnimationTimeline::new(100);

        let states = vec![make_test_state(0)];
        timeline.add(states, Some(Duration::from_secs(10)), None);

        assert!(timeline.is_animating());

        timeline.cancel();

        assert!(!timeline.is_animating());
        assert!(timeline.get_interpolated().is_empty());
        assert!(timeline.dirty_chains().is_empty());
        assert!(timeline.dirty_residues().is_empty());
    }

    #[test]
    fn test_timeline_preemption() {
        let mut timeline = AnimationTimeline::new(100);

        // Add first animation
        let states1 = vec![make_test_state(0)];
        timeline.add(states1, Some(Duration::from_secs(10)), None);

        // Add second animation (should preempt the first)
        let states2 = vec![make_test_state(1)];
        timeline.add(states2, Some(Duration::from_secs(10)), None);

        // Update to process preemption
        timeline.update(Instant::now());

        // After update, only the second animation should be active
        // (first was marked done during add, then removed during update)
        assert_eq!(timeline.active_count(), 1);
    }

    #[test]
    fn test_timeline_dirty_tracking() {
        let mut timeline = AnimationTimeline::new(100);

        let states = vec![make_test_state(5), make_test_state(10)];
        timeline.add(states, Some(Duration::from_millis(100)), None);

        timeline.update(Instant::now());

        // Dirty residues should contain the animated residue indices
        assert!(timeline.dirty_residues().contains(&5));
        assert!(timeline.dirty_residues().contains(&10));

        // Clear dirty tracking
        timeline.clear_dirty();
        assert!(timeline.dirty_residues().is_empty());
    }

    #[test]
    fn test_timeline_reserve() {
        let mut timeline = AnimationTimeline::new(100);

        // Reserve more capacity
        timeline.reserve(500);

        // Should be able to handle more residues now
        let mut states = Vec::new();
        for i in 0..200 {
            states.push(make_test_state(i));
        }
        timeline.add(states, None, None);
        timeline.update(Instant::now());

        assert_eq!(timeline.get_interpolated().len(), 200);
    }

    #[test]
    fn test_animation_config_default() {
        let config = AnimationConfig::default();
        assert_eq!(config.duration, Duration::from_millis(300));
        assert_eq!(config.easing, EasingFunction::DEFAULT);
        assert!(config.enabled);
    }

    #[test]
    fn test_active_animation_progress() {
        let start = Instant::now();
        let duration = Duration::from_millis(100);
        let anim = ActiveAnimation::new(
            start,
            duration,
            EasingFunction::Linear,
            vec![make_test_state(0)],
        );

        // At start, progress should be ~0
        let progress = anim.progress(start);
        assert!(progress < 0.1);

        // After duration, progress should be 1.0
        let end = start + duration;
        let progress = anim.progress(end);
        assert!((progress - 1.0).abs() < 0.01);

        // Past duration should still clamp to 1.0
        let past = start + duration + Duration::from_millis(50);
        let progress = anim.progress(past);
        assert!((progress - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_disabled_config_prevents_add() {
        let mut timeline = AnimationTimeline::new(100);
        timeline.config.enabled = false;

        let states = vec![make_test_state(0)];
        timeline.add(states, None, None);

        // Animation should not be added when disabled
        assert!(!timeline.is_animating());
    }

    #[test]
    fn test_empty_states_not_added() {
        let mut timeline = AnimationTimeline::new(100);

        timeline.add(vec![], None, None);

        assert!(!timeline.is_animating());
    }
}
