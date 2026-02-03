//! Structure animator composed from smaller components.
//!
//! - `StructureState`: Holds current and target visual states
//! - `AnimationRunner`: Executes a single animation
//! - `AnimationController`: Handles preemption and action->behavior mapping

mod controller;
mod runner;
mod state;

pub use controller::AnimationController;
pub use runner::AnimationRunner;
pub use state::StructureState;

use std::time::Instant;

use glam::Vec3;

use super::preferences::AnimationAction;

/// Structure animator manages animation state and applies behaviors.
///
/// This is a thin facade composing:
/// - `StructureState` for current/target visual state
/// - `AnimationRunner` for executing animations
/// - `AnimationController` for preemption and preferences
///
/// # Usage
///
/// ```ignore
/// let mut animator = StructureAnimator::new();
///
/// // Set target when data changes
/// animator.set_target(&backbone_chains, AnimationAction::Wiggle);
///
/// // Each frame
/// animator.update(Instant::now());
/// let visual = animator.get_backbone();
/// ```
pub struct StructureAnimator {
    state: StructureState,
    runner: Option<AnimationRunner>,
    controller: AnimationController,
}

impl StructureAnimator {
    /// Create a new animator with default preferences.
    pub fn new() -> Self {
        Self {
            state: StructureState::new(),
            runner: None,
            controller: AnimationController::new(),
        }
    }

    /// Create with custom controller (for custom preferences).
    pub fn with_controller(controller: AnimationController) -> Self {
        Self {
            state: StructureState::new(),
            runner: None,
            controller,
        }
    }

    /// Get mutable access to the controller (for changing preferences).
    pub fn controller_mut(&mut self) -> &mut AnimationController {
        &mut self.controller
    }

    /// Get the controller.
    pub fn controller(&self) -> &AnimationController {
        &self.controller
    }

    /// Enable or disable animations.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.controller.set_enabled(enabled);
        if !enabled {
            self.runner = None;
        }
    }

    /// Check if animations are enabled.
    pub fn is_enabled(&self) -> bool {
        self.controller.is_enabled()
    }

    /// Check if an animation is currently active.
    pub fn is_animating(&self) -> bool {
        self.runner.is_some()
    }

    /// Get the current animation progress (0.0 to 1.0).
    /// Returns 1.0 if no animation is active.
    pub fn progress(&self) -> f32 {
        self.runner
            .as_ref()
            .map(|r| r.progress(Instant::now()))
            .unwrap_or(1.0)
    }

    /// Set a new target state, potentially triggering an animation.
    pub fn set_target(&mut self, backbone_chains: &[Vec<Vec3>], action: AnimationAction) {
        let new_target = StructureState::from_backbone(backbone_chains);

        // Let controller decide what to do
        let maybe_runner = self.controller.handle_new_target(
            &mut self.state,
            &new_target,
            self.runner.as_ref(),
            action,
        );

        if let Some(runner) = maybe_runner {
            self.runner = Some(runner);
        }

        self.state.set_target(new_target);
    }

    /// Update animations for the current frame.
    ///
    /// Returns `true` if animations are still active.
    pub fn update(&mut self, now: Instant) -> bool {
        let Some(ref runner) = self.runner else {
            return false;
        };

        // Apply interpolated states
        runner.apply_to_state(&mut self.state, now);

        // Check if complete
        if runner.is_complete(now) {
            self.state.snap_to_target();
            self.runner = None;
            return false;
        }

        true
    }

    /// Skip current animation to end state.
    pub fn skip(&mut self) {
        self.state.snap_to_target();
        self.runner = None;
    }

    /// Cancel current animation, staying at current visual position.
    pub fn cancel(&mut self) {
        self.runner = None;
    }

    /// Get the current visual backbone state as chains.
    pub fn get_backbone(&self) -> Vec<Vec<Vec3>> {
        self.state.to_backbone_chains()
    }

    /// Get the number of residues.
    pub fn residue_count(&self) -> usize {
        self.state.residue_count()
    }

    /// Get the underlying state (for advanced usage).
    pub fn state(&self) -> &StructureState {
        &self.state
    }
}

impl Default for StructureAnimator {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for StructureAnimator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StructureAnimator")
            .field("residue_count", &self.state.residue_count())
            .field("is_animating", &self.runner.is_some())
            .field("controller", &self.controller)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_backbone(y: f32) -> Vec<Vec<Vec3>> {
        vec![vec![
            Vec3::new(0.0, y, 0.0),
            Vec3::new(1.0, y, 0.0),
            Vec3::new(2.0, y, 0.0),
            Vec3::new(3.0, y, 0.0),
            Vec3::new(4.0, y, 0.0),
            Vec3::new(5.0, y, 0.0),
        ]]
    }

    #[test]
    fn test_animator_initial_state() {
        let animator = StructureAnimator::new();
        assert!(animator.is_enabled());
        assert!(!animator.is_animating());
        assert_eq!(animator.residue_count(), 0);
    }

    #[test]
    fn test_animator_first_target_snaps() {
        let mut animator = StructureAnimator::new();
        animator.set_target(&make_backbone(0.0), AnimationAction::Wiggle);

        assert!(!animator.is_animating());
        assert_eq!(animator.residue_count(), 2);
    }

    #[test]
    fn test_animator_animates_on_change() {
        let mut animator = StructureAnimator::new();
        animator.set_target(&make_backbone(0.0), AnimationAction::Wiggle);
        animator.set_target(&make_backbone(10.0), AnimationAction::Wiggle);

        assert!(animator.is_animating());
    }

    #[test]
    fn test_animator_skip() {
        let mut animator = StructureAnimator::new();
        animator.set_target(&make_backbone(0.0), AnimationAction::Wiggle);
        animator.set_target(&make_backbone(10.0), AnimationAction::Wiggle);

        animator.skip();

        assert!(!animator.is_animating());
        let backbone = animator.get_backbone();
        assert!((backbone[0][0].y - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_animator_completes() {
        let mut animator = StructureAnimator::new();
        animator.set_target(&make_backbone(0.0), AnimationAction::Load);
        animator.set_target(&make_backbone(10.0), AnimationAction::Wiggle);

        let future = Instant::now() + Duration::from_secs(1);
        let still_animating = animator.update(future);

        assert!(!still_animating);
        assert!(!animator.is_animating());
    }
}
