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
    /// Start sidechain positions (animation begin state)
    start_sidechain_positions: Vec<Vec3>,
    /// Target sidechain positions (animation end state)
    target_sidechain_positions: Vec<Vec3>,
    /// Residue index for each sidechain atom (for collapse point lookup)
    sidechain_residue_indices: Vec<u32>,
    /// CA positions per residue for collapse animation (indexed by residue)
    start_ca_positions: Vec<Vec3>,
    /// Target CA positions per residue
    target_ca_positions: Vec<Vec3>,
    /// Flag indicating sidechains changed (for triggering animation even when backbone is static)
    sidechains_changed: bool,
    /// Pending action for sidechain-only animation
    pending_sidechain_action: Option<AnimationAction>,
}

impl StructureAnimator {
    /// Create a new animator with default preferences.
    pub fn new() -> Self {
        Self {
            state: StructureState::new(),
            runner: None,
            controller: AnimationController::new(),
            start_sidechain_positions: Vec::new(),
            target_sidechain_positions: Vec::new(),
            sidechain_residue_indices: Vec::new(),
            start_ca_positions: Vec::new(),
            target_ca_positions: Vec::new(),
            sidechains_changed: false,
            pending_sidechain_action: None,
        }
    }

    /// Create with custom controller (for custom preferences).
    pub fn with_controller(controller: AnimationController) -> Self {
        Self {
            state: StructureState::new(),
            runner: None,
            controller,
            start_sidechain_positions: Vec::new(),
            target_sidechain_positions: Vec::new(),
            sidechain_residue_indices: Vec::new(),
            start_ca_positions: Vec::new(),
            target_ca_positions: Vec::new(),
            sidechains_changed: false,
            pending_sidechain_action: None,
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

        // Check if we have pending sidechain changes that should force animation
        let force_animation = self.sidechains_changed;
        let effective_action = if force_animation {
            // Use the pending sidechain action if available, otherwise use the provided action
            self.pending_sidechain_action.take().unwrap_or(action)
        } else {
            action
        };

        // Let controller decide what to do
        let maybe_runner = self.controller.handle_new_target(
            &mut self.state,
            &new_target,
            self.runner.as_ref(),
            effective_action,
            force_animation,
        );

        if let Some(runner) = maybe_runner {
            self.runner = Some(runner);
        }

        // Reset sidechain change flag after processing
        self.sidechains_changed = false;

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

    /// Set sidechain target positions for animation.
    ///
    /// Call this alongside `set_target` when sidechain data changes.
    /// The residue_indices map each sidechain atom to its residue for collapse animation.
    /// Pass an action to enable sidechain-only animation triggering.
    pub fn set_sidechain_target(
        &mut self,
        positions: &[Vec3],
        residue_indices: &[u32],
        ca_positions: &[Vec3],
    ) {
        self.set_sidechain_target_with_action(positions, residue_indices, ca_positions, None);
    }

    /// Set sidechain target positions with an explicit action for sidechain-only animations.
    ///
    /// If sidechains change but backbone doesn't, this action will be used to trigger
    /// an animation. Call this BEFORE `set_target()` for proper animation triggering.
    pub fn set_sidechain_target_with_action(
        &mut self,
        positions: &[Vec3],
        residue_indices: &[u32],
        ca_positions: &[Vec3],
        action: Option<AnimationAction>,
    ) {
        // Check if sidechains actually changed
        let sidechains_changed = self.sidechains_differ(positions);

        // If sizes match, capture current target as new start
        if self.target_sidechain_positions.len() == positions.len() {
            self.start_sidechain_positions = self.target_sidechain_positions.clone();
            self.start_ca_positions = self.target_ca_positions.clone();
        } else {
            // Size changed - snap to new positions
            self.start_sidechain_positions = positions.to_vec();
            self.start_ca_positions = ca_positions.to_vec();
        }

        self.target_sidechain_positions = positions.to_vec();
        self.target_ca_positions = ca_positions.to_vec();
        self.sidechain_residue_indices = residue_indices.to_vec();

        // Store sidechain change state for set_target() to use
        self.sidechains_changed = sidechains_changed;
        self.pending_sidechain_action = action;
    }

    /// Check if new sidechain positions differ from current target.
    fn sidechains_differ(&self, new_positions: &[Vec3]) -> bool {
        // Size change means difference
        if self.target_sidechain_positions.len() != new_positions.len() {
            return !new_positions.is_empty();
        }

        // Empty means no change
        if new_positions.is_empty() {
            return false;
        }

        // Compare positions with small epsilon
        const EPSILON: f32 = 0.001;
        for (old, new) in self.target_sidechain_positions.iter().zip(new_positions.iter()) {
            if (*old - *new).length_squared() > EPSILON * EPSILON {
                return true;
            }
        }

        false
    }

    /// Get interpolated sidechain positions using the current animation behavior.
    ///
    /// This applies the same interpolation logic as backbone animation,
    /// including collapse/expand for mutations.
    pub fn get_sidechain_positions(&self) -> Vec<Vec3> {
        let t = self.progress();

        // If no runner or animation complete, return target positions
        if self.runner.is_none() || t >= 1.0 {
            return self.target_sidechain_positions.clone();
        }

        let runner = self.runner.as_ref().unwrap();
        let behavior = runner.behavior();

        self.start_sidechain_positions
            .iter()
            .zip(self.target_sidechain_positions.iter())
            .enumerate()
            .map(|(i, (start, end))| {
                // Get the collapse point (CA position) for this atom's residue
                let res_idx = self.sidechain_residue_indices.get(i).copied().unwrap_or(0) as usize;
                let start_ca = self.start_ca_positions.get(res_idx).copied().unwrap_or(*start);
                let end_ca = self.target_ca_positions.get(res_idx).copied().unwrap_or(*end);
                // Interpolate CA position for collapse point
                let collapse_point = start_ca + (end_ca - start_ca) * t;

                behavior.interpolate_position(t, *start, *end, collapse_point)
            })
            .collect()
    }

    /// Check if sidechain animation state is valid (has data).
    pub fn has_sidechain_data(&self) -> bool {
        !self.target_sidechain_positions.is_empty()
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

    #[test]
    fn test_animator_sidechain_only_change_triggers_animation() {
        let mut animator = StructureAnimator::new();

        // Set initial backbone
        animator.set_target(&make_backbone(5.0), AnimationAction::Load);
        assert!(!animator.is_animating(), "Initial backbone should snap");

        // Set initial sidechains
        let sidechain_pos = vec![Vec3::new(1.0, 5.0, 1.0), Vec3::new(2.0, 5.0, 1.0)];
        let residue_indices = vec![0, 1];
        let ca_positions = vec![Vec3::new(1.0, 5.0, 0.0), Vec3::new(4.0, 5.0, 0.0)];
        animator.set_sidechain_target_with_action(
            &sidechain_pos,
            &residue_indices,
            &ca_positions,
            Some(AnimationAction::Shake),
        );

        // Set same backbone again - should trigger animation due to sidechain change
        animator.set_target(&make_backbone(5.0), AnimationAction::Load);

        // Now change sidechains with same backbone
        let new_sidechain_pos = vec![Vec3::new(1.0, 5.0, 2.0), Vec3::new(2.0, 5.0, 2.0)];
        animator.set_sidechain_target_with_action(
            &new_sidechain_pos,
            &residue_indices,
            &ca_positions,
            Some(AnimationAction::Shake),
        );

        // Set same backbone - should animate because sidechains changed
        animator.set_target(&make_backbone(5.0), AnimationAction::Load);
        assert!(animator.is_animating(), "Sidechain-only change should trigger animation");
    }

    #[test]
    fn test_sidechains_differ_detects_changes() {
        let mut animator = StructureAnimator::new();

        // Set initial sidechains
        let sidechain_pos = vec![Vec3::new(1.0, 2.0, 3.0)];
        let residue_indices = vec![0];
        let ca_positions = vec![Vec3::ZERO];
        animator.set_sidechain_target(&sidechain_pos, &residue_indices, &ca_positions);

        // Same positions should not differ
        assert!(!animator.sidechains_differ(&sidechain_pos));

        // Different positions should differ
        let different_pos = vec![Vec3::new(1.0, 2.0, 4.0)];
        assert!(animator.sidechains_differ(&different_pos));

        // Very small change should not differ (within epsilon)
        let tiny_change = vec![Vec3::new(1.0, 2.0, 3.0005)];
        assert!(!animator.sidechains_differ(&tiny_change));
    }
}
