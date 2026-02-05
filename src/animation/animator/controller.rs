//! Animation controller handles preemption and action->behavior mapping.

use std::time::Instant;

use crate::animation::behaviors::PreemptionStrategy;
use crate::animation::preferences::{AnimationAction, AnimationPreferences};

use super::runner::{AnimationRunner, ResidueAnimationData};
use super::state::StructureState;

/// Controls animation lifecycle: when to start, preempt, or ignore new targets.
///
/// Responsibilities:
/// - Map actions to behaviors via preferences
/// - Decide preemption strategy when new target arrives
/// - Build AnimationRunner when animation should start
#[derive(Clone)]
pub struct AnimationController {
    preferences: AnimationPreferences,
    enabled: bool,
}

impl AnimationController {
    /// Create with default preferences.
    pub fn new() -> Self {
        Self {
            preferences: AnimationPreferences::default(),
            enabled: true,
        }
    }

    /// Create with custom preferences.
    pub fn with_preferences(preferences: AnimationPreferences) -> Self {
        Self {
            preferences,
            enabled: true,
        }
    }

    /// Get mutable access to preferences.
    pub fn preferences_mut(&mut self) -> &mut AnimationPreferences {
        &mut self.preferences
    }

    /// Get preferences.
    pub fn preferences(&self) -> &AnimationPreferences {
        &self.preferences
    }

    /// Enable or disable animations.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if animations are enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Handle arrival of a new target state.
    ///
    /// Decides whether to:
    /// - Start a new animation
    /// - Ignore the new target (if current animation has Ignore preemption)
    /// - Preempt current animation (syncing visual state first)
    ///
    /// The `force` parameter allows forcing animation creation even when backbone
    /// hasn't changed (e.g., for sidechain-only animations like Shake or MPNN).
    ///
    /// Returns `Some(AnimationRunner)` if a new animation should start,
    /// `None` if the new target should be ignored or no animation needed.
    pub fn handle_new_target(
        &self,
        current_state: &mut StructureState,
        new_target: &StructureState,
        current_runner: Option<&AnimationRunner>,
        action: AnimationAction,
        force: bool,
    ) -> Option<AnimationRunner> {
        // Disabled means no animation
        if !self.enabled {
            current_state.snap_to_target();
            return None;
        }

        // Structure size changed - snap, no animation
        if current_state.size_differs(new_target) {
            // Replace current state entirely
            *current_state = new_target.clone();
            return None;
        }

        // First time (empty state) - snap, no animation
        if current_state.is_empty() {
            *current_state = new_target.clone();
            return None;
        }

        // Get behavior for this action
        let behavior = self.preferences.get(action).clone();

        // Handle preemption if animation is in progress
        if let Some(runner) = current_runner {
            match behavior.preemption() {
                PreemptionStrategy::Ignore => {
                    // Keep current animation, ignore new target
                    return None;
                }
                PreemptionStrategy::Restart | PreemptionStrategy::Blend => {
                    // Sync current state to visual position before starting new animation
                    let t = runner.progress(Instant::now());
                    runner.apply_to_state(current_state, t);
                }
            }
        }

        // Check if backbone target actually changed (compare against previous target)
        let backbone_changed = current_state.target_differs(new_target);

        // If neither backbone nor sidechains changed (and not forced), skip animation
        if !backbone_changed && !force {
            return None;
        }

        // Find residues that need animation (backbone differences)
        let differing = current_state.differing_residues(new_target);

        // If backbone changed, we need residue data for the animation
        // If only sidechains changed (force=true, backbone_changed=false),
        // create a minimal runner that will drive sidechain interpolation
        if differing.is_empty() && !force {
            return None;
        }

        // Build residue animation data from differing backbone residues
        let residue_data: Vec<ResidueAnimationData> = differing
            .into_iter()
            .filter_map(|idx| {
                let start = current_state.get_current(idx)?.clone();
                let target = new_target.get_target(idx)?.clone();
                Some(ResidueAnimationData {
                    residue_idx: idx,
                    start,
                    target,
                })
            })
            .collect();

        // If we have backbone residue data, create animation from it
        // If forcing (sidechain-only), create a minimal animation runner
        if residue_data.is_empty() && !force {
            return None;
        }

        // For sidechain-only animations (force=true, no backbone changes),
        // create a runner with empty residue data - it will still provide
        // timing/progress for sidechain interpolation
        Some(AnimationRunner::new(behavior, residue_data))
    }
}

impl Default for AnimationController {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for AnimationController {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnimationController")
            .field("enabled", &self.enabled)
            .field("preferences", &self.preferences)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::behaviors::{Snap, shared};
    use glam::Vec3;

    fn make_backbone(y: f32, num_residues: usize) -> Vec<Vec<Vec3>> {
        let atoms: Vec<Vec3> = (0..num_residues)
            .flat_map(|r| {
                let x = r as f32 * 3.0;
                vec![
                    Vec3::new(x, y, 0.0),
                    Vec3::new(x + 1.0, y, 0.0),
                    Vec3::new(x + 2.0, y, 0.0),
                ]
            })
            .collect();
        vec![atoms]
    }

    #[test]
    fn test_controller_first_target_snaps() {
        let controller = AnimationController::new();
        let mut state = StructureState::new();
        let new_target = StructureState::from_backbone(&make_backbone(0.0, 2));

        let runner = controller.handle_new_target(
            &mut state,
            &new_target,
            None,
            AnimationAction::Wiggle,
            false,
        );

        assert!(runner.is_none());
        assert_eq!(state.residue_count(), 2);
    }

    #[test]
    fn test_controller_creates_animation_on_change() {
        let controller = AnimationController::new();
        let mut state = StructureState::from_backbone(&make_backbone(0.0, 2));
        let new_target = StructureState::from_backbone(&make_backbone(10.0, 2));

        let runner = controller.handle_new_target(
            &mut state,
            &new_target,
            None,
            AnimationAction::Wiggle,
            false,
        );

        assert!(runner.is_some());
        assert_eq!(runner.unwrap().residue_count(), 2);
    }

    #[test]
    fn test_controller_disabled_snaps() {
        let mut controller = AnimationController::new();
        controller.set_enabled(false);

        let mut state = StructureState::from_backbone(&make_backbone(0.0, 2));
        let new_target = StructureState::from_backbone(&make_backbone(10.0, 2));

        let runner = controller.handle_new_target(
            &mut state,
            &new_target,
            None,
            AnimationAction::Wiggle,
            false,
        );

        assert!(runner.is_none());
    }

    #[test]
    fn test_controller_respects_snap_behavior() {
        let mut controller = AnimationController::new();
        controller.preferences_mut().set(
            AnimationAction::Load,
            shared(Snap),
        );

        let mut state = StructureState::from_backbone(&make_backbone(0.0, 2));
        let new_target = StructureState::from_backbone(&make_backbone(10.0, 2));

        let runner = controller.handle_new_target(
            &mut state,
            &new_target,
            None,
            AnimationAction::Load,
            false,
        );

        // Snap behavior has zero duration, so it completes instantly
        // But we still get a runner (it just finishes immediately)
        assert!(runner.is_some());
    }

    #[test]
    fn test_controller_size_change_snaps() {
        let controller = AnimationController::new();
        let mut state = StructureState::from_backbone(&make_backbone(0.0, 2));
        let new_target = StructureState::from_backbone(&make_backbone(0.0, 5)); // Different size

        let runner = controller.handle_new_target(
            &mut state,
            &new_target,
            None,
            AnimationAction::Wiggle,
            false,
        );

        assert!(runner.is_none());
        assert_eq!(state.residue_count(), 5); // Snapped to new size
    }

    #[test]
    fn test_controller_no_animation_when_unchanged() {
        let controller = AnimationController::new();
        let mut state = StructureState::from_backbone(&make_backbone(5.0, 2));
        let new_target = StructureState::from_backbone(&make_backbone(5.0, 2)); // Same

        let runner = controller.handle_new_target(
            &mut state,
            &new_target,
            None,
            AnimationAction::Wiggle,
            false,
        );

        assert!(runner.is_none());
    }

    #[test]
    fn test_controller_force_creates_animation_with_same_backbone() {
        let controller = AnimationController::new();
        let mut state = StructureState::from_backbone(&make_backbone(5.0, 2));
        let new_target = StructureState::from_backbone(&make_backbone(5.0, 2)); // Same backbone

        // With force=true, should create animation even with same backbone
        let runner = controller.handle_new_target(
            &mut state,
            &new_target,
            None,
            AnimationAction::Shake,
            true, // force animation for sidechain-only change
        );

        assert!(runner.is_some());
    }
}
