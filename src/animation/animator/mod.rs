//! Structure animator composed from smaller components.
//!
//! - `StructureState`: Holds current and target visual states
//! - `AnimationRunner`: Executes a single animation
//! - `AnimationController`: Handles preemption and transition-based behavior

mod controller;
mod runner;
mod state;

use std::time::Instant;

pub use controller::AnimationController;
use glam::Vec3;
pub use runner::AnimationRunner;
pub use state::StructureState;

use super::{interpolation::InterpolationContext, transition::Transition};

/// Composes [`StructureState`], [`AnimationRunner`], and
/// [`AnimationController`] to animate backbone/sidechain transitions.
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
    /// Flag indicating sidechains changed (for triggering animation even when
    /// backbone is static)
    sidechains_changed: bool,
    /// Pending transition for sidechain-only animation
    pending_sidechain_transition: Option<Transition>,
    /// Current frame's animation progress (0.0 to 1.0), set by update().
    current_frame_progress: f32,
    /// Residue ranges that have been snapped (non-targeted entities).
    /// Sidechain atoms in these ranges skip interpolation entirely.
    snapped_residue_ranges: Vec<(usize, usize)>,
}

impl StructureAnimator {
    /// Animator with default settings.
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
            pending_sidechain_transition: None,
            current_frame_progress: 1.0,
            snapped_residue_ranges: Vec::new(),
        }
    }

    /// Enable or disable animations.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.controller.set_enabled(enabled);
        if !enabled {
            self.runner = None;
        }
    }

    /// Whether animations are enabled.
    pub fn is_enabled(&self) -> bool {
        self.controller.is_enabled()
    }

    /// Whether an animation is currently in progress.
    pub fn is_animating(&self) -> bool {
        self.runner.is_some()
    }

    /// Animation progress (0.0 to 1.0), computed in the last `update()` call.
    pub fn progress(&self) -> f32 {
        self.current_frame_progress
    }

    /// Set a new target state, potentially triggering an animation.
    pub fn set_target(
        &mut self,
        backbone_chains: &[Vec<Vec3>],
        transition: &Transition,
    ) {
        let new_target = StructureState::from_backbone(backbone_chains);

        // Use the pending sidechain transition if sidechains changed,
        // otherwise use the provided transition
        let effective_transition = if self.sidechains_changed {
            self.pending_sidechain_transition
                .take()
                .unwrap_or_else(|| transition.clone())
        } else {
            transition.clone()
        };

        // Let controller decide about backbone animation
        let maybe_runner = self.controller.handle_new_target(
            &mut self.state,
            &new_target,
            self.runner.as_ref(),
            &effective_transition,
        );

        if let Some(runner) = maybe_runner {
            self.runner = Some(runner);
        } else if self.sidechains_changed {
            // Sidechain-only change: create a timing-only runner
            // (empty residue data — just provides progress/behavior for
            // sidechain interpolation)
            self.runner = Some(AnimationRunner::new(
                effective_transition.behavior.clone(),
                vec![],
            ));
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
            self.current_frame_progress = 1.0;
            return false;
        };

        self.current_frame_progress = runner.progress(now);

        // Apply interpolated states using the same progress value
        runner.apply_to_state(&mut self.state, self.current_frame_progress);

        // Check if complete
        if self.current_frame_progress >= 1.0 {
            self.current_frame_progress = 1.0;
            self.state.snap_to_target();
            self.runner = None;
            return false;
        }

        true
    }

    /// Skip current animation to end state.
    pub fn skip(&mut self) {
        self.current_frame_progress = 1.0;
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

    /// Total number of residues in the structure.
    pub fn residue_count(&self) -> usize {
        self.state.residue_count()
    }

    /// Get the current structure state.
    pub fn state(&self) -> &StructureState {
        &self.state
    }

    /// Get the active animation runner, if any.
    pub fn runner(&self) -> Option<&AnimationRunner> {
        self.runner.as_ref()
    }

    /// Current interpolation context (unified across backbone, sidechains,
    /// bonds).
    pub fn interpolation_context(&self) -> InterpolationContext {
        let raw_t = self.progress();
        match self.runner.as_ref() {
            Some(runner) if raw_t < 1.0 => {
                runner.behavior().compute_context(raw_t)
            }
            _ => InterpolationContext::identity(),
        }
    }

    /// Set sidechain target positions for animation.
    ///
    /// Call this alongside `set_target` when sidechain data changes.
    /// The residue_indices map each sidechain atom to its residue for collapse
    /// animation.
    pub fn set_sidechain_target(
        &mut self,
        positions: &[Vec3],
        residue_indices: &[u32],
        ca_positions: &[Vec3],
    ) {
        self.set_sidechain_target_with_transition(
            positions,
            residue_indices,
            ca_positions,
            None,
        );
    }

    /// Set sidechain target positions with an explicit transition for
    /// sidechain-only animations.
    ///
    /// If sidechains change but backbone doesn't, this transition will be used
    /// to trigger an animation. Call this BEFORE `set_target()` for proper
    /// animation triggering.
    pub fn set_sidechain_target_with_transition(
        &mut self,
        positions: &[Vec3],
        residue_indices: &[u32],
        ca_positions: &[Vec3],
        transition: Option<&Transition>,
    ) {
        // Clear per-entity snap ranges (will be re-set by
        // remove_non_targeted_from_runner if needed)
        self.snapped_residue_ranges.clear();

        // Check if sidechains actually changed
        let sidechains_changed = self.sidechains_differ(positions);

        // Capture current visual state as the new animation start.
        // Three cases: animating (sync to interpolated), static (use
        // previous target), or size-changed (collapse or snap).
        let sizes_match =
            self.target_sidechain_positions.len() == positions.len();

        if sizes_match
            && self.is_animating()
            && !self.target_sidechain_positions.is_empty()
        {
            // Animation in progress — sync to current interpolated
            // positions to prevent jumps during rapid updates (pulls).
            self.start_sidechain_positions = self.get_sidechain_positions();
            let ctx = self.interpolation_context();
            self.start_ca_positions = self
                .start_ca_positions
                .iter()
                .zip(self.target_ca_positions.iter())
                .map(|(start, target)| {
                    *start + (*target - *start) * ctx.eased_t
                })
                .collect();
        } else if sizes_match {
            // No animation — use previous target as new start.
            self.start_sidechain_positions =
                self.target_sidechain_positions.clone();
            self.start_ca_positions = self.target_ca_positions.clone();
        } else if transition.is_some_and(|t| t.allows_size_change) {
            // Size changed with resize-capable transition — start each
            // sidechain atom at its residue's CA (collapsed) so
            // CollapseExpand can animate them expanding outward.
            self.start_sidechain_positions = residue_indices
                .iter()
                .map(|&ri| {
                    ca_positions.get(ri as usize).copied().unwrap_or(Vec3::ZERO)
                })
                .collect();
            self.start_ca_positions = ca_positions.to_vec();
        } else {
            // Size changed, no resize animation — snap to target.
            self.start_sidechain_positions = positions.to_vec();
            self.start_ca_positions = ca_positions.to_vec();
        }

        self.target_sidechain_positions = positions.to_vec();
        self.target_ca_positions = ca_positions.to_vec();
        self.sidechain_residue_indices = residue_indices.to_vec();

        // Store sidechain change state for set_target() to use
        self.sidechains_changed = sidechains_changed;
        self.pending_sidechain_transition = transition.cloned();
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
        for (old, new) in self
            .target_sidechain_positions
            .iter()
            .zip(new_positions.iter())
        {
            if (*old - *new).length_squared() > EPSILON * EPSILON {
                return true;
            }
        }

        false
    }

    /// Get interpolated sidechain positions using the current animation
    /// behavior.
    ///
    /// This applies the same interpolation logic as backbone animation,
    /// including collapse/expand for mutations.
    pub fn get_sidechain_positions(&self) -> Vec<Vec3> {
        let raw_t = self.progress();

        // If no runner or animation complete, return target positions
        let Some(runner) = self.runner.as_ref() else {
            return self.target_sidechain_positions.clone();
        };
        if raw_t >= 1.0 {
            return self.target_sidechain_positions.clone();
        }
        let behavior = runner.behavior();

        let ctx = behavior.compute_context(raw_t);

        self.start_sidechain_positions
            .iter()
            .zip(self.target_sidechain_positions.iter())
            .enumerate()
            .map(|(i, (start, end))| {
                let res_idx =
                    self.sidechain_residue_indices.get(i).copied().unwrap_or(0)
                        as usize;

                // Skip interpolation for snapped (non-targeted) entities —
                // CollapseExpand's 3-point path (start→CA→end) produces visible
                // motion even when start==end, so we must bypass it entirely.
                if self
                    .snapped_residue_ranges
                    .iter()
                    .any(|&(s, e)| res_idx >= s && res_idx < e)
                {
                    return *end;
                }

                // Get the collapse point (CA position) for this atom's residue
                let start_ca = self
                    .start_ca_positions
                    .get(res_idx)
                    .copied()
                    .unwrap_or(*start);
                let end_ca = self
                    .target_ca_positions
                    .get(res_idx)
                    .copied()
                    .unwrap_or(*end);

                let collapse_point =
                    start_ca + (end_ca - start_ca) * ctx.eased_t;

                behavior.interpolate_position(
                    raw_t,
                    *start,
                    *end,
                    collapse_point,
                )
            })
            .collect()
    }

    /// Snap sidechain and CA positions for entities NOT covered by any
    /// transition.
    ///
    /// For each entity range whose id is NOT in `active_entities`, sets
    /// `start = target` so those residues produce zero displacement during
    /// interpolation (they snap instantly).
    ///
    /// Call this AFTER `set_sidechain_target_with_transition` and `set_target`
    /// when per-entity transitions are in use.
    pub fn snap_entities_without_action<K: std::hash::Hash + Eq + Copy>(
        &mut self,
        entity_residue_ranges: &[(K, u32, u32)],
        active_entities: &std::collections::HashSet<K>,
    ) {
        for &(ref entity_id, start_residue, residue_count) in
            entity_residue_ranges
        {
            if active_entities.contains(entity_id) {
                continue; // This entity has a transition, let it animate
            }

            let res_start = start_residue as usize;
            let res_end = (start_residue + residue_count) as usize;

            // Snap CA positions for this group's residues
            for r in res_start..res_end.min(self.start_ca_positions.len()) {
                if let Some(target) = self.target_ca_positions.get(r) {
                    self.start_ca_positions[r] = *target;
                }
            }

            // Snap sidechain positions for atoms belonging to this group's
            // residues
            for (i, &res_idx) in
                self.sidechain_residue_indices.iter().enumerate()
            {
                let r = res_idx as usize;
                if !(res_start..res_end).contains(&r) {
                    continue;
                }
                if let (Some(target), Some(start)) = (
                    self.target_sidechain_positions.get(i),
                    self.start_sidechain_positions.get_mut(i),
                ) {
                    *start = *target;
                }
            }
        }
    }

    /// Remove non-targeted entity residues from the active AnimationRunner.
    ///
    /// For entities NOT in `active_entities`, removes their residues from
    /// the runner's list so `apply_to_state` never touches them. Also snaps
    /// their backbone `current = target` so they show no visual motion.
    ///
    /// Call this AFTER `set_target` (which creates the runner) and AFTER
    /// `snap_entities_without_action` when per-entity transitions are in use.
    pub fn remove_non_targeted_from_runner<K: std::hash::Hash + Eq + Copy>(
        &mut self,
        entity_residue_ranges: &[(K, u32, u32)],
        active_entities: &std::collections::HashSet<K>,
    ) {
        // Collect residue ranges for non-targeted entities
        let snap_ranges: Vec<(usize, usize)> = entity_residue_ranges
            .iter()
            .filter(|(id, _, _)| !active_entities.contains(id))
            .map(|(_, start, count)| {
                (*start as usize, (*start + *count) as usize)
            })
            .collect();

        if snap_ranges.is_empty() {
            return;
        }

        // Snap backbone state so current = target for those residues
        for &(start, end) in &snap_ranges {
            for r in start..end {
                if let Some(target) = self.state.get_target(r).copied() {
                    self.state.set_current(r, target);
                }
            }
        }

        // Remove from runner so apply_to_state never overwrites them
        if let Some(ref mut runner) = self.runner {
            runner.remove_residue_ranges(&snap_ranges);
        }

        // Store snapped ranges so get_sidechain_positions() can skip them
        self.snapped_residue_ranges = snap_ranges;
    }

    /// Check if sidechain animation state is valid (has data).
    pub fn has_sidechain_data(&self) -> bool {
        !self.target_sidechain_positions.is_empty()
    }

    /// Whether sidechains should be included in animation frames right now.
    ///
    /// Multi-phase behaviors (BackboneThenExpand) hide sidechains during
    /// the backbone-lerp phase so new atoms don't flash at their final
    /// positions.
    pub fn should_include_sidechains(&self) -> bool {
        self.runner.as_ref().is_none_or(|r| {
            r.behavior().should_include_sidechains(self.progress())
        })
    }

    /// Get the CA position of a residue by index.
    /// Returns the interpolated position during animation.
    pub fn get_ca_position(&self, residue_idx: usize) -> Option<Vec3> {
        let target = self.target_ca_positions.get(residue_idx)?;

        // If not animating, return target position
        let Some(runner) = self.runner.as_ref() else {
            return Some(*target);
        };
        let raw_t = self.progress();
        if raw_t >= 1.0 {
            return Some(*target);
        }

        // Use unified context for consistent interpolation with
        // backbone/sidechains
        let start = self.start_ca_positions.get(residue_idx).unwrap_or(target);
        let ctx = runner.behavior().compute_context(raw_t);
        Some(*start + (*target - *start) * ctx.eased_t)
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
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

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
        animator.set_target(&make_backbone(0.0), &Transition::snap());

        assert!(!animator.is_animating());
        assert_eq!(animator.residue_count(), 2);
    }

    #[test]
    fn test_animator_animates_on_change() {
        let mut animator = StructureAnimator::new();
        animator.set_target(&make_backbone(0.0), &Transition::smooth());
        animator.set_target(&make_backbone(10.0), &Transition::smooth());

        assert!(animator.is_animating());
    }

    #[test]
    fn test_animator_skip() {
        let mut animator = StructureAnimator::new();
        animator.set_target(&make_backbone(0.0), &Transition::smooth());
        animator.set_target(&make_backbone(10.0), &Transition::smooth());

        animator.skip();

        assert!(!animator.is_animating());
        let backbone = animator.get_backbone();
        assert!((backbone[0][0].y - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_animator_completes() {
        let mut animator = StructureAnimator::new();
        animator.set_target(&make_backbone(0.0), &Transition::snap());
        animator.set_target(&make_backbone(10.0), &Transition::smooth());

        let future = Instant::now() + Duration::from_secs(1);
        let still_animating = animator.update(future);

        assert!(!still_animating);
        assert!(!animator.is_animating());
    }

    #[test]
    fn test_animator_sidechain_only_change_triggers_animation() {
        let mut animator = StructureAnimator::new();

        // Set initial backbone
        animator.set_target(&make_backbone(5.0), &Transition::snap());
        assert!(!animator.is_animating(), "Initial backbone should snap");

        // Set initial sidechains
        let sidechain_pos =
            vec![Vec3::new(1.0, 5.0, 1.0), Vec3::new(2.0, 5.0, 1.0)];
        let residue_indices = vec![0, 1];
        let ca_positions =
            vec![Vec3::new(1.0, 5.0, 0.0), Vec3::new(4.0, 5.0, 0.0)];
        animator.set_sidechain_target_with_transition(
            &sidechain_pos,
            &residue_indices,
            &ca_positions,
            Some(&Transition::smooth()),
        );

        // Set same backbone again - should trigger animation due to sidechain
        // change
        animator.set_target(&make_backbone(5.0), &Transition::snap());

        // Now change sidechains with same backbone
        let new_sidechain_pos =
            vec![Vec3::new(1.0, 5.0, 2.0), Vec3::new(2.0, 5.0, 2.0)];
        animator.set_sidechain_target_with_transition(
            &new_sidechain_pos,
            &residue_indices,
            &ca_positions,
            Some(&Transition::smooth()),
        );

        // Set same backbone - should animate because sidechains changed
        animator.set_target(&make_backbone(5.0), &Transition::snap());
        assert!(
            animator.is_animating(),
            "Sidechain-only change should trigger animation"
        );
    }

    #[test]
    fn test_sidechains_differ_detects_changes() {
        let mut animator = StructureAnimator::new();

        // Set initial sidechains
        let sidechain_pos = vec![Vec3::new(1.0, 2.0, 3.0)];
        let residue_indices = vec![0];
        let ca_positions = vec![Vec3::ZERO];
        animator.set_sidechain_target(
            &sidechain_pos,
            &residue_indices,
            &ca_positions,
        );

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
