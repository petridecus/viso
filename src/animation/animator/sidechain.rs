//! Sidechain animation data and interpolation.
//!
//! Encapsulates all sidechain-related animation state and methods,
//! following the same pattern as [`StructureState`](super::StructureState)
//! for backbone data.

use glam::Vec3;

use super::runner::AnimationRunner;
use crate::animation::interpolation::InterpolationContext;
use crate::animation::transition::Transition;

/// Animation state from the active runner, passed to sidechain methods
/// that need interpolation info.
#[derive(Clone, Copy)]
pub(super) struct AnimationContext<'a> {
    /// Raw animation progress (0.0 to 1.0).
    pub raw_t: f32,
    /// The active animation runner, if any.
    pub runner: Option<&'a AnimationRunner>,
    /// Whether any animation (global or per-entity) is in progress.
    pub is_animating: bool,
}

/// Sidechain target data passed to
/// [`SidechainAnimationData::set_target`].
pub(super) struct SidechainTarget<'a> {
    /// Per-atom sidechain positions.
    pub positions: &'a [Vec3],
    /// Residue index for each sidechain atom.
    pub residue_indices: &'a [u32],
    /// CA positions per residue (indexed by residue index).
    pub ca_positions: &'a [Vec3],
}

/// Sidechain animation state: start/target positions, residue indices,
/// CA positions for collapse, snapped ranges, and change tracking.
pub(super) struct SidechainAnimationData {
    /// Start sidechain positions (animation begin state).
    start_positions: Vec<Vec3>,
    /// Target sidechain positions (animation end state).
    target_positions: Vec<Vec3>,
    /// Residue index for each sidechain atom (for collapse point lookup).
    residue_indices: Vec<u32>,
    /// CA positions per residue for collapse animation (indexed by residue).
    start_ca: Vec<Vec3>,
    /// Target CA positions per residue.
    target_ca: Vec<Vec3>,
    /// Residue ranges that have been snapped (non-targeted entities).
    /// Sidechain atoms in these ranges skip interpolation entirely.
    snapped_ranges: Vec<(usize, usize)>,
    /// Flag indicating sidechains changed (for triggering animation even
    /// when backbone is static).
    pub changed: bool,
    /// Pending transition for sidechain-only animation.
    pub pending_transition: Option<Transition>,
}

impl SidechainAnimationData {
    /// Create empty sidechain animation data.
    pub fn new() -> Self {
        Self {
            start_positions: Vec::new(),
            target_positions: Vec::new(),
            residue_indices: Vec::new(),
            start_ca: Vec::new(),
            target_ca: Vec::new(),
            snapped_ranges: Vec::new(),
            changed: false,
            pending_transition: None,
        }
    }

    /// Whether sidechain data is present.
    pub fn has_data(&self) -> bool {
        !self.target_positions.is_empty()
    }

    /// Take the pending sidechain transition, leaving `None`.
    pub fn take_pending_transition(&mut self) -> Option<Transition> {
        self.pending_transition.take()
    }

    /// Check if new sidechain positions differ from current target.
    fn differs(&self, new_positions: &[Vec3]) -> bool {
        // Size change means difference
        if self.target_positions.len() != new_positions.len() {
            return !new_positions.is_empty();
        }

        // Empty means no change
        if new_positions.is_empty() {
            return false;
        }

        // Compare positions with small epsilon
        const EPSILON: f32 = 0.001;
        for (old, new) in self.target_positions.iter().zip(new_positions.iter())
        {
            if (*old - *new).length_squared() > EPSILON * EPSILON {
                return true;
            }
        }

        false
    }

    /// Set sidechain target positions with an explicit transition for
    /// sidechain-only animations.
    ///
    /// If sidechains change but backbone doesn't, this transition will be
    /// used to trigger an animation. Call this BEFORE
    /// `StructureAnimator::set_target()` for proper animation triggering.
    pub fn set_target(
        &mut self,
        target: &SidechainTarget<'_>,
        transition: Option<&Transition>,
        anim: AnimationContext<'_>,
    ) {
        // Clear per-entity snap ranges (will be re-set by
        // remove_non_targeted_from_runner if needed)
        self.snapped_ranges.clear();

        // Check if sidechains actually changed
        let sidechains_changed = self.differs(target.positions);

        // Capture current visual state as the new animation start.
        // Three cases: animating (sync to interpolated), static (use
        // previous target), or size-changed (collapse or snap).
        let sizes_match = self.target_positions.len() == target.positions.len();

        if sizes_match && anim.is_animating && !self.target_positions.is_empty()
        {
            // Animation in progress — sync to current interpolated
            // positions to prevent jumps during rapid updates (pulls).
            self.start_positions = self.get_positions(anim);
            let ctx = match anim.runner {
                Some(runner) if anim.raw_t < 1.0 => {
                    runner.behavior().compute_context(anim.raw_t)
                }
                _ => InterpolationContext::identity(),
            };
            self.start_ca = self
                .start_ca
                .iter()
                .zip(self.target_ca.iter())
                .map(|(start, end)| *start + (*end - *start) * ctx.eased_t)
                .collect();
        } else if sizes_match {
            // No animation — use previous target as new start.
            self.start_positions = self.target_positions.clone();
            self.start_ca = self.target_ca.clone();
        } else if transition.is_some_and(|t| t.allows_size_change) {
            // Size changed with resize-capable transition — start each
            // sidechain atom at its residue's CA (collapsed) so
            // CollapseExpand can animate them expanding outward.
            self.start_positions = target
                .residue_indices
                .iter()
                .map(|&ri| {
                    target
                        .ca_positions
                        .get(ri as usize)
                        .copied()
                        .unwrap_or(Vec3::ZERO)
                })
                .collect();
            self.start_ca = target.ca_positions.to_vec();
        } else {
            // Size changed, no resize animation — snap to target.
            self.start_positions = target.positions.to_vec();
            self.start_ca = target.ca_positions.to_vec();
        }

        self.target_positions = target.positions.to_vec();
        self.target_ca = target.ca_positions.to_vec();
        self.residue_indices = target.residue_indices.to_vec();

        // Store sidechain change state for set_target() to use
        self.changed = sidechains_changed;
        self.pending_transition = transition.cloned();
    }

    /// Get interpolated sidechain positions using the current animation
    /// behavior.
    ///
    /// Applies the same interpolation logic as backbone animation,
    /// including collapse/expand for mutations.
    pub fn get_positions(&self, anim: AnimationContext<'_>) -> Vec<Vec3> {
        // If no runner or animation complete, return target positions
        let Some(runner) = anim.runner else {
            return self.target_positions.clone();
        };
        if anim.raw_t >= 1.0 {
            return self.target_positions.clone();
        }
        let behavior = runner.behavior();

        let ctx = behavior.compute_context(anim.raw_t);

        self.start_positions
            .iter()
            .zip(self.target_positions.iter())
            .enumerate()
            .map(|(i, (start, end))| {
                let res_idx =
                    self.residue_indices.get(i).copied().unwrap_or(0) as usize;

                // Skip interpolation for snapped (non-targeted) entities —
                // CollapseExpand's 3-point path (start→CA→end) produces
                // visible motion even when start==end, so we must bypass
                // it entirely.
                if self
                    .snapped_ranges
                    .iter()
                    .any(|&(s, e)| res_idx >= s && res_idx < e)
                {
                    return *end;
                }

                // Get the collapse point (CA position) for this atom's
                // residue
                let start_ca =
                    self.start_ca.get(res_idx).copied().unwrap_or(*start);
                let end_ca =
                    self.target_ca.get(res_idx).copied().unwrap_or(*end);

                let collapse_point =
                    start_ca + (end_ca - start_ca) * ctx.eased_t;

                behavior.interpolate_position(
                    anim.raw_t,
                    *start,
                    *end,
                    collapse_point,
                )
            })
            .collect()
    }

    /// Get the CA position of a residue by index.
    /// Returns the interpolated position during animation.
    pub fn get_ca(
        &self,
        residue_idx: usize,
        anim: AnimationContext<'_>,
    ) -> Option<Vec3> {
        let target = self.target_ca.get(residue_idx)?;

        // If not animating, return target position
        let Some(runner) = anim.runner else {
            return Some(*target);
        };
        if anim.raw_t >= 1.0 {
            return Some(*target);
        }

        // Use unified context for consistent interpolation with
        // backbone/sidechains
        let start = self.start_ca.get(residue_idx).unwrap_or(target);
        let ctx = runner.behavior().compute_context(anim.raw_t);
        Some(*start + (*target - *start) * ctx.eased_t)
    }

    /// Snap sidechain and CA positions for entities NOT covered by any
    /// transition.
    ///
    /// For each entity range whose id is NOT in `active_entities`, sets
    /// `start = target` so those residues produce zero displacement during
    /// interpolation (they snap instantly).
    ///
    /// Call this AFTER `set_target` and `StructureAnimator::set_target`
    /// when per-entity transitions are in use.
    pub fn snap_non_targeted(
        &mut self,
        entity_residue_ranges: &[crate::scene::EntityResidueRange],
        active_entities: &std::collections::HashSet<u32>,
    ) {
        for range in entity_residue_ranges {
            if active_entities.contains(&range.entity_id) {
                continue; // This entity has a transition, let it animate
            }

            let res_start = range.start as usize;
            let res_end = range.end() as usize;

            // Snap CA positions for this group's residues
            for r in res_start..res_end.min(self.start_ca.len()) {
                if let Some(target) = self.target_ca.get(r) {
                    self.start_ca[r] = *target;
                }
            }

            // Snap sidechain positions for atoms belonging to this
            // group's residues
            for (i, &res_idx) in self.residue_indices.iter().enumerate() {
                let r = res_idx as usize;
                if !(res_start..res_end).contains(&r) {
                    continue;
                }
                if let (Some(target), Some(start)) = (
                    self.target_positions.get(i),
                    self.start_positions.get_mut(i),
                ) {
                    *start = *target;
                }
            }
        }
    }

    /// Store snapped ranges so `get_positions()` can skip them.
    pub fn set_snapped_ranges(&mut self, ranges: Vec<(usize, usize)>) {
        self.snapped_ranges = ranges;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sidechains_differ_detects_changes() {
        let mut sc = SidechainAnimationData::new();

        // Set initial sidechains via set_target
        let sidechain_pos = vec![Vec3::new(1.0, 2.0, 3.0)];
        let residue_indices = vec![0];
        let ca_positions = vec![Vec3::ZERO];
        let no_anim = AnimationContext {
            raw_t: 1.0,
            runner: None,
            is_animating: false,
        };
        let target = SidechainTarget {
            positions: &sidechain_pos,
            residue_indices: &residue_indices,
            ca_positions: &ca_positions,
        };
        sc.set_target(&target, None, no_anim);

        // Same positions should not differ
        assert!(!sc.differs(&sidechain_pos));

        // Different positions should differ
        let different_pos = vec![Vec3::new(1.0, 2.0, 4.0)];
        assert!(sc.differs(&different_pos));

        // Very small change should not differ (within epsilon)
        let tiny_change = vec![Vec3::new(1.0, 2.0, 3.0005)];
        assert!(!sc.differs(&tiny_change));
    }
}
