//! Animation system for smooth structural transitions.
//!
//! # Architecture
//!
//! The animation system is composed of several layers:
//!
//! - **Behaviors** (`behaviors/`): Define how animations look (easing, phasing, etc.)
//! - **Preferences** (`preferences.rs`): Map actions to behaviors (user-configurable)
//! - **Animator** (`animator/`): Manages state and executes animations
//!
//! # Usage
//!
//! ```ignore
//! use foldit_render::animation::{StructureAnimator, AnimationAction};
//!
//! let mut animator = StructureAnimator::new();
//!
//! // When structure changes
//! animator.set_target(&backbone_chains, AnimationAction::Wiggle);
//!
//! // Each frame
//! animator.update(Instant::now());
//! let visual = animator.get_backbone();
//! ```

pub mod animator;
pub mod behaviors;
pub mod interpolation;
pub mod preferences;

// Re-export commonly used types
pub use animator::{AnimationController, AnimationRunner, StructureAnimator, StructureState};
pub use behaviors::{
    AnimationBehavior, Cascade, CollapseExpand, PreemptionStrategy, ResidueVisualState,
    SharedBehavior, SmoothInterpolation, Snap, shared,
};
pub use interpolation::{InterpolationContext, lerp_position, lerp_f32};
pub use preferences::{AnimationAction, AnimationPreferences};
