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
pub(crate) mod sidechain_state;

// Re-export commonly used types
pub use animator::{AnimationController, AnimationRunner, StructureAnimator, StructureState};
pub use behaviors::{
    shared, AnimationBehavior, Cascade, CollapseExpand, PreemptionStrategy, ResidueVisualState,
    SharedBehavior, SmoothInterpolation, Snap,
};
pub use interpolation::{lerp_f32, lerp_position, InterpolationContext};
pub use preferences::{AnimationAction, AnimationPreferences};
