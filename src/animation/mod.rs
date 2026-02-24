//! Animation system for smooth structural transitions.
//!
//! # Architecture
//!
//! The animation system is composed of several layers:
//!
//! - **Behaviors** (`behaviors/`): Define how animations look (easing, phasing,
//!   etc.)
//! - **Transition** (`transition.rs`): Bundles a behavior with metadata flags
//! - **Animator** (`animator/`): Manages state and executes animations
//!
//! # Usage
//!
//! ```ignore
//! use viso::{Transition, StructureAnimator};
//!
//! let mut animator = StructureAnimator::new();
//!
//! // When structure changes
//! animator.set_target(&backbone_chains, &Transition::smooth());
//!
//! // Each frame
//! animator.update(Instant::now());
//! let visual = animator.get_backbone();
//! ```

pub mod animator;
pub mod behaviors;
pub mod interpolation;
pub(crate) mod sidechain_state;
pub mod transition;
