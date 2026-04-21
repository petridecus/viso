//! Animation system for smooth structural transitions.

pub(crate) mod animator;
pub(crate) mod runner;
pub(crate) mod state;
pub mod transition;

pub(crate) use animator::StructureAnimator;
pub(crate) use state::AnimationState;
