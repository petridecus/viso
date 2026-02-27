//! Animation system for smooth structural transitions.

mod runner;
pub mod transition;

pub(crate) mod animator;
pub(crate) mod easing;

pub(crate) use animator::StructureAnimator;
pub(crate) use runner::SidechainAnimPositions;
