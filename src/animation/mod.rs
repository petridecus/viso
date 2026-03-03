//! Animation system for smooth structural transitions.

mod runner;
pub(crate) mod state;
pub mod transition;

pub(crate) mod animator;

pub(crate) use animator::{
    AnimationFrame, EntitySidechainData, StructureAnimator,
};
pub(crate) use runner::SidechainAnimPositions;
pub(crate) use state::AnimationState;
