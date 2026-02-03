//! Animation behaviors define how structural changes are animated.
//!
//! Behaviors are decoupled from actions - the same behavior can be used
//! for different actions, and users can customize which behavior is used
//! for each action type.

mod cascade;
mod collapse_expand;
mod smooth;
mod snap;
mod state;
mod traits;

pub use cascade::Cascade;
pub use collapse_expand::CollapseExpand;
pub use smooth::SmoothInterpolation;
pub use snap::Snap;
pub use state::ResidueVisualState;
pub use traits::{AnimationBehavior, PreemptionStrategy, SharedBehavior, shared};
