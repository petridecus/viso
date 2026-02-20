//! Input handling for keyboard actions and mouse interaction.
//!
//! Defines key-bindable actions and a multi-click state machine for
//! residue selection.

/// Key-bindable actions for camera and display toggles.
pub mod keyboard;
/// Multi-click state machine and mouse position tracking.
pub mod mouse;

pub use keyboard::KeyAction;
pub use mouse::{ClickResult, InputState};
