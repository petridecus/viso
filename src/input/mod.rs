//! Input handling for keyboard actions and mouse interaction.
//!
//! Defines key-bindable actions, a multi-click state machine for
//! residue selection, and platform-agnostic input events.

/// Platform-agnostic input events.
pub mod event;
/// Key-bindable actions for camera and display toggles.
pub mod keyboard;
/// Multi-click state machine and mouse position tracking.
pub mod mouse;

pub use event::{InputEvent, MouseButton};
pub use keyboard::KeyAction;
pub use mouse::{ClickResult, InputState};
