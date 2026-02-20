//! Input handling for keyboard actions and mouse interaction.
//!
//! Defines key-bindable actions and a multi-click state machine for
//! residue selection.

pub mod keyboard;
pub mod mouse;

pub use keyboard::KeyAction;
pub use mouse::{ClickResult, InputState};
