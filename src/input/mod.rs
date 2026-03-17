//! Input handling: event types, state machines, and the input processor
//! that converts raw window events into engine commands.

/// Platform-agnostic input events.
pub mod event;
/// Multi-click state machine and mouse position tracking.
pub(crate) mod mouse;
/// Converts raw events into engine commands.
pub mod processor;

pub use event::{InputEvent, MouseButton};
pub use processor::{InputProcessor, KeyBindings};
