//! Input handling: event types, state machines, and the input processor
//! that converts raw window events into engine commands.

/// Platform-agnostic input events.
pub(crate) mod event;
/// Multi-click state machine and mouse position tracking.
pub(crate) mod mouse;
/// Converts raw events into engine commands.
pub(crate) mod processor;

pub use event::{InputEvent, MouseButton};
pub use processor::{InputProcessor, KeyBindings};
