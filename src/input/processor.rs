//! Converts raw platform events into engine commands.
//!
//! The `InputProcessor` owns all transient input state (mouse tracking,
//! drag detection, multi-click timing, modifier keys) and the key-binding
//! map.  It is the only thing that sits between raw window events and the
//! engine's [`execute`](crate::VisoEngine::execute) method.

use std::collections::HashMap;

use glam::Vec2;
use serde::{Deserialize, Serialize};

use super::event::{InputEvent, MouseButton};
use super::mouse::{ClickResult, InputState};
use crate::engine::command::VisoCommand;
use crate::renderer::picking::PickTarget;

/// Maps physical key strings to [`VisoCommand`] variants.
///
/// Key strings use the `winit::keyboard::KeyCode` debug format:
/// `"KeyQ"`, `"Tab"`, `"Escape"`, etc.
///
/// Only *discrete* commands (toggles, actions) make sense as key
/// bindings — parameterized commands like `RotateCamera` are produced
/// by the mouse gesture interpreter, not key lookups.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct KeyBindings {
    /// Forward map: key string → command tag.
    bindings: HashMap<String, KeyCommandTag>,
}

/// Serializable tag for the subset of [`VisoCommand`] that can be
/// key-bound (discrete, parameterless actions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KeyCommandTag {
    /// Re-center camera on the current focus.
    RecenterCamera,
    /// Toggle trajectory playback.
    ToggleTrajectory,
    /// Cycle focus through entities.
    CycleFocus,
    /// Toggle turntable auto-rotation.
    ToggleAutoRotate,
    /// Reset focus to session level.
    ResetFocus,
    /// Cancel / clear selection.
    Cancel,
}

impl KeyCommandTag {
    /// Convert to the corresponding parameterless [`VisoCommand`].
    fn to_command(self) -> VisoCommand {
        match self {
            Self::RecenterCamera => VisoCommand::RecenterCamera,
            Self::ToggleTrajectory => VisoCommand::ToggleTrajectory,
            Self::CycleFocus => VisoCommand::CycleFocus,
            Self::ToggleAutoRotate => VisoCommand::ToggleAutoRotate,
            Self::ResetFocus => VisoCommand::ResetFocus,
            Self::Cancel => VisoCommand::ClearSelection,
        }
    }
}

impl Default for KeyBindings {
    fn default() -> Self {
        let bindings = HashMap::from([
            ("KeyQ".into(), KeyCommandTag::RecenterCamera),
            ("KeyT".into(), KeyCommandTag::ToggleTrajectory),
            ("Tab".into(), KeyCommandTag::CycleFocus),
            ("KeyR".into(), KeyCommandTag::ToggleAutoRotate),
            ("Backquote".into(), KeyCommandTag::ResetFocus),
            ("Escape".into(), KeyCommandTag::Cancel),
        ]);
        Self { bindings }
    }
}

impl KeyBindings {
    /// Look up the command for a physical key string.
    #[must_use]
    pub fn lookup(&self, key: &str) -> Option<VisoCommand> {
        self.bindings.get(key).map(|tag| tag.to_command())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// InputProcessor
// ─────────────────────────────────────────────────────────────────────────────

/// Converts raw window events into [`VisoCommand`]s.
///
/// Owns all transient input state (mouse position, drag detection,
/// multi-click timing, modifier keys) and the keyboard binding map.
///
/// # Usage
///
/// ```ignore
/// // In the event loop:
/// for cmd in input_processor.handle_event(event, engine.hovered_target()) {
///     engine.execute(cmd);
/// }
///
/// if let Some(cmd) = input_processor.handle_key_press("KeyQ") {
///     engine.execute(cmd);
/// }
/// ```
pub struct InputProcessor {
    /// Mouse tracking and multi-click state machine.
    state: InputState,
    /// Whether the primary mouse button is currently held.
    mouse_pressed: bool,
    /// Whether the shift modifier is currently held.
    shift_pressed: bool,
    /// Key string → command mapping.
    key_bindings: KeyBindings,
}

impl InputProcessor {
    /// Create a new processor with default key bindings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: InputState::new(),
            mouse_pressed: false,
            shift_pressed: false,
            key_bindings: KeyBindings::default(),
        }
    }

    /// Create a processor with custom key bindings.
    #[must_use]
    pub fn with_key_bindings(key_bindings: KeyBindings) -> Self {
        Self {
            key_bindings,
            ..Self::new()
        }
    }

    /// Current cursor position in physical pixels.
    #[must_use]
    pub fn mouse_pos(&self) -> (f32, f32) {
        self.state.mouse_pos
    }

    /// Whether the primary mouse button is pressed.
    #[must_use]
    pub fn mouse_pressed(&self) -> bool {
        self.mouse_pressed
    }

    /// Whether the shift modifier is held.
    #[must_use]
    pub fn shift_pressed(&self) -> bool {
        self.shift_pressed
    }

    /// Read-only access to the key bindings.
    #[must_use]
    pub fn key_bindings(&self) -> &KeyBindings {
        &self.key_bindings
    }

    /// Mutable access to the key bindings for reconfiguration.
    pub fn key_bindings_mut(&mut self) -> &mut KeyBindings {
        &mut self.key_bindings
    }

    /// Release the mouse button without triggering click detection.
    ///
    /// Used by consumers that intercept mouse events for external drag
    /// operations (e.g. pull/band drag in Foldit) and need to release
    /// the mouse cleanly.
    pub fn release_mouse_state(&mut self) {
        self.mouse_pressed = false;
    }

    /// Look up a key press and return the corresponding command, if bound.
    #[must_use]
    pub fn handle_key_press(&self, key: &str) -> Option<VisoCommand> {
        self.key_bindings.lookup(key)
    }

    /// Process a raw input event and return zero or one commands.
    ///
    /// `hovered` is the pick target currently under the cursor (from the
    /// engine's GPU picking system).
    pub fn handle_event(
        &mut self,
        event: InputEvent,
        hovered: PickTarget,
    ) -> Option<VisoCommand> {
        match event {
            InputEvent::CursorMoved { x, y } => self.handle_cursor_moved(x, y),
            InputEvent::MouseButton { button, pressed } => {
                self.handle_mouse_button(button, pressed, hovered)
            }
            InputEvent::Scroll { delta } => Some(VisoCommand::Zoom { delta }),
            InputEvent::ModifiersChanged { shift } => {
                self.shift_pressed = shift;
                None
            }
        }
    }

    /// Cursor moved — compute delta, possibly produce a camera command.
    fn handle_cursor_moved(&mut self, x: f32, y: f32) -> Option<VisoCommand> {
        let (delta_x, delta_y) = self.state.handle_mouse_position(x, y);

        // Camera rotate/pan only when dragging on background
        if self.mouse_pressed && self.state.mouse_down_target.is_none() {
            let delta = Vec2::new(delta_x, delta_y);
            if delta.length_squared() > 1.0 {
                self.state.mark_dragging();
            }
            if self.shift_pressed {
                return Some(VisoCommand::PanCamera { delta });
            }
            return Some(VisoCommand::RotateCamera { delta });
        }

        None
    }

    /// Mouse button press/release — track state, produce selection commands
    /// on release.
    fn handle_mouse_button(
        &mut self,
        button: MouseButton,
        pressed: bool,
        hovered: PickTarget,
    ) -> Option<VisoCommand> {
        if button != MouseButton::Left {
            return None;
        }

        if pressed {
            self.state.handle_mouse_down(hovered);
            self.mouse_pressed = true;
            return None;
        }

        // Release
        self.mouse_pressed = false;
        self.process_mouse_up(hovered)
    }

    /// Convert a mouse-up into a selection command (if any).
    fn process_mouse_up(&mut self, hovered: PickTarget) -> Option<VisoCommand> {
        let click = self.state.process_mouse_up(hovered, self.shift_pressed);

        match click {
            ClickResult::NoAction => None,
            ClickResult::SingleClick { target, shift_held } => {
                Some(VisoCommand::SelectResidue {
                    index: target.as_residue_i32(),
                    extend: shift_held,
                })
            }
            ClickResult::DoubleClick { target, shift_held } => {
                Some(VisoCommand::SelectSegment {
                    index: target.as_residue_i32(),
                    extend: shift_held,
                })
            }
            ClickResult::TripleClick { target, shift_held } => {
                Some(VisoCommand::SelectChain {
                    index: target.as_residue_i32(),
                    extend: shift_held,
                })
            }
            ClickResult::ClearSelection => Some(VisoCommand::ClearSelection),
        }
    }
}

impl Default for InputProcessor {
    fn default() -> Self {
        Self::new()
    }
}
