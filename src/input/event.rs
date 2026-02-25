/// Platform-agnostic input events for the render engine.
///
/// Consumers forward raw window events as [`InputEvent`] variants;
/// the engine internally dispatches them to camera, picking, and selection.
///
/// # Example
///
/// ```ignore
/// engine.handle_input(InputEvent::CursorMoved { x: 100.0, y: 200.0 });
/// engine.handle_input(InputEvent::MouseButton {
///     button: MouseButton::Left,
///     pressed: true,
/// });
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InputEvent {
    /// Cursor moved to absolute screen position.
    CursorMoved {
        /// Horizontal position in physical pixels.
        x: f32,
        /// Vertical position in physical pixels.
        y: f32,
    },
    /// Mouse button pressed or released.
    MouseButton {
        /// Which button changed.
        button: MouseButton,
        /// `true` for press, `false` for release.
        pressed: bool,
    },
    /// Scroll wheel (positive = zoom in).
    Scroll {
        /// Scroll amount (positive = zoom in, negative = zoom out).
        delta: f32,
    },
    /// Modifier key state changed.
    ModifiersChanged {
        /// Whether the shift key is held.
        shift: bool,
    },
}

/// Platform-agnostic mouse button identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MouseButton {
    /// Primary (left) mouse button.
    Left,
    /// Secondary (right) mouse button.
    Right,
    /// Middle mouse button (wheel click).
    Middle,
}

#[cfg(feature = "viewer")]
impl From<winit::event::MouseButton> for MouseButton {
    fn from(button: winit::event::MouseButton) -> Self {
        match button {
            winit::event::MouseButton::Right => Self::Right,
            winit::event::MouseButton::Middle => Self::Middle,
            _ => Self::Left,
        }
    }
}
