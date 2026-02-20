use serde::{Deserialize, Serialize};

/// Engine-level actions that can be bound to keys.
///
/// Serde serializes as `snake_case` strings so TOML presets stay readable:
/// ```toml
/// [keybindings]
/// cycle_focus = "Tab"
/// toggle_waters = "KeyU"
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KeyAction {
    /// Re-center camera on the current focus.
    RecenterCamera,
    /// Toggle trajectory playback.
    ToggleTrajectory,
    /// Toggle ion visibility.
    ToggleIons,
    /// Toggle water molecule visibility.
    ToggleWaters,
    /// Toggle solvent visibility.
    ToggleSolvent,
    /// Toggle lipid visibility.
    ToggleLipids,
    /// Cycle focus through groups and entities.
    CycleFocus,
    /// Toggle turntable auto-rotation.
    ToggleAutoRotate,
    /// Reset focus to session level.
    ResetFocus,
    /// Cancel current operation.
    Cancel,
}
