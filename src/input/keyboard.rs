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
    RecenterCamera,
    ToggleTrajectory,
    ToggleIons,
    ToggleWaters,
    ToggleSolvent,
    ToggleLipids,
    CycleFocus,
    ToggleAutoRotate,
    ResetFocus,
    Cancel,
}
