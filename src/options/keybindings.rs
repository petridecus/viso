use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::input::KeyAction;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
/// Configurable keyboard bindings mapping actions to key codes.
pub struct KeybindingOptions {
    /// Maps action → key string (e.g. `CycleFocus` → `"Tab"`).
    pub bindings: HashMap<KeyAction, String>,
    /// Reverse lookup cache (key string → action). Rebuilt on load.
    #[serde(skip)]
    key_to_action: HashMap<String, KeyAction>,
}

impl Default for KeybindingOptions {
    fn default() -> Self {
        let bindings = HashMap::from([
            (KeyAction::RecenterCamera, "KeyQ".into()),
            (KeyAction::ToggleTrajectory, "KeyT".into()),
            (KeyAction::ToggleIons, "KeyI".into()),
            (KeyAction::ToggleWaters, "KeyU".into()),
            (KeyAction::ToggleSolvent, "KeyO".into()),
            (KeyAction::ToggleLipids, "KeyL".into()),
            (KeyAction::CycleFocus, "Tab".into()),
            (KeyAction::ToggleAutoRotate, "KeyR".into()),
            (KeyAction::ResetFocus, "Backquote".into()),
            (KeyAction::Cancel, "Escape".into()),
        ]);

        let mut opts = Self {
            bindings,
            key_to_action: HashMap::new(),
        };
        opts.rebuild_reverse_map();
        opts
    }
}

impl KeybindingOptions {
    /// Rebuild the reverse lookup map (key string → action).
    pub fn rebuild_reverse_map(&mut self) {
        self.key_to_action.clear();
        for (action, key) in &self.bindings {
            let _ = self.key_to_action.insert(key.clone(), *action);
        }
    }

    /// Look up the action for a key string.
    #[must_use]
    pub fn lookup(&self, key: &str) -> Option<KeyAction> {
        self.key_to_action.get(key).copied()
    }
}
