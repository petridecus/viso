//! Platform-agnostic bridge between the viso engine and viso-ui.
//!
//! Contains the IPC action types, parsing, entity serialization, structure
//! parsing, and the shared bridge JavaScript that both native (wry) and web
//! (wasm) hosts inject into viso-ui.

use crate::engine::command::VisoCommand;
use crate::engine::scene::Focus;
use crate::VisoEngine;

// ── Panel layout model ──────────────────────────────────────────────────

/// Panel axis: right sidebar or bottom bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelAxis {
    /// Panel docked to the right edge (landscape orientation).
    Right,
    /// Panel docked to the bottom edge (portrait orientation).
    Bottom,
}

impl PanelAxis {
    /// Determine axis from window dimensions.
    #[must_use]
    pub fn from_dimensions(width: u32, height: u32) -> Self {
        if height > width {
            Self::Bottom
        } else {
            Self::Right
        }
    }

    /// Returns `"portrait"` or `"landscape"` for the viso-ui orientation
    /// event.
    #[must_use]
    pub fn orientation_str(self) -> &'static str {
        match self {
            Self::Bottom => "portrait",
            Self::Right => "landscape",
        }
    }
}

/// Default panel size in physical pixels.
pub const DEFAULT_PANEL_SIZE: u32 = 340;

/// Minimum panel size for resize.
pub const MIN_PANEL_SIZE: u32 = 220;

/// Maximum panel size for resize.
pub const MAX_PANEL_SIZE: u32 = 700;

/// Panel size when collapsed (just the toggle arrow strip).
///
/// Used by the web host to set the iframe's collapsed dimension.
#[cfg(target_arch = "wasm32")]
pub const COLLAPSED_SIZE: u32 = 32;

// ── UiAction ─────────────────────────────────────────────────────────────

/// Actions sent from the viso-ui WASM app to the host engine.
#[derive(Debug)]
pub enum UiAction {
    /// Set a single option field: `options[section][field] = value`.
    SetOption {
        /// Top-level section key (e.g. `"lighting"`).
        path: String,
        /// Field key within the section (e.g. `"roughness"`).
        field: String,
        /// New JSON value.
        value: serde_json::Value,
    },
    /// Load a structure file by path.
    LoadFile {
        /// Filesystem path to the `.cif` or `.pdb` file.
        path: String,
    },
    /// Fetch a PDB structure by ID from a remote database.
    FetchPdb {
        /// 4-character PDB identifier (e.g. `"1crn"`).
        id: String,
        /// Source database: `"rcsb"` or `"pdb-redo"`.
        source: String,
    },
    /// Open a native file dialog to pick a local structure file.
    OpenFileDialog,
    /// Toggle the panel between pinned and unpinned.
    TogglePanel,
    /// Resize the panel along its current axis.
    ResizePanel {
        /// New panel size in physical pixels (width or height depending
        /// on axis).
        size: u32,
    },
    /// A key press forwarded from the webview (e.g. Tab on Windows).
    KeyPress {
        /// Physical key name matching winit's `KeyCode` debug format.
        key: String,
    },
    /// An engine command to forward via `engine.execute()`.
    Command(VisoCommand),
}

// ── IPC parsing ──────────────────────────────────────────────────────────

/// Parse an IPC message from viso-ui into a [`UiAction`].
pub fn parse_action(msg: &serde_json::Value) -> Option<UiAction> {
    let action = msg.get("action")?.as_str()?;
    match action {
        "set_option" => {
            let path = msg.get("path")?.as_str()?.to_owned();
            let field = msg.get("field")?.as_str()?.to_owned();
            let value = msg.get("value")?.clone();
            Some(UiAction::SetOption { path, field, value })
        }
        "load_file" => {
            let path = msg.get("path")?.as_str()?.to_owned();
            Some(UiAction::LoadFile { path })
        }
        "fetch_pdb" => {
            let id = msg.get("id")?.as_str()?.to_owned();
            let source = msg.get("source")?.as_str()?.to_owned();
            Some(UiAction::FetchPdb { id, source })
        }
        "open_file_dialog" => Some(UiAction::OpenFileDialog),
        "key" => {
            let key = msg.get("key")?.as_str()?.to_owned();
            Some(UiAction::KeyPress { key })
        }
        "toggle_panel" => Some(UiAction::TogglePanel),
        "resize_panel" => {
            let size =
                msg.get("size").or_else(|| msg.get("width"))?.as_u64()? as u32;
            Some(UiAction::ResizePanel { size })
        }
        "focus_entity" => {
            let id = msg.get("id")?.as_u64()? as u32;
            Some(UiAction::Command(VisoCommand::FocusEntity { id }))
        }
        "toggle_entity_visibility" => {
            let id = msg.get("id")?.as_u64()? as u32;
            Some(UiAction::Command(VisoCommand::ToggleEntityVisibility {
                id,
            }))
        }
        "remove_entity" => {
            let id = msg.get("id")?.as_u64()? as u32;
            Some(UiAction::Command(VisoCommand::RemoveEntity { id }))
        }
        _ => None,
    }
}

// ── JS escaping ──────────────────────────────────────────────────────────

/// Escape a string for safe embedding in a JavaScript single-quoted
/// literal.
pub fn escape_for_js(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\'', "\\'")
}

// ── Entity summaries ─────────────────────────────────────────────────────

/// Build a JSON-serializable summary of all entities for the viso-ui
/// panel.
pub fn entity_summaries(engine: &VisoEngine) -> Vec<serde_json::Value> {
    use molex::types::entity::MoleculeType;

    let focus = engine.entities.focus();
    engine
        .entities
        .entities()
        .iter()
        .map(|se| {
            let mol_type = match se.entity.molecule_type {
                MoleculeType::Protein => "Protein",
                MoleculeType::DNA => "DNA",
                MoleculeType::RNA => "RNA",
                _ => "Ligand",
            };
            let chain_ids: Vec<String> =
                se.entity.as_polymer().map_or_else(Vec::new, |data| {
                    data.chains
                        .iter()
                        .map(|c| String::from(c.chain_id as char))
                        .collect()
                });
            let focused = matches!(
                focus,
                Focus::Entity(eid) if *eid == se.id()
            );
            serde_json::json!({
                "id": se.id(),
                "molecule_type": mol_type,
                "label": se.entity.label(),
                "visible": se.visible,
                "atom_count": se.entity.atom_count(),
                "chain_ids": chain_ids,
                "focused": focused,
                "focusable": se.entity.is_focusable(),
            })
        })
        .collect()
}

// ── Structure parsing ────────────────────────────────────────────────────

/// Parse a structure file from in-memory bytes.
///
/// `format_hint` should be the file extension (e.g. `"cif"`, `"pdb"`),
/// with or without a leading dot.
#[cfg(target_arch = "wasm32")]
pub fn parse_structure_bytes(
    bytes: &[u8],
    format_hint: &str,
) -> Result<Vec<molex::types::entity::MoleculeEntity>, String> {
    let hint = format_hint.to_ascii_lowercase();
    let hint = hint.trim_start_matches('.');
    match hint {
        "cif" | "mmcif" => {
            let text = std::str::from_utf8(bytes)
                .map_err(|e| format!("Invalid UTF-8 in CIF: {e}"))?;
            molex::adapters::pdb::mmcif_str_to_entities(text)
                .map_err(|e| format!("CIF parse error: {e}"))
        }
        "pdb" | "ent" => {
            let text = std::str::from_utf8(bytes)
                .map_err(|e| format!("Invalid UTF-8 in PDB: {e}"))?;
            molex::adapters::pdb::pdb_str_to_entities(text)
                .map_err(|e| format!("PDB parse error: {e}"))
        }
        other => {
            Err(format!("Unsupported format '{other}'. Use 'cif' or 'pdb'."))
        }
    }
}

// ── Bridge JavaScript ────────────────────────────────────────────────────

/// JavaScript bridge injected into viso-ui that defines the
/// `window.__viso_push_*` functions and dispatches `CustomEvent`s.
///
/// Both native (wry) and web (wasm) hosts use this same core script.
/// The native host appends additional platform-specific code for Tab key
/// forwarding and `addEventListener` replay.
pub const BRIDGE_JS: &str = r"
(function() {
    var pending = {};
    function dispatch(name, json) {
        window.dispatchEvent(new CustomEvent(name, { detail: json }));
    }
    function makePush(key, eventName) {
        window['__viso_push_' + key] = function(json) {
            pending[key] = json;
            dispatch(eventName, json);
        };
    }
    makePush('schema', 'viso-schema');
    makePush('options', 'viso-options');
    makePush('stats', 'viso-stats');
    makePush('panel_pinned', 'viso-panel-pinned');
    makePush('load_status', 'viso-load-status');
    makePush('scene_entities', 'viso-scene-entities');
    makePush('orientation', 'viso-orientation');

    // Allow late listeners (e.g. dioxus WASM) to replay any values
    // that were pushed before they registered.
    window.__viso_replay_pending = function() {
        for (var k in pending) {
            var fn = window['__viso_push_' + k];
            if (fn) fn(pending[k]);
        }
    };

    // Replay any early pushes
    if (window.__viso_early) {
        var e = window.__viso_early;
        for (var k in e) {
            if (window['__viso_push_' + k]) {
                window['__viso_push_' + k](e[k]);
            }
        }
        delete window.__viso_early;
    }
})();
";
