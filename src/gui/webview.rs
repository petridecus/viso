//! Wry webview child of the winit window.
//!
//! Creates a [`wry::WebView`] positioned at the right edge of the window,
//! loads the viso-ui WASM bundle via a custom `viso://` protocol, and
//! bridges IPC between the Dioxus web app and the native engine.

use std::{borrow::Cow, sync::mpsc};

use rust_embed::RustEmbed;
use wry::{
    dpi,
    http::{header::CONTENT_TYPE, Response},
    Rect, WebView, WebViewBuilder,
};

use crate::options::Options;

/// Embedded viso-ui dist output (built by `trunk build`).
#[derive(RustEmbed)]
#[folder = "crates/viso-ui/dist/"]
struct UiAssets;

/// Width of the options panel in logical pixels.
pub const PANEL_WIDTH: u32 = 350;

/// Actions sent from the webview WASM app to the native engine.
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
    /// Toggle the panel between pinned and unpinned.
    TogglePanel,
    /// Resize the panel to a new width.
    ResizePanel {
        /// New panel width in physical pixels.
        width: u32,
    },
}

/// Create the wry webview as a child of the given window.
///
/// Returns `(webview, action_rx)` — the receiver yields [`UiAction`]s
/// from the WASM app.
pub fn create_webview<W: wry::raw_window_handle::HasWindowHandle>(
    window: &W,
    window_width: u32,
    window_height: u32,
    panel_width: u32,
) -> Result<(WebView, mpsc::Receiver<UiAction>), wry::Error> {
    let (tx, rx) = mpsc::channel();

    let bounds = panel_bounds(window_width, window_height, panel_width);

    let webview = WebViewBuilder::new()
        .with_bounds(bounds)
        .with_transparent(true)
        .with_custom_protocol("viso".into(), |_id, request| {
            let path = request.uri().path();
            // Default to index.html for the root path.
            let path = if path == "/" {
                "index.html"
            } else {
                &path[1..]
            };

            match UiAssets::get(path) {
                Some(asset) => {
                    let mime = mime_guess::from_path(path)
                        .first_or_octet_stream()
                        .to_string();
                    Response::builder()
                        .header(CONTENT_TYPE, mime)
                        .body(Cow::from(asset.data.to_vec()))
                        .unwrap_or_else(|_| {
                            Response::new(Cow::from(Vec::new()))
                        })
                }
                None => Response::builder()
                    .status(404)
                    .body(Cow::from(Vec::new()))
                    .unwrap_or_else(|_| Response::new(Cow::from(Vec::new()))),
            }
        })
        .with_url("viso://localhost/")
        .with_initialization_script(BRIDGE_JS)
        .with_ipc_handler(move |req| {
            let body = req.body();
            if let Ok(msg) = serde_json::from_str::<serde_json::Value>(body) {
                if let Some(action) = parse_action(&msg) {
                    let _ = tx.send(action);
                }
            }
        })
        .build_as_child(window)?;

    Ok((webview, rx))
}

/// Compute the [`Rect`] for the pinned panel at the right edge of the
/// window.
#[must_use]
pub fn panel_bounds(
    window_width: u32,
    window_height: u32,
    panel_width: u32,
) -> Rect {
    let x = window_width.saturating_sub(panel_width);
    Rect {
        position: dpi::Position::Physical(dpi::PhysicalPosition::new(
            x as i32, 0,
        )),
        size: dpi::Size::Physical(dpi::PhysicalSize::new(
            panel_width.min(window_width),
            window_height,
        )),
    }
}

/// Compute the [`Rect`] for the floating (unpinned) panel, inset by
/// `margin` on all sides.
#[must_use]
pub fn panel_bounds_floating(
    window_width: u32,
    window_height: u32,
    panel_width: u32,
    margin: u32,
) -> Rect {
    let x = window_width.saturating_sub(panel_width + margin);
    Rect {
        position: dpi::Position::Physical(dpi::PhysicalPosition::new(
            x as i32, margin as i32,
        )),
        size: dpi::Size::Physical(dpi::PhysicalSize::new(
            panel_width.min(window_width),
            window_height.saturating_sub(margin * 2),
        )),
    }
}

/// Push the Options JSON schema to the webview (call once after creation).
pub fn push_schema(webview: &WebView, options: &Options) {
    let schema = schemars::schema_for!(Options);
    let json = serde_json::to_string(&schema).unwrap_or_default();
    let escaped = json.replace('\\', "\\\\").replace('\'', "\\'");
    let _ = webview
        .evaluate_script(&format!("window.__viso_push_schema('{escaped}')"));

    push_options(webview, options);
}

/// Push the current Options state to the webview.
pub fn push_options(webview: &WebView, options: &Options) {
    let json = serde_json::to_string(options).unwrap_or_default();
    let escaped = json.replace('\\', "\\\\").replace('\'', "\\'");
    let _ = webview
        .evaluate_script(&format!("window.__viso_push_options('{escaped}')"));
}

/// Push stats (e.g. FPS) to the webview.
pub fn push_stats(webview: &WebView, fps: f32) {
    let json = serde_json::json!({ "fps": fps });
    let s = json.to_string().replace('\\', "\\\\").replace('\'', "\\'");
    let _ =
        webview.evaluate_script(&format!("window.__viso_push_stats('{s}')"));
}

/// Push the panel pinned state to the webview.
pub fn push_panel_pinned(webview: &WebView, pinned: bool) {
    let _ = webview.evaluate_script(&format!(
        "window.__viso_push_panel_pinned('{}')",
        if pinned { "true" } else { "false" }
    ));
}

// ── Internals ────────────────────────────────────────────────────────────

/// JavaScript injected before page load. Defines the bridge functions that
/// the Dioxus WASM code calls, and dispatches `CustomEvent`s.
///
/// Calls that arrive before the WASM app has registered listeners are
/// buffered. When a listener attaches it replays any pending data.
const BRIDGE_JS: &str = r#"
(function() {
    var pending = { schema: null, options: null, stats: null, panel_pinned: null };

    function dispatch(name, json) {
        window.dispatchEvent(new CustomEvent(name, { detail: json }));
    }

    window.__viso_push_schema = function(json) {
        pending.schema = json;
        dispatch('viso-schema', json);
    };
    window.__viso_push_options = function(json) {
        pending.options = json;
        dispatch('viso-options', json);
    };
    window.__viso_push_stats = function(json) {
        pending.stats = json;
        dispatch('viso-stats', json);
    };
    window.__viso_push_panel_pinned = function(val) {
        pending.panel_pinned = val;
        dispatch('viso-panel-pinned', val);
    };

    // When the WASM app adds a listener, replay buffered data.
    var origAdd = EventTarget.prototype.addEventListener;
    EventTarget.prototype.addEventListener = function(type, fn, opts) {
        origAdd.call(this, type, fn, opts);
        if (this === window && type === 'viso-schema' && pending.schema) {
            dispatch('viso-schema', pending.schema);
        }
        if (this === window && type === 'viso-options' && pending.options) {
            dispatch('viso-options', pending.options);
        }
        if (this === window && type === 'viso-stats' && pending.stats) {
            dispatch('viso-stats', pending.stats);
        }
        if (this === window && type === 'viso-panel-pinned' && pending.panel_pinned !== null) {
            dispatch('viso-panel-pinned', pending.panel_pinned);
        }
    };
})();
"#;

/// Parse an IPC message from the WASM side into a [`UiAction`].
fn parse_action(msg: &serde_json::Value) -> Option<UiAction> {
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
        "toggle_panel" => Some(UiAction::TogglePanel),
        "resize_panel" => {
            let width = msg.get("width")?.as_u64()? as u32;
            Some(UiAction::ResizePanel { width })
        }
        _ => None,
    }
}
