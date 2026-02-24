//! IPC bridge between the wry webview (native) and the Dioxus WASM app.
//!
//! **Inbound** (native → WASM): the native side calls
//! `window.__viso_push_schema(json)` and `window.__viso_push_options(json)`,
//! which dispatch `CustomEvent`s that we listen to here.
//!
//! **Outbound** (WASM → native): we call `window.ipc.postMessage(json)` to
//! send [`UiAction`]s back to the engine.

use dioxus::signals::{Signal, Writable};
use serde_json::Value;
use wasm_bindgen::prelude::*;

// ── Inbound listeners ────────────────────────────────────────────────────

/// Register `CustomEvent` listeners that push schema and options JSON into
/// the provided signals. Call once at app startup.
pub fn register_listeners(
    mut schema_sig: Signal<Option<Value>>,
    mut options_sig: Signal<Option<Value>>,
) {
    // viso-schema
    let on_schema = Closure::<dyn FnMut(web_sys::CustomEvent)>::new(
        move |evt: web_sys::CustomEvent| {
            if let Some(json_str) = evt.detail().as_string() {
                if let Ok(val) = serde_json::from_str::<Value>(&json_str) {
                    schema_sig.set(Some(val));
                }
            }
        },
    );
    web_sys::window()
        .expect("no global window")
        .add_event_listener_with_callback(
            "viso-schema",
            on_schema.as_ref().unchecked_ref(),
        )
        .expect("failed to add viso-schema listener");
    on_schema.forget();

    // viso-options
    let on_options = Closure::<dyn FnMut(web_sys::CustomEvent)>::new(
        move |evt: web_sys::CustomEvent| {
            if let Some(json_str) = evt.detail().as_string() {
                if let Ok(val) = serde_json::from_str::<Value>(&json_str) {
                    options_sig.set(Some(val));
                }
            }
        },
    );
    web_sys::window()
        .expect("no global window")
        .add_event_listener_with_callback(
            "viso-options",
            on_options.as_ref().unchecked_ref(),
        )
        .expect("failed to add viso-options listener");
    on_options.forget();
}

/// Register a listener for stats updates (FPS, etc.) from the native
/// engine.
pub fn register_stats_listener(mut stats_sig: Signal<Option<Value>>) {
    let on_stats = Closure::<dyn FnMut(web_sys::CustomEvent)>::new(
        move |evt: web_sys::CustomEvent| {
            if let Some(json_str) = evt.detail().as_string() {
                if let Ok(val) = serde_json::from_str::<Value>(&json_str) {
                    stats_sig.set(Some(val));
                }
            }
        },
    );
    web_sys::window()
        .expect("no global window")
        .add_event_listener_with_callback(
            "viso-stats",
            on_stats.as_ref().unchecked_ref(),
        )
        .expect("failed to add viso-stats listener");
    on_stats.forget();
}

/// Register a listener for panel pinned state changes from the native
/// engine.
pub fn register_panel_listener(mut pinned_sig: Signal<bool>) {
    let on_pinned = Closure::<dyn FnMut(web_sys::CustomEvent)>::new(
        move |evt: web_sys::CustomEvent| {
            if let Some(val_str) = evt.detail().as_string() {
                pinned_sig.set(val_str == "true");
            }
        },
    );
    web_sys::window()
        .expect("no global window")
        .add_event_listener_with_callback(
            "viso-panel-pinned",
            on_pinned.as_ref().unchecked_ref(),
        )
        .expect("failed to add viso-panel-pinned listener");
    on_pinned.forget();
}

// ── Outbound actions ─────────────────────────────────────────────────────

/// Send a `toggle_panel` action to the native engine.
pub fn send_toggle_panel() {
    let msg = serde_json::json!({ "action": "toggle_panel" });
    post_message(&msg.to_string());
}

/// Send a `resize_panel` action to the native engine.
pub fn send_resize_panel(width: u32) {
    let msg = serde_json::json!({ "action": "resize_panel", "width": width });
    post_message(&msg.to_string());
}

/// Send a `set_option` action to the native engine.
pub fn send_set_option(path: &str, field: &str, value: &Value) {
    let msg = serde_json::json!({
        "action": "set_option",
        "path": path,
        "field": field,
        "value": value,
    });
    post_message(&msg.to_string());
}

/// Send a `load_file` action to the native engine.
pub fn send_load_file(path: &str) {
    let msg = serde_json::json!({
        "action": "load_file",
        "path": path,
    });
    post_message(&msg.to_string());
}

/// Call `window.ipc.postMessage(json)` to send a message to the native
/// wry IPC handler.
fn post_message(json: &str) {
    let js = format!(
        "window.ipc.postMessage('{}')",
        json.replace('\\', "\\\\").replace('\'', "\\'")
    );
    let _ = js_sys::eval(&js);
}
