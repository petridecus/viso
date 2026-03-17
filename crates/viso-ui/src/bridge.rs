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

/// Register a listener for scene-entity updates from the native engine.
pub fn register_scene_entities_listener(
    mut entities_sig: Signal<Option<Value>>,
) {
    let on_entities = Closure::<dyn FnMut(web_sys::CustomEvent)>::new(
        move |evt: web_sys::CustomEvent| {
            if let Some(json_str) = evt.detail().as_string() {
                if let Ok(val) = serde_json::from_str::<Value>(&json_str) {
                    entities_sig.set(Some(val));
                }
            }
        },
    );
    web_sys::window()
        .expect("no global window")
        .add_event_listener_with_callback(
            "viso-scene-entities",
            on_entities.as_ref().unchecked_ref(),
        )
        .expect("failed to add viso-scene-entities listener");
    on_entities.forget();
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

/// Send a `fetch_pdb` action to the native engine.
pub fn send_fetch_pdb(id: &str, source: &str) {
    let msg = serde_json::json!({
        "action": "fetch_pdb",
        "id": id,
        "source": source,
    });
    post_message(&msg.to_string());
}

/// Open a file dialog. On native, delegates to `rfd::FileDialog` via IPC.
/// On web, uses a browser `<input type="file">` and passes bytes to the
/// engine via `window.parent.__viso_load_bytes()`.
pub fn send_open_file_dialog() {
    if has_load_bytes() {
        open_browser_file_dialog();
    } else {
        let msg = serde_json::json!({ "action": "open_file_dialog" });
        post_message(&msg.to_string());
    }
}

/// Check whether `window.parent.__viso_load_bytes` exists (web context).
fn has_load_bytes() -> bool {
    let Some(win) = web_sys::window() else {
        return false;
    };
    let Ok(parent) = js_sys::Reflect::get(&win, &JsValue::from_str("parent"))
    else {
        return false;
    };
    js_sys::Reflect::get(&parent, &JsValue::from_str("__viso_load_bytes"))
        .map(|v| v.is_function())
        .unwrap_or(false)
}

/// Create a hidden `<input type="file">`, trigger it, read the selected
/// file via `FileReader`, and pass the bytes to the engine.
fn open_browser_file_dialog() {
    let Some(document) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let Ok(el) = document.create_element("input") else {
        return;
    };
    let input: web_sys::HtmlInputElement = el.unchecked_into();
    input.set_type("file");
    input.set_accept(".cif,.pdb");

    let on_change = Closure::<dyn FnMut()>::once({
        let input = input.clone();
        move || {
            let Some(files) = input.files() else { return };
            let Some(file) = files.get(0) else { return };
            let name = file.name();
            let ext = name.rsplit('.').next().unwrap_or("cif").to_owned();

            let Ok(reader) = web_sys::FileReader::new() else {
                return;
            };
            let reader_clone = reader.clone();
            let on_load = Closure::<dyn FnMut()>::once(move || {
                let Ok(result) = reader_clone.result() else {
                    return;
                };
                let array = js_sys::Uint8Array::new(&result);
                // Call window.parent.__viso_load_bytes(bytes, ext)
                let Some(win) = web_sys::window() else { return };
                let Ok(parent) =
                    js_sys::Reflect::get(&win, &JsValue::from_str("parent"))
                else {
                    return;
                };
                let Ok(func) = js_sys::Reflect::get(
                    &parent,
                    &JsValue::from_str("__viso_load_bytes"),
                ) else {
                    return;
                };
                if func.is_function() {
                    let func: js_sys::Function = func.unchecked_into();
                    let _ =
                        func.call2(&parent, &array, &JsValue::from_str(&ext));
                }
            });
            reader.set_onload(Some(on_load.as_ref().unchecked_ref()));
            on_load.forget();
            let _ = reader.read_as_array_buffer(&file);
        }
    });
    let _ = input.add_event_listener_with_callback(
        "change",
        on_change.as_ref().unchecked_ref(),
    );
    on_change.forget();
    input.click();
}

/// Send a `focus_entity` action to the native engine.
pub fn send_focus_entity(id: u64) {
    let msg = serde_json::json!({ "action": "focus_entity", "id": id });
    post_message(&msg.to_string());
}

/// Send a `toggle_entity_visibility` action to the native engine.
pub fn send_toggle_entity_visibility(id: u64) {
    let msg =
        serde_json::json!({ "action": "toggle_entity_visibility", "id": id });
    post_message(&msg.to_string());
}

/// Send a `remove_entity` action to the native engine.
pub fn send_remove_entity(id: u64) {
    let msg = serde_json::json!({ "action": "remove_entity", "id": id });
    post_message(&msg.to_string());
}

/// Register a listener for load-status events from the native engine.
pub fn register_load_status_listener(mut status_sig: Signal<Option<Value>>) {
    let on_status = Closure::<dyn FnMut(web_sys::CustomEvent)>::new(
        move |evt: web_sys::CustomEvent| {
            if let Some(json_str) = evt.detail().as_string() {
                if let Ok(val) = serde_json::from_str::<Value>(&json_str) {
                    status_sig.set(Some(val));
                }
            }
        },
    );
    web_sys::window()
        .expect("no global window")
        .add_event_listener_with_callback(
            "viso-load-status",
            on_status.as_ref().unchecked_ref(),
        )
        .expect("failed to add viso-load-status listener");
    on_status.forget();
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
