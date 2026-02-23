//! Dioxus web app for the viso options panel.
//!
//! Compiled to WASM and loaded into a wry webview by the native viso engine.
//! Communicates with the engine via a JSON IPC bridge.

mod bridge;
mod schema_ui;

use dioxus::prelude::*;
use serde_json::Value;

fn main() {
    dioxus::launch(app);
}

fn app() -> Element {
    let schema: Signal<Option<Value>> = use_signal(|| None);
    let options: Signal<Option<Value>> = use_signal(|| None);

    // Register IPC listeners once on mount.
    use_effect(move || {
        bridge::register_listeners(schema, options);
    });

    let schema_val = schema.read();
    let options_val = options.read();

    match (&*schema_val, &*options_val) {
        (Some(s), Some(o)) => rsx! {
            schema_ui::SchemaPanel {
                schema: s.clone(),
                options: o.clone(),
            }
        },
        _ => rsx! {
            div {
                style: "padding: 16px; color: #585b70;",
                "Waiting for engine..."
            }
        },
    }
}
