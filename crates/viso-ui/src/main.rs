//! Dioxus web app for the viso options panel.
//!
//! Compiled to WASM and loaded into a wry webview by the native viso engine.
//! Communicates with the engine via a JSON IPC bridge.

mod bridge;
mod load_ui;
mod scene_ui;
mod schema_ui;

use dioxus::prelude::*;
use serde_json::Value;

fn main() {
    dioxus::launch(app);
}

fn app() -> Element {
    let schema: Signal<Option<Value>> = use_signal(|| None);
    let options: Signal<Option<Value>> = use_signal(|| None);
    let stats: Signal<Option<Value>> = use_signal(|| None);
    let panel_pinned: Signal<bool> = use_signal(|| true);
    let load_status: Signal<Option<Value>> = use_signal(|| None);
    let scene_entities: Signal<Option<Value>> = use_signal(|| None);

    // "load", "scene", or "options"
    let mut top_tab: Signal<String> = use_signal(|| "load".to_string());

    // Register IPC listeners once on mount.
    use_effect(move || {
        bridge::register_listeners(schema, options);
        bridge::register_stats_listener(stats);
        bridge::register_panel_listener(panel_pinned);
        bridge::register_load_status_listener(load_status);
        bridge::register_scene_entities_listener(scene_entities);
    });

    let pinned = *panel_pinned.read();
    let panel_class = if pinned {
        "side-panel"
    } else {
        "side-panel floating"
    };

    // Resize drag state: (start_screen_x, start_body_width)
    let mut drag = use_signal::<Option<(f64, f64)>>(|| None);

    let current_tab = top_tab.read().clone();

    rsx! {
        div { class: "{panel_class}",
            div {
                class: "resize-handle",
                onpointerdown: move |evt: PointerEvent| {
                    let sx = evt.screen_coordinates().x;
                    // clientWidth is CSS pixels; native expects physical pixels.
                    let dpr = web_sys::window()
                        .map(|w| w.device_pixel_ratio())
                        .unwrap_or(1.0);
                    let w = web_sys::window()
                        .and_then(|w| w.document())
                        .and_then(|d| d.body())
                        .map(|b| b.client_width() as f64 * dpr)
                        .unwrap_or(350.0);
                    drag.set(Some((sx, w)));
                    let id = evt.pointer_id();
                    let js = format!(
                        "document.querySelector('.resize-handle')\
                         .setPointerCapture({})",
                        id
                    );
                    let _ = js_sys::eval(&js);
                },
                onpointermove: move |evt: PointerEvent| {
                    if let Some((start_x, start_w)) = *drag.read() {
                        let dpr = web_sys::window()
                            .map(|w| w.device_pixel_ratio())
                            .unwrap_or(1.0);
                        // screen_coordinates are CSS pixels; scale delta to
                        // physical pixels to match the native panel width.
                        let delta =
                            (start_x - evt.screen_coordinates().x) * dpr;
                        let new_w = (start_w + delta).clamp(220.0, 700.0);
                        bridge::send_resize_panel(new_w as u32);
                    }
                },
                onpointerup: move |evt: PointerEvent| {
                    if drag.read().is_some() {
                        drag.set(None);
                        let id = evt.pointer_id();
                        let js = format!(
                            "document.querySelector('.resize-handle')\
                             .releasePointerCapture({})",
                            id
                        );
                        let _ = js_sys::eval(&js);
                    }
                },
            }
            div { class: "panel-header",
                div { class: "top-tabs",
                    button {
                        class: if current_tab == "load" { "top-tab active" } else { "top-tab" },
                        onclick: move |_| top_tab.set("load".into()),
                        "Load"
                    }
                    button {
                        class: if current_tab == "scene" { "top-tab active" } else { "top-tab" },
                        onclick: move |_| top_tab.set("scene".into()),
                        "Scene"
                    }
                    button {
                        class: if current_tab == "options" { "top-tab active" } else { "top-tab" },
                        onclick: move |_| top_tab.set("options".into()),
                        "Options"
                    }
                }
                button {
                    class: "panel-toggle-btn",
                    title: if pinned { "Unpin sidebar" } else { "Pin sidebar" },
                    onclick: move |_| {
                        bridge::send_toggle_panel();
                    },
                    svg {
                        width: "16",
                        height: "16",
                        view_box: "0 0 16 16",
                        fill: "none",
                        rect {
                            x: "1",
                            y: "2",
                            width: "14",
                            height: "12",
                            rx: "2",
                            stroke: "currentColor",
                            stroke_width: "1.5",
                            fill: "none",
                        }
                        line {
                            x1: "10",
                            y1: "2",
                            x2: "10",
                            y2: "14",
                            stroke: "currentColor",
                            stroke_width: "1.5",
                        }
                        if pinned {
                            rect {
                                x: "10",
                                y: "2",
                                width: "5",
                                height: "12",
                                rx: "0",
                                fill: "currentColor",
                                opacity: "0.4",
                            }
                        }
                    }
                }
            }
            // ── Tab content ──
            match current_tab.as_str() {
                "options" => {
                    let schema_val = schema.read();
                    let options_val = options.read();
                    match (&*schema_val, &*options_val) {
                        (Some(s), Some(o)) => rsx! {
                            schema_ui::OptionsPanel {
                                schema: s.clone(),
                                options: o.clone(),
                                stats_sig: stats,
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
                "scene" => rsx! {
                    scene_ui::ScenePanel { scene_entities: scene_entities }
                },
                _ => rsx! {
                    load_ui::LoadPanel { load_status: load_status }
                },
            }
        }
    }
}
