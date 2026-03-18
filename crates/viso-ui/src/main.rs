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

    let mut collapsed: Signal<bool> = use_signal(|| false);
    let mut top_tab: Signal<String> = use_signal(|| "load".to_string());

    // Resize drag state: (start_screen_coord, start_panel_size)
    let mut drag = use_signal::<Option<(f64, f64)>>(|| None);

    use_effect(move || {
        bridge::register_listeners(schema, options);
        bridge::register_stats_listener(stats);
        bridge::register_panel_listener(panel_pinned);
        bridge::register_load_status_listener(load_status);
        bridge::register_scene_entities_listener(scene_entities);

        // The host pushes orientation via a 'viso-orientation' custom event.
        // Listen for it and set the body class accordingly.
        bridge::register_orientation_listener();
    });

    let is_collapsed = *collapsed.read();
    let current_tab = top_tab.read().clone();

    let arrow_title = if is_collapsed {
        "Expand panel"
    } else {
        "Collapse panel"
    };
    let arrow_icon_class = if is_collapsed {
        "arrow-icon expanded"
    } else {
        "arrow-icon"
    };

    rsx! {
        div { class: "panel-root",
            // ── Persistent arrow column ──
            button {
                class: "arrow-column",
                title: "{arrow_title}",
                onclick: move |_| {
                    let next = !is_collapsed;
                    collapsed.set(next);
                    bridge::send_toggle_panel();
                },
                // Generic right-pointing chevron; CSS rotates it based
                // on orientation and collapsed state.
                svg {
                    class: "{arrow_icon_class}",
                    width: "16",
                    height: "16",
                    view_box: "0 0 16 16",
                    fill: "none",
                    path {
                        d: "M6 3L11 8L6 13",
                        stroke: "currentColor",
                        stroke_width: "2",
                        stroke_linecap: "round",
                        stroke_linejoin: "round",
                    }
                }
            }

            // ── Settings content (hidden when collapsed) ──
            if !is_collapsed {
                div { class: "settings-content",
                    div {
                        class: "resize-handle",
                        onpointerdown: move |evt: PointerEvent| {
                            let screen = evt.screen_coordinates();
                            let portrait = bridge::is_portrait();
                            // Capture starting screen coord and current
                            // panel size (the iframe's own dimension
                            // along the panel axis).
                            let start_coord = if portrait {
                                screen.y
                            } else {
                                screen.x
                            };
                            let current_size = js_sys::eval(
                                if portrait {
                                    "window.innerHeight"
                                } else {
                                    "window.innerWidth"
                                },
                            )
                            .ok()
                            .and_then(|v| v.as_f64())
                            .unwrap_or(340.0);
                            drag.set(Some((start_coord, current_size)));

                            let id = evt.pointer_id();
                            let js = format!(
                                "document.querySelector('.resize-handle')\
                                 .setPointerCapture({})",
                                id
                            );
                            let _ = js_sys::eval(&js);
                        },
                        onpointermove: move |evt: PointerEvent| {
                            if let Some((start_coord, start_size)) =
                                *drag.read()
                            {
                                let screen = evt.screen_coordinates();
                                let coord = if bridge::is_portrait() {
                                    screen.y
                                } else {
                                    screen.x
                                };
                                // Dragging inward (toward canvas) makes
                                // the panel larger.
                                let delta = start_coord - coord;
                                let new_size =
                                    (start_size + delta).clamp(220.0, 700.0);
                                bridge::send_resize_panel(new_size as u32);
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
                    }
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
                            scene_ui::ScenePanel {
                                scene_entities: scene_entities,
                            }
                        },
                        _ => rsx! {
                            load_ui::LoadPanel { load_status: load_status }
                        },
                    }
                }
            }
        }
    }
}
