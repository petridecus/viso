//! Scene panel UI showing all entities in the current scene.

use dioxus::prelude::*;
use serde_json::Value;

use crate::bridge;

/// Scene panel: lists all entities with visibility toggles and focus.
#[component]
pub fn ScenePanel(scene_entities: Signal<Option<Value>>) -> Element {
    let entities = scene_entities.read();
    let list = entities.as_ref().and_then(Value::as_array);

    rsx! {
        div { class: "scene-panel",
            if let Some(items) = list {
                if items.is_empty() {
                    div { class: "scene-empty", "No entities in scene" }
                } else {
                    for item in items.iter() {
                        {entity_row(item)}
                    }
                }
            } else {
                div { class: "scene-empty", "Waiting for scene data..." }
            }
        }
    }
}

/// Render a single entity row.
fn entity_row(entity: &Value) -> Element {
    let id = entity.get("id").and_then(Value::as_u64).unwrap_or(0);
    let mol_type = entity
        .get("molecule_type")
        .and_then(Value::as_str)
        .unwrap_or("Unknown");
    let visible = entity
        .get("visible")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let atom_count = entity
        .get("atom_count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let focused = entity
        .get("focused")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let chain_ids = entity
        .get("chain_ids")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();

    let type_class = match mol_type {
        "Protein" => "entity-type-protein",
        "DNA" => "entity-type-dna",
        "RNA" => "entity-type-rna",
        _ => "entity-type-ligand",
    };

    let row_class = if focused {
        format!("entity-row focused {type_class}")
    } else {
        format!("entity-row {type_class}")
    };

    let label = if chain_ids.is_empty() {
        mol_type.to_string()
    } else {
        format!("{mol_type} {chain_ids}")
    };

    let subtitle = format!("{atom_count} atoms");

    let opacity_class = if visible { "" } else { " entity-hidden" };

    rsx! {
        div {
            key: "{id}",
            class: "{row_class}{opacity_class}",
            div { class: "entity-type-indicator" }
            div {
                class: "entity-info",
                onclick: move |_| {
                    bridge::send_focus_entity(id);
                },
                div { class: "entity-label", "{label}" }
                div { class: "entity-subtitle", "{subtitle}" }
            }
            div { class: "entity-actions",
                button {
                    class: "entity-action-btn",
                    title: if visible { "Hide" } else { "Show" },
                    onclick: move |_| {
                        bridge::send_toggle_entity_visibility(id);
                    },
                    // Eye icon
                    svg {
                        width: "16",
                        height: "16",
                        view_box: "0 0 24 24",
                        fill: "none",
                        stroke: "currentColor",
                        stroke_width: "2",
                        stroke_linecap: "round",
                        stroke_linejoin: "round",
                        if visible {
                            path { d: "M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" }
                            circle { cx: "12", cy: "12", r: "3" }
                        } else {
                            path { d: "M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" }
                            path { d: "M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" }
                            line { x1: "1", y1: "1", x2: "23", y2: "23" }
                        }
                    }
                }
                button {
                    class: "entity-action-btn entity-remove-btn",
                    title: "Remove",
                    onclick: move |_| {
                        bridge::send_remove_entity(id);
                    },
                    // X icon
                    svg {
                        width: "14",
                        height: "14",
                        view_box: "0 0 24 24",
                        fill: "none",
                        stroke: "currentColor",
                        stroke_width: "2",
                        stroke_linecap: "round",
                        stroke_linejoin: "round",
                        line { x1: "18", y1: "6", x2: "6", y2: "18" }
                        line { x1: "6", y1: "6", x2: "18", y2: "18" }
                    }
                }
            }
        }
    }
}
