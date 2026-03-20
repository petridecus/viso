//! Scene panel UI showing all entities in the current scene.

use std::collections::HashSet;

use dioxus::prelude::*;
use serde_json::Value;

use crate::bridge;

/// Scene panel: lists all entities with visibility toggles and focus.
///
/// `expanded_ids` tracks which entity rows have their options panel open.
/// It's owned by the parent `app()` so it survives tab switches.
#[component]
pub fn ScenePanel(
    scene_entities: Signal<Option<Value>>,
    expanded_ids: Signal<HashSet<u64>>,
) -> Element {
    let entities = scene_entities.read();
    let list = entities.as_ref().and_then(Value::as_array);

    rsx! {
        div { class: "scene-panel",
            if let Some(items) = list {
                if items.is_empty() {
                    div { class: "scene-empty", "No entities in scene" }
                } else {
                    for item in items.iter() {
                        {entity_row(item, expanded_ids)}
                    }
                }
            } else {
                div { class: "scene-empty", "Waiting for scene data..." }
            }
        }
    }
}

/// Render a single entity row with optional expandable options.
fn entity_row(
    entity: &Value,
    mut expanded_ids: Signal<HashSet<u64>>,
) -> Element {
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

    // Read effective values from top-level entity fields (set by
    // engine's entity_summaries — authoritative, survives round-trips).
    let drawing_mode = entity
        .get("drawing_mode")
        .and_then(Value::as_str)
        .unwrap_or("cartoon")
        .to_owned();
    let helix_style = entity
        .get("helix_style")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let sheet_style = entity
        .get("sheet_style")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();
    let show_sidechains_override = entity
        .get("show_sidechains_override")
        .and_then(Value::as_bool);
    let color_scheme = entity
        .get("color_scheme")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_owned();

    let has_overrides = entity
        .get("display_overrides")
        .and_then(Value::as_object)
        .is_some_and(|o| !o.is_empty());

    let is_cartoon_capable = matches!(mol_type, "Protein" | "DNA" | "RNA");
    let is_protein = mol_type == "Protein";

    let is_expanded = expanded_ids.read().contains(&id);

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
                // Expand/collapse toggle
                if is_cartoon_capable {
                    button {
                        class: "entity-action-btn entity-expand-btn",
                        title: if is_expanded { "Collapse options" } else { "Expand options" },
                        onclick: move |_| {
                            let mut ids = expanded_ids.write();
                            if ids.contains(&id) {
                                ids.remove(&id);
                            } else {
                                ids.insert(id);
                            }
                        },
                        svg {
                            width: "14",
                            height: "14",
                            view_box: "0 0 24 24",
                            fill: "none",
                            stroke: "currentColor",
                            stroke_width: "2",
                            stroke_linecap: "round",
                            stroke_linejoin: "round",
                            if is_expanded {
                                path { d: "M18 15l-6-6-6 6" }
                            } else {
                                path { d: "M6 9l6 6 6-6" }
                            }
                        }
                    }
                }
                button {
                    class: "entity-action-btn",
                    title: if visible { "Hide" } else { "Show" },
                    onclick: move |_| {
                        bridge::send_toggle_entity_visibility(id);
                    },
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
        // Expanded entity options panel
        if is_expanded && is_cartoon_capable {
            div { class: "entity-options",
                // Drawing Mode dropdown
                {entity_option_select(
                    id,
                    "Drawing Mode",
                    "drawing_mode",
                    &drawing_mode,
                    if is_protein {
                        &[("cartoon", "Cartoon"), ("stick", "Stick"), ("thin_stick", "Thin Stick"), ("ball_and_stick", "Ball & Stick")]
                    } else {
                        &[("cartoon", "Cartoon"), ("stick", "Stick"), ("thin_stick", "Thin Stick"), ("ball_and_stick", "Ball & Stick")]
                    },
                )}
                // Color scheme (applies to all drawing modes)
                {entity_option_select(
                    id,
                    "Color",
                    "backbone_color_scheme",
                    &color_scheme,
                    &[
                        ("", "Default"),
                        ("chain", "Chain"),
                        ("entity", "Entity"),
                        ("secondary_structure", "Secondary Structure"),
                        ("residue_index", "Residue Index"),
                        ("solid", "Solid"),
                    ],
                )}
                // Helix/Sheet Style (only for proteins in Cartoon mode)
                if is_protein && drawing_mode == "cartoon" {
                    {entity_option_select(
                        id,
                        "Helix Style",
                        "helix_style",
                        &helix_style,
                        &[("", "Default"), ("ribbon", "Ribbon"), ("tube", "Tube"), ("cylinder", "Cylinder")],
                    )}
                    {entity_option_select(
                        id,
                        "Sheet Style",
                        "sheet_style",
                        &sheet_style,
                        &[("", "Default"), ("ribbon", "Ribbon"), ("tube", "Tube")],
                    )}
                    {entity_option_toggle(
                        id,
                        "Show Sidechains",
                        "show_sidechains",
                        show_sidechains_override,
                    )}
                }
                // Reset to defaults button
                if has_overrides {
                    div { class: "entity-option-row",
                        button {
                            class: "entity-reset-btn",
                            onclick: move |_| {
                                bridge::send_clear_entity_option(id);
                            },
                            "Reset to defaults"
                        }
                    }
                }
            }
        }
    }
}

/// Render a labeled select dropdown for a per-entity option.
fn entity_option_select(
    id: u64,
    label: &str,
    field: &str,
    current: &str,
    choices: &[(&str, &str)],
) -> Element {
    let field = field.to_owned();
    let options: Vec<(String, String)> = choices
        .iter()
        .map(|(v, l)| ((*v).to_owned(), (*l).to_owned()))
        .collect();
    let current = current.to_owned();
    rsx! {
        div { class: "entity-option-row",
            label { class: "entity-option-label", "{label}" }
            select {
                class: "entity-option-select",
                value: "{current}",
                onchange: move |evt: Event<FormData>| {
                    let val = evt.value();
                    if val.is_empty() {
                        // Empty string means "use default" → send null to clear
                        bridge::send_set_entity_option(
                            id,
                            &field,
                            &Value::Null,
                        );
                    } else {
                        bridge::send_set_entity_option(
                            id,
                            &field,
                            &Value::String(val),
                        );
                    }
                },
                for (value, display_label) in options.iter() {
                    option {
                        value: "{value}",
                        selected: *value == current,
                        "{display_label}"
                    }
                }
            }
        }
    }
}

/// Render a labeled toggle for a per-entity boolean option.
fn entity_option_toggle(
    id: u64,
    label: &str,
    field: &str,
    current: Option<bool>,
) -> Element {
    let field = field.to_owned();
    let display_val = match current {
        None => "default",
        Some(true) => "on",
        Some(false) => "off",
    };
    rsx! {
        div { class: "entity-option-row",
            label { class: "entity-option-label", "{label}" }
            select {
                class: "entity-option-select",
                value: "{display_val}",
                onchange: move |evt: Event<FormData>| {
                    let val = evt.value();
                    match val.as_str() {
                        "on" => bridge::send_set_entity_option(
                            id,
                            &field,
                            &Value::Bool(true),
                        ),
                        "off" => bridge::send_set_entity_option(
                            id,
                            &field,
                            &Value::Bool(false),
                        ),
                        _ => bridge::send_set_entity_option(
                            id,
                            &field,
                            &Value::Null,
                        ),
                    }
                },
                option { value: "default", selected: current.is_none(), "Default" }
                option { value: "on", selected: current == Some(true), "On" }
                option { value: "off", selected: current == Some(false), "Off" }
            }
        }
    }
}
