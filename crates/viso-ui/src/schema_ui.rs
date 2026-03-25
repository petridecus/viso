//! Schema-driven UI generation.
//!
//! Walks a JSON Schema object produced by `schemars` and renders Dioxus
//! controls that match each field's type. When a user changes a value, the
//! bridge sends a `set_option` IPC message to the native engine.
//!
//! The panel uses a tabbed layout with icon tabs on the left edge and
//! grouped cards within each tab.

use dioxus::prelude::*;
use serde_json::Value;

use crate::bridge;

/// Desired tab order (left to right in the tab bar).
const TAB_ORDER: &[&str] =
    &["lighting", "post_processing", "camera", "geometry", "debug"];

/// Convert a `snake_case` string to `Title Case`.
fn display_name(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(c) => {
                    let upper: String = c.to_uppercase().collect();
                    format!("{upper}{}", chars.as_str())
                }
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Leaf component that reads the stats signal. Only this component
/// re-renders when FPS updates, leaving the rest of the panel untouched.
#[component]
fn FpsLabel(stats_sig: Signal<Option<Value>>) -> Element {
    let stats = stats_sig.read();
    let fps_text = stats
        .as_ref()
        .and_then(|s| s.get("fps"))
        .and_then(Value::as_f64)
        .map(|f| format!("{:.0}", f))
        .unwrap_or_else(|| "--".to_string());

    rsx! {
        div { class: "field-row",
            label { class: "field-label", "FPS" }
            span { class: "fps-value", "{fps_text}" }
        }
    }
}

/// Return an SVG icon element for a section key.
fn tab_icon(section_key: &str) -> Element {
    match section_key {
        "display" => rsx! {
            svg {
                width: "16", height: "16", view_box: "0 0 16 16", fill: "none",
                path {
                    d: "M1 8s2.5-4 7-4 7 4 7 4-2.5 4-7 4-7-4-7-4z",
                    stroke: "currentColor", stroke_width: "1.5",
                }
                circle {
                    cx: "8", cy: "8", r: "2",
                    stroke: "currentColor", stroke_width: "1.5",
                }
            }
        },
        "lighting" => rsx! {
            svg {
                width: "16", height: "16", view_box: "0 0 16 16", fill: "none",
                circle {
                    cx: "8", cy: "8", r: "3",
                    stroke: "currentColor", stroke_width: "1.5",
                }
                path {
                    d: "M8 1.5v1.5M8 13v1.5M1.5 8H3M13 8h1.5M3.4 3.4l1.1 1.1M11.5 11.5l1.1 1.1M3.4 12.6l1.1-1.1M11.5 4.5l1.1-1.1",
                    stroke: "currentColor", stroke_width: "1.5", stroke_linecap: "round",
                }
            }
        },
        "post_processing" => rsx! {
            svg {
                width: "16", height: "16", view_box: "0 0 16 16", fill: "none",
                path {
                    d: "M8 1l1.8 4.7L15 8l-5.2 2.3L8 15l-1.8-4.7L1 8l5.2-2.3z",
                    stroke: "currentColor", stroke_width: "1.5", stroke_linejoin: "round",
                }
            }
        },
        "camera" => rsx! {
            svg {
                width: "16", height: "16", view_box: "0 0 16 16", fill: "none",
                rect {
                    x: "1", y: "4", width: "14", height: "10", rx: "2",
                    stroke: "currentColor", stroke_width: "1.5",
                }
                circle {
                    cx: "8", cy: "9", r: "2.5",
                    stroke: "currentColor", stroke_width: "1.5",
                }
                path {
                    d: "M5.5 4L6.5 2h3l1 2",
                    stroke: "currentColor", stroke_width: "1.5",
                }
            }
        },
        "geometry" => rsx! {
            svg {
                width: "16", height: "16", view_box: "0 0 16 16", fill: "none",
                path {
                    d: "M8 1.5l5.5 3v7L8 14.5l-5.5-3v-7z",
                    stroke: "currentColor", stroke_width: "1.5", stroke_linejoin: "round",
                }
                path {
                    d: "M8 7.5l5.5-3M8 7.5v7M8 7.5L2.5 4.5",
                    stroke: "currentColor", stroke_width: "1.5",
                }
            }
        },
        "debug" => rsx! {
            svg {
                width: "16", height: "16", view_box: "0 0 16 16", fill: "none",
                ellipse {
                    cx: "8", cy: "9.5", rx: "3.5", ry: "4.5",
                    stroke: "currentColor", stroke_width: "1.5",
                }
                path {
                    d: "M8 4V1.5M4.5 6.5L2 5.5M11.5 6.5L14 5.5M4.5 10.5H2M11.5 10.5H14M5 13.5L3.5 15M11 13.5l1.5 1.5",
                    stroke: "currentColor", stroke_width: "1.5", stroke_linecap: "round",
                }
                line {
                    x1: "8", y1: "5", x2: "8", y2: "14",
                    stroke: "currentColor", stroke_width: "1", opacity: "0.5",
                }
            }
        },
        _ => rsx! {
            svg {
                width: "16", height: "16", view_box: "0 0 16 16", fill: "none",
                circle {
                    cx: "8", cy: "8", r: "6",
                    stroke: "currentColor", stroke_width: "1.5",
                }
            }
        },
    }
}

/// Options panel content: icon tab bar on the left + section controls.
///
/// This component renders only the inner `.panel-layout` area.
/// The outer panel shell (resize handle, header, pin button, top-level tabs)
/// is owned by the parent in `main.rs`.
#[component]
pub fn OptionsPanel(
    schema: Value,
    options: Value,
    stats_sig: Signal<Option<Value>>,
) -> Element {
    let properties = schema.pointer("/properties").and_then(Value::as_object);

    let Some(props) = properties else {
        return rsx! { p { "No schema loaded" } };
    };

    // Ordered list of section keys that actually exist in the schema.
    let section_keys: Vec<String> = TAB_ORDER
        .iter()
        .filter(|k| props.contains_key(**k))
        .map(|k| (*k).to_owned())
        .collect();

    let first_key = section_keys.first().cloned().unwrap_or_default();
    let mut active_tab = use_signal(|| first_key);

    rsx! {
        div { class: "panel-layout",
            // Tab bar — vertical strip of icon buttons
            div { class: "tab-bar",
                for key in section_keys.iter() {
                    {
                        let key_owned = key.clone();
                        let is_active = *active_tab.read() == *key;
                        let title = props.get(key.as_str())
                            .and_then(|s| s.get("title"))
                            .and_then(Value::as_str)
                            .map(String::from)
                            .unwrap_or_else(|| display_name(key));
                        let btn_class = if is_active { "tab-btn active" } else { "tab-btn" };
                        rsx! {
                            button {
                                class: "{btn_class}",
                                title: "{title}",
                                onclick: move |_| {
                                    active_tab.set(key_owned.clone());
                                },
                                {tab_icon(key)}
                            }
                        }
                    }
                }
            }
            // Tab content — active section only
            div { class: "tab-content",
                {
                    let key = active_tab.read().clone();
                    if let Some(section_schema) = props.get(key.as_str()) {
                        render_section(
                            &key,
                            section_schema,
                            options.get(key.as_str()),
                            &schema,
                            if key == "debug" { Some(stats_sig) } else { None },
                        )
                    } else {
                        rsx! {}
                    }
                }
            }
        }
    }
}

/// Render a section's content with sub-group cards.
///
/// Fields with `x-group` metadata are collected into named card groups.
/// Fields without a group render flat at the section level.
/// When `stats_sig` is provided (Debug section), an FPS label appears at top.
fn render_section(
    key: &str,
    schema: &Value,
    current: Option<&Value>,
    root: &Value,
    stats_sig: Option<Signal<Option<Value>>>,
) -> Element {
    let title = schema
        .get("title")
        .and_then(Value::as_str)
        .map(String::from)
        .unwrap_or_else(|| display_name(key));

    let properties = schema
        .pointer("/properties")
        .or_else(|| schema.pointer("/allOf/0/properties"))
        .and_then(Value::as_object);

    // Collect fields into ordered groups: (Option<group_name>, fields).
    let mut groups: Vec<(Option<String>, Vec<(String, Value)>)> = Vec::new();

    if let Some(props) = properties {
        for (field_key, field_schema) in props.iter() {
            let group_name = field_schema
                .get("x-group")
                .and_then(Value::as_str)
                .map(String::from);

            match &group_name {
                Some(name) => {
                    // Append to existing group or create new one.
                    if let Some(group) = groups
                        .iter_mut()
                        .find(|(n, _)| n.as_deref() == Some(name.as_str()))
                    {
                        group.1.push((field_key.clone(), field_schema.clone()));
                    } else {
                        groups.push((
                            Some(name.clone()),
                            vec![(field_key.clone(), field_schema.clone())],
                        ));
                    }
                }
                None => {
                    groups.push((
                        None,
                        vec![(field_key.clone(), field_schema.clone())],
                    ));
                }
            }
        }
    }

    rsx! {
        div { class: "section-title", "{title}" }
        if let Some(sig) = stats_sig {
            FpsLabel { stats_sig: sig }
        }
        for (group_name, fields) in groups.iter() {
            {render_field_group(
                key, group_name.as_deref(), fields, current, root,
            )}
        }
    }
}

/// Render a group of fields — either as a card (named group) or flat.
fn render_field_group(
    section: &str,
    name: Option<&str>,
    fields: &[(String, Value)],
    current: Option<&Value>,
    root: &Value,
) -> Element {
    if let Some(name) = name {
        rsx! {
            div { class: "field-group",
                div { class: "field-group-header", "{name}" }
                for (field_key, field_schema) in fields.iter() {
                    {render_field(
                        section,
                        field_key,
                        field_schema,
                        current.and_then(|c| c.get(field_key.as_str())),
                        root,
                    )}
                }
            }
        }
    } else {
        rsx! {
            for (field_key, field_schema) in fields.iter() {
                {render_field(
                    section,
                    field_key,
                    field_schema,
                    current.and_then(|c| c.get(field_key.as_str())),
                    root,
                )}
            }
        }
    }
}

/// Resolve a `$ref` pointer (e.g. `"#/$defs/BackboneColorMode"`) against
/// the root schema, returning the referenced sub-schema. Returns the
/// input schema unchanged if there is no `$ref`.
fn resolve_ref<'a>(schema: &'a Value, root: &'a Value) -> &'a Value {
    if let Some(ref_str) = schema.get("$ref").and_then(Value::as_str) {
        // Convert "#/$defs/Foo" → "/$defs/Foo" for JSON pointer lookup.
        let pointer = ref_str.strip_prefix('#').unwrap_or(ref_str);
        root.pointer(pointer).unwrap_or(schema)
    } else {
        schema
    }
}

/// Render a single field control based on its schema type.
fn render_field(
    section: &str,
    field: &str,
    raw_schema: &Value,
    current: Option<&Value>,
    root: &Value,
) -> Element {
    let schema = resolve_ref(raw_schema, root);

    let label = schema
        .get("title")
        .or_else(|| raw_schema.get("title"))
        .and_then(Value::as_str)
        .map(String::from)
        .unwrap_or_else(|| display_name(field));

    let field_type = schema.get("type").and_then(Value::as_str);
    let has_enum =
        schema.get("enum").is_some() || schema.get("oneOf").is_some();
    let section_owned = section.to_owned();
    let field_owned = field.to_owned();

    rsx! {
        div { class: "field-row",
            label { class: "field-label",
                "{label}"
            }
            {match field_type {
                Some("number" | "integer") => {
                    render_number_field(
                        &section_owned,
                        &field_owned,
                        schema,
                        current,
                    )
                }
                Some("boolean") => {
                    render_bool_field(
                        &section_owned,
                        &field_owned,
                        current,
                    )
                }
                Some("string") if has_enum => {
                    render_enum_field(
                        &section_owned,
                        &field_owned,
                        schema,
                        current,
                    )
                }
                Some("string") => {
                    render_string_field(
                        &section_owned,
                        &field_owned,
                        current,
                    )
                }
                // No "type" but has enum variants (schemars oneOf pattern).
                None if has_enum => {
                    render_enum_field(
                        &section_owned,
                        &field_owned,
                        schema,
                        current,
                    )
                }
                _ => rsx! {
                    span { class: "text-xs text-neutral-600",
                        "(unsupported type)"
                    }
                },
            }}
        }
    }
}

/// Number input with optional min/max from schema, plus value readout.
fn render_number_field(
    section: &str,
    field: &str,
    schema: &Value,
    current: Option<&Value>,
) -> Element {
    let current_val = current.and_then(Value::as_f64).unwrap_or(0.0);
    let min = schema.get("minimum").and_then(Value::as_f64);
    let max = schema.get("maximum").and_then(Value::as_f64);
    let is_int = schema.get("type").and_then(Value::as_str) == Some("integer");

    // Use step from schema extension, fall back to sensible default.
    let step_val = schema.get("step").and_then(Value::as_f64);
    let step = step_val.map(|s| format!("{s}")).unwrap_or_else(|| {
        if is_int {
            "1".into()
        } else {
            "0.01".into()
        }
    });

    // Format the displayed value with appropriate precision.
    let display_val = if is_int {
        format!("{}", current_val as i64)
    } else {
        let decimals = step_val
            .map(|s| {
                let s_str = format!("{s}");
                s_str.find('.').map_or(0, |dot| s_str.len() - dot - 1)
            })
            .unwrap_or(2);
        format!("{current_val:.decimals$}")
    };

    let section = section.to_owned();
    let field = field.to_owned();

    rsx! {
        div { class: "slider-row",
            input {
                r#type: "range",
                value: "{current_val}",
                step: "{step}",
                min: min.map(|v| format!("{v}")).unwrap_or_default(),
                max: max.map(|v| format!("{v}")).unwrap_or_default(),
                oninput: move |evt: Event<FormData>| {
                    if let Ok(v) = evt.value().parse::<f64>() {
                        let val = if is_int {
                            Value::from(v as i64)
                        } else {
                            Value::from(v)
                        };
                        bridge::send_set_option(&section, &field, &val);
                    }
                },
            }
            span { class: "slider-value", "{display_val}" }
        }
    }
}

/// Boolean toggle checkbox.
fn render_bool_field(
    section: &str,
    field: &str,
    current: Option<&Value>,
) -> Element {
    let checked = current.and_then(Value::as_bool).unwrap_or(false);
    let section = section.to_owned();
    let field = field.to_owned();

    rsx! {
        div { class: "options-checkbox",
            input {
                r#type: "checkbox",
                checked: "{checked}",
                onchange: move |evt: Event<FormData>| {
                    let val = Value::Bool(evt.value() == "true");
                    bridge::send_set_option(&section, &field, &val);
                },
            }
        }
    }
}

/// Enum dropdown (string with enum constraint).
///
/// Uses a custom HTML/CSS dropdown instead of a native `<select>` because
/// WebKitGTK renders native selects as GTK popups that escape the webview.
///
/// Handles both `{ "enum": [...] }` and `{ "oneOf": [{"const": ...}, ...] }`
/// patterns produced by schemars.
fn render_enum_field(
    section: &str,
    field: &str,
    schema: &Value,
    current: Option<&Value>,
) -> Element {
    // Try "enum" array first, then "oneOf" with "const" entries.
    let variants: Vec<String> = if let Some(arr) =
        schema.get("enum").and_then(Value::as_array)
    {
        arr.iter()
            .filter_map(Value::as_str)
            .map(String::from)
            .collect()
    } else if let Some(arr) = schema.get("oneOf").and_then(Value::as_array) {
        arr.iter()
            .filter_map(|v| {
                v.get("const")
                    .or_else(|| v.get("enum").and_then(|e| e.get(0)))
                    .and_then(Value::as_str)
                    .map(String::from)
            })
            .collect()
    } else {
        Vec::new()
    };

    let current_str = current.and_then(Value::as_str).unwrap_or("");

    let section = section.to_owned();
    let field = field.to_owned();

    let mut open = use_signal(|| false);

    rsx! {
        div { class: "dropdown",
            div {
                class: "dropdown-trigger",
                onclick: move |_| { let v = *open.read(); open.set(!v); },
                span { {display_name(current_str)} }
                span { class: "dropdown-arrow", "▾" }
            }
            if *open.read() {
                div { class: "dropdown-menu",
                    for variant in &variants {
                        div {
                            class: if *variant == current_str { "dropdown-item selected" } else { "dropdown-item" },
                            onclick: {
                                let section = section.clone();
                                let field = field.clone();
                                let v = variant.clone();
                                move |_| {
                                    let val = Value::String(v.clone());
                                    bridge::send_set_option(
                                        &section, &field, &val,
                                    );
                                    open.set(false);
                                }
                            },
                            {display_name(variant)}
                        }
                    }
                }
            }
        }
    }
}

/// Plain text input for string fields without enum.
fn render_string_field(
    section: &str,
    field: &str,
    current: Option<&Value>,
) -> Element {
    let current_str = current.and_then(Value::as_str).unwrap_or("");

    let section = section.to_owned();
    let field = field.to_owned();

    rsx! {
        input {
            r#type: "text",
            class: "text-input",
            value: "{current_str}",
            onchange: move |evt: Event<FormData>| {
                let val = Value::String(evt.value().to_string());
                bridge::send_set_option(&section, &field, &val);
            },
        }
    }
}
