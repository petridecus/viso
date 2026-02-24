//! Schema-driven UI generation.
//!
//! Walks a JSON Schema object produced by `schemars` and renders Dioxus
//! controls that match each field's type. When a user changes a value, the
//! bridge sends a `set_option` IPC message to the native engine.

use dioxus::prelude::*;
use serde_json::Value;

use crate::bridge;

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

/// Top-level component: renders collapsible sections for each schema
/// property group.
#[component]
pub fn SchemaPanel(
    schema: Value,
    options: Value,
    stats_sig: Signal<Option<Value>>,
) -> Element {
    let properties = schema.pointer("/properties").and_then(Value::as_object);

    let Some(props) = properties else {
        return rsx! { p { "No schema loaded" } };
    };

    rsx! {
        div { class: "side-panel",
            for (section_key, section_schema) in props.iter() {
                {render_section(
                    section_key,
                    section_schema,
                    options.get(section_key),
                    &schema,
                    if section_key == "debug" { Some(stats_sig) } else { None },
                )}
            }
        }
    }
}

/// Render a collapsible section (one top-level Options field).
///
/// When `stats_sig` is provided (for the Debug section), an `FpsLabel`
/// component is rendered at the top of the section body.
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

    rsx! {
        details { open: true,
            summary { class: "section-header",
                "{title}"
            }
            div { class: "section-body",
                if let Some(sig) = stats_sig {
                    FpsLabel { stats_sig: sig }
                }
                if let Some(props) = properties {
                    for (field_key, field_schema) in props.iter() {
                        {render_field(
                            key,
                            field_key,
                            field_schema,
                            current.and_then(|c| c.get(field_key)),
                            root,
                        )}
                    }
                }
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

/// Number input with optional min/max from schema.
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
    let step = if is_int { "1" } else { "0.01" };

    let section = section.to_owned();
    let field = field.to_owned();

    rsx! {
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

    rsx! {
        div { class: "options-dropdown",
            select {
                value: "{current_str}",
                onchange: move |evt: Event<FormData>| {
                    let val = Value::String(evt.value().to_string());
                    bridge::send_set_option(&section, &field, &val);
                },
                for variant in &variants {
                    option {
                        value: "{variant}",
                        selected: variant == current_str,
                        {display_name(variant)}
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
