// Hover / selection highlight — shared by all geometry shaders.
//
// Extracts the duplicated hover-brighten + selection-tint blocks and the
// post-lighting selection edge-darkening into reusable functions.

#define_import_path viso::highlight

/// Additive brightness applied on hover.
const HOVER_BRIGHTNESS: f32 = 0.3;
/// Blend factor for the original color when selected.
const SELECTION_BLEND: f32 = 0.5;
/// Tint color added to selected geometry (Foldit blue).
const SELECTION_TINT: vec3<f32> = vec3<f32>(0.0, 0.0, 1.0);

/// Apply hover and selection highlighting to `base_color`.
///
/// Returns `vec4(modified_color, outline_factor)` where `outline_factor` is
/// 1.0 when selected (for edge-darkening) and 0.0 otherwise.
fn apply_highlight(
    base_color: vec3<f32>,
    is_hovered: bool,
    is_sel: bool,
) -> vec4<f32> {
    var color = base_color;
    if (is_hovered) {
        color = color + vec3<f32>(HOVER_BRIGHTNESS);
    }
    var outline = 0.0;
    if (is_sel) {
        color = color * SELECTION_BLEND + SELECTION_TINT;
        outline = 1.0;
    }
    return vec4<f32>(color, outline);
}

/// Edge-darkening for selected geometry.  Call after lighting with the
/// `outline_factor` returned by `apply_highlight`.
fn apply_selection_edge(
    lit_color: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    outline_factor: f32,
) -> vec3<f32> {
    if (outline_factor > 0.0) {
        let edge = pow(1.0 - max(dot(normal, view_dir), 0.0), 2.0);
        return mix(lit_color, vec3<f32>(0.0, 0.0, 0.0), edge * 0.6);
    }
    return lit_color;
}
