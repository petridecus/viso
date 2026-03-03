// Full PBR shading orchestration — single source of truth for all geometry
// shaders. Texture sampling happens in the caller; this module receives the
// pre-sampled IBL values as arguments (same pattern as viso::selection).

#define_import_path viso::shade

#import viso::lighting::{LightingUniform, compute_rim}
#import viso::pbr::pbr_direct_light
#import viso::highlight::apply_selection_edge
#import viso::constants::{LUMINANCE_REC709, DIELECTRIC_F0}

struct ShadingResult {
    color: vec3<f32>,
    ambient_ratio: f32,
};

/// Evaluate full PBR shading: F0, IBL ambient, rim, 2-light direct,
/// specular IBL, selection edge, and ambient ratio.
fn shade_geometry(
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    base_color: vec3<f32>,
    outline_factor: f32,
    lighting: LightingUniform,
    irradiance: vec3<f32>,
    prefiltered_color: vec3<f32>,
    brdf_sample: vec2<f32>,
) -> ShadingResult {
    let NdotV = max(dot(normal, view_dir), 0.0);

    let F0 = mix(vec3<f32>(DIELECTRIC_F0), base_color, lighting.metalness);
    let roughness = lighting.roughness;

    // IBL diffuse
    let ibl_diffuse = irradiance * lighting.ibl_strength;
    let ambient_light = mix(vec3(lighting.ambient), ibl_diffuse, lighting.ibl_strength);

    // Rim lighting
    let rim = compute_rim(
        normal, view_dir,
        lighting.rim_power, lighting.rim_intensity,
        lighting.rim_directionality, lighting.rim_color, lighting.rim_dir,
    );

    // PBR direct lighting (key + fill)
    var Lo = vec3<f32>(0.0);
    Lo += pbr_direct_light(
        normal, view_dir, lighting.light1_dir, lighting.light1_intensity,
        F0, roughness, lighting.metalness, NdotV, base_color,
    );
    Lo += pbr_direct_light(
        normal, view_dir, lighting.light2_dir, lighting.light2_intensity,
        F0, roughness, lighting.metalness, NdotV, base_color,
    );

    // Specular IBL
    let specular_ibl = prefiltered_color * (F0 * brdf_sample.x + brdf_sample.y)
        * lighting.ibl_strength;

    let ambient_contribution = base_color * ambient_light + specular_ibl;
    let direct_contribution = Lo + rim;
    let lit_color = ambient_contribution + direct_contribution;

    // Edge darkening for selected
    let final_color = apply_selection_edge(lit_color, normal, view_dir, outline_factor);

    // Ambient ratio for ambient-only AO in composite pass
    let total_lum = max(dot(final_color, LUMINANCE_REC709), 0.0001);
    let ambient_lum = dot(ambient_contribution, LUMINANCE_REC709);

    return ShadingResult(final_color, clamp(ambient_lum / total_lum, 0.0, 1.0));
}
