// PBR direct-light evaluation — shared by all geometry shaders.
//
// Replaces the per-shader key/fill light blocks with a single function call
// per light source.

#define_import_path viso::pbr

#import viso::lighting::{PI, distribution_ggx, geometry_smith, fresnel_schlick}

/// Evaluate Cook-Torrance BRDF for a single directional light.
///
/// Returns the outgoing radiance contribution (diffuse + specular) from one
/// light source.  Call once per light and sum the results into `Lo`.
fn pbr_direct_light(
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    light_dir: vec3<f32>,
    light_intensity: f32,
    F0: vec3<f32>,
    roughness: f32,
    metalness: f32,
    NdotV: f32,
    base_color: vec3<f32>,
) -> vec3<f32> {
    let L = light_dir;
    let H = normalize(L + view_dir);
    let NdotL = max(dot(normal, L), 0.0);
    let NdotH = max(dot(normal, H), 0.0);
    let HdotV = max(dot(H, view_dir), 0.0);

    let D = distribution_ggx(NdotH, roughness);
    let G = geometry_smith(NdotV, NdotL, roughness);
    let F = fresnel_schlick(HdotV, F0);

    let numerator = D * G * F;
    let denominator = 4.0 * NdotV * NdotL + 0.0001;
    let specular = numerator / denominator;

    let kD = (vec3<f32>(1.0) - F) * (1.0 - metalness);
    return (kD * base_color / PI + specular) * light_intensity * NdotL;
}
