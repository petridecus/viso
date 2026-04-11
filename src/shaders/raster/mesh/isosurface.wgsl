#import viso::camera::CameraUniform
#import viso::lighting::{LightingUniform, compute_rim}
#import viso::shade::{shade_geometry, ShadingResult}
#import viso::constants::MAX_IBL_MIP

// Mirrors `renderer::geometry::isosurface::isosurface_kind` in Rust.
// Keep these in sync — they're the per-vertex source-kind discriminator.
const ISO_KIND_SURFACE: u32 = 0u;
const ISO_KIND_CAVITY: u32 = 1u;
const ISO_KIND_DENSITY: u32 = 2u;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) kind: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) vertex_color: vec4<f32>,
    // Integer attributes can't be linearly interpolated across a
    // primitive — must be flat.
    @location(3) @interpolate(flat) kind: u32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> lighting: LightingUniform;
@group(1) @binding(1) var irradiance_map: texture_cube<f32>;
@group(1) @binding(2) var env_sampler: sampler;
@group(1) @binding(3) var prefiltered_map: texture_cube<f32>;
@group(1) @binding(4) var brdf_lut: texture_2d<f32>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.world_position = in.position;
    out.world_normal = in.normal;
    out.vertex_color = in.color;
    out.kind = in.kind;
    return out;
}

struct FragOutput {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
    let normal = normalize(in.world_normal);
    let view_dir = normalize(camera.position - in.world_position);

    // Pre-sample IBL textures
    let NdotV = max(dot(normal, view_dir), 0.0);
    let R = reflect(-view_dir, normal);
    let irradiance = textureSample(irradiance_map, env_sampler, normal).rgb;
    let prefiltered = textureSampleLevel(prefiltered_map, env_sampler, R,
        lighting.roughness * MAX_IBL_MIP).rgb;
    let brdf = textureSample(brdf_lut, env_sampler, vec2<f32>(NdotV, lighting.roughness)).rg;

    let result = shade_geometry(normal, view_dir, in.vertex_color.rgb, 0.0,
        lighting, irradiance, prefiltered, brdf);

    var final_color = result.color;
    var final_alpha = in.vertex_color.a;

    // Cavity-specific effects: extra pulsing rim layered over the
    // standard PBR pass, plus an alpha breath in sync with the rim.
    // Gated on the per-vertex kind so SES / Gaussian / density meshes
    // are unaffected.
    //
    // Both rim and alpha must span enough range that the pulse is
    // visible AFTER tonemapping. Rim color is kept in LDR (≤1) and
    // intensity sweeps low→high so the pulse is perceptible.
    if (in.kind == ISO_KIND_CAVITY) {
        // ~0.19 Hz pulse, easing between 0.0 and 1.0 (period ≈ 5.2s).
        let pulse = 0.5 + 0.5 * sin(camera.time * 1.2);

        // Wider, softer rim than the global one (lower power = broader
        // halo). Color stays in LDR; intensity holds a bright floor and
        // breathes upward so the cavity never goes dim.
        let cavity_rim = compute_rim(
            normal,
            view_dir,
            2.0,                         // softer falloff
            0.55 + 0.45 * pulse,         // intensity breathes 0.55 → 1.0
            0.0,                         // omni-directional
            vec3<f32>(0.40, 0.60, 1.00), // cyan-blue glow (LDR)
            vec3<f32>(0.0, -1.0, 0.0),   // unused with directionality=0
        );
        final_color = final_color + cavity_rim;

        // Alpha breath — narrower, with a higher floor: 0.75 → 1.0.
        final_alpha = final_alpha * (0.75 + 0.25 * pulse);
    }

    var out: FragOutput;
    if (camera.debug_mode == 1u) {
        out.color = vec4<f32>(normal * 0.5 + 0.5, final_alpha);
    } else {
        out.color = vec4<f32>(final_color, final_alpha);
    }
    out.normal = vec4<f32>(normal, result.ambient_ratio);
    return out;
}
