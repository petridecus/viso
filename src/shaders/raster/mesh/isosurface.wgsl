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
    @location(4) cavity_center: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) vertex_color: vec4<f32>,
    @location(3) @interpolate(flat) kind: u32,
    @location(4) view_z: f32,
};

// ── Lava-lamp displacement helpers ────────────────────────────────────
//
// Cheap 3D value noise used as a SPATIAL liveliness map (no time
// component) so the noise field stays static and the cavity doesn't
// drift. The time-varying motion comes from the sum-of-sines block.

fn hash13(p: vec3<f32>) -> f32 {
    let q = vec3<f32>(
        dot(p, vec3<f32>(127.1, 311.7,  74.7)),
        dot(p, vec3<f32>(269.5, 183.3, 246.1)),
        dot(p, vec3<f32>(113.5, 271.9, 124.6)),
    );
    return fract(sin(q.x + q.y + q.z) * 43758.5453);
}

fn noise3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);  // smoothstep interpolation

    let n000 = hash13(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash13(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash13(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash13(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash13(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash13(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash13(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash13(i + vec3<f32>(1.0, 1.0, 1.0));

    let nx00 = mix(n000, n100, u.x);
    let nx10 = mix(n010, n110, u.x);
    let nx01 = mix(n001, n101, u.x);
    let nx11 = mix(n011, n111, u.x);
    let nxy0 = mix(nx00, nx10, u.y);
    let nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z);
}

/// Compute the lava-lamp displacement for a cavity vertex.
///
/// Three motion modes layered:
///   1. Surface undulation: sum of sines along the vertex normal. Both
///      AMPLITUDE and PHASE come from spatial noise — amplitude varies
///      where the surface is "lively", phase varies so different
///      regions peak at different times. No linear wavefronts, no
///      lockstep.
///   2. Radial breath: whole cavity grows/shrinks from its centroid.
///      Frequency and phase are derived from noise sampled at the
///      cavity centroid, so each cavity breathes on its own rhythm.
///   3. (No drift — all noise inputs are spatial, no time component.
///      Time only enters through the sine arguments.)
///
/// Returns the world-space offset to add to the rest position.
fn cavity_displacement(
    rest_pos: vec3<f32>,
    normal: vec3<f32>,
    cavity_center: vec3<f32>,
    time: f32,
) -> vec3<f32> {
    let TAU = 6.28318530718;

    // Per-region amplitude for the surface undulation.
    let liveliness = noise3d(rest_pos * 0.5);
    let surf_amp = mix(0.08, 0.30, liveliness);

    // Per-region PHASES for the surface undulation. Sampling noise
    // (instead of dot products) eliminates the planar-wavefront look
    // that linear phases give — neighboring regions wobble independently.
    let phase_a = noise3d(rest_pos * 0.3 + vec3<f32>(13.7,  0.0,  0.0)) * TAU;
    let phase_b = noise3d(rest_pos * 0.4 + vec3<f32>( 0.0, 41.2,  0.0)) * TAU;
    let phase_c = noise3d(rest_pos * 0.6 + vec3<f32>( 0.0,  0.0, 27.5)) * TAU;
    let surf_wave =
          sin(time * 1.10 + phase_a) * 0.5
        + sin(time * 1.65 + phase_b) * 0.3
        + sin(time * 0.75 + phase_c) * 0.2;

    // Per-cavity phase + frequency for the radial breath. Both come
    // from noise sampled at the centroid, so two cavities at different
    // positions get independent breathing rhythms. Frequencies span
    // [0.60, 1.50] rad/s (periods ~4.2s to ~10.5s).
    let cavity_phase = noise3d(cavity_center * 0.7) * TAU;
    let cavity_freq  = 0.60 + noise3d(cavity_center * 0.9 + vec3<f32>(7.7)) * 0.90;

    // Radial breath — whole cavity inflates/deflates from its centroid.
    let to_center = rest_pos - cavity_center;
    let radial_dir = to_center / max(length(to_center), 0.0001);
    let radial = sin(time * cavity_freq + cavity_phase) * 0.15;

    return normal * surf_wave * surf_amp + radial_dir * radial;
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> lighting: LightingUniform;
@group(1) @binding(1) var irradiance_map: texture_cube<f32>;
@group(1) @binding(2) var env_sampler: sampler;
@group(1) @binding(3) var prefiltered_map: texture_cube<f32>;
@group(1) @binding(4) var brdf_lut: texture_2d<f32>;
@group(2) @binding(0) var backface_depth_tex: texture_2d<f32>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Lava-lamp displacement: bounded sinusoidal motion around the rest
    // position, gated on cavity kind. Surface meshes (SES / Gaussian /
    // density) are unaffected.
    var pos = in.position;
    if (in.kind == ISO_KIND_CAVITY) {
        pos = pos + cavity_displacement(
            in.position, in.normal, in.cavity_center, camera.time,
        );
    }

    out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    out.world_position = pos;
    out.world_normal = in.normal;
    out.vertex_color = in.color;
    out.kind = in.kind;
    out.view_z = dot(pos - camera.position, camera.forward);
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

    // Beer-Lambert thickness absorption. The back-face pre-pass wrote
    // linear view-space z for every isosurface back-face into
    // `backface_depth_tex`. We sample it at this fragment's screen
    // position and subtract this fragment's view_z to get the
    // thickness of the isosurface slab along the view ray. Per-kind
    // absorption coefficients produce the right look for each kind:
    //
    //   - CAVITY : strong blue-biased absorption → deep saturated blue
    //              centers, clear edges. Makes cavities read as dense
    //              pockets of glowing gel.
    //   - SURFACE / DENSITY : mild neutral absorption → center slightly
    //              more opaque than edges, gives the translucent shell
    //              a sense of depth without heavy tinting.
    let pixel_coord = vec2<i32>(in.clip_position.xy);
    let back_z = textureLoad(backface_depth_tex, pixel_coord, 0).r;
    let thickness = max(back_z - in.view_z, 0.0);

    var absorption: f32;
    if (in.kind == ISO_KIND_CAVITY) {
        absorption = 0.35;
    } else {
        absorption = 0.10;
    }
    let opacity = 1.0 - exp(-thickness * absorption);
    final_alpha = in.vertex_color.a * opacity;

    // Cavity-specific rim, layered over the PBR pass. At the silhouette
    // thickness ≈ 0 so the Beer-Lambert opacity also ≈ 0, which would
    // make the rim invisible after alpha blending. Use the rim
    // magnitude itself as a floor on final_alpha so the brightest part
    // of the rim always gets enough alpha to be visible.
    if (in.kind == ISO_KIND_CAVITY) {
        let cavity_rim = compute_rim(
            normal,
            view_dir,
            2.0,
            1.0,
            0.0,
            vec3<f32>(0.40, 0.60, 1.00),
            vec3<f32>(0.0, -1.0, 0.0),
        );
        final_color = final_color + cavity_rim;

        let rim_strength = (cavity_rim.r + cavity_rim.g + cavity_rim.b) / 3.0;
        final_alpha = max(final_alpha, rim_strength);
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
