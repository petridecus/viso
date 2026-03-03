#import viso::camera::CameraUniform
#import viso::lighting::{LightingUniform, compute_rim}
#import viso::selection::check_selection
#import viso::highlight::{apply_highlight, apply_selection_edge}
#import viso::pbr::pbr_direct_light

fn is_selected(residue_idx: u32) -> bool {
    return check_selection(residue_idx, arrayLength(&selection), selection[residue_idx / 32u]);
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) residue_idx: u32,
    @location(4) center_pos: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) vertex_color: vec3<f32>,
    @location(3) @interpolate(flat) residue_idx: u32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> lighting: LightingUniform;
@group(1) @binding(1) var irradiance_map: texture_cube<f32>;
@group(1) @binding(2) var env_sampler: sampler;
@group(1) @binding(3) var prefiltered_map: texture_cube<f32>;
@group(1) @binding(4) var brdf_lut: texture_2d<f32>;
@group(2) @binding(0) var<storage, read> selection: array<u32>;
@group(3) @binding(0) var<storage, read> residue_colors: array<vec4<f32>>;

fn lookup_residue_color(residue_idx: u32) -> vec3<f32> {
    if (residue_idx < arrayLength(&residue_colors)) {
        return residue_colors[residue_idx].xyz;
    }
    return vec3<f32>(0.5, 0.5, 0.5);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Expand selected residues outward along normal for 1.4x radius (matches Foldit)
    var position = in.position;
    if (is_selected(in.residue_idx)) {
        position = position + in.normal * 0.24;  // ~1.4x expansion (tube radius ~0.6)
    }

    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
    out.world_normal = in.normal;
    out.world_position = position;
    out.vertex_color = lookup_residue_color(in.residue_idx);
    out.residue_idx = in.residue_idx;
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

    // Highlight
    let hovered = camera.hovered_residue >= 0 && u32(camera.hovered_residue) == in.residue_idx;
    let highlighted = apply_highlight(in.vertex_color, hovered, is_selected(in.residue_idx));
    var base_color = highlighted.xyz;
    let outline_factor = highlighted.w;

    let NdotV = max(dot(normal, view_dir), 0.0);

    // PBR setup
    let F0 = mix(vec3<f32>(0.04), base_color, lighting.metalness);
    let roughness = lighting.roughness;

    // IBL diffuse
    let irradiance = textureSample(irradiance_map, env_sampler, normal).rgb;
    let ibl_diffuse = irradiance * lighting.ibl_strength;
    let ambient_light = mix(vec3(lighting.ambient), ibl_diffuse, lighting.ibl_strength);

    // Rim lighting
    let rim = compute_rim(normal, view_dir, lighting.rim_power, lighting.rim_intensity, lighting.rim_directionality, lighting.rim_color, lighting.rim_dir);

    // PBR direct lighting
    var Lo = vec3<f32>(0.0);
    Lo += pbr_direct_light(normal, view_dir, lighting.light1_dir, lighting.light1_intensity, F0, roughness, lighting.metalness, NdotV, base_color);
    Lo += pbr_direct_light(normal, view_dir, lighting.light2_dir, lighting.light2_intensity, F0, roughness, lighting.metalness, NdotV, base_color);

    // Specular IBL
    let R = reflect(-view_dir, normal);
    let max_mip = 5.0;
    let prefiltered_color = textureSampleLevel(prefiltered_map, env_sampler, R, roughness * max_mip).rgb;
    let brdf_sample = textureSample(brdf_lut, env_sampler, vec2<f32>(NdotV, roughness)).rg;
    let specular_ibl = prefiltered_color * (F0 * brdf_sample.x + brdf_sample.y) * lighting.ibl_strength;

    let ambient_contribution = base_color * ambient_light + specular_ibl;
    let direct_contribution = Lo + rim;
    let lit_color = ambient_contribution + direct_contribution;

    // Edge darkening for selected
    let final_color = apply_selection_edge(lit_color, normal, view_dir, outline_factor);

    // Ambient ratio for ambient-only AO in composite pass
    let total_lum = max(dot(final_color, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0001);
    let ambient_lum = dot(ambient_contribution, vec3<f32>(0.2126, 0.7152, 0.0722));
    let ambient_ratio = clamp(ambient_lum / total_lum, 0.0, 1.0);

    var out: FragOutput;
    if (camera.debug_mode == 1u) {
        out.color = vec4<f32>(normal * 0.5 + 0.5, 1.0);
    } else {
        out.color = vec4<f32>(final_color, 1.0);
    }
    out.normal = vec4<f32>(normal, ambient_ratio);
    return out;
}
