// Ray-marched sphere impostors for ball-and-stick rendering
// Each sphere is a billboard quad with per-pixel ray-sphere intersection

#import viso::camera::CameraUniform
#import viso::lighting::{LightingUniform, PI, distribution_ggx, geometry_smith, fresnel_schlick, compute_rim}

// Selection bit-array lookup (inlined — requires global `selection` storage buffer)
fn is_selected(residue_idx: u32) -> bool {
    let word_idx = residue_idx / 32u;
    let bit_idx = residue_idx % 32u;
    if (word_idx >= arrayLength(&selection)) {
        return false;
    }
    return (selection[word_idx] & (1u << bit_idx)) != 0u;
}

// Per-instance data for sphere
// center: xyz = position, w = radius
// color: xyz = RGB, w = entity_id (packed as float)
struct SphereInstance {
    center: vec4<f32>,
    color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) sphere_center: vec3<f32>,
    @location(2) radius: f32,
    @location(3) color: vec3<f32>,
    @location(4) @interpolate(flat) entity_id: u32,
};

struct FragOut {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> spheres: array<SphereInstance>;
@group(1) @binding(0) var<uniform> camera: CameraUniform;
@group(2) @binding(0) var<uniform> lighting: LightingUniform;
@group(2) @binding(1) var irradiance_map: texture_cube<f32>;
@group(2) @binding(2) var env_sampler: sampler;
@group(2) @binding(3) var prefiltered_map: texture_cube<f32>;
@group(2) @binding(4) var brdf_lut: texture_2d<f32>;
@group(3) @binding(0) var<storage, read> selection: array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) vidx: u32,
    @builtin(instance_index) iidx: u32
) -> VertexOutput {
    let quad = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0)
    );

    let sph = spheres[iidx];
    let center = sph.center.xyz;
    let radius = sph.center.w;
    let color = sph.color.xyz;
    let entity_id = u32(sph.color.w);

    let to_camera = normalize(camera.position - center);

    // Build billboard basis
    var right = cross(to_camera, vec3<f32>(0.0, 1.0, 0.0));
    if (length(right) < 0.001) {
        right = cross(to_camera, vec3<f32>(0.0, 0.0, 1.0));
    }
    right = normalize(right);
    let up = normalize(cross(right, to_camera));

    let half_size = radius * 1.6;

    let local_uv = quad[vidx];
    let world_offset = right * local_uv.x * half_size + up * local_uv.y * half_size;
    let world_pos = center + world_offset;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.sphere_center = center;
    out.radius = radius;
    out.color = color;
    out.entity_id = entity_id;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragOut {
    let ray_origin = camera.position;
    let ray_dir = normalize(in.world_pos - camera.position);

    // Ray-sphere intersection
    let oc = ray_origin - in.sphere_center;
    let a = dot(ray_dir, ray_dir);
    let b = 2.0 * dot(oc, ray_dir);
    let c = dot(oc, oc) - in.radius * in.radius;
    let disc = b * b - 4.0 * a * c;

    if (disc < 0.0) {
        discard;
    }

    let t = (-b - sqrt(disc)) / (2.0 * a);
    if (t < 0.001) {
        discard;
    }

    let world_hit = ray_origin + ray_dir * t;
    let normal = normalize(world_hit - in.sphere_center);
    let view_dir = normalize(camera.position - world_hit);

    // Base color
    var base_color = in.color;

    // Hover highlight
    if (camera.hovered_residue >= 0 && u32(camera.hovered_residue) == in.entity_id) {
        base_color = base_color + vec3<f32>(0.3, 0.3, 0.3);
    }

    // Selection highlight
    var outline_factor = 0.0;
    if (is_selected(in.entity_id)) {
        base_color = base_color * 0.5 + vec3<f32>(0.0, 0.0, 1.0);
        outline_factor = 1.0;
    }

    let NdotV = max(dot(normal, view_dir), 0.0);

    // PBR: Fresnel reflectance at normal incidence
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

    // Key light
    {
        let L = lighting.light1_dir;
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

        let kD = (vec3<f32>(1.0) - F) * (1.0 - lighting.metalness);
        Lo += (kD * base_color / PI + specular) * lighting.light1_intensity * NdotL;
    }

    // Fill light
    {
        let L = lighting.light2_dir;
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

        let kD = (vec3<f32>(1.0) - F) * (1.0 - lighting.metalness);
        Lo += (kD * base_color / PI + specular) * lighting.light2_intensity * NdotL;
    }

    // Separate ambient and direct contributions for ambient-only AO
    // Specular IBL: sample prefiltered environment map at roughness-dependent mip
    let R = reflect(-view_dir, normal);
    let max_mip = 5.0; // 6 mip levels (0-5)
    let prefiltered_color = textureSampleLevel(prefiltered_map, env_sampler, R, roughness * max_mip).rgb;
    let brdf_sample = textureSample(brdf_lut, env_sampler, vec2<f32>(NdotV, roughness)).rg;
    let specular_ibl = prefiltered_color * (F0 * brdf_sample.x + brdf_sample.y) * lighting.ibl_strength;

    let ambient_contribution = base_color * ambient_light + specular_ibl;
    let direct_contribution = Lo + rim;
    let lit_color = ambient_contribution + direct_contribution;

    // Edge darkening for selected
    var final_color = lit_color;
    if (outline_factor > 0.0) {
        let edge = pow(1.0 - max(dot(normal, view_dir), 0.0), 2.0);
        final_color = mix(final_color, vec3<f32>(0.0, 0.0, 0.0), edge * 0.6);
    }

    // Compute ambient ratio for ambient-only AO in composite pass
    let total_lum = max(dot(final_color, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0001);
    let ambient_lum = dot(ambient_contribution, vec3<f32>(0.2126, 0.7152, 0.0722));
    let ambient_ratio = clamp(ambient_lum / total_lum, 0.0, 1.0);

    let clip_pos = camera.view_proj * vec4<f32>(world_hit, 1.0);
    let ndc_depth = clip_pos.z / clip_pos.w;

    // SDF edge AA: smooth alpha at sphere boundary for alpha-to-coverage MSAA
    let sdf = length(in.world_pos - in.sphere_center) - in.radius;
    let aa_edge = fwidth(sdf);
    let alpha = smoothstep(aa_edge, -aa_edge, sdf);

    var out: FragOut;
    out.depth = ndc_depth;
    if (camera.debug_mode == 1u) {
        out.color = vec4<f32>(normal * 0.5 + 0.5, alpha);
    } else {
        out.color = vec4<f32>(final_color, alpha);
    }
    out.normal = vec4<f32>(normal, ambient_ratio);
    return out;
}
