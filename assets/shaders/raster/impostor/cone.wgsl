// Ray-marched cone impostor for pull arrow rendering
// Cone points from base (atom) toward tip (mouse target)

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

// Per-instance data for cone
// base: xyz = base position (wide end), w = base radius
// tip: xyz = tip position (point), w = residue_idx (packed as float)
// color: xyz = RGB, w = unused
struct ConeInstance {
    base: vec4<f32>,
    tip: vec4<f32>,
    color: vec4<f32>,
    _pad: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) base: vec3<f32>,
    @location(2) tip: vec3<f32>,
    @location(3) base_radius: f32,
    @location(4) color: vec3<f32>,
    @location(5) @interpolate(flat) residue_idx: u32,
};

struct FragOut {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> cones: array<ConeInstance>;
@group(1) @binding(0) var<uniform> camera: CameraUniform;
@group(2) @binding(0) var<uniform> lighting: LightingUniform;
@group(2) @binding(1) var irradiance_map: texture_cube<f32>;
@group(2) @binding(2) var env_sampler: sampler;
@group(2) @binding(3) var prefiltered_map: texture_cube<f32>;
@group(2) @binding(4) var brdf_lut: texture_2d<f32>;
@group(3) @binding(0) var<storage, read> selection: array<u32>;

// Ray-cone intersection
// Returns vec3(t, u, hit_type) where:
//   t = ray parameter (distance)
//   u = position along axis (0 = base, 1 = tip)
//   hit_type = 1.0 for cone surface, 2.0 for base cap, 0.0 for miss
fn intersect_cone(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    base: vec3<f32>,
    tip: vec3<f32>,
    base_radius: f32
) -> vec3<f32> {
    let axis = tip - base;
    let height = length(axis);

    if (height < 0.0001) {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }

    let axis_dir = axis / height;

    // Cone equation: at height h along axis, radius = base_radius * (1 - h/height)
    // This creates a cone that tapers from base_radius at base to 0 at tip
    let cos_angle = height / sqrt(height * height + base_radius * base_radius);
    let sin_angle = base_radius / sqrt(height * height + base_radius * base_radius);
    let cos2 = cos_angle * cos_angle;
    let sin2 = sin_angle * sin_angle;

    // Transform ray to cone space (tip at origin, axis along +Y)
    let co = ray_origin - tip;

    let d_dot_v = dot(ray_dir, axis_dir);
    let co_dot_v = dot(co, axis_dir);

    // Quadratic coefficients for infinite cone
    let a = d_dot_v * d_dot_v - cos2;
    let b = 2.0 * (d_dot_v * co_dot_v - dot(ray_dir, co) * cos2);
    let c_coef = co_dot_v * co_dot_v - dot(co, co) * cos2;

    var best_t: f32 = 1e20;
    var best_u: f32 = 0.0;
    var best_type: f32 = 0.0;

    // Check cone surface
    let disc = b * b - 4.0 * a * c_coef;
    if (disc >= 0.0 && abs(a) > 0.0001) {
        let sqrt_disc = sqrt(disc);
        let t1 = (-b - sqrt_disc) / (2.0 * a);
        let t2 = (-b + sqrt_disc) / (2.0 * a);

        for (var i = 0; i < 2; i++) {
            let t = select(t2, t1, i == 0);
            if (t > 0.001 && t < best_t) {
                let hit = ray_origin + ray_dir * t;
                let hit_along_axis = dot(hit - base, axis_dir);
                // Check if hit is within cone bounds (between base and tip)
                if (hit_along_axis >= 0.0 && hit_along_axis <= height) {
                    best_t = t;
                    best_u = hit_along_axis / height;
                    best_type = 1.0;
                }
            }
        }
    }

    // Check base cap (disc at base)
    let denom = dot(ray_dir, axis_dir);
    if (abs(denom) > 0.0001) {
        let t = dot(base - ray_origin, axis_dir) / denom;
        if (t > 0.001 && t < best_t) {
            let hit = ray_origin + ray_dir * t;
            let dist_from_axis = length(hit - base - axis_dir * dot(hit - base, axis_dir));
            if (dist_from_axis <= base_radius) {
                best_t = t;
                best_u = 0.0;
                best_type = 2.0;
            }
        }
    }

    if (best_type == 0.0) {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }

    return vec3<f32>(best_t, best_u, best_type);
}

fn cone_normal(
    point: vec3<f32>,
    base: vec3<f32>,
    tip: vec3<f32>,
    base_radius: f32,
    hit_type: f32
) -> vec3<f32> {
    let axis = tip - base;
    let height = length(axis);
    let axis_dir = axis / height;

    if (hit_type == 2.0) {
        // Base cap - normal points away from tip
        return -axis_dir;
    } else {
        // Cone surface
        // Normal is perpendicular to surface, pointing outward
        let to_point = point - base;
        let along_axis = dot(to_point, axis_dir);
        let radial = to_point - axis_dir * along_axis;
        let radial_dir = normalize(radial);

        // The normal tilts inward toward the axis as we go up
        // tan(half_angle) = base_radius / height
        let slope = base_radius / height;
        return normalize(radial_dir + axis_dir * slope);
    }
}

@vertex
fn vs_main(
    @builtin(vertex_index) vidx: u32,
    @builtin(instance_index) iidx: u32
) -> VertexOutput {
    let quad = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0)
    );

    let cone = cones[iidx];
    let base = cone.base.xyz;
    let tip = cone.tip.xyz;
    let base_radius = cone.base.w;
    let color = cone.color.xyz;
    let residue_idx = u32(cone.tip.w);

    let center = (base + tip) * 0.5;
    let axis = tip - base;
    let height = length(axis);
    let axis_dir = select(vec3<f32>(0.0, 1.0, 0.0), axis / height, height > 0.0001);

    let to_camera = normalize(camera.position - center);

    var right = cross(axis_dir, to_camera);
    let right_len = length(right);
    if (right_len < 0.001) {
        right = cross(axis_dir, vec3<f32>(0.0, 0.0, 1.0));
        if (length(right) < 0.001) {
            right = vec3<f32>(1.0, 0.0, 0.0);
        }
    }
    right = normalize(right);

    let up = axis_dir;

    // Billboard size needs to encompass the whole cone
    let half_width = base_radius * 1.5;
    let half_height = height * 0.5 + base_radius * 0.5;

    let local_uv = quad[vidx];
    let world_offset = right * local_uv.x * half_width + up * local_uv.y * half_height;
    let world_pos = center + world_offset;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.base = base;
    out.tip = tip;
    out.base_radius = base_radius;
    out.color = color;
    out.residue_idx = residue_idx;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragOut {
    let ray_origin = camera.position;
    let ray_dir = normalize(in.world_pos - camera.position);

    let hit = intersect_cone(ray_origin, ray_dir, in.base, in.tip, in.base_radius);

    if (hit.x < 0.0) {
        discard;
    }

    let t = hit.x;
    let axis_param = hit.y;
    let hit_type = hit.z;

    let world_hit = ray_origin + ray_dir * t;

    let normal = cone_normal(world_hit, in.base, in.tip, in.base_radius, hit_type);
    let view_dir = normalize(camera.position - world_hit);

    var base_color = in.color;

    // Hover highlight
    if (camera.hovered_residue >= 0 && u32(camera.hovered_residue) == in.residue_idx) {
        base_color = base_color + vec3<f32>(0.3, 0.3, 0.3);
    }

    // Selection highlight
    if (is_selected(in.residue_idx)) {
        base_color = base_color * 0.5 + vec3<f32>(0.0, 0.0, 1.0);
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
    let final_color = lit_color;

    // Compute ambient ratio for ambient-only AO in composite pass
    let total_lum = max(dot(final_color, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0001);
    let ambient_lum = dot(ambient_contribution, vec3<f32>(0.2126, 0.7152, 0.0722));
    let ambient_ratio = clamp(ambient_lum / total_lum, 0.0, 1.0);

    let clip_pos = camera.view_proj * vec4<f32>(world_hit, 1.0);
    let ndc_depth = clip_pos.z / clip_pos.w;

    var out: FragOut;
    out.depth = ndc_depth;
    out.color = vec4<f32>(final_color, 1.0);
    out.normal = vec4<f32>(normal, ambient_ratio);
    return out;
}
