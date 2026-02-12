// Ray-marched capsule impostors for sidechain rendering
// Capsules = cylinders with hemispherical caps

struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    aspect: f32,
    forward: vec3<f32>,
    fovy: f32,
    hovered_residue: i32,
};

struct LightingUniform {
    light1_dir: vec3<f32>,
    _pad1: f32,
    light2_dir: vec3<f32>,
    _pad2: f32,
    light1_intensity: f32,
    light2_intensity: f32,
    ambient: f32,
    specular_intensity: f32,
    shininess: f32,
    rim_power: f32,
    rim_intensity: f32,
    rim_directionality: f32,
    rim_color: vec3<f32>,
    ibl_strength: f32,
    rim_dir: vec3<f32>,
    _pad3: f32,
    roughness: f32,
    metalness: f32,
    _pad4: f32,
    _pad5: f32,
};

// Per-instance data for capsule
// endpoint_a: xyz = position, w = radius
// endpoint_b: xyz = position, w = residue_idx (packed as float)
// color_a: xyz = RGB at endpoint A, w = unused
// color_b: xyz = RGB at endpoint B, w = unused
struct CapsuleInstance {
    endpoint_a: vec4<f32>,
    endpoint_b: vec4<f32>,
    color_a: vec4<f32>,
    color_b: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) endpoint_a: vec3<f32>,
    @location(2) endpoint_b: vec3<f32>,
    @location(3) radius: f32,
    @location(4) color_a: vec3<f32>,
    @location(5) color_b: vec3<f32>,
    @location(6) @interpolate(flat) residue_idx: u32,
};

struct FragOut {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

const TUBE_RADIUS: f32 = 0.3;

@group(0) @binding(0) var<storage, read> capsules: array<CapsuleInstance>;
@group(1) @binding(0) var<uniform> camera: CameraUniform;
@group(2) @binding(0) var<uniform> lighting: LightingUniform;
@group(2) @binding(1) var irradiance_map: texture_cube<f32>;
@group(2) @binding(2) var env_sampler: sampler;
@group(2) @binding(3) var prefiltered_map: texture_cube<f32>;
@group(2) @binding(4) var brdf_lut: texture_2d<f32>;
@group(3) @binding(0) var<storage, read> selection: array<u32>;

// Check if a residue is selected (bit array lookup)
fn is_selected(residue_idx: u32) -> bool {
    let word_idx = residue_idx / 32u;
    let bit_idx = residue_idx % 32u;
    if (word_idx >= arrayLength(&selection)) {
        return false;
    }
    return (selection[word_idx] & (1u << bit_idx)) != 0u;
}

const PI: f32 = 3.14159265359;

// GGX/Trowbridge-Reitz normal distribution function
fn distribution_ggx(NdotH: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Schlick-GGX geometry function (single direction)
fn geometry_schlick_ggx(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

// Smith's geometry function (both view and light)
fn geometry_smith(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(NdotV, roughness) * geometry_schlick_ggx(NdotL, roughness);
}

// Fresnel-Schlick approximation
fn fresnel_schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Compute rim lighting: view-dependent + optional directional back-light
fn compute_rim(normal: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    let NdotV = max(dot(normal, view_dir), 0.0);
    let rim = pow(1.0 - NdotV, lighting.rim_power);

    let back_factor = max(dot(normal, -lighting.rim_dir), 0.0);
    let directional_rim = rim * mix(1.0, back_factor, lighting.rim_directionality);

    return directional_rim * lighting.rim_intensity * lighting.rim_color;
}

// Ray-capsule intersection
fn intersect_capsule(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    cap_a: vec3<f32>,
    cap_b: vec3<f32>,
    radius: f32
) -> vec3<f32> {
    let ba = cap_b - cap_a;
    let ba_len = length(ba);

    if (ba_len < 0.0001) {
        let oc = ray_origin - cap_a;
        let a = dot(ray_dir, ray_dir);
        let b = 2.0 * dot(oc, ray_dir);
        let c = dot(oc, oc) - radius * radius;
        let disc = b * b - 4.0 * a * c;
        if (disc < 0.0) { return vec3<f32>(-1.0, 0.0, 0.0); }
        let t = (-b - sqrt(disc)) / (2.0 * a);
        if (t > 0.001) { return vec3<f32>(t, 0.5, 2.0); }
        return vec3<f32>(-1.0, 0.0, 0.0);
    }

    let ba_dir = ba / ba_len;
    let oa = ray_origin - cap_a;

    let ray_par = dot(ray_dir, ba_dir);
    let ray_perp = ray_dir - ba_dir * ray_par;
    let oa_par = dot(oa, ba_dir);
    let oa_perp = oa - ba_dir * oa_par;

    let a = dot(ray_perp, ray_perp);
    let b = 2.0 * dot(oa_perp, ray_perp);
    let c = dot(oa_perp, oa_perp) - radius * radius;

    var best_t: f32 = 1e20;
    var best_axis: f32 = 0.0;
    var best_type: f32 = 0.0;

    let disc_cyl = b * b - 4.0 * a * c;
    if (disc_cyl >= 0.0 && a > 0.0001) {
        let sqrt_disc = sqrt(disc_cyl);
        let t1 = (-b - sqrt_disc) / (2.0 * a);
        let t2 = (-b + sqrt_disc) / (2.0 * a);

        for (var i = 0; i < 2; i++) {
            let t = select(t2, t1, i == 0);
            if (t > 0.001 && t < best_t) {
                let axis_pos = oa_par + ray_par * t;
                if (axis_pos >= 0.0 && axis_pos <= ba_len) {
                    best_t = t;
                    best_axis = axis_pos / ba_len;
                    best_type = 1.0;
                }
            }
        }
    }

    // Check cap A
    {
        let oc = ray_origin - cap_a;
        let a_sph = dot(ray_dir, ray_dir);
        let b_sph = 2.0 * dot(oc, ray_dir);
        let c_sph = dot(oc, oc) - radius * radius;
        let disc = b_sph * b_sph - 4.0 * a_sph * c_sph;
        if (disc >= 0.0) {
            let t = (-b_sph - sqrt(disc)) / (2.0 * a_sph);
            if (t > 0.001 && t < best_t) {
                let hit = ray_origin + ray_dir * t;
                let axis_pos = dot(hit - cap_a, ba_dir);
                if (axis_pos <= 0.0) {
                    best_t = t;
                    best_axis = 0.0;
                    best_type = 2.0;
                }
            }
        }
    }

    // Check cap B
    {
        let oc = ray_origin - cap_b;
        let a_sph = dot(ray_dir, ray_dir);
        let b_sph = 2.0 * dot(oc, ray_dir);
        let c_sph = dot(oc, oc) - radius * radius;
        let disc = b_sph * b_sph - 4.0 * a_sph * c_sph;
        if (disc >= 0.0) {
            let t = (-b_sph - sqrt(disc)) / (2.0 * a_sph);
            if (t > 0.001 && t < best_t) {
                let hit = ray_origin + ray_dir * t;
                let axis_pos = dot(hit - cap_a, ba_dir);
                if (axis_pos >= ba_len) {
                    best_t = t;
                    best_axis = 1.0;
                    best_type = 3.0;
                }
            }
        }
    }

    if (best_type == 0.0) {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }

    return vec3<f32>(best_t, best_axis, best_type);
}

fn capsule_normal(
    point: vec3<f32>,
    cap_a: vec3<f32>,
    cap_b: vec3<f32>,
    hit_type: f32
) -> vec3<f32> {
    if (hit_type == 2.0) {
        return normalize(point - cap_a);
    } else if (hit_type == 3.0) {
        return normalize(point - cap_b);
    } else {
        let axis = normalize(cap_b - cap_a);
        let to_point = point - cap_a;
        let proj = axis * dot(to_point, axis);
        return normalize(to_point - proj);
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

    let cap = capsules[iidx];
    let endpoint_a = cap.endpoint_a.xyz;
    let endpoint_b = cap.endpoint_b.xyz;
    let color_a = cap.color_a.xyz;
    let color_b = cap.color_b.xyz;
    // Residue index packed in endpoint_b.w
    let residue_idx = u32(cap.endpoint_b.w);

    // Use radius from instance data (packed in endpoint_a.w)
    // Fall back to TUBE_RADIUS if instance radius is 0 (legacy compatibility)
    var radius = cap.endpoint_a.w;
    if (radius < 0.001) {
        radius = TUBE_RADIUS;
    }
    // Make selected residues 1.4x larger (matches Foldit SEL_THICKNESS_MULTIPLIER)
    if (is_selected(residue_idx)) {
        radius = radius * 1.4;
    }

    let center = (endpoint_a + endpoint_b) * 0.5;
    let axis = endpoint_b - endpoint_a;
    let seg_length = length(axis);
    let axis_dir = select(vec3<f32>(0.0, 1.0, 0.0), axis / seg_length, seg_length > 0.0001);

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

    let half_width = radius * 1.6;
    let half_height = seg_length * 0.5 + radius * 1.6;

    let local_uv = quad[vidx];
    let world_offset = right * local_uv.x * half_width + up * local_uv.y * half_height;
    let world_pos = center + world_offset;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.endpoint_a = endpoint_a;
    out.endpoint_b = endpoint_b;
    out.radius = radius;
    out.color_a = color_a;
    out.color_b = color_b;
    out.residue_idx = residue_idx;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragOut {
    let ray_origin = camera.position;
    let ray_dir = normalize(in.world_pos - camera.position);

    let hit = intersect_capsule(ray_origin, ray_dir, in.endpoint_a, in.endpoint_b, in.radius);

    if (hit.x < 0.0) {
        discard;
    }

    let t = hit.x;
    let axis_param = hit.y;
    let hit_type = hit.z;

    let world_hit = ray_origin + ray_dir * t;

    let normal = capsule_normal(world_hit, in.endpoint_a, in.endpoint_b, hit_type);
    let view_dir = normalize(camera.position - world_hit);

    // Interpolate base color along capsule axis
    var base_color = mix(in.color_a, in.color_b, axis_param);

    // Hover highlight: make brighter with additive white
    if (camera.hovered_residue >= 0 && u32(camera.hovered_residue) == in.residue_idx) {
        base_color = base_color + vec3<f32>(0.3, 0.3, 0.3);
    }

    // Selection highlight: blend original color with blue (matches Foldit)
    // Foldit uses: color * 0.5 + SELECT_COLOR * 1.0 where SELECT_COLOR = (0,0,1)
    var outline_factor = 0.0;
    if (is_selected(in.residue_idx)) {
        base_color = base_color * 0.5 + vec3<f32>(0.0, 0.0, 1.0);
        outline_factor = 1.0;
    }

    let NdotV = max(dot(normal, view_dir), 0.0);

    // PBR: Fresnel reflectance at normal incidence
    let F0 = mix(vec3<f32>(0.04), base_color, lighting.metalness);
    let roughness = lighting.roughness;

    // IBL diffuse: sample irradiance cubemap with surface normal
    let irradiance = textureSample(irradiance_map, env_sampler, normal).rgb;
    let ibl_diffuse = irradiance * lighting.ibl_strength;
    let ambient_light = mix(vec3(lighting.ambient), ibl_diffuse, lighting.ibl_strength);

    // Rim lighting
    let rim = compute_rim(normal, view_dir);

    // PBR direct lighting: accumulate from both lights
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

    // SDF edge AA: smooth alpha at capsule boundary for alpha-to-coverage MSAA
    let pa = in.world_pos - in.endpoint_a;
    let ba = in.endpoint_b - in.endpoint_a;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    let sdf = length(pa - ba * h) - in.radius;
    let aa_edge = fwidth(sdf);
    let alpha = smoothstep(aa_edge, -aa_edge, sdf);

    var out: FragOut;
    out.depth = ndc_depth;
    out.color = vec4<f32>(final_color, alpha);
    out.normal = vec4<f32>(normal, ambient_ratio);
    return out;
}
