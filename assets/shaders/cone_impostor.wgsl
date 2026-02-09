// Ray-marched cone impostor for pull arrow rendering
// Cone points from base (atom) toward tip (mouse target)

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
};

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

// Compute rim lighting: view-dependent + optional directional back-light
fn compute_rim(normal: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    let NdotV = max(dot(normal, view_dir), 0.0);
    let rim = pow(1.0 - NdotV, lighting.rim_power);

    let back_factor = max(dot(normal, -lighting.rim_dir), 0.0);
    let directional_rim = rim * mix(1.0, back_factor, lighting.rim_directionality);

    return directional_rim * lighting.rim_intensity * lighting.rim_color;
}

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

    // Lighting
    let key_diff = max(dot(normal, lighting.light1_dir), 0.0) * lighting.light1_intensity;
    let fill_diff = max(dot(normal, lighting.light2_dir), 0.0) * lighting.light2_intensity;

    let half_vec = normalize(lighting.light1_dir + view_dir);
    let specular = pow(max(dot(normal, half_vec), 0.0), lighting.shininess) * lighting.specular_intensity;

    // IBL diffuse: sample irradiance cubemap with surface normal
    let irradiance = textureSample(irradiance_map, env_sampler, normal).rgb;
    let ibl_diffuse = irradiance * lighting.ibl_strength;
    let ambient_light = mix(vec3(lighting.ambient), ibl_diffuse, lighting.ibl_strength);

    // Rim lighting (replaces old Fresnel)
    let rim = compute_rim(normal, view_dir);

    let total_light = ambient_light + key_diff + fill_diff;

    // Rim is additive (light from behind, not modulated by surface color)
    let lit_color = base_color * total_light + vec3<f32>(specular) + rim;
    let final_color = lit_color;

    let clip_pos = camera.view_proj * vec4<f32>(world_hit, 1.0);
    let ndc_depth = clip_pos.z / clip_pos.w;

    var out: FragOut;
    out.depth = ndc_depth;
    out.color = vec4<f32>(final_color, 1.0);
    out.normal = vec4<f32>(normal, 0.0);
    return out;
}
