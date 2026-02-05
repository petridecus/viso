// Ray-marched capsule impostors for sidechain rendering
// Capsules = cylinders with hemispherical caps

struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    aspect: f32,
    forward: vec3<f32>,
    fovy: f32,
    hovered_residue: i32,
    fog_start: f32,
    fog_density: f32,
    _pad: f32,
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
    fresnel_power: f32,
    fresnel_intensity: f32,
    _pad3: f32,
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
};

const TUBE_RADIUS: f32 = 0.3;

@group(0) @binding(0) var<storage, read> capsules: array<CapsuleInstance>;
@group(1) @binding(0) var<uniform> camera: CameraUniform;
@group(2) @binding(0) var<uniform> lighting: LightingUniform;
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
    
    // Lighting
    let key_diff = max(dot(normal, lighting.light1_dir), 0.0) * lighting.light1_intensity;
    let fill_diff = max(dot(normal, lighting.light2_dir), 0.0) * lighting.light2_intensity;
    
    let half_vec = normalize(lighting.light1_dir + view_dir);
    let specular = pow(max(dot(normal, half_vec), 0.0), lighting.shininess) * lighting.specular_intensity;
    
    let fresnel = pow(1.0 - max(dot(normal, view_dir), 0.0), lighting.fresnel_power);
    let fresnel_boost = fresnel * lighting.fresnel_intensity;
    
    let total_light = lighting.ambient + key_diff + fill_diff + fresnel_boost;
    
    // Fog
    let world_depth = length(camera.position - world_hit);
    let fog_distance = max(world_depth - camera.fog_start, 0.0);
    let fog_factor = exp(-fog_distance * camera.fog_density);
    
    let lit_color = base_color * total_light + vec3<f32>(specular);
    
    // Edge darkening for selected
    var final_color = lit_color;
    if (outline_factor > 0.0) {
        let edge = pow(1.0 - max(dot(normal, view_dir), 0.0), 2.0);
        final_color = mix(final_color, vec3<f32>(0.0, 0.0, 0.0), edge * 0.6);
    }
    
    final_color = final_color * fog_factor;
    
    let clip_pos = camera.view_proj * vec4<f32>(world_hit, 1.0);
    let ndc_depth = clip_pos.z / clip_pos.w;
    
    var out: FragOut;
    out.depth = ndc_depth;
    out.color = vec4<f32>(final_color, 1.0);
    return out;
}
