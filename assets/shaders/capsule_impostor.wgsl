// Ray-marched capsule impostors for backbone tube rendering
// Capsules = cylinders with hemispherical caps
// Provides per-pixel normals matching sphere quality
// Capsules naturally blend at joints for smooth tube appearance

struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    aspect: f32,
    forward: vec3<f32>,
    fovy: f32,
    selected_atom_index: i32,
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
// endpoint_b: xyz = position, w = unused
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
};

struct FragOut {
    @builtin(frag_depth) depth: f32,
    @location(0) color: vec4<f32>,
};

const TUBE_RADIUS: f32 = 0.3;

@group(0) @binding(0) var<storage, read> capsules: array<CapsuleInstance>;
@group(1) @binding(0) var<uniform> camera: CameraUniform;
@group(2) @binding(0) var<uniform> lighting: LightingUniform;

// Ray-capsule intersection using SDF approach
// A capsule is the set of all points within 'radius' of the line segment AB
// Returns (t, axis_param) where axis_param is 0 at A, 1 at B, <0 or >1 for caps
fn intersect_capsule(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    cap_a: vec3<f32>,
    cap_b: vec3<f32>,
    radius: f32
) -> vec3<f32> {  // Returns (t, axis_param, hit_type) where hit_type: 0=miss, 1=cylinder, 2=cap_a, 3=cap_b
    let ba = cap_b - cap_a;
    let ba_len = length(ba);
    
    if (ba_len < 0.0001) {
        // Degenerate capsule = sphere
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
    
    // Decompose ray direction and offset into components parallel and perpendicular to axis
    let ray_par = dot(ray_dir, ba_dir);
    let ray_perp = ray_dir - ba_dir * ray_par;
    let oa_par = dot(oa, ba_dir);
    let oa_perp = oa - ba_dir * oa_par;
    
    // Solve quadratic for infinite cylinder
    let a = dot(ray_perp, ray_perp);
    let b = 2.0 * dot(oa_perp, ray_perp);
    let c = dot(oa_perp, oa_perp) - radius * radius;
    
    var best_t: f32 = 1e20;
    var best_axis: f32 = 0.0;
    var best_type: f32 = 0.0;
    
    // Check cylinder body intersection
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
    
    // Check cap A (sphere at cap_a)
    {
        let oc = ray_origin - cap_a;
        let a_sph = dot(ray_dir, ray_dir);
        let b_sph = 2.0 * dot(oc, ray_dir);
        let c_sph = dot(oc, oc) - radius * radius;
        let disc = b_sph * b_sph - 4.0 * a_sph * c_sph;
        if (disc >= 0.0) {
            let t = (-b_sph - sqrt(disc)) / (2.0 * a_sph);
            if (t > 0.001 && t < best_t) {
                // Check that hit is on the cap side (axis_param <= 0)
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
    
    // Check cap B (sphere at cap_b)
    {
        let oc = ray_origin - cap_b;
        let a_sph = dot(ray_dir, ray_dir);
        let b_sph = 2.0 * dot(oc, ray_dir);
        let c_sph = dot(oc, oc) - radius * radius;
        let disc = b_sph * b_sph - 4.0 * a_sph * c_sph;
        if (disc >= 0.0) {
            let t = (-b_sph - sqrt(disc)) / (2.0 * a_sph);
            if (t > 0.001 && t < best_t) {
                // Check that hit is on the cap side (axis_param >= 1)
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

// Compute capsule normal at a point
fn capsule_normal(
    point: vec3<f32>,
    cap_a: vec3<f32>,
    cap_b: vec3<f32>,
    hit_type: f32
) -> vec3<f32> {
    if (hit_type == 2.0) {
        // Cap A: sphere normal
        return normalize(point - cap_a);
    } else if (hit_type == 3.0) {
        // Cap B: sphere normal
        return normalize(point - cap_b);
    } else {
        // Cylinder body: radial normal
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
    // Quad vertices
    let quad = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(1.0, 1.0)
    );
    
    let cap = capsules[iidx];
    let endpoint_a = cap.endpoint_a.xyz;
    let endpoint_b = cap.endpoint_b.xyz;
    let radius = TUBE_RADIUS;
    let color_a = cap.color_a.xyz;
    let color_b = cap.color_b.xyz;
    
    // Capsule center and axis
    let center = (endpoint_a + endpoint_b) * 0.5;
    let axis = endpoint_b - endpoint_a;
    let seg_length = length(axis);
    let axis_dir = select(vec3<f32>(0.0, 1.0, 0.0), axis / seg_length, seg_length > 0.0001);
    
    // Build oriented billboard
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
    
    // Quad size: encompasses capsule + margin for perspective
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
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragOut {
    // Ray from camera through this fragment
    let ray_origin = camera.position;
    let ray_dir = normalize(in.world_pos - camera.position);
    
    // Intersect ray with capsule
    let hit = intersect_capsule(ray_origin, ray_dir, in.endpoint_a, in.endpoint_b, in.radius);
    
    if (hit.x < 0.0) {
        discard;
    }
    
    let t = hit.x;
    let axis_param = hit.y;  // 0 at A, 1 at B
    let hit_type = hit.z;
    
    let world_hit = ray_origin + ray_dir * t;
    
    // Compute normal
    let normal = capsule_normal(world_hit, in.endpoint_a, in.endpoint_b, hit_type);
    let view_dir = normalize(camera.position - world_hit);
    
    // Interpolate color along capsule axis
    let base_color = mix(in.color_a, in.color_b, axis_param);
    
    // === LIGHTING (matches sphere shader exactly) ===
    let key_diff = max(dot(normal, lighting.light1_dir), 0.0) * lighting.light1_intensity;
    let fill_diff = max(dot(normal, lighting.light2_dir), 0.0) * lighting.light2_intensity;
    
    // Specular (Blinn-Phong)
    let half_vec = normalize(lighting.light1_dir + view_dir);
    let specular = pow(max(dot(normal, half_vec), 0.0), lighting.shininess) * lighting.specular_intensity;
    
    // Fresnel edge glow
    let fresnel = pow(1.0 - max(dot(normal, view_dir), 0.0), lighting.fresnel_power);
    let fresnel_boost = fresnel * lighting.fresnel_intensity;
    
    let total_light = lighting.ambient + key_diff + fill_diff + fresnel_boost;
    
    // Fog
    let world_depth = length(camera.position - world_hit);
    let fog_distance = max(world_depth - camera.fog_start, 0.0);
    let fog_factor = exp(-fog_distance * camera.fog_density);
    
    let lit_color = base_color * total_light + vec3<f32>(specular);
    let final_color = lit_color * fog_factor;
    
    // Compute proper depth
    let clip_pos = camera.view_proj * vec4<f32>(world_hit, 1.0);
    let ndc_depth = clip_pos.z / clip_pos.w;
    
    var out: FragOut;
    out.depth = ndc_depth;
    out.color = vec4<f32>(final_color, 1.0);
    return out;
}
