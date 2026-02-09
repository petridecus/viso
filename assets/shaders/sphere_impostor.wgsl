// Ray-marched sphere impostors for ball-and-stick rendering
// Each sphere is a billboard quad with per-pixel ray-sphere intersection

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

    // Lighting (same model as capsule_impostor.wgsl)
    let key_diff = max(dot(normal, lighting.light1_dir), 0.0) * lighting.light1_intensity;
    let fill_diff = max(dot(normal, lighting.light2_dir), 0.0) * lighting.light2_intensity;

    let half_vec = normalize(lighting.light1_dir + view_dir);
    let specular = pow(max(dot(normal, half_vec), 0.0), lighting.shininess) * lighting.specular_intensity;

    // IBL diffuse
    let irradiance = textureSample(irradiance_map, env_sampler, normal).rgb;
    let ibl_diffuse = irradiance * lighting.ibl_strength;
    let ambient_light = mix(vec3(lighting.ambient), ibl_diffuse, lighting.ibl_strength);

    // Rim lighting
    let rim = compute_rim(normal, view_dir);

    let total_light = ambient_light + key_diff + fill_diff;
    let lit_color = base_color * total_light + vec3<f32>(specular) + rim;

    // Edge darkening for selected
    var final_color = lit_color;
    if (outline_factor > 0.0) {
        let edge = pow(1.0 - max(dot(normal, view_dir), 0.0), 2.0);
        final_color = mix(final_color, vec3<f32>(0.0, 0.0, 0.0), edge * 0.6);
    }

    let clip_pos = camera.view_proj * vec4<f32>(world_hit, 1.0);
    let ndc_depth = clip_pos.z / clip_pos.w;

    var out: FragOut;
    out.depth = ndc_depth;
    out.color = vec4<f32>(final_color, 1.0);
    out.normal = vec4<f32>(normal, 0.0);
    return out;
}
