// Picking shader for sphere impostors - renders pick IDs to a picking buffer

#import viso::camera::CameraUniform

struct SphereInstance {
    center: vec4<f32>,
    color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) sphere_center: vec3<f32>,
    @location(2) radius: f32,
    @location(3) @interpolate(flat) pick_id: u32,
};

struct FragOut {
    @builtin(frag_depth) depth: f32,
    @location(0) pick_id: u32,
};

@group(0) @binding(0) var<storage, read> spheres: array<SphereInstance>;
@group(1) @binding(0) var<uniform> camera: CameraUniform;

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
    let pick_id = u32(sph.color.w);

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
    out.pick_id = pick_id;

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

    let clip_pos = camera.view_proj * vec4<f32>(world_hit, 1.0);
    let ndc_depth = clip_pos.z / clip_pos.w;

    var out: FragOut;
    out.depth = ndc_depth;
    // Output pick_id + 1 (so 0 means "no hit")
    out.pick_id = in.pick_id + 1u;
    return out;
}
