// Picking shader for capsule impostors - renders residue indices to a picking buffer

#import viso::camera::CameraUniform
#import viso::sdf::intersect_capsule

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
    @location(4) @interpolate(flat) residue_idx: u32,
};

struct FragOut {
    @builtin(frag_depth) depth: f32,
    @location(0) residue_id: u32,
};

const TUBE_RADIUS: f32 = 0.3;

@group(0) @binding(0) var<storage, read> capsules: array<CapsuleInstance>;
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

    let cap = capsules[iidx];
    let endpoint_a = cap.endpoint_a.xyz;
    let endpoint_b = cap.endpoint_b.xyz;
    let residue_idx = u32(cap.endpoint_b.w);
    let radius = TUBE_RADIUS;

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
    let world_hit = ray_origin + ray_dir * t;

    let clip_pos = camera.view_proj * vec4<f32>(world_hit, 1.0);
    let ndc_depth = clip_pos.z / clip_pos.w;

    var out: FragOut;
    out.depth = ndc_depth;
    // Output residue index + 1 (so 0 means "no hit")
    out.residue_id = in.residue_idx + 1u;
    return out;
}
