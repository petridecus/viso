struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    aspect: f32,
    forward: vec3<f32>,
    fovy: f32,
    selected_atom_index: i32,
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
    _pad3: vec3<f32>,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // Instance model matrix (4 vec4 columns)
    @location(2) model_0: vec4<f32>,
    @location(3) model_1: vec4<f32>,
    @location(4) model_2: vec4<f32>,
    @location(5) model_3: vec4<f32>,
    // Instance color (hydrophobic = blue, hydrophilic = orange)
    @location(6) instance_color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) depth: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> lighting: LightingUniform;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    // Reconstruct model matrix from columns
    let model = mat4x4<f32>(
        in.model_0,
        in.model_1,
        in.model_2,
        in.model_3,
    );

    // Transform position
    let world_pos = model * vec4<f32>(in.position, 1.0);

    // Transform normal (using upper 3x3 of model matrix)
    // For proper normal transformation, we should use inverse transpose,
    // but for uniform scaling this approximation works
    let normal_matrix = mat3x3<f32>(
        model[0].xyz,
        model[1].xyz,
        model[2].xyz,
    );
    let world_normal = normalize(normal_matrix * in.normal);

    var out: VertexOutput;
    out.clip_position = camera.view_proj * world_pos;
    out.world_normal = world_normal;
    out.world_position = world_pos.xyz;
    out.color = in.instance_color;
    out.depth = length(camera.position - world_pos.xyz);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);
    let view_dir = normalize(camera.position - in.world_position);

    // Camera forward direction from uniform
    let cam_forward = camera.forward;

    // Build camera coordinate frame
    let world_up = vec3<f32>(0.0, 1.0, 0.0);
    var cam_right = cross(cam_forward, world_up);
    if (length(cam_right) < 0.001) {
        cam_right = vec3<f32>(1.0, 0.0, 0.0);
    } else {
        cam_right = normalize(cam_right);
    }
    let cam_up = normalize(cross(cam_right, cam_forward));

    // Strong key light (upper-right), weaker fill (upper-left)
    let key_light = normalize(cam_forward + cam_right * 0.4 + cam_up * 0.6);
    let fill_light = normalize(cam_forward - cam_right * 0.3 + cam_up * 0.4);

    // Diffuse - strong directional contrast
    let key_diff = max(dot(normal, key_light), 0.0) * 0.8;
    let fill_diff = max(dot(normal, fill_light), 0.0) * 0.3;

    // Specular (Blinn-Phong) - tight highlight for cylindrical shape
    let half_vec = normalize(key_light + view_dir);
    let specular = pow(max(dot(normal, half_vec), 0.0), 64.0) * 0.5;

    // Low ambient to allow dark sides
    let ambient = 0.1;
    let total_light = ambient + key_diff + fill_diff;

    // Use instance color (hydrophobic blue or hydrophilic orange)
    let base_color = in.color;

    // Depth fog - fade to black with distance (depth computed in vertex shader)
    let fog_start = 100.0;
    let fog_end = 500.0;
    let fog_factor = clamp((fog_end - in.depth) / (fog_end - fog_start), 0.0, 1.0);

    let lit_color = base_color * total_light + vec3<f32>(specular);
    let final_color = lit_color * fog_factor;

    return vec4<f32>(final_color, 1.0);
}
