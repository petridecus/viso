struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    aspect: f32,
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
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec3<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> lighting: LightingUniform;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.world_normal = in.normal;
    out.world_position = in.position;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);

    // Headlight: always comes from camera direction (ensures visible surfaces are lit)
    let view_dir = normalize(camera.position - in.world_position);
    let headlight_diff = max(dot(normal, view_dir), 0.0) * 0.5;

    // Fill lights from fixed world-space directions
    let light1 = normalize(lighting.light1_dir);
    let light2 = normalize(lighting.light2_dir);
    let diff1 = max(dot(normal, light1), 0.0) * lighting.light1_intensity;
    let diff2 = max(dot(normal, light2), 0.0) * lighting.light2_intensity;

    let total_diffuse = headlight_diff + diff1 + diff2;

    // Backbone color - pale green
    let base_color = vec3<f32>(0.6, 0.85, 0.6);

    let final_color = base_color * (lighting.ambient + total_diffuse);

    return vec4<f32>(final_color, 1.0);
}
