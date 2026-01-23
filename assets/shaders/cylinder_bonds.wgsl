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
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);

    // Two-light setup from shared lighting uniform (fixed world-space directions)
    let light1 = normalize(lighting.light1_dir);
    let light2 = normalize(lighting.light2_dir);

    // Simple Lambertian diffuse - consistent from all angles
    let diff1 = max(dot(normal, light1), 0.0) * lighting.light1_intensity;
    let diff2 = max(dot(normal, light2), 0.0) * lighting.light2_intensity;
    let total_diffuse = diff1 + diff2;

    // Use instance color (hydrophobic blue or hydrophilic orange)
    let base_color = in.color;

    // Ambient + diffuse only (no specular for consistent matte look)
    let final_color = base_color * (lighting.ambient + total_diffuse);

    return vec4<f32>(final_color, 1.0);
}
