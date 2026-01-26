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

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) depth: f32,
    @location(3) vertex_color: vec3<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> lighting: LightingUniform;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.world_normal = in.normal;
    out.world_position = in.position;
    // Compute depth once in vertex shader (cheaper than per-fragment length())
    out.depth = length(camera.position - in.position);
    out.vertex_color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // World-space normal from mesh geometry (already in world space)
    let normal = normalize(in.world_normal);
    let view_dir = normalize(camera.position - in.world_position);

    // HEADLAMP LIGHTING:
    // Light directions follow camera orientation (updated each frame on CPU).
    // All shaders use consistent world-space normals for unified appearance.
    let key_diff = max(dot(normal, lighting.light1_dir), 0.0) * lighting.light1_intensity;
    let fill_diff = max(dot(normal, lighting.light2_dir), 0.0) * lighting.light2_intensity;

    // Specular (Blinn-Phong with tight highlight for jewel-like look)
    let half_vec = normalize(lighting.light1_dir + view_dir);
    let specular = pow(max(dot(normal, half_vec), 0.0), lighting.shininess) * lighting.specular_intensity;

    // Fresnel edge glow - edges facing away from viewer glow brighter
    let fresnel = pow(1.0 - max(dot(normal, view_dir), 0.0), lighting.fresnel_power);
    let fresnel_boost = fresnel * lighting.fresnel_intensity;

    let total_light = lighting.ambient + key_diff + fill_diff + fresnel_boost;

    // Use vertex color for secondary structure coloring
    let base_color = in.vertex_color;

    // EXPONENTIAL FOG - keeps foreground crisp, gradual falloff
    // Only applies fog beyond fog_start (front of protein is full color)
    let fog_distance = max(in.depth - camera.fog_start, 0.0);
    let fog_factor = exp(-fog_distance * camera.fog_density);
    // fog_factor: 1.0 = no fog (close), 0.0 = full fog (far)

    let lit_color = base_color * total_light + vec3<f32>(specular);
    // Blend toward background color (black) for silhouette effect
    let final_color = lit_color * fog_factor;

    return vec4<f32>(final_color, 1.0);
}
