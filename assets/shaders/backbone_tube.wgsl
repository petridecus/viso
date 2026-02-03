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

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) residue_idx: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) depth: f32,
    @location(3) vertex_color: vec3<f32>,
    @location(4) @interpolate(flat) residue_idx: u32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> lighting: LightingUniform;
@group(2) @binding(0) var<storage, read> selection: array<u32>;

// Check if a residue is selected (bit array lookup)
fn is_selected(residue_idx: u32) -> bool {
    let word_idx = residue_idx / 32u;
    let bit_idx = residue_idx % 32u;
    if (word_idx >= arrayLength(&selection)) {
        return false;
    }
    return (selection[word_idx] & (1u << bit_idx)) != 0u;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Expand selected residues outward along normal for 1.4x radius (matches Foldit)
    var position = in.position;
    if (is_selected(in.residue_idx)) {
        position = position + in.normal * 0.24;  // ~1.4x expansion (tube radius ~0.6)
    }

    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
    out.world_normal = in.normal;
    out.world_position = position;
    out.depth = length(camera.position - position);
    out.vertex_color = in.color;
    out.residue_idx = in.residue_idx;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal = normalize(in.world_normal);
    let view_dir = normalize(camera.position - in.world_position);

    let key_diff = max(dot(normal, lighting.light1_dir), 0.0) * lighting.light1_intensity;
    let fill_diff = max(dot(normal, lighting.light2_dir), 0.0) * lighting.light2_intensity;

    let half_vec = normalize(lighting.light1_dir + view_dir);
    let specular = pow(max(dot(normal, half_vec), 0.0), lighting.shininess) * lighting.specular_intensity;

    let fresnel = pow(1.0 - max(dot(normal, view_dir), 0.0), lighting.fresnel_power);
    let fresnel_boost = fresnel * lighting.fresnel_intensity;

    let total_light = lighting.ambient + key_diff + fill_diff + fresnel_boost;

    // Start with vertex color
    var base_color = in.vertex_color;

    // Hover highlight: make brighter with additive white
    if (camera.hovered_residue >= 0 && u32(camera.hovered_residue) == in.residue_idx) {
        base_color = base_color + vec3<f32>(0.3, 0.3, 0.3);  // Add brightness
    }

    // Selection highlight: blend original color with blue (matches Foldit)
    // Foldit uses: color * 0.5 + SELECT_COLOR * 1.0 where SELECT_COLOR = (0,0,1)
    var outline_factor = 0.0;
    if (is_selected(in.residue_idx)) {
        base_color = base_color * 0.5 + vec3<f32>(0.0, 0.0, 1.0);
        outline_factor = 1.0;
    }

    let fog_distance = max(in.depth - camera.fog_start, 0.0);
    let fog_factor = exp(-fog_distance * camera.fog_density);

    let lit_color = base_color * total_light + vec3<f32>(specular);

    // For selected residues, add dark edge effect
    var final_color = lit_color;
    if (outline_factor > 0.0) {
        let edge = pow(1.0 - max(dot(normal, view_dir), 0.0), 2.0);
        final_color = mix(final_color, vec3<f32>(0.0, 0.0, 0.0), edge * 0.6);
    }

    final_color = final_color * fog_factor;

    return vec4<f32>(final_color, 1.0);
}
