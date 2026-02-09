// Instanced tube rendering shader
// Uses pre-computed RMF frames from compute shader
// Template mesh is transformed per-instance for each tube span

// Camera uniform (same as backbone_tube.wgsl)
struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    aspect: f32,
    forward: vec3<f32>,
    fovy: f32,
    hovered_residue: i32,
};

// Lighting uniform (same as backbone_tube.wgsl)
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

// Spline frame from compute shader
struct SplineFrame {
    position: vec4<f32>,
    tangent: vec4<f32>,
    normal: vec4<f32>,
    binormal: vec4<f32>,
}

// Tube rendering parameters
struct TubeParams {
    radius: f32,
    frames_per_span: u32,  // Matches segments_per_span from compute
    total_frames: u32,
    _pad: u32,
}

// Per-span instance data
struct SpanData {
    color: vec4<f32>,       // xyz = color, w = unused
    residue_idx: u32,       // Residue index for picking/selection
    start_frame: u32,       // First frame index for this span
    _pad0: u32,
    _pad1: u32,
}

// Template vertex input (cross-section of tube)
struct VertexInput {
    @location(0) position: vec3<f32>,  // x,y = cross-section position, z = t (0-1 along span)
    @location(1) normal: vec3<f32>,    // Cross-section normal (will be transformed by frame)
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) vertex_color: vec3<f32>,
    @location(3) @interpolate(flat) residue_idx: u32,
}

// Bind group 0: Tube-specific data
@group(0) @binding(0) var<storage, read> frames: array<SplineFrame>;
@group(0) @binding(1) var<uniform> tube_params: TubeParams;
@group(0) @binding(2) var<storage, read> span_data: array<SpanData>;

// Bind group 1: Camera
@group(1) @binding(0) var<uniform> camera: CameraUniform;

// Bind group 2: Lighting
@group(2) @binding(0) var<uniform> lighting: LightingUniform;

// Bind group 3: Selection
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

@vertex
fn vs_main(
    vertex: VertexInput,
    @builtin(instance_index) span_idx: u32,
) -> VertexOutput {
    var out: VertexOutput;

    let span = span_data[span_idx];
    let t = vertex.position.z;  // 0-1 along span

    // Calculate frame indices for interpolation
    let frames_per_span = f32(tube_params.frames_per_span);
    let frame_offset = f32(span.start_frame);
    let frame_t = t * frames_per_span;
    let frame_idx0 = u32(frame_offset + floor(frame_t));
    let frame_idx1 = min(frame_idx0 + 1u, tube_params.total_frames - 1u);
    let lerp_t = fract(frame_t);

    let frame0 = frames[frame_idx0];
    let frame1 = frames[frame_idx1];

    // Interpolate frame
    let pos = mix(frame0.position.xyz, frame1.position.xyz, lerp_t);
    let normal_frame = normalize(mix(frame0.normal.xyz, frame1.normal.xyz, lerp_t));
    let binormal_frame = normalize(mix(frame0.binormal.xyz, frame1.binormal.xyz, lerp_t));

    // Transform cross-section vertex using frame
    // vertex.position.xy is the local cross-section position
    let offset = vertex.position.x * normal_frame + vertex.position.y * binormal_frame;
    var world_pos = pos + offset * tube_params.radius;

    // Transform normal from local to world space
    let world_normal = normalize(vertex.normal.x * normal_frame + vertex.normal.y * binormal_frame);

    let residue_idx = span.residue_idx;

    // Expand selected residues outward along normal (matches original tube shader)
    if (is_selected(residue_idx)) {
        world_pos = world_pos + world_normal * 0.24;  // ~1.4x expansion
    }

    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_normal = world_normal;
    out.world_position = world_pos;
    out.vertex_color = span.color.xyz;
    out.residue_idx = residue_idx;

    return out;
}

struct FragOutput {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

@fragment
fn fs_main(in: VertexOutput) -> FragOutput {
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
        base_color = base_color + vec3<f32>(0.3, 0.3, 0.3);
    }

    // Selection highlight: blend with blue
    var outline_factor = 0.0;
    if (is_selected(in.residue_idx)) {
        base_color = base_color * 0.5 + vec3<f32>(0.0, 0.0, 1.0);
        outline_factor = 1.0;
    }

    let lit_color = base_color * total_light + vec3<f32>(specular);

    // For selected residues, add dark edge effect
    var final_color = lit_color;
    if (outline_factor > 0.0) {
        let edge = pow(1.0 - max(dot(normal, view_dir), 0.0), 2.0);
        final_color = mix(final_color, vec3<f32>(0.0, 0.0, 0.0), edge * 0.6);
    }

    var out: FragOutput;
    out.color = vec4<f32>(final_color, 1.0);
    out.normal = vec4<f32>(normal, 0.0);
    return out;
}
