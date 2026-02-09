// Composite pass - applies SSAO and silhouette outlines to the rendered scene

struct CompositeParams {
    screen_size: vec2<f32>,
    outline_thickness: f32,
    outline_strength: f32,
    ao_strength: f32,
    near: f32,
    far: f32,
    fog_start: f32,
    fog_density: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var color_texture: texture_2d<f32>;
@group(0) @binding(1) var ssao_texture: texture_2d<f32>;
@group(0) @binding(2) var depth_texture: texture_depth_2d;
@group(0) @binding(3) var tex_sampler: sampler;
@group(0) @binding(4) var depth_sampler: sampler;
@group(0) @binding(5) var<uniform> params: CompositeParams;

// Full-screen triangle (more efficient than quad - no diagonal edge)
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    // Generate oversized triangle that covers the screen
    // Vertices: (-1, -1), (3, -1), (-1, 3)
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    // UV: (0, 1), (2, 1), (0, -1) -> clips to (0,0)-(1,1)
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// Linearize depth from NDC to view-space distance
fn linearize_depth(d: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - d * (far - near));
}

// Depth-relative edge detection with adaptive threshold
// Key insight: 0.1A at front = clear edge; 0.1A at 500A away = noise
fn detect_edges(uv: vec2<f32>, texel_size: vec2<f32>, thickness: f32) -> f32 {
    let offset = texel_size * thickness;

    // Sample depth at center and 4 neighbors
    let d_c = textureSample(depth_texture, depth_sampler, uv);
    let d_t = textureSample(depth_texture, depth_sampler, uv + vec2(0.0, -offset.y));
    let d_b = textureSample(depth_texture, depth_sampler, uv + vec2(0.0, offset.y));
    let d_l = textureSample(depth_texture, depth_sampler, uv + vec2(-offset.x, 0.0));
    let d_r = textureSample(depth_texture, depth_sampler, uv + vec2(offset.x, 0.0));

    // Linearize all depths for meaningful distance comparison
    let l_c = linearize_depth(d_c, params.near, params.far);
    let l_t = linearize_depth(d_t, params.near, params.far);
    let l_b = linearize_depth(d_b, params.near, params.far);
    let l_l = linearize_depth(d_l, params.near, params.far);
    let l_r = linearize_depth(d_r, params.near, params.far);

    // Find maximum depth difference from center
    let max_diff = max(
        max(abs(l_c - l_t), abs(l_c - l_b)),
        max(abs(l_c - l_l), abs(l_c - l_r))
    );

    // DEPTH-RELATIVE THRESHOLD: scale by linearized depth
    // Close objects need smaller absolute diff to trigger edge
    // Far objects need larger absolute diff (prevents noise)
    let relative_diff = max_diff / max(l_c, 0.1);

    // Soft threshold with smoothstep for anti-aliased edges
    return smoothstep(0.008, 0.03, relative_diff);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(color_texture, tex_sampler, in.uv);
    let ao = textureSample(ssao_texture, tex_sampler, in.uv).r;
    let depth = textureSample(depth_texture, depth_sampler, in.uv);

    // Background early-out (no outline on empty space)
    if (depth > 0.9999) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    // === STEP 1: Apply SSAO (darken crevices) ===
    let adjusted_ao = mix(1.0, ao, params.ao_strength);
    var final_color = color.rgb * adjusted_ao;

    // === STEP 2: Exponential fog from linearized depth ===
    let linear_depth = linearize_depth(depth, params.near, params.far);
    let fog_distance = max(linear_depth - params.fog_start, 0.0);
    let fog_factor = exp(-fog_distance * params.fog_density);
    final_color = final_color * fog_factor;

    // === STEP 3: Apply outline LAST (on top of fogged geometry) ===
    let texel_size = 1.0 / params.screen_size;
    let edge = detect_edges(in.uv, texel_size, params.outline_thickness);

    // Outline strength attenuates with depth to prevent
    // pitch-black outlines on fogged distant geometry
    // Use a gentler falloff than the fog itself
    let depth_attenuation = 1.0 / (1.0 + linear_depth * 0.002);
    let attenuated_edge = edge * params.outline_strength * depth_attenuation;

    // Darken toward black for outline effect
    final_color = mix(final_color, vec3<f32>(0.0), attenuated_edge);

    return vec4<f32>(final_color, color.a);
}
