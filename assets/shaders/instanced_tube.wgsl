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
    rim_power: f32,
    rim_intensity: f32,
    rim_directionality: f32,
    rim_color: vec3<f32>,
    ibl_strength: f32,
    rim_dir: vec3<f32>,
    _pad3: f32,
    roughness: f32,
    metalness: f32,
    _pad4: f32,
    _pad5: f32,
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
@group(2) @binding(1) var irradiance_map: texture_cube<f32>;
@group(2) @binding(2) var env_sampler: sampler;
@group(2) @binding(3) var prefiltered_map: texture_cube<f32>;
@group(2) @binding(4) var brdf_lut: texture_2d<f32>;

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

const PI: f32 = 3.14159265359;

// GGX/Trowbridge-Reitz normal distribution function
fn distribution_ggx(NdotH: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Schlick-GGX geometry function (single direction)
fn geometry_schlick_ggx(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

// Smith's geometry function (both view and light)
fn geometry_smith(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    return geometry_schlick_ggx(NdotV, roughness) * geometry_schlick_ggx(NdotL, roughness);
}

// Fresnel-Schlick approximation
fn fresnel_schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
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

    let NdotV = max(dot(normal, view_dir), 0.0);

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

    // PBR: Fresnel reflectance at normal incidence
    let F0 = mix(vec3<f32>(0.04), base_color, lighting.metalness);
    let roughness = lighting.roughness;

    // IBL diffuse: sample irradiance cubemap with surface normal
    let irradiance = textureSample(irradiance_map, env_sampler, normal).rgb;
    let ibl_diffuse = irradiance * lighting.ibl_strength;
    let ambient_light = mix(vec3(lighting.ambient), ibl_diffuse, lighting.ibl_strength);

    // Rim lighting
    let rim = compute_rim(normal, view_dir);

    // PBR direct lighting: accumulate from both lights
    var Lo = vec3<f32>(0.0);

    // Key light
    {
        let L = lighting.light1_dir;
        let H = normalize(L + view_dir);
        let NdotL = max(dot(normal, L), 0.0);
        let NdotH = max(dot(normal, H), 0.0);
        let HdotV = max(dot(H, view_dir), 0.0);

        let D = distribution_ggx(NdotH, roughness);
        let G = geometry_smith(NdotV, NdotL, roughness);
        let F = fresnel_schlick(HdotV, F0);

        let numerator = D * G * F;
        let denominator = 4.0 * NdotV * NdotL + 0.0001;
        let specular = numerator / denominator;

        let kD = (vec3<f32>(1.0) - F) * (1.0 - lighting.metalness);
        Lo += (kD * base_color / PI + specular) * lighting.light1_intensity * NdotL;
    }

    // Fill light
    {
        let L = lighting.light2_dir;
        let H = normalize(L + view_dir);
        let NdotL = max(dot(normal, L), 0.0);
        let NdotH = max(dot(normal, H), 0.0);
        let HdotV = max(dot(H, view_dir), 0.0);

        let D = distribution_ggx(NdotH, roughness);
        let G = geometry_smith(NdotV, NdotL, roughness);
        let F = fresnel_schlick(HdotV, F0);

        let numerator = D * G * F;
        let denominator = 4.0 * NdotV * NdotL + 0.0001;
        let specular = numerator / denominator;

        let kD = (vec3<f32>(1.0) - F) * (1.0 - lighting.metalness);
        Lo += (kD * base_color / PI + specular) * lighting.light2_intensity * NdotL;
    }

    // Separate ambient and direct contributions for ambient-only AO
    // Specular IBL: sample prefiltered environment map at roughness-dependent mip
    let R = reflect(-view_dir, normal);
    let max_mip = 5.0; // 6 mip levels (0-5)
    let prefiltered_color = textureSampleLevel(prefiltered_map, env_sampler, R, roughness * max_mip).rgb;
    let brdf_sample = textureSample(brdf_lut, env_sampler, vec2<f32>(NdotV, roughness)).rg;
    let specular_ibl = prefiltered_color * (F0 * brdf_sample.x + brdf_sample.y) * lighting.ibl_strength;

    let ambient_contribution = base_color * ambient_light + specular_ibl;
    let direct_contribution = Lo + rim;
    let lit_color = ambient_contribution + direct_contribution;

    // For selected residues, add dark edge effect
    var final_color = lit_color;
    if (outline_factor > 0.0) {
        let edge = pow(1.0 - max(dot(normal, view_dir), 0.0), 2.0);
        final_color = mix(final_color, vec3<f32>(0.0, 0.0, 0.0), edge * 0.6);
    }

    // Compute ambient ratio for ambient-only AO in composite pass
    let total_lum = max(dot(final_color, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0001);
    let ambient_lum = dot(ambient_contribution, vec3<f32>(0.2126, 0.7152, 0.0722));
    let ambient_ratio = clamp(ambient_lum / total_lum, 0.0, 1.0);

    var out: FragOutput;
    out.color = vec4<f32>(final_color, 1.0);
    out.normal = vec4<f32>(normal, ambient_ratio);
    return out;
}
