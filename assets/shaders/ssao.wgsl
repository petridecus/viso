// SSAO (Screen Space Ambient Occlusion) shader
// Works in VIEW SPACE for camera-independent results

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct Kernel {
    samples: array<vec4<f32>, 32>,
};

struct SsaoParams {
    // Inverse projection matrix for reconstructing view-space positions
    inv_proj: mat4x4<f32>,
    // Projection matrix for projecting samples back to screen
    proj: mat4x4<f32>,
    // View matrix for transforming world-space normals to view-space
    view: mat4x4<f32>,
    // Screen dimensions
    screen_size: vec2<f32>,
    // Near/far planes
    near: f32,
    far: f32,
};

@group(0) @binding(0) var depth_texture: texture_depth_2d;
@group(0) @binding(1) var noise_texture: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var noise_sampler: sampler;
@group(0) @binding(4) var<uniform> kernel: Kernel;
@group(0) @binding(5) var<uniform> params: SsaoParams;
@group(0) @binding(6) var normal_texture: texture_2d<f32>;

const RADIUS: f32 = 2.0;
const BIAS: f32 = 0.05;

// Load raw depth via textureLoad (bypasses sampler — works on Vulkan and GL/GLES)
fn load_depth(uv: vec2<f32>) -> f32 {
    let dims = vec2<f32>(textureDimensions(depth_texture, 0));
    let texel = vec2<i32>(clamp(uv * dims, vec2<f32>(0.0), dims - 1.0));
    return textureLoad(depth_texture, texel, 0);
}

// Full-screen triangle
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;

    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);

    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);

    return out;
}

// Reconstruct view-space position from UV and depth using inverse projection
fn get_view_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    // Convert UV to NDC (-1 to 1)
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);

    // Unproject to view space
    var view_pos = params.inv_proj * ndc;
    view_pos = view_pos / view_pos.w;

    return view_pos.xyz;
}

// Read world-space normal from G-buffer and transform to view-space
fn get_view_normal(uv: vec2<f32>, texel_size: vec2<f32>) -> vec3<f32> {
    let world_normal = textureSample(normal_texture, tex_sampler, uv).xyz;
    // Transform world-space normal to view-space (w=0 for direction vector)
    return normalize((params.view * vec4<f32>(world_normal, 0.0)).xyz);
}

// Project view-space position to screen UV
fn project_to_uv(view_pos: vec3<f32>) -> vec2<f32> {
    var clip = params.proj * vec4<f32>(view_pos, 1.0);
    clip = clip / clip.w;
    return clip.xy * 0.5 + 0.5;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) f32 {
    let texel_size = 1.0 / params.screen_size;
    let noise_scale = params.screen_size / 4.0;

    // Sample depth
    let depth = load_depth(in.uv);

    // Skip background (depth buffer cleared to 1.0)
    if (depth > 0.9999) {
        return 1.0;
    }

    // Reconstruct view-space position and normal
    let frag_pos = get_view_pos(in.uv, depth);
    let normal = get_view_normal(in.uv, texel_size);

    // Get noise for random rotation (tiled across screen)
    let noise_uv = in.uv * noise_scale;
    let random_vec = normalize(textureSample(noise_texture, noise_sampler, noise_uv).xyz * 2.0 - 1.0);

    // Create TBN matrix to orient hemisphere along normal
    let tangent = normalize(random_vec - normal * dot(random_vec, normal));
    let bitangent = cross(normal, tangent);
    let tbn = mat3x3<f32>(tangent, bitangent, normal);

    // Sample hemisphere and accumulate occlusion
    var occlusion = 0.0;

    for (var i = 0; i < 32; i++) {
        // Transform sample from tangent space to view space
        let sample_offset = tbn * kernel.samples[i].xyz;
        let sample_pos = frag_pos + sample_offset * RADIUS;

        // Project sample to screen space
        let sample_uv = project_to_uv(sample_pos);

        // Skip samples outside screen
        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            continue;
        }

        // Get depth at sample position
        let sample_depth = load_depth(sample_uv);
        let sample_view_pos = get_view_pos(sample_uv, sample_depth);

        // Range check - only count occlusion from nearby geometry
        let range_check = smoothstep(0.0, 1.0, RADIUS / abs(frag_pos.z - sample_view_pos.z));

        // Occlusion test: is the actual surface closer than our sample point?
        // In view space, more negative Z is further from camera
        if (sample_view_pos.z > sample_pos.z + BIAS) {
            occlusion += range_check;
        }
    }

    occlusion = 1.0 - (occlusion / 32.0);

    return pow(occlusion, 2.0);
}
