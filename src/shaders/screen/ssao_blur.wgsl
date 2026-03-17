// Bilateral blur for SSAO — preserves edges across depth and normal discontinuities
// Uses a 7x7 kernel with Gaussian spatial weights, depth similarity, and normal similarity

#import viso::fullscreen::{FullscreenVertexOutput, fullscreen_vertex}
#import viso::depth::{SsaoParams, linearize_depth}

// Load raw depth via textureLoad (bypasses sampler — works on Vulkan and GL/GLES)
fn load_depth(uv: vec2<f32>) -> f32 {
    let dims = vec2<f32>(textureDimensions(depth_texture, 0));
    let texel = vec2<i32>(clamp(uv * dims, vec2<f32>(0.0), dims - 1.0));
    return textureLoad(depth_texture, texel, 0);
}

@group(0) @binding(0) var ssao_texture: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(0) @binding(2) var depth_texture: texture_depth_2d;
@group(0) @binding(3) var normal_texture: texture_2d<f32>;
@group(0) @binding(4) var<uniform> params: SsaoParams;

// Full-screen triangle
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> FullscreenVertexOutput {
    return fullscreen_vertex(vertex_index);
}

// Load normal from G-buffer
fn load_normal(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(normal_texture, tex_sampler, uv, 0.0).xyz;
}

// Gaussian weight for spatial distance (pre-computed sigma = 3.0 for 7x7 kernel)
fn gaussian_weight(offset: vec2<f32>) -> f32 {
    let sigma = 3.0;
    let d2 = dot(offset, offset);
    return exp(-d2 / (2.0 * sigma * sigma));
}

@fragment
fn fs_main(in: FullscreenVertexOutput) -> @location(0) f32 {
    let texel_size = vec2<f32>(1.0) / vec2<f32>(textureDimensions(ssao_texture));

    // Center sample
    let center_ao = textureSampleLevel(ssao_texture, tex_sampler, in.uv, 0.0).r;
    let center_depth = linearize_depth(load_depth(in.uv), params.near, params.far);
    let center_normal = load_normal(in.uv);

    // Skip background
    if (load_depth(in.uv) > 0.9999) {
        return 1.0;
    }

    var total_weight = 0.0;
    var total_ao = 0.0;

    // 7x7 bilateral blur
    for (var x = -3; x <= 3; x++) {
        for (var y = -3; y <= 3; y++) {
            let offset = vec2<f32>(f32(x), f32(y));
            let sample_uv = in.uv + offset * texel_size;

            // Spatial (Gaussian) weight
            let spatial_w = gaussian_weight(offset);

            // Sample AO, depth, and normal
            let sample_ao = textureSampleLevel(ssao_texture, tex_sampler, sample_uv, 0.0).r;
            let sample_depth = linearize_depth(load_depth(sample_uv), params.near, params.far);
            let sample_normal = load_normal(sample_uv);

            // Depth similarity weight: reject samples at very different depths
            let depth_diff = abs(center_depth - sample_depth);
            let depth_w = exp(-depth_diff * depth_diff / max(center_depth * 0.02, 0.01));

            // Normal similarity weight: reject samples with very different normals
            let normal_similarity = max(dot(center_normal, sample_normal), 0.0);
            let normal_w = pow(normal_similarity, 32.0);

            let w = spatial_w * depth_w * normal_w;
            total_ao += sample_ao * w;
            total_weight += w;
        }
    }

    return total_ao / max(total_weight, 0.0001);
}
