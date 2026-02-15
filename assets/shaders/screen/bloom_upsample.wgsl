// Bloom upsample passthrough (bilinear sample only)

#import viso::fullscreen::{FullscreenVertexOutput, fullscreen_vertex}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> FullscreenVertexOutput {
    return fullscreen_vertex(vertex_index);
}

@fragment
fn fs_main(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    return textureSample(input_texture, tex_sampler, in.uv);
}
