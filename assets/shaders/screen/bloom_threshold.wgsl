// Bloom threshold extraction - extracts bright pixels from HDR color buffer

#import viso::fullscreen::{FullscreenVertexOutput, fullscreen_vertex}

@group(0) @binding(0) var color_texture: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(0) @binding(2) var<uniform> threshold: f32;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> FullscreenVertexOutput {
    return fullscreen_vertex(vertex_index);
}

@fragment
fn fs_main(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(color_texture, tex_sampler, in.uv).rgb;
    let luminance = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));

    // Soft knee: gradually ramp up contribution above threshold
    let knee = 0.1;
    let soft = luminance - threshold + knee;
    let contribution = max(soft * soft / (4.0 * knee + 0.0001), max(luminance - threshold, 0.0));
    let weight = contribution / max(luminance, 0.0001);

    return vec4<f32>(color * weight, 1.0);
}
