// Separable Gaussian blur for bloom downsample/upsample chain

struct BlurParams {
    texel_size: vec2<f32>,
    horizontal: u32,
    _pad: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(0) @binding(2) var<uniform> params: BlurParams;

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// 9-tap Gaussian weights (sigma ~= 4)
const WEIGHTS: array<f32, 5> = array<f32, 5>(
    0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216
);

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var direction: vec2<f32>;
    if (params.horizontal != 0u) {
        direction = vec2<f32>(params.texel_size.x, 0.0);
    } else {
        direction = vec2<f32>(0.0, params.texel_size.y);
    }

    var result = textureSample(input_texture, tex_sampler, in.uv).rgb * WEIGHTS[0];

    for (var i = 1; i < 5; i++) {
        let offset = direction * f32(i);
        result += textureSample(input_texture, tex_sampler, in.uv + offset).rgb * WEIGHTS[i];
        result += textureSample(input_texture, tex_sampler, in.uv - offset).rgb * WEIGHTS[i];
    }

    return vec4<f32>(result, 1.0);
}
