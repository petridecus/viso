// Shared depth utilities for post-processing shaders.

#define_import_path viso::depth

struct SsaoParams {
    inv_proj: mat4x4<f32>,
    proj: mat4x4<f32>,
    view: mat4x4<f32>,
    screen_size: vec2<f32>,
    near: f32,
    far: f32,
    radius: f32,
    bias: f32,
    power: f32,
    _pad: f32,
};

fn linearize_depth(d: f32, near: f32, far: f32) -> f32 {
    return near * far / (far - d * (far - near));
}
