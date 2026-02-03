// Picking shader - renders residue indices to a picking buffer
// Uses the same geometry as backbone_tube.wgsl but outputs residue_idx as color

struct CameraUniform {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    aspect: f32,
    forward: vec3<f32>,
    fovy: f32,
    hovered_residue: i32,
    fog_start: f32,
    fog_density: f32,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) residue_idx: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) residue_idx: u32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.residue_idx = in.residue_idx;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) u32 {
    // Output residue index + 1 (so 0 means "no hit")
    return in.residue_idx + 1u;
}
