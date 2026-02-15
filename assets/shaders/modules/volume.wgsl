#define_import_path viso::volume

// Map a world-space position to [0,1] texture coordinates within an axis-aligned box.
fn world_to_uvw(pos: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec3<f32> {
    return (pos - box_min) / (box_max - box_min);
}

// Sample a scalar value from a 3D volume texture.
fn sample_volume(tex: texture_3d<f32>, samp: sampler, uvw: vec3<f32>) -> f32 {
    return textureSampleLevel(tex, samp, uvw, 0.0).r;
}

// Estimate the gradient (surface normal direction) at a point in volume space
// using central differences with the given epsilon step size (in UVW space).
fn volume_gradient(tex: texture_3d<f32>, samp: sampler, uvw: vec3<f32>, epsilon: f32) -> vec3<f32> {
    let dx = sample_volume(tex, samp, uvw + vec3<f32>(epsilon, 0.0, 0.0))
           - sample_volume(tex, samp, uvw - vec3<f32>(epsilon, 0.0, 0.0));
    let dy = sample_volume(tex, samp, uvw + vec3<f32>(0.0, epsilon, 0.0))
           - sample_volume(tex, samp, uvw - vec3<f32>(0.0, epsilon, 0.0));
    let dz = sample_volume(tex, samp, uvw + vec3<f32>(0.0, 0.0, epsilon))
           - sample_volume(tex, samp, uvw - vec3<f32>(0.0, 0.0, epsilon));
    return vec3<f32>(dx, dy, dz);
}
