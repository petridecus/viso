// Compute shader for Rotation-Minimizing Frame (RMF) calculation
// Uses the double-reflection method (Wang et al. 2008)
//
// This shader computes spline points and their orientation frames
// for instanced tube rendering of protein backbones.

// Spline frame data structure - position and orientation
struct SplineFrame {
    position: vec4<f32>,   // xyz = position, w = unused
    tangent: vec4<f32>,    // xyz = tangent, w = unused
    normal: vec4<f32>,     // xyz = normal (RMF), w = unused
    binormal: vec4<f32>,   // xyz = binormal (RMF), w = unused
}

// Compute parameters
struct ComputeParams {
    num_ca: u32,           // Number of CA atoms
    segments_per_span: u32, // Axial segments between CA atoms
    total_frames: u32,      // Total number of spline frames
    _pad: u32,
}

// Input: CA positions (from backbone chains)
@group(0) @binding(0) var<storage, read> ca_positions: array<vec4<f32>>;

// Output: Computed spline frames with RMF orientation
@group(0) @binding(1) var<storage, read_write> frames: array<SplineFrame>;

// Uniform parameters
@group(0) @binding(2) var<uniform> params: ComputeParams;

// Cubic Hermite basis functions for position
fn hermite_point(p0: vec3<f32>, m0: vec3<f32>, p1: vec3<f32>, m1: vec3<f32>, t: f32) -> vec3<f32> {
    let t2 = t * t;
    let t3 = t2 * t;

    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;

    return p0 * h00 + m0 * h10 + p1 * h01 + m1 * h11;
}

// Cubic Hermite derivative for tangent
fn hermite_tangent(p0: vec3<f32>, m0: vec3<f32>, p1: vec3<f32>, m1: vec3<f32>, t: f32) -> vec3<f32> {
    let t2 = t * t;

    let dh00 = 6.0 * t2 - 6.0 * t;
    let dh10 = 3.0 * t2 - 4.0 * t + 1.0;
    let dh01 = -6.0 * t2 + 6.0 * t;
    let dh11 = 3.0 * t2 - 2.0 * t;

    return p0 * dh00 + m0 * dh10 + p1 * dh01 + m1 * dh11;
}

// Compute tangent at CA index using Catmull-Rom style
fn compute_tangent(i: u32, n: u32) -> vec3<f32> {
    if (i == 0u) {
        return ca_positions[1].xyz - ca_positions[0].xyz;
    } else if (i == n - 1u) {
        return ca_positions[n - 1u].xyz - ca_positions[n - 2u].xyz;
    } else {
        return (ca_positions[i + 1u].xyz - ca_positions[i - 1u].xyz) * 0.5;
    }
}

// Main compute kernel - runs sequentially due to RMF dependency
@compute @workgroup_size(1)
fn main() {
    let n = params.num_ca;
    let segs = params.segments_per_span;

    if (n < 2u) {
        return;
    }

    // Step 1: Generate spline points with cubic Hermite interpolation
    var frame_idx = 0u;
    for (var i = 0u; i < n - 1u; i++) {
        let p0 = ca_positions[i].xyz;
        let p1 = ca_positions[i + 1u].xyz;
        let m0 = compute_tangent(i, n);
        let m1 = compute_tangent(i + 1u, n);

        for (var j = 0u; j < segs; j++) {
            let t = f32(j) / f32(segs);
            let pos = hermite_point(p0, m0, p1, m1, t);
            let tang = normalize(hermite_tangent(p0, m0, p1, m1, t));

            frames[frame_idx].position = vec4<f32>(pos, 0.0);
            frames[frame_idx].tangent = vec4<f32>(tang, 0.0);
            frame_idx++;
        }
    }

    // Add final point
    let last_pos = ca_positions[n - 1u].xyz;
    let last_tang = normalize(compute_tangent(n - 1u, n));
    frames[frame_idx].position = vec4<f32>(last_pos, 0.0);
    frames[frame_idx].tangent = vec4<f32>(last_tang, 0.0);

    // Step 2: Initialize first frame with arbitrary orthonormal basis
    let t0 = frames[0].tangent.xyz;
    var arbitrary: vec3<f32>;
    if (abs(t0.x) < 0.9) {
        arbitrary = vec3<f32>(1.0, 0.0, 0.0);
    } else {
        arbitrary = vec3<f32>(0.0, 1.0, 0.0);
    }
    let n0 = normalize(cross(t0, arbitrary));
    let b0 = normalize(cross(t0, n0));

    frames[0].normal = vec4<f32>(n0, 0.0);
    frames[0].binormal = vec4<f32>(b0, 0.0);

    // Step 3: Propagate RMF using double-reflection method (Wang et al. 2008)
    let total_frames = frame_idx + 1u;
    for (var i = 0u; i < total_frames - 1u; i++) {
        let x_i = frames[i].position.xyz;
        let x_i1 = frames[i + 1u].position.xyz;
        let t_i = frames[i].tangent.xyz;
        let t_i1 = frames[i + 1u].tangent.xyz;
        let r_i = frames[i].normal.xyz;

        let v1 = x_i1 - x_i;
        let c1 = dot(v1, v1);

        var r_i1: vec3<f32>;
        if (c1 < 1e-10) {
            // Points are coincident, copy frame
            r_i1 = r_i;
        } else {
            // First reflection: reflect r_i and t_i across plane perpendicular to v1
            let r_i_l = r_i - (2.0 / c1) * dot(v1, r_i) * v1;
            let t_i_l = t_i - (2.0 / c1) * dot(v1, t_i) * v1;

            // Second reflection
            let v2 = t_i1 - t_i_l;
            let c2 = dot(v2, v2);

            if (c2 < 1e-10) {
                r_i1 = r_i_l;
            } else {
                r_i1 = r_i_l - (2.0 / c2) * dot(v2, r_i_l) * v2;
            }
        }

        // Ensure orthonormality
        r_i1 = normalize(r_i1 - t_i1 * dot(t_i1, r_i1));
        let s_i1 = normalize(cross(t_i1, r_i1));

        frames[i + 1u].normal = vec4<f32>(r_i1, 0.0);
        frames[i + 1u].binormal = vec4<f32>(s_i1, 0.0);
    }
}
