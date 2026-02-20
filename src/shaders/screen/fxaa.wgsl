// FXAA 3.11 Quality — screen-space anti-aliasing post-process pass
//
// Based on Timothy Lottes' FXAA 3.11 algorithm.
// Performs luminance-based edge detection and sub-pixel anti-aliasing.

#import viso::fullscreen::{FullscreenVertexOutput, fullscreen_vertex}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(0) @binding(2) var<uniform> screen_size: vec2<f32>;

// Full-screen triangle (same pattern as composite.wgsl)
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> FullscreenVertexOutput {
    return fullscreen_vertex(vertex_index);
}

// Perceptual luminance (Rec. 709)
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.299, 0.587, 0.114));
}

// FXAA quality settings
const FXAA_EDGE_THRESHOLD: f32 = 0.0625;      // 1/16 — minimum edge contrast to process
const FXAA_EDGE_THRESHOLD_MIN: f32 = 0.0312;   // 1/32 — skip very dark edges (noise)
const FXAA_SUBPIX_QUALITY: f32 = 0.75;         // sub-pixel AA strength (0=off, 1=full)
const FXAA_SEARCH_STEPS: i32 = 12;             // edge-walk iterations per direction

@fragment
fn fs_main(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let texel_size = 1.0 / screen_size;
    let uv = in.uv;

    // Sample center and 4 direct neighbors
    let rgbM  = textureSample(input_texture, tex_sampler, uv).rgb;
    let rgbN  = textureSample(input_texture, tex_sampler, uv + vec2<f32>( 0.0, -texel_size.y)).rgb;
    let rgbS  = textureSample(input_texture, tex_sampler, uv + vec2<f32>( 0.0,  texel_size.y)).rgb;
    let rgbW  = textureSample(input_texture, tex_sampler, uv + vec2<f32>(-texel_size.x,  0.0)).rgb;
    let rgbE  = textureSample(input_texture, tex_sampler, uv + vec2<f32>( texel_size.x,  0.0)).rgb;

    let lumM = luminance(rgbM);
    let lumN = luminance(rgbN);
    let lumS = luminance(rgbS);
    let lumW = luminance(rgbW);
    let lumE = luminance(rgbE);

    let lumMin = min(lumM, min(min(lumN, lumS), min(lumW, lumE)));
    let lumMax = max(lumM, max(max(lumN, lumS), max(lumW, lumE)));
    let lumRange = lumMax - lumMin;

    // Early exit: no edge detected
    if lumRange < max(FXAA_EDGE_THRESHOLD_MIN, lumMax * FXAA_EDGE_THRESHOLD) {
        return vec4<f32>(rgbM, 1.0);
    }

    // Sample corners for sub-pixel aliasing detection
    let rgbNW = textureSample(input_texture, tex_sampler, uv + vec2<f32>(-texel_size.x, -texel_size.y)).rgb;
    let rgbNE = textureSample(input_texture, tex_sampler, uv + vec2<f32>( texel_size.x, -texel_size.y)).rgb;
    let rgbSW = textureSample(input_texture, tex_sampler, uv + vec2<f32>(-texel_size.x,  texel_size.y)).rgb;
    let rgbSE = textureSample(input_texture, tex_sampler, uv + vec2<f32>( texel_size.x,  texel_size.y)).rgb;

    let lumNW = luminance(rgbNW);
    let lumNE = luminance(rgbNE);
    let lumSW = luminance(rgbSW);
    let lumSE = luminance(rgbSE);

    // Sub-pixel aliasing detection
    let lumNS = lumN + lumS;
    let lumWE = lumW + lumE;
    let lumCorners = lumNW + lumNE + lumSW + lumSE;
    let subpixA = (2.0 * lumNS + 2.0 * lumWE + lumCorners) / 12.0;
    var subpixB = abs(subpixA - lumM) / lumRange;
    subpixB = clamp(subpixB, 0.0, 1.0);
    let subpixC = (-2.0 * subpixB + 3.0) * subpixB * subpixB; // smoothstep
    let subpixBlend = subpixC * subpixC * FXAA_SUBPIX_QUALITY;

    // Determine edge direction (horizontal vs vertical)
    let edgeH = abs(-2.0 * lumN + lumNW + lumNE) +
                abs(-2.0 * lumM + lumW  + lumE ) * 2.0 +
                abs(-2.0 * lumS + lumSW + lumSE);
    let edgeV = abs(-2.0 * lumW + lumNW + lumSW) +
                abs(-2.0 * lumM + lumN  + lumS ) * 2.0 +
                abs(-2.0 * lumE + lumNE + lumSE);
    let isHorizontal = edgeH >= edgeV;

    // Choose step direction perpendicular to edge
    var stepLength: f32;
    var lumP: f32; // positive direction neighbor lum
    var lumN2: f32; // negative direction neighbor lum
    if isHorizontal {
        stepLength = texel_size.y;
        lumP = lumS;
        lumN2 = lumN;
    } else {
        stepLength = texel_size.x;
        lumP = lumE;
        lumN2 = lumW;
    }

    let gradP = abs(lumP - lumM);
    let gradN = abs(lumN2 - lumM);

    // Step toward the steeper gradient
    var pixelStep: f32;
    var lumEnd: f32;
    if gradP < gradN {
        pixelStep = -stepLength;
        lumEnd = lumN2;
    } else {
        pixelStep = stepLength;
        lumEnd = lumP;
    }

    let gradientScaled = max(gradP, gradN) * 0.25;
    let lumAvg = (lumM + lumEnd) * 0.5;

    // Walk along the edge in both directions to find endpoints
    var edgeUv = uv;
    if isHorizontal {
        edgeUv.y += pixelStep * 0.5;
    } else {
        edgeUv.x += pixelStep * 0.5;
    }

    var edgeStep: vec2<f32>;
    if isHorizontal {
        edgeStep = vec2<f32>(texel_size.x, 0.0);
    } else {
        edgeStep = vec2<f32>(0.0, texel_size.y);
    }

    // Search positive direction
    var uvP = edgeUv + edgeStep;
    var lumDeltaP = luminance(textureSample(input_texture, tex_sampler, uvP).rgb) - lumAvg;
    var doneP = abs(lumDeltaP) >= gradientScaled;

    for (var i = 1; i < FXAA_SEARCH_STEPS && !doneP; i = i + 1) {
        uvP += edgeStep;
        lumDeltaP = luminance(textureSample(input_texture, tex_sampler, uvP).rgb) - lumAvg;
        doneP = abs(lumDeltaP) >= gradientScaled;
    }

    // Search negative direction
    var uvN = edgeUv - edgeStep;
    var lumDeltaN = luminance(textureSample(input_texture, tex_sampler, uvN).rgb) - lumAvg;
    var doneN = abs(lumDeltaN) >= gradientScaled;

    for (var i = 1; i < FXAA_SEARCH_STEPS && !doneN; i = i + 1) {
        uvN -= edgeStep;
        lumDeltaN = luminance(textureSample(input_texture, tex_sampler, uvN).rgb) - lumAvg;
        doneN = abs(lumDeltaN) >= gradientScaled;
    }

    // Compute distances to edge endpoints
    var distP: f32;
    var distN: f32;
    if isHorizontal {
        distP = uvP.x - uv.x;
        distN = uv.x - uvN.x;
    } else {
        distP = uvP.y - uv.y;
        distN = uv.y - uvN.y;
    }

    let spanLength = distP + distN;
    let isCloserToP = distP < distN;
    let closerDist = min(distP, distN);
    let edgeBlend = 0.5 - closerDist / spanLength;

    // Check if edge endpoint has the right sign (prevents over-blending)
    let goodSpan = select(lumDeltaN < 0.0, lumDeltaP < 0.0, isCloserToP) != (lumM - lumAvg < 0.0);
    let finalEdgeBlend = select(0.0, edgeBlend, goodSpan);

    // Take max of sub-pixel and edge-direction blend
    let finalBlend = max(finalEdgeBlend, subpixBlend);

    // Apply blend by shifting UV perpendicular to edge
    var finalUv = uv;
    if isHorizontal {
        finalUv.y += pixelStep * finalBlend;
    } else {
        finalUv.x += pixelStep * finalBlend;
    }

    let result = textureSample(input_texture, tex_sampler, finalUv).rgb;
    return vec4<f32>(result, 1.0);
}
