// Procedurally extruded polygon for nucleic acid base rings.
//
// Each instance encodes up to 6 coplanar vertices, a face normal, and a
// half-thickness.  The vertex shader generates:
//   vid  0..17  top face fan  (center + 6 edges = 18 verts for 6 tris)
//   vid 18..35  bottom face fan (reversed winding)
//   vid 36..71  side quads (6 quads × 2 tris × 3 verts = 36 verts)
//
// Pentagon instances set vertex_count = 5 and v5 = centroid, so the last
// triangle of each fan collapses to zero area (degenerate).
//
// draw(0..72, 0..instance_count)

#import viso::camera::CameraUniform
#import viso::lighting::{LightingUniform, PI, distribution_ggx, geometry_smith, fresnel_schlick, compute_rim}

// Selection bit-array lookup
fn is_selected(residue_idx: u32) -> bool {
    let word_idx = residue_idx / 32u;
    let bit_idx = residue_idx % 32u;
    if (word_idx >= arrayLength(&selection)) {
        return false;
    }
    return (selection[word_idx] & (1u << bit_idx)) != 0u;
}

struct PolygonInstance {
    v0: vec4<f32>,  // xyz=position, w=vertex_count
    v1: vec4<f32>,  // xyz=position, w=half_thickness
    v2: vec4<f32>,
    v3: vec4<f32>,
    v4: vec4<f32>,
    v5: vec4<f32>,
    normal: vec4<f32>,  // xyz=face normal, w=residue_idx
    color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) base_color: vec3<f32>,
    @location(3) @interpolate(flat) residue_idx: u32,
};

struct FragOut {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> polygons: array<PolygonInstance>;
@group(1) @binding(0) var<uniform> camera: CameraUniform;
@group(2) @binding(0) var<uniform> lighting: LightingUniform;
@group(2) @binding(1) var irradiance_map: texture_cube<f32>;
@group(2) @binding(2) var env_sampler: sampler;
@group(2) @binding(3) var prefiltered_map: texture_cube<f32>;
@group(2) @binding(4) var brdf_lut: texture_2d<f32>;
@group(3) @binding(0) var<storage, read> selection: array<u32>;

// Access the i-th vertex (0..5) of the polygon instance
fn get_vertex(inst: PolygonInstance, i: u32) -> vec3<f32> {
    switch (i) {
        case 0u: { return inst.v0.xyz; }
        case 1u: { return inst.v1.xyz; }
        case 2u: { return inst.v2.xyz; }
        case 3u: { return inst.v3.xyz; }
        case 4u: { return inst.v4.xyz; }
        default: { return inst.v5.xyz; }
    }
}

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32
) -> VertexOutput {
    let inst = polygons[iid];
    let n = u32(inst.v0.w);    // 5 or 6
    let half_t = inst.v1.w;
    let face_n = inst.normal.xyz;
    let residue_idx = u32(inst.normal.w);
    let offset = face_n * half_t;

    // Compute centroid
    var centroid = vec3<f32>(0.0);
    for (var i = 0u; i < n; i++) {
        centroid += get_vertex(inst, i);
    }
    centroid /= f32(n);

    var pos: vec3<f32>;
    var normal: vec3<f32>;

    if (vid < 18u) {
        // --- Top face fan: 6 triangles, 3 verts each ---
        let tri = vid / 3u;       // which triangle (0..5)
        let vert = vid % 3u;      // which vertex in triangle
        if (vert == 0u) {
            pos = centroid + offset;
        } else if (vert == 1u) {
            pos = get_vertex(inst, tri % n) + offset;
        } else {
            pos = get_vertex(inst, (tri + 1u) % n) + offset;
        }
        normal = face_n;
    } else if (vid < 36u) {
        // --- Bottom face fan (reversed winding): 6 triangles ---
        let local = vid - 18u;
        let tri = local / 3u;
        let vert = local % 3u;
        if (vert == 0u) {
            pos = centroid - offset;
        } else if (vert == 1u) {
            // Reversed winding: swap edge order
            pos = get_vertex(inst, (tri + 1u) % n) - offset;
        } else {
            pos = get_vertex(inst, tri % n) - offset;
        }
        normal = -face_n;
    } else {
        // --- Side walls: 6 quads = 12 triangles = 36 verts ---
        let local = vid - 36u;
        let quad = local / 6u;    // which edge (0..5)
        let qv = local % 6u;     // vertex within quad (2 tris)

        let i0 = quad % n;
        let i1 = (quad + 1u) % n;
        let p0 = get_vertex(inst, i0);
        let p1 = get_vertex(inst, i1);

        let t0 = p0 + offset;
        let t1 = p1 + offset;
        let b0 = p0 - offset;
        let b1 = p1 - offset;

        // Outward normal for this edge
        let edge = p1 - p0;
        let side_n = normalize(cross(edge, face_n));

        // Two triangles: (t0, t1, b0) and (b0, t1, b1)
        switch (qv) {
            case 0u: { pos = t0; }
            case 1u: { pos = t1; }
            case 2u: { pos = b0; }
            case 3u: { pos = b0; }
            case 4u: { pos = t1; }
            default: { pos = b1; }
        }
        normal = side_n;
    }

    // Expand selected residues outward
    if (is_selected(residue_idx)) {
        pos = pos + normal * 0.15;
    }

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(pos, 1.0);
    out.world_pos = pos;
    out.world_normal = normal;
    out.base_color = inst.color.xyz;
    out.residue_idx = residue_idx;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragOut {
    let normal = normalize(in.world_normal);
    let view_dir = normalize(camera.position - in.world_pos);

    var base_color = in.base_color;

    // Hover highlight
    if (camera.hovered_residue >= 0 && u32(camera.hovered_residue) == in.residue_idx) {
        base_color = base_color + vec3<f32>(0.3, 0.3, 0.3);
    }

    // Selection highlight
    var outline_factor = 0.0;
    if (is_selected(in.residue_idx)) {
        base_color = base_color * 0.5 + vec3<f32>(0.0, 0.0, 1.0);
        outline_factor = 1.0;
    }

    let NdotV = max(dot(normal, view_dir), 0.0);

    // PBR: Fresnel reflectance at normal incidence
    let F0 = mix(vec3<f32>(0.04), base_color, lighting.metalness);
    let roughness = lighting.roughness;

    // IBL diffuse
    let irradiance = textureSample(irradiance_map, env_sampler, normal).rgb;
    let ibl_diffuse = irradiance * lighting.ibl_strength;
    let ambient_light = mix(vec3(lighting.ambient), ibl_diffuse, lighting.ibl_strength);

    // Rim lighting
    let rim = compute_rim(normal, view_dir, lighting.rim_power, lighting.rim_intensity, lighting.rim_directionality, lighting.rim_color, lighting.rim_dir);

    var Lo = vec3<f32>(0.0);

    // Key light
    {
        let L = lighting.light1_dir;
        let H = normalize(L + view_dir);
        let NdotL = max(dot(normal, L), 0.0);
        let NdotH = max(dot(normal, H), 0.0);
        let HdotV = max(dot(H, view_dir), 0.0);

        let D = distribution_ggx(NdotH, roughness);
        let G = geometry_smith(NdotV, NdotL, roughness);
        let F = fresnel_schlick(HdotV, F0);

        let numerator = D * G * F;
        let denominator = 4.0 * NdotV * NdotL + 0.0001;
        let specular = numerator / denominator;

        let kD = (vec3<f32>(1.0) - F) * (1.0 - lighting.metalness);
        Lo += (kD * base_color / PI + specular) * lighting.light1_intensity * NdotL;
    }

    // Fill light
    {
        let L = lighting.light2_dir;
        let H = normalize(L + view_dir);
        let NdotL = max(dot(normal, L), 0.0);
        let NdotH = max(dot(normal, H), 0.0);
        let HdotV = max(dot(H, view_dir), 0.0);

        let D = distribution_ggx(NdotH, roughness);
        let G = geometry_smith(NdotV, NdotL, roughness);
        let F = fresnel_schlick(HdotV, F0);

        let numerator = D * G * F;
        let denominator = 4.0 * NdotV * NdotL + 0.0001;
        let specular = numerator / denominator;

        let kD = (vec3<f32>(1.0) - F) * (1.0 - lighting.metalness);
        Lo += (kD * base_color / PI + specular) * lighting.light2_intensity * NdotL;
    }

    // Specular IBL
    let R = reflect(-view_dir, normal);
    let max_mip = 5.0;
    let prefiltered_color = textureSampleLevel(prefiltered_map, env_sampler, R, roughness * max_mip).rgb;
    let brdf_sample = textureSample(brdf_lut, env_sampler, vec2<f32>(NdotV, roughness)).rg;
    let specular_ibl = prefiltered_color * (F0 * brdf_sample.x + brdf_sample.y) * lighting.ibl_strength;

    let ambient_contribution = base_color * ambient_light + specular_ibl;
    let direct_contribution = Lo + rim;
    let lit_color = ambient_contribution + direct_contribution;

    var final_color = lit_color;
    if (outline_factor > 0.0) {
        let edge = pow(1.0 - max(dot(normal, view_dir), 0.0), 2.0);
        final_color = mix(final_color, vec3<f32>(0.0, 0.0, 0.0), edge * 0.6);
    }

    let total_lum = max(dot(final_color, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0001);
    let ambient_lum = dot(ambient_contribution, vec3<f32>(0.2126, 0.7152, 0.0722));
    let ambient_ratio = clamp(ambient_lum / total_lum, 0.0, 1.0);

    var out: FragOut;
    if (camera.debug_mode == 1u) {
        out.color = vec4<f32>(normal * 0.5 + 0.5, 1.0);
    } else {
        out.color = vec4<f32>(final_color, 1.0);
    }
    out.normal = vec4<f32>(normal, ambient_ratio);
    return out;
}
