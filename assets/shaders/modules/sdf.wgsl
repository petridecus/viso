#define_import_path viso::sdf

// Ray-capsule intersection.
// Returns vec3(t, axis_param, hit_type) where:
//   t = ray parameter (distance), negative means miss
//   axis_param = 0..1 position along capsule axis
//   hit_type = 1.0 cylinder, 2.0 cap A, 3.0 cap B
fn intersect_capsule(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    cap_a: vec3<f32>,
    cap_b: vec3<f32>,
    radius: f32
) -> vec3<f32> {
    let ba = cap_b - cap_a;
    let ba_len = length(ba);

    if (ba_len < 0.0001) {
        let oc = ray_origin - cap_a;
        let a = dot(ray_dir, ray_dir);
        let b = 2.0 * dot(oc, ray_dir);
        let c = dot(oc, oc) - radius * radius;
        let disc = b * b - 4.0 * a * c;
        if (disc < 0.0) { return vec3<f32>(-1.0, 0.0, 0.0); }
        let t = (-b - sqrt(disc)) / (2.0 * a);
        if (t > 0.001) { return vec3<f32>(t, 0.5, 2.0); }
        return vec3<f32>(-1.0, 0.0, 0.0);
    }

    let ba_dir = ba / ba_len;
    let oa = ray_origin - cap_a;

    let ray_par = dot(ray_dir, ba_dir);
    let ray_perp = ray_dir - ba_dir * ray_par;
    let oa_par = dot(oa, ba_dir);
    let oa_perp = oa - ba_dir * oa_par;

    let a = dot(ray_perp, ray_perp);
    let b = 2.0 * dot(oa_perp, ray_perp);
    let c = dot(oa_perp, oa_perp) - radius * radius;

    var best_t: f32 = 1e20;
    var best_axis: f32 = 0.0;
    var best_type: f32 = 0.0;

    let disc_cyl = b * b - 4.0 * a * c;
    if (disc_cyl >= 0.0 && a > 0.0001) {
        let sqrt_disc = sqrt(disc_cyl);
        let t1 = (-b - sqrt_disc) / (2.0 * a);
        let t2 = (-b + sqrt_disc) / (2.0 * a);

        for (var i = 0; i < 2; i++) {
            let t = select(t2, t1, i == 0);
            if (t > 0.001 && t < best_t) {
                let axis_pos = oa_par + ray_par * t;
                if (axis_pos >= 0.0 && axis_pos <= ba_len) {
                    best_t = t;
                    best_axis = axis_pos / ba_len;
                    best_type = 1.0;
                }
            }
        }
    }

    // Check cap A
    {
        let oc = ray_origin - cap_a;
        let a_sph = dot(ray_dir, ray_dir);
        let b_sph = 2.0 * dot(oc, ray_dir);
        let c_sph = dot(oc, oc) - radius * radius;
        let disc = b_sph * b_sph - 4.0 * a_sph * c_sph;
        if (disc >= 0.0) {
            let t = (-b_sph - sqrt(disc)) / (2.0 * a_sph);
            if (t > 0.001 && t < best_t) {
                let hit = ray_origin + ray_dir * t;
                let axis_pos = dot(hit - cap_a, ba_dir);
                if (axis_pos <= 0.0) {
                    best_t = t;
                    best_axis = 0.0;
                    best_type = 2.0;
                }
            }
        }
    }

    // Check cap B
    {
        let oc = ray_origin - cap_b;
        let a_sph = dot(ray_dir, ray_dir);
        let b_sph = 2.0 * dot(oc, ray_dir);
        let c_sph = dot(oc, oc) - radius * radius;
        let disc = b_sph * b_sph - 4.0 * a_sph * c_sph;
        if (disc >= 0.0) {
            let t = (-b_sph - sqrt(disc)) / (2.0 * a_sph);
            if (t > 0.001 && t < best_t) {
                let hit = ray_origin + ray_dir * t;
                let axis_pos = dot(hit - cap_a, ba_dir);
                if (axis_pos >= ba_len) {
                    best_t = t;
                    best_axis = 1.0;
                    best_type = 3.0;
                }
            }
        }
    }

    if (best_type == 0.0) {
        return vec3<f32>(-1.0, 0.0, 0.0);
    }

    return vec3<f32>(best_t, best_axis, best_type);
}

// Compute the surface normal at a point on a capsule
fn capsule_normal(
    point: vec3<f32>,
    cap_a: vec3<f32>,
    cap_b: vec3<f32>,
    hit_type: f32
) -> vec3<f32> {
    if (hit_type == 2.0) {
        return normalize(point - cap_a);
    } else if (hit_type == 3.0) {
        return normalize(point - cap_b);
    } else {
        let axis = normalize(cap_b - cap_a);
        let to_point = point - cap_a;
        let proj = axis * dot(to_point, axis);
        return normalize(to_point - proj);
    }
}
