#define_import_path viso::raymarch

struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
};

// Ray-AABB intersection via slab method.
// Returns (t_near, t_far). Negative t_near means miss.
fn intersect_aabb(ray: Ray, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    let inv_dir = 1.0 / ray.direction;
    let t0 = (box_min - ray.origin) * inv_dir;
    let t1 = (box_max - ray.origin) * inv_dir;
    let t_min = min(t0, t1);
    let t_max = max(t0, t1);
    let t_near = max(t_min.x, max(t_min.y, t_min.z));
    let t_far = min(t_max.x, min(t_max.y, t_max.z));
    if (t_near > t_far || t_far < 0.0) {
        return vec2<f32>(-1.0, -1.0);
    }
    return vec2<f32>(t_near, t_far);
}

// Ray-AABB entry/exit with t_near clamped to 0 when ray origin is inside the box.
fn ray_box_entry_exit(ray: Ray, box_min: vec3<f32>, box_max: vec3<f32>) -> vec2<f32> {
    let t = intersect_aabb(ray, box_min, box_max);
    if (t.x < 0.0 && t.y < 0.0) {
        return t;
    }
    return vec2<f32>(max(t.x, 0.0), t.y);
}
