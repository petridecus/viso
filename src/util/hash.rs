use std::hash::{Hash, Hasher};

use glam::Vec3;

/// Hash a single [`Vec3`] by converting each component to bits.
pub fn hash_vec3(v: &Vec3, hasher: &mut impl Hasher) {
    v.x.to_bits().hash(hasher);
    v.y.to_bits().hash(hasher);
    v.z.to_bits().hash(hasher);
}

/// Hash a slice of [`Vec3`] by sampling first, middle, and last points.
///
/// Provides good change detection without hashing every element.
pub fn hash_vec3_slice_summary(slice: &[Vec3], hasher: &mut impl Hasher) {
    slice.len().hash(hasher);
    if let Some(first) = slice.first() {
        hash_vec3(first, hasher);
    }
    if slice.len() > 2 {
        hash_vec3(&slice[slice.len() / 2], hasher);
    }
    if let Some(last) = slice.last() {
        hash_vec3(last, hasher);
    }
}

/// Hash multiple [`Vec3`] slices, returning a `u64` digest.
///
/// Convenience wrapper over [`hash_vec3_slice_summary`] for the common case
/// where no additional data needs to be mixed into the hasher.
pub fn hash_vec3_slices(slices: &[Vec<Vec3>]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    slices.len().hash(&mut hasher);
    for slice in slices {
        hash_vec3_slice_summary(slice, &mut hasher);
    }
    hasher.finish()
}
