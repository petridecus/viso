//! View frustum for culling
//!
//! Extracts frustum planes from the view-projection matrix and provides
//! intersection tests for points and spheres.

use glam::{Mat4, Vec3, Vec4};

/// A plane in 3D space, represented as (normal.x, normal.y, normal.z, distance)
/// where the plane equation is: ax + by + cz + d = 0
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    /// Unit normal pointing into the positive half-space.
    pub normal: Vec3,
    /// Signed distance from origin (`n · p + d = 0`).
    pub distance: f32,
}

impl Plane {
    /// Create a plane from coefficients and normalize it
    pub fn from_coefficients(a: f32, b: f32, c: f32, d: f32) -> Self {
        let len = (a * a + b * b + c * c).sqrt();
        if len > 0.0 {
            Self {
                normal: Vec3::new(a / len, b / len, c / len),
                distance: d / len,
            }
        } else {
            Self {
                normal: Vec3::ZERO,
                distance: 0.0,
            }
        }
    }

    /// Signed distance from point to plane (positive = in front, negative =
    /// behind)
    #[inline]
    pub fn distance_to_point(&self, point: Vec3) -> f32 {
        self.normal.dot(point) + self.distance
    }
}

/// View frustum consisting of 6 planes
#[derive(Debug, Clone)]
pub struct Frustum {
    /// Six clipping planes: left, right, bottom, top, near, far.
    pub planes: [Plane; 6],
}

impl Frustum {
    /// Extract frustum planes from a view-projection matrix.
    /// Uses the Gribb/Hartmann method for plane extraction.
    /// Planes point inward (positive half-space is inside the frustum).
    pub fn from_view_projection(vp: Mat4) -> Self {
        // Get matrix rows (glam stores column-major, so we transpose
        // conceptually)
        let row0 =
            Vec4::new(vp.x_axis.x, vp.y_axis.x, vp.z_axis.x, vp.w_axis.x);
        let row1 =
            Vec4::new(vp.x_axis.y, vp.y_axis.y, vp.z_axis.y, vp.w_axis.y);
        let row2 =
            Vec4::new(vp.x_axis.z, vp.y_axis.z, vp.z_axis.z, vp.w_axis.z);
        let row3 =
            Vec4::new(vp.x_axis.w, vp.y_axis.w, vp.z_axis.w, vp.w_axis.w);

        // Extract planes (Gribb/Hartmann method)
        // For right-handed system with [0,1] depth range (wgpu/Vulkan)
        let left = row3 + row0;
        let right = row3 - row0;
        let bottom = row3 + row1;
        let top = row3 - row1;
        let near = row2; // [0,1] depth: near plane is just row2
        let far = row3 - row2;

        Self {
            planes: [
                Plane::from_coefficients(left.x, left.y, left.z, left.w),
                Plane::from_coefficients(right.x, right.y, right.z, right.w),
                Plane::from_coefficients(
                    bottom.x, bottom.y, bottom.z, bottom.w,
                ),
                Plane::from_coefficients(top.x, top.y, top.z, top.w),
                Plane::from_coefficients(near.x, near.y, near.z, near.w),
                Plane::from_coefficients(far.x, far.y, far.z, far.w),
            ],
        }
    }

    /// Test if a point is inside the frustum
    #[inline]
    pub fn contains_point(&self, point: Vec3) -> bool {
        for plane in &self.planes {
            if plane.distance_to_point(point) < 0.0 {
                return false;
            }
        }
        true
    }

    /// Test if a sphere intersects or is inside the frustum
    #[inline]
    pub fn intersects_sphere(&self, center: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            if plane.distance_to_point(center) < -radius {
                return false;
            }
        }
        true
    }

    /// Test if a sphere is completely inside the frustum (not just
    /// intersecting)
    #[inline]
    pub fn contains_sphere(&self, center: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            if plane.distance_to_point(center) < radius {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use glam::Mat4;

    use super::*;

    #[test]
    fn test_frustum_contains_origin() {
        // Simple orthographic-like frustum centered at origin
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 1.0, 0.1, 100.0);
        let view =
            Mat4::look_at_rh(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, Vec3::Y);
        let vp = proj * view;
        let frustum = Frustum::from_view_projection(vp);

        // Origin should be inside the frustum
        assert!(frustum.contains_point(Vec3::ZERO));

        // Point far behind camera should be outside
        assert!(!frustum.contains_point(Vec3::new(0.0, 0.0, 20.0)));
    }

    #[test]
    fn test_sphere_intersection() {
        let proj = Mat4::perspective_rh(45.0_f32.to_radians(), 1.0, 0.1, 100.0);
        let view =
            Mat4::look_at_rh(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, Vec3::Y);
        let vp = proj * view;
        let frustum = Frustum::from_view_projection(vp);

        // Sphere at origin should intersect
        assert!(frustum.intersects_sphere(Vec3::ZERO, 1.0));

        // Large sphere behind camera that doesn't reach frustum
        assert!(!frustum.intersects_sphere(Vec3::new(0.0, 0.0, 50.0), 1.0));
    }
}
