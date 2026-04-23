use encase::ShaderType;
use glam::{Mat4, Vec3};

/// Perspective camera defined by eye position, target, and projection
/// parameters.
pub(crate) struct Camera {
    /// Eye (camera) position in world space.
    pub(crate) eye: Vec3,
    /// Look-at target position.
    pub(crate) target: Vec3,
    /// Up direction vector.
    pub(crate) up: Vec3,
    /// Viewport aspect ratio (width / height).
    pub(crate) aspect: f32,
    /// Vertical field of view in degrees.
    pub(crate) fovy: f32,
    /// Near clipping plane distance.
    pub(crate) znear: f32,
    /// Far clipping plane distance.
    pub(crate) zfar: f32,
}

/// GPU uniform buffer holding the view-projection matrix and camera metadata.
///
/// Layout matches the WGSL `CameraUniform` struct (112 bytes, std140).
/// Padding is handled automatically by encase.
#[derive(Debug, Copy, Clone, ShaderType)]
pub(crate) struct CameraUniform {
    /// Combined view-projection matrix.
    pub(crate) view_proj: Mat4,
    /// Camera world-space position.
    pub(crate) position: Vec3,
    /// Viewport aspect ratio.
    pub(crate) aspect: f32,
    /// Camera forward direction for lighting.
    pub(crate) forward: Vec3,
    /// Vertical field of view in degrees.
    pub(crate) fovy: f32,
    /// Currently hovered residue index (-1 if none).
    pub(crate) hovered_residue: i32,
    /// Debug visualization mode (0 = off, 1 = show normals).
    pub(crate) debug_mode: u32,
    /// Wall-clock elapsed time in seconds (for shader animations).
    pub(crate) time: f32,
    /// Padding to maintain 16-byte alignment.
    pub(crate) _pad: f32,
}

impl Camera {
    /// Build the combined view-projection matrix.
    pub(crate) fn build_matrix(&self) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye, self.target, self.up);
        // perspective_rh already uses [0,1] depth range (wgpu/Vulkan
        // convention)
        let proj = Mat4::perspective_rh(
            self.fovy.to_radians(),
            self.aspect,
            self.znear,
            self.zfar,
        );
        proj * view
    }

    /// Get just the projection matrix for SSAO
    pub(crate) fn build_projection(&self) -> Mat4 {
        // perspective_rh already uses [0,1] depth range (wgpu/Vulkan
        // convention)
        Mat4::perspective_rh(
            self.fovy.to_radians(),
            self.aspect,
            self.znear,
            self.zfar,
        )
    }
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self::new()
    }
}

impl CameraUniform {
    /// Create a new camera uniform with identity view-projection.
    pub(crate) fn new() -> Self {
        Self {
            view_proj: Mat4::IDENTITY,
            position: Vec3::ZERO,
            aspect: 1.6,
            forward: Vec3::new(0.0, 0.0, -1.0),
            fovy: 45.0,
            hovered_residue: -1,
            debug_mode: 0,
            time: 0.0,
            _pad: 0.0,
        }
    }

    /// Update uniform fields from the given camera's current state.
    pub(crate) fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_matrix();
        self.position = camera.eye;
        self.aspect = camera.aspect;
        self.forward = (camera.target - camera.eye).normalize();
        self.fovy = camera.fovy;
    }
}
