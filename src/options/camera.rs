use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
#[schemars(title = "Camera", inline)]
#[serde(default)]
/// Camera projection and control parameters.
pub struct CameraOptions {
    /// Vertical field of view in degrees.
    #[schemars(title = "Field of View", range(min = 20.0, max = 90.0), extend("step" = 1.0))]
    pub fovy: f32,
    /// Near clipping plane distance.
    #[schemars(skip)]
    pub znear: f32,
    /// Far clipping plane distance.
    #[schemars(skip)]
    pub zfar: f32,
    /// Rotation sensitivity multiplier.
    #[schemars(title = "Rotate Speed", range(min = 0.1, max = 2.0), extend("step" = 0.05))]
    pub rotate_speed: f32,
    /// Pan sensitivity multiplier.
    #[schemars(title = "Pan Speed", range(min = 0.1, max = 2.0), extend("step" = 0.05))]
    pub pan_speed: f32,
    /// Zoom sensitivity multiplier.
    #[schemars(title = "Zoom Speed", range(min = 0.01, max = 0.5), extend("step" = 0.01))]
    pub zoom_speed: f32,
}

impl Default for CameraOptions {
    fn default() -> Self {
        Self {
            fovy: 45.0,
            znear: 5.0,
            zfar: 2000.0,
            rotate_speed: 0.5,
            pan_speed: 0.5,
            zoom_speed: 0.1,
        }
    }
}
