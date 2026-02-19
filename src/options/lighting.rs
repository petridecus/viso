use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
#[schemars(title = "Lighting", inline)]
#[serde(default)]
pub struct LightingOptions {
    #[schemars(title = "Key Light", range(min = 0.0, max = 3.5), extend("step" = 0.05))]
    pub light1_intensity: f32,
    #[schemars(title = "Fill Light", range(min = 0.0, max = 2.0), extend("step" = 0.05))]
    pub light2_intensity: f32,
    #[schemars(title = "Ambient", range(min = 0.0, max = 0.7), extend("step" = 0.01))]
    pub ambient: f32,
    #[schemars(skip)]
    pub specular_intensity: f32,
    #[schemars(skip)]
    pub shininess: f32,
    #[schemars(title = "Rim Power", range(min = 0.5, max = 10.0), extend("step" = 0.1))]
    pub rim_power: f32,
    #[schemars(title = "Rim Intensity", range(min = 0.0, max = 1.0), extend("step" = 0.01))]
    pub rim_intensity: f32,
    #[schemars(skip)]
    pub rim_directionality: f32,
    #[schemars(skip)]
    pub rim_color: [f32; 3],
    #[schemars(title = "IBL Strength", range(min = 0.0, max = 1.0), extend("step" = 0.05))]
    pub ibl_strength: f32,
    #[schemars(title = "Roughness", range(min = 0.05, max = 1.0), extend("step" = 0.01))]
    pub roughness: f32,
    #[schemars(title = "Metalness", range(min = 0.0, max = 1.0), extend("step" = 0.01))]
    pub metalness: f32,
}

impl Default for LightingOptions {
    fn default() -> Self {
        Self {
            light1_intensity: 2.0,
            light2_intensity: 1.1,
            ambient: 0.45,
            specular_intensity: 0.35,
            shininess: 38.0,
            rim_power: 5.0,
            rim_intensity: 0.3,
            rim_directionality: 0.3,
            rim_color: [1.0, 0.85, 0.7],
            ibl_strength: 0.6,
            roughness: 0.35,
            metalness: 0.15,
        }
    }
}
