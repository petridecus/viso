use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
#[schemars(title = "Effects", inline)]
#[serde(default)]
pub struct PostProcessingOptions {
    #[schemars(title = "Outline Thickness", range(min = 0.0, max = 2.0), extend("step" = 0.1))]
    pub outline_thickness: f32,
    #[schemars(title = "Outline Strength", range(min = 0.0, max = 1.0), extend("step" = 0.01))]
    pub outline_strength: f32,
    #[schemars(title = "AO Strength", range(min = 0.0, max = 1.5), extend("step" = 0.05))]
    pub ao_strength: f32,
    #[schemars(title = "AO Radius", range(min = 0.1, max = 2.0), extend("step" = 0.05))]
    pub ao_radius: f32,
    #[schemars(skip)]
    pub ao_bias: f32,
    #[schemars(skip)]
    pub ao_power: f32,
    #[schemars(title = "Fog Start", range(min = 10.0, max = 500.0), extend("step" = 5.0))]
    pub fog_start: f32,
    #[schemars(title = "Fog Density", range(min = 0.0, max = 0.02), extend("step" = 0.001))]
    pub fog_density: f32,
    #[schemars(title = "Exposure", range(min = 0.5, max = 2.0), extend("step" = 0.05))]
    pub exposure: f32,
    #[schemars(title = "Normal Outline", range(min = 0.0, max = 1.0), extend("step" = 0.01))]
    pub normal_outline_strength: f32,
    #[schemars(title = "Bloom Intensity", range(min = 0.0, max = 0.5), extend("step" = 0.01))]
    pub bloom_intensity: f32,
    #[schemars(title = "Bloom Threshold", range(min = 0.5, max = 2.0), extend("step" = 0.05))]
    pub bloom_threshold: f32,
}

impl Default for PostProcessingOptions {
    fn default() -> Self {
        Self {
            outline_thickness: 1.0,
            outline_strength: 0.7,
            ao_strength: 0.85,
            ao_radius: 0.5,
            ao_bias: 0.025,
            ao_power: 2.0,
            fog_start: 100.0,
            fog_density: 0.005,
            exposure: 1.0,
            normal_outline_strength: 0.5,
            bloom_intensity: 0.0,
            bloom_threshold: 1.0,
        }
    }
}
