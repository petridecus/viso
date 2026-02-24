//! Centralized rendering/display options with TOML preset support.
//!
//! All tweakable settings (lighting, post-processing, camera, colors, geometry,
//! keybindings, display toggles) are consolidated here. Options serialize
//! to/from TOML for view presets stored in `assets/view_presets/`.

mod camera;
mod colors;
mod debug;
mod display;
mod geometry;
mod keybindings;
mod lighting;
mod post_processing;

use std::path::Path;

pub use camera::CameraOptions;
pub use colors::ColorOptions;
pub use debug::DebugOptions;
pub use display::{
    BackboneColorMode, DisplayOptions, LipidMode, NaColorMode,
    SidechainColorMode,
};
pub use geometry::{
    lod_params, lod_scaled, select_chain_lod_tier, select_lod_tier,
    GeometryOptions,
};
pub use keybindings::KeybindingOptions;
pub use lighting::LightingOptions;
pub use post_processing::PostProcessingOptions;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::VisoError;

/// Top-level options container. All sub-structs use `#[serde(default)]` so
/// partial TOML files (e.g. only overriding `[lighting]`) work correctly.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Default, JsonSchema,
)]
#[serde(default)]
pub struct Options {
    /// Display toggles and coloring modes.
    pub display: DisplayOptions,
    /// Lighting parameters.
    pub lighting: LightingOptions,
    /// Post-processing effect parameters.
    pub post_processing: PostProcessingOptions,
    /// Camera projection and control parameters.
    pub camera: CameraOptions,
    /// Color palette options.
    #[schemars(skip)]
    pub colors: ColorOptions,
    /// Backbone and ligand geometry options.
    pub geometry: GeometryOptions,
    /// Keyboard binding options.
    #[schemars(skip)]
    pub keybindings: KeybindingOptions,
    /// Debug visualization options.
    pub debug: DebugOptions,
}

impl Options {
    /// Generate JSON Schema describing the UI-exposed options.
    #[must_use]
    pub fn json_schema() -> schemars::Schema {
        schemars::schema_for!(Options)
    }

    /// Load options from a TOML file. Missing fields use defaults.
    pub fn load(path: &Path) -> Result<Self, VisoError> {
        let content = std::fs::read_to_string(path).map_err(VisoError::Io)?;
        toml::from_str(&content)
            .map_err(|e| VisoError::OptionsParse(e.to_string()))
    }

    /// Save options to a TOML file (pretty-printed).
    pub fn save(&self, path: &Path) -> Result<(), VisoError> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| VisoError::OptionsParse(e.to_string()))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(VisoError::Io)?;
        }
        std::fs::write(path, content).map_err(VisoError::Io)
    }

    /// List available preset names (TOML file stems) in a directory.
    #[must_use]
    pub fn list_presets(dir: &Path) -> Vec<String> {
        let mut names = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "toml") {
                    if let Some(stem) =
                        path.file_stem().and_then(|s| s.to_str())
                    {
                        names.push(stem.to_owned());
                    }
                }
            }
        }
        names.sort();
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_round_trips_through_toml() {
        let opts = Options::default();
        let toml_str = toml::to_string_pretty(&opts).unwrap();
        let parsed: Options = toml::from_str(&toml_str).unwrap();
        assert_eq!(opts, parsed);
    }

    #[test]
    fn partial_toml_fills_defaults() {
        let toml_str = r"
[lighting]
shininess = 80.0
";
        let opts: Options = toml::from_str(toml_str).unwrap();
        assert_eq!(opts.lighting.shininess, 80.0);
        // Everything else should be default
        assert_eq!(opts.lighting.ambient, 0.45);
        assert_eq!(opts.display.lipid_mode, LipidMode::Coarse);
    }

    #[test]
    fn keybinding_lookup() {
        use crate::input::KeyAction;
        let opts = Options::default();
        assert_eq!(
            opts.keybindings.lookup("KeyQ"),
            Some(KeyAction::RecenterCamera)
        );
        assert_eq!(opts.keybindings.lookup("Tab"), Some(KeyAction::CycleFocus));
        assert_eq!(opts.keybindings.lookup("KeyZ"), None);
    }

    #[test]
    fn cofactor_tint_lookup() {
        let colors = ColorOptions::default();
        assert_eq!(colors.cofactor_tint("CLA"), [0.2, 0.7, 0.3]);
        assert_eq!(colors.cofactor_tint("UNKNOWN"), [0.5, 0.5, 0.5]);
    }

    #[test]
    fn schema_has_expected_properties() {
        let schema_value =
            serde_json::to_value(Options::json_schema()).unwrap();
        let props = schema_value["properties"].as_object().unwrap();

        // UI-exposed sections should be present
        assert!(props.contains_key("display"));
        assert!(props.contains_key("lighting"));
        assert!(props.contains_key("post_processing"));
        assert!(props.contains_key("camera"));

        // Skipped sections should be absent
        assert!(!props.contains_key("colors"));
        assert!(!props.contains_key("keybindings"));

        // Geometry and debug should be present (exposed in UI)
        assert!(props.contains_key("geometry"));
        assert!(props.contains_key("debug"));

        // Lighting should have exposed fields but not skipped ones
        let lighting = &props["lighting"]["properties"];
        assert!(lighting.get("light1_intensity").is_some());
        assert!(lighting.get("ambient").is_some());
        assert!(lighting.get("light1_dir").is_none());
        assert!(lighting.get("specular_intensity").is_none());
    }
}
