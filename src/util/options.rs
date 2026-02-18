//! Centralized rendering/display options with TOML preset support.
//!
//! All tweakable settings (lighting, post-processing, camera, colors, geometry,
//! keybindings, display toggles) are consolidated here. Options serialize to/from
//! TOML for view presets stored in `assets/view_presets/`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Top-level options container. All sub-structs use `#[serde(default)]` so
/// partial TOML files (e.g. only overriding `[lighting]`) work correctly.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Options {
    pub display: DisplayOptions,
    pub lighting: LightingOptions,
    pub post_processing: PostProcessingOptions,
    pub camera: CameraOptions,
    pub colors: ColorOptions,
    pub geometry: GeometryOptions,
    pub keybindings: KeybindingOptions,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            display: DisplayOptions::default(),
            lighting: LightingOptions::default(),
            post_processing: PostProcessingOptions::default(),
            camera: CameraOptions::default(),
            colors: ColorOptions::default(),
            geometry: GeometryOptions::default(),
            keybindings: KeybindingOptions::default(),
        }
    }
}

impl Options {
    /// Load options from a TOML file. Missing fields use defaults.
    pub fn load(path: &Path) -> Result<Self, String> {
        let content =
            std::fs::read_to_string(path).map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
        toml::from_str(&content).map_err(|e| format!("Failed to parse {}: {}", path.display(), e))
    }

    /// Save options to a TOML file (pretty-printed).
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let content =
            toml::to_string_pretty(self).map_err(|e| format!("Failed to serialize options: {}", e))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory {}: {}", parent.display(), e))?;
        }
        std::fs::write(path, content).map_err(|e| format!("Failed to write {}: {}", path.display(), e))
    }

    /// List available preset names (TOML file stems) in a directory.
    pub fn list_presets(dir: &Path) -> Vec<String> {
        let mut names = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|ext| ext == "toml") {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        names.push(stem.to_string());
                    }
                }
            }
        }
        names.sort();
        names
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

/// How protein backbone is colored.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum BackboneColorMode {
    #[default]
    Score,
    ScoreRelative,
    SecondaryStructure,
}

/// How sidechains are colored.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SidechainColorMode {
    #[default]
    Hydrophobicity,
}

/// How nucleic acid backbone is colored.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum NaColorMode {
    #[default]
    Uniform,
}

/// Lipid display style.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum LipidMode {
    /// Coarse-grained spheres.
    #[default]
    Coarse,
    /// Full ball-and-stick representation.
    BallAndStick,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct DisplayOptions {
    pub show_waters: bool,
    pub show_ions: bool,
    pub show_solvent: bool,
    pub lipid_mode: LipidMode,
    pub show_sidechains: bool,
    pub show_hydrogens: bool,
    /// Backbone coloring mode.
    pub backbone_color_mode: BackboneColorMode,
    /// Sidechain coloring mode.
    pub sidechain_color_mode: SidechainColorMode,
    /// Nucleic acid coloring mode.
    pub na_color_mode: NaColorMode,
}

impl Default for DisplayOptions {
    fn default() -> Self {
        Self {
            show_waters: false,
            show_ions: false,
            show_solvent: false,
            lipid_mode: LipidMode::default(),
            show_sidechains: true,
            show_hydrogens: false,
            backbone_color_mode: BackboneColorMode::default(),
            sidechain_color_mode: SidechainColorMode::default(),
            na_color_mode: NaColorMode::default(),
        }
    }
}

impl DisplayOptions {
    /// Whether lipid mode uses full ball-and-stick representation.
    pub fn lipid_ball_and_stick(&self) -> bool {
        matches!(self.lipid_mode, LipidMode::BallAndStick)
    }
}

// ---------------------------------------------------------------------------
// Lighting
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct LightingOptions {
    pub light1_dir: [f32; 3],
    pub light2_dir: [f32; 3],
    pub light1_intensity: f32,
    pub light2_intensity: f32,
    pub ambient: f32,
    pub specular_intensity: f32,
    pub shininess: f32,
    pub rim_power: f32,
    pub rim_intensity: f32,
    pub rim_directionality: f32,
    pub rim_color: [f32; 3],
    pub ibl_strength: f32,
    pub rim_dir: [f32; 3],
    pub roughness: f32,
    pub metalness: f32,
}

impl Default for LightingOptions {
    fn default() -> Self {
        Self {
            light1_dir: [-0.3, 0.9, -0.3],
            light2_dir: [0.3, 0.6, -0.4],
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
            rim_dir: [0.0, -0.7, 0.5],
            roughness: 0.35,
            metalness: 0.15,
        }
    }
}

// ---------------------------------------------------------------------------
// Post-processing
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PostProcessingOptions {
    pub outline_thickness: f32,
    pub outline_strength: f32,
    pub ao_strength: f32,
    pub ao_radius: f32,
    pub ao_bias: f32,
    pub ao_power: f32,
    pub fog_start: f32,
    pub fog_density: f32,
    pub exposure: f32,
    pub normal_outline_strength: f32,
    pub bloom_intensity: f32,
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

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CameraOptions {
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
    pub rotate_speed: f32,
    pub pan_speed: f32,
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

// ---------------------------------------------------------------------------
// Colors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ColorOptions {
    pub lipid_carbon_tint: [f32; 3],
    pub hydrophobic_sidechain: [f32; 3],
    pub hydrophilic_sidechain: [f32; 3],
    pub nucleic_acid: [f32; 3],
    pub band_default: [f32; 3],
    pub band_backbone: [f32; 3],
    pub band_disulfide: [f32; 3],
    pub band_hbond: [f32; 3],
    pub solvent_color: [f32; 3],
    pub cofactor_tints: HashMap<String, [f32; 3]>,
}

impl Default for ColorOptions {
    fn default() -> Self {
        let mut cofactor_tints = HashMap::new();
        cofactor_tints.insert("CLA".to_string(), [0.2, 0.7, 0.3]);
        cofactor_tints.insert("CHL".to_string(), [0.2, 0.6, 0.35]);
        cofactor_tints.insert("BCR".to_string(), [0.9, 0.5, 0.1]);
        cofactor_tints.insert("BCB".to_string(), [0.9, 0.5, 0.1]);
        cofactor_tints.insert("HEM".to_string(), [0.7, 0.15, 0.15]);
        cofactor_tints.insert("HEC".to_string(), [0.7, 0.15, 0.15]);
        cofactor_tints.insert("HEA".to_string(), [0.7, 0.15, 0.15]);
        cofactor_tints.insert("HEB".to_string(), [0.7, 0.15, 0.15]);
        cofactor_tints.insert("PHO".to_string(), [0.5, 0.7, 0.3]);
        cofactor_tints.insert("PL9".to_string(), [0.6, 0.5, 0.2]);
        cofactor_tints.insert("PLQ".to_string(), [0.6, 0.5, 0.2]);

        Self {
            lipid_carbon_tint: [0.76, 0.70, 0.50],
            hydrophobic_sidechain: [0.3, 0.5, 0.9],
            hydrophilic_sidechain: [0.95, 0.6, 0.2],
            nucleic_acid: [0.45, 0.55, 0.85],
            band_default: [0.5, 0.0, 0.5],
            band_backbone: [1.0, 0.75, 0.0],
            band_disulfide: [0.5, 1.0, 0.0],
            band_hbond: [0.0, 0.75, 1.0],
            solvent_color: [0.6, 0.6, 0.6],
            cofactor_tints,
        }
    }
}

impl ColorOptions {
    /// Look up cofactor carbon tint by 3-letter residue name. Falls back to neutral gray.
    pub fn cofactor_tint(&self, res_name: &str) -> [f32; 3] {
        self.cofactor_tints
            .get(res_name.trim())
            .copied()
            .unwrap_or([0.5, 0.5, 0.5])
    }
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct GeometryOptions {
    pub tube_radius: f32,
    pub tube_radial_segments: u32,
    pub solvent_radius: f32,
    pub ligand_sphere_radius: f32,
    pub ligand_bond_radius: f32,
}

impl Default for GeometryOptions {
    fn default() -> Self {
        Self {
            tube_radius: 0.3,
            tube_radial_segments: 8,
            solvent_radius: 0.15,
            ligand_sphere_radius: 0.3,
            ligand_bond_radius: 0.12,
        }
    }
}

// ---------------------------------------------------------------------------
// Keybindings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct KeybindingOptions {
    /// Maps action name → key string (e.g. "recenter_camera" → "KeyQ").
    pub bindings: HashMap<String, String>,
    /// Reverse lookup cache (key string → action name). Rebuilt on load.
    #[serde(skip)]
    key_to_action: HashMap<String, String>,
}

impl Default for KeybindingOptions {
    fn default() -> Self {
        let mut bindings = HashMap::new();
        bindings.insert("recenter_camera".to_string(), "KeyQ".to_string());
        bindings.insert("toggle_trajectory".to_string(), "KeyT".to_string());
        bindings.insert("toggle_ions".to_string(), "KeyI".to_string());
        bindings.insert("toggle_waters".to_string(), "KeyU".to_string());
        bindings.insert("toggle_solvent".to_string(), "KeyO".to_string());
        bindings.insert("toggle_lipids".to_string(), "KeyL".to_string());
        bindings.insert("cycle_focus".to_string(), "Tab".to_string());
        bindings.insert("toggle_auto_rotate".to_string(), "KeyR".to_string());
        bindings.insert("reset_focus".to_string(), "Backquote".to_string());
        bindings.insert("cancel".to_string(), "Escape".to_string());

        let mut opts = Self {
            bindings,
            key_to_action: HashMap::new(),
        };
        opts.rebuild_reverse_map();
        opts
    }
}

impl KeybindingOptions {
    /// Rebuild the reverse lookup map (key → action).
    pub fn rebuild_reverse_map(&mut self) {
        self.key_to_action.clear();
        for (action, key) in &self.bindings {
            self.key_to_action.insert(key.clone(), action.clone());
        }
    }

    /// Look up the action name for a given key string.
    pub fn lookup(&self, key: &str) -> Option<&str> {
        self.key_to_action.get(key).map(|s| s.as_str())
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
        let toml_str = r#"
[lighting]
shininess = 80.0
"#;
        let opts: Options = toml::from_str(toml_str).unwrap();
        assert_eq!(opts.lighting.shininess, 80.0);
        // Everything else should be default
        assert_eq!(opts.lighting.ambient, 0.45);
        assert_eq!(opts.display.lipid_mode, LipidMode::Coarse);
    }

    #[test]
    fn keybinding_lookup() {
        let opts = Options::default();
        assert_eq!(opts.keybindings.lookup("KeyQ"), Some("recenter_camera"));
        assert_eq!(opts.keybindings.lookup("Tab"), Some("cycle_focus"));
        assert_eq!(opts.keybindings.lookup("KeyZ"), None);
    }

    #[test]
    fn cofactor_tint_lookup() {
        let colors = ColorOptions::default();
        assert_eq!(colors.cofactor_tint("CLA"), [0.2, 0.7, 0.3]);
        assert_eq!(colors.cofactor_tint("UNKNOWN"), [0.5, 0.5, 0.5]);
    }
}
