use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// How protein backbone is colored.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Default, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum BackboneColorMode {
    Score,
    ScoreRelative,
    SecondaryStructure,
    /// Each chain gets a distinct color, interpolated blue→red.
    #[default]
    Chain,
}

/// How sidechains are colored.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Default, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum SidechainColorMode {
    #[default]
    Hydrophobicity,
}

/// How nucleic acid backbone is colored.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Default, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum NaColorMode {
    #[default]
    Uniform,
}

/// Lipid display style.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Default, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum LipidMode {
    /// Coarse-grained spheres.
    #[default]
    Coarse,
    /// Full ball-and-stick representation.
    BallAndStick,
}

#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Default, JsonSchema,
)]
#[schemars(title = "Display", inline)]
#[serde(default)]
pub struct DisplayOptions {
    #[schemars(title = "Show Waters")]
    pub show_waters: bool,
    #[schemars(title = "Show Ions")]
    pub show_ions: bool,
    #[schemars(title = "Show Solvent")]
    pub show_solvent: bool,
    #[schemars(title = "Lipid Mode")]
    pub lipid_mode: LipidMode,
    #[schemars(title = "Show Sidechains")]
    pub show_sidechains: bool,
    #[schemars(title = "Show Hydrogens")]
    pub show_hydrogens: bool,
    #[schemars(title = "Backbone Color")]
    pub backbone_color_mode: BackboneColorMode,
    #[schemars(title = "Sidechain Color")]
    pub sidechain_color_mode: SidechainColorMode,
    #[schemars(title = "Nucleic Acid Color")]
    pub na_color_mode: NaColorMode,
}

impl DisplayOptions {
    /// Whether lipid mode uses full ball-and-stick representation.
    pub fn lipid_ball_and_stick(&self) -> bool {
        matches!(self.lipid_mode, LipidMode::BallAndStick)
    }
}
