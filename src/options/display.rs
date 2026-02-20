use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// How protein backbone is colored.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Default, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum BackboneColorMode {
    /// Color by absolute score.
    Score,
    /// Color by relative score.
    ScoreRelative,
    /// Color by secondary structure type.
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
    /// Color by hydrophobicity.
    #[default]
    Hydrophobicity,
}

/// How nucleic acid backbone is colored.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Default, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum NaColorMode {
    /// Single uniform color.
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
/// Display toggles and coloring mode selections.
pub struct DisplayOptions {
    /// Whether to render water molecules.
    #[schemars(title = "Show Waters")]
    pub show_waters: bool,
    /// Whether to render ion atoms.
    #[schemars(title = "Show Ions")]
    pub show_ions: bool,
    /// Whether to render solvent molecules.
    #[schemars(title = "Show Solvent")]
    pub show_solvent: bool,
    /// Lipid rendering style.
    #[schemars(title = "Lipid Mode")]
    pub lipid_mode: LipidMode,
    /// Whether to render amino acid sidechains.
    #[schemars(title = "Show Sidechains")]
    pub show_sidechains: bool,
    /// Whether to render hydrogen atoms.
    #[schemars(title = "Show Hydrogens")]
    pub show_hydrogens: bool,
    /// Backbone coloring strategy.
    #[schemars(title = "Backbone Color")]
    pub backbone_color_mode: BackboneColorMode,
    /// Sidechain coloring strategy.
    #[schemars(title = "Sidechain Color")]
    pub sidechain_color_mode: SidechainColorMode,
    /// Nucleic acid coloring strategy.
    #[schemars(title = "Nucleic Acid Color")]
    pub na_color_mode: NaColorMode,
}

impl DisplayOptions {
    /// Whether lipid mode uses full ball-and-stick representation.
    pub fn lipid_ball_and_stick(&self) -> bool {
        matches!(self.lipid_mode, LipidMode::BallAndStick)
    }
}
