use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::geometry::{CartoonStyle, GeometryOptions};
use super::palette::{Palette, PaletteMode, PalettePreset};

/// How protein backbone is colored.
///
/// Legacy enum retained for backward compatibility. New code should prefer
/// [`ColorScheme`].
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default, JsonSchema,
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

/// What property drives coloring.
///
/// The scheme determines *what data* maps to color. The palette (a separate
/// field) determines *which colors* are used. Any scheme can be combined with
/// any palette.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum ColorScheme {
    /// Each chain gets a distinct color from the palette.
    #[default]
    Chain,
    /// Each entity gets a distinct color from the palette.
    Entity,
    /// Color by molecule type (protein, RNA, DNA, ligand, etc.).
    EntityType,
    /// Color by secondary structure type (helix, sheet, coil).
    SecondaryStructure,
    /// N-to-C gradient per chain using the palette.
    ResidueIndex,
    /// Gradient by crystallographic B-factor.
    BFactor,
    /// Gradient by Kyte-Doolittle hydrophobicity scale.
    Hydrophobicity,
    /// Absolute Rosetta energy score.
    Score,
    /// Relative score (5th/95th percentile normalized).
    ScoreRelative,
    /// Single uniform color (uses first color from palette stops).
    Solid,
}

impl From<&BackboneColorMode> for ColorScheme {
    fn from(mode: &BackboneColorMode) -> Self {
        match mode {
            BackboneColorMode::Chain => Self::Chain,
            BackboneColorMode::Score => Self::Score,
            BackboneColorMode::ScoreRelative => Self::ScoreRelative,
            BackboneColorMode::SecondaryStructure => Self::SecondaryStructure,
        }
    }
}

/// How sidechains are colored.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum SidechainColorMode {
    /// Color by hydrophobicity.
    Hydrophobicity,
    /// Match the backbone color of the corresponding residue.
    #[default]
    Backbone,
}

/// How nucleic acid backbone is colored.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum NaColorMode {
    /// Single uniform color.
    Uniform,
    /// Color each residue's backbone segment to match its base (A/T/G/C/U).
    #[default]
    BaseColor,
}

/// Lipid display style.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum LipidMode {
    /// Coarse-grained spheres.
    #[default]
    Coarse,
    /// Full ball-and-stick representation.
    BallAndStick,
}

/// Surface presentation mode.
///
/// Not all modes are supported on every platform. If the requested mode is
/// unavailable, the engine falls back to [`PresentMode::Fifo`] (always
/// supported).
#[derive(
    Debug,
    Clone,
    Copy,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    Default,
    JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum PresentMode {
    /// VSync — capped to display refresh rate, no tearing.
    #[default]
    Fifo,
    /// Immediate — lowest latency, may tear.
    Immediate,
    /// Mailbox — low-latency VSync (triple-buffered).
    Mailbox,
}

impl PresentMode {
    /// Convert to the corresponding wgpu present mode.
    #[must_use]
    pub fn to_wgpu(self) -> wgpu::PresentMode {
        match self {
            Self::Fifo => wgpu::PresentMode::Fifo,
            Self::Immediate => wgpu::PresentMode::Immediate,
            Self::Mailbox => wgpu::PresentMode::Mailbox,
        }
    }
}

#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default, JsonSchema,
)]
#[schemars(title = "Display", inline)]
#[serde(default)]
#[allow(clippy::struct_excessive_bools)]
/// Display toggles and coloring mode selections.
pub struct DisplayOptions {
    /// Whether to render water molecules (controlled via Scene entity
    /// visibility, not the Options panel).
    #[schemars(skip)]
    pub show_waters: bool,
    /// Whether to render ion atoms (controlled via Scene entity
    /// visibility, not the Options panel).
    #[schemars(skip)]
    pub show_ions: bool,
    /// Whether to render solvent molecules (controlled via Scene entity
    /// visibility, not the Options panel).
    #[schemars(skip)]
    pub show_solvent: bool,
    /// Lipid rendering style.
    #[schemars(title = "Lipid Mode", extend("x-group" = "Visibility"))]
    pub lipid_mode: LipidMode,
    /// Whether to render amino acid sidechains.
    #[schemars(title = "Show Sidechains", extend("x-group" = "Visibility"))]
    pub show_sidechains: bool,
    /// Whether to render hydrogen atoms.
    #[schemars(title = "Show Hydrogens", extend("x-group" = "Visibility"))]
    pub show_hydrogens: bool,
    /// Backbone coloring strategy (legacy field — prefer
    /// `backbone_color_scheme`).
    #[schemars(skip)]
    pub backbone_color_mode: BackboneColorMode,
    /// What property backbone color maps to.
    #[schemars(title = "Backbone Color", extend("x-group" = "Coloring"))]
    pub backbone_color_scheme: ColorScheme,
    /// Named color palette preset for backbone coloring.
    #[schemars(title = "Backbone Palette", extend("x-group" = "Coloring"))]
    pub backbone_palette_preset: PalettePreset,
    /// How backbone palette colors are distributed.
    #[schemars(title = "Palette Mode", extend("x-group" = "Coloring"))]
    pub backbone_palette_mode: PaletteMode,
    /// Sidechain coloring strategy.
    #[schemars(title = "Sidechain Color", extend("x-group" = "Coloring"))]
    pub sidechain_color_mode: SidechainColorMode,
    /// Nucleic acid coloring strategy.
    #[schemars(title = "Nucleic Acid Color", extend("x-group" = "Coloring"))]
    pub na_color_mode: NaColorMode,
    /// Surface presentation mode (VSync, immediate, mailbox).
    #[schemars(title = "Present Mode", extend("x-group" = "Presentation"))]
    pub present_mode: PresentMode,
}

/// Per-entity display overrides. `None` fields use the session default.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct EntityDisplayOverride {
    /// Override backbone color scheme.
    pub backbone_color_scheme: Option<ColorScheme>,
    /// Override backbone palette preset.
    pub backbone_palette_preset: Option<PalettePreset>,
    /// Override backbone palette distribution mode.
    pub backbone_palette_mode: Option<PaletteMode>,
    /// Override sidechain visibility.
    pub show_sidechains: Option<bool>,
    /// Override sidechain coloring strategy.
    pub sidechain_color_mode: Option<SidechainColorMode>,
    /// Override cartoon rendering style.
    pub cartoon_style: Option<CartoonStyle>,
}

impl EntityDisplayOverride {
    /// Produce a [`DisplayOptions`] by patching overridden fields onto `base`.
    #[must_use]
    pub fn resolve_display(&self, base: &DisplayOptions) -> DisplayOptions {
        let mut out = base.clone();
        if let Some(ref v) = self.backbone_color_scheme {
            out.backbone_color_scheme = v.clone();
        }
        if let Some(ref v) = self.backbone_palette_preset {
            out.backbone_palette_preset = v.clone();
        }
        if let Some(ref v) = self.backbone_palette_mode {
            out.backbone_palette_mode = v.clone();
        }
        if let Some(v) = self.show_sidechains {
            out.show_sidechains = v;
        }
        if let Some(ref v) = self.sidechain_color_mode {
            out.sidechain_color_mode = v.clone();
        }
        out
    }

    /// Produce a [`GeometryOptions`] by patching the cartoon style onto `base`.
    #[must_use]
    pub fn resolve_geometry(&self, base: &GeometryOptions) -> GeometryOptions {
        self.cartoon_style.as_ref().map_or_else(
            || base.clone(),
            |style| {
                let mut out = base.clone();
                out.cartoon_style = style.clone();
                out
            },
        )
    }

    /// Whether all fields are `None` (no overrides).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.backbone_color_scheme.is_none()
            && self.backbone_palette_preset.is_none()
            && self.backbone_palette_mode.is_none()
            && self.show_sidechains.is_none()
            && self.sidechain_color_mode.is_none()
            && self.cartoon_style.is_none()
    }
}

impl DisplayOptions {
    /// Whether lipid mode uses full ball-and-stick representation.
    #[must_use]
    pub fn lipid_ball_and_stick(&self) -> bool {
        matches!(self.lipid_mode, LipidMode::BallAndStick)
    }

    /// Construct a [`Palette`] from the flattened preset/mode fields.
    #[must_use]
    pub fn backbone_palette(&self) -> Palette {
        Palette {
            preset: self.backbone_palette_preset.clone(),
            mode: self.backbone_palette_mode.clone(),
            stops: Vec::new(),
        }
    }
}
