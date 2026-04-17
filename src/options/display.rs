use molex::MoleculeType;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ── Structural bond options ──────────────────────────────────────────────

/// Visual style for structural bond rendering (H-bonds, disulfides).
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
pub enum BondStyle {
    /// Solid cylinder with hemispherical caps (Foldit default).
    #[default]
    Solid,
    /// Dashed segments along the bond axis.
    Dashed,
    /// Solid cylinder with stipple pattern (fragment discard).
    Stippled,
}

/// How structural bonds of a given type are sourced.
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
pub enum BondSource {
    /// Auto-detect from atomic geometry (distance + angle cutoffs).
    Auto,
    /// Only show bonds explicitly provided by the caller.
    #[default]
    Manual,
    /// Auto-detect, but caller-provided bonds take precedence for
    /// overlapping atom pairs.
    Both,
}

/// Display options for a single category of structural bond.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Default, JsonSchema,
)]
#[serde(default)]
pub struct BondTypeOptions {
    /// Whether to show this bond type at all.
    pub visible: bool,
    /// Visual rendering style.
    pub style: BondStyle,
    /// Base radius in Angstroms.
    pub radius: f32,
    /// How bonds are sourced (auto-detect, manual, or both).
    pub source: BondSource,
}

/// Options controlling structural bond visualization (H-bonds,
/// disulfides).
///
/// Colors are configured separately in [`super::ColorOptions`]
/// (`band_hbond`, `band_disulfide`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
#[serde(default)]
pub struct BondOptions {
    /// Hydrogen bond display settings.
    pub hydrogen_bonds: BondTypeOptions,
    /// Disulfide bond display settings.
    pub disulfide_bonds: BondTypeOptions,
}

impl Default for BondOptions {
    fn default() -> Self {
        Self {
            hydrogen_bonds: BondTypeOptions {
                visible: false,
                style: BondStyle::Solid,
                radius: 0.1,
                source: BondSource::Auto,
            },
            disulfide_bonds: BondTypeOptions {
                visible: false,
                style: BondStyle::Solid,
                radius: 0.15,
                source: BondSource::Auto,
            },
        }
    }
}

use super::palette::{Palette, PaletteMode, PalettePreset};

/// Top-level drawing mode for an entity.
///
/// Determines whether the entity is rendered as a cartoon backbone,
/// stick model, or ball-and-stick model.
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
pub enum DrawingMode {
    /// Cartoon backbone with optional sidechains (default for proteins/NA).
    #[default]
    Cartoon,
    /// Bonds as capsules (sidechain thickness), small joint spheres.
    Stick,
    /// Thin bonds with tiny joint spheres (wire-like).
    ThinStick,
    /// Full ball-and-stick with vdW-scaled atom spheres.
    BallAndStick,
}

impl DrawingMode {
    /// Return the appropriate default drawing mode for a molecule type.
    #[must_use]
    pub fn default_for(mol_type: MoleculeType) -> Self {
        match mol_type {
            MoleculeType::Protein | MoleculeType::DNA | MoleculeType::RNA => {
                Self::Cartoon
            }
            _ => Self::BallAndStick,
        }
    }
}

/// Helix rendering style within Cartoon mode.
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
pub enum HelixStyle {
    /// Flat ribbon (default).
    #[default]
    Ribbon,
    /// Round tube.
    Tube,
    /// Solid cylinder.
    Cylinder,
}

/// Sheet rendering style within Cartoon mode.
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
pub enum SheetStyle {
    /// Flat ribbon (default).
    #[default]
    Ribbon,
    /// Round tube.
    Tube,
}

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
    /// Each entity gets a distinct color from the palette.
    #[default]
    Entity,
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
            BackboneColorMode::Chain => Self::Entity,
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

/// Global surface type option for the Display panel.
///
/// When set to something other than `None`, all entities without a per-entity
/// surface override will get this surface type applied automatically.
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
pub enum SurfaceKindOption {
    /// No global surface.
    #[default]
    None,
    /// Smooth Gaussian blob surface.
    Gaussian,
    /// Solvent-excluded / Connolly surface.
    Ses,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
#[serde(default)]
#[allow(clippy::struct_excessive_bools)]
/// Display toggles and coloring mode selections.
///
/// This is the single canonical appearance struct. Per-entity overrides
/// are stored as [`super::EntityAppearance`] (thin `Option<T>` wrapper)
/// and resolved against this via
/// [`EntityAppearance::to_display_options`](super::EntityAppearance::to_display_options).
///
/// Rendered in the Scene tab's Global Appearance card, not the Options
/// tab (hence `#[schemars(skip)]` at struct level).
pub struct DisplayOptions {
    // --- Per-entity overridable fields ---
    /// Top-level drawing mode (Cartoon / Stick / BallAndStick).
    pub drawing_mode: DrawingMode,
    /// What property backbone color maps to.
    pub backbone_color_scheme: ColorScheme,
    /// Whether to render amino acid sidechains.
    pub show_sidechains: bool,
    /// Global molecular surface type.
    pub surface_kind: SurfaceKindOption,
    /// Global surface opacity (0.0–1.0).
    #[serde(default = "default_surface_opacity")]
    pub surface_opacity: f32,
    /// Whether to render internal cavity meshes extracted from the same
    /// SDF pipeline as the molecular surface. Cavities use a fixed
    /// bluish translucent color and are not configurable per-entity.
    pub show_cavities: bool,
    /// Helix rendering style within Cartoon mode.
    pub helix_style: HelixStyle,
    /// Sheet rendering style within Cartoon mode.
    pub sheet_style: SheetStyle,
    /// Sidechain coloring strategy.
    pub sidechain_color_mode: SidechainColorMode,
    /// Nucleic acid coloring strategy.
    pub na_color_mode: NaColorMode,
    /// Lipid rendering style.
    pub lipid_mode: LipidMode,
    /// Whether to render hydrogen atoms.
    pub show_hydrogens: bool,
    /// Named color palette preset for backbone coloring.
    pub backbone_palette_preset: PalettePreset,
    /// How backbone palette colors are distributed.
    pub backbone_palette_mode: PaletteMode,

    // --- Ambient visibility (synced with EntityStore, not per-entity) ---
    /// Whether to render water molecules.
    pub show_waters: bool,
    /// Whether to render ion atoms.
    pub show_ions: bool,
    /// Whether to render solvent molecules.
    pub show_solvent: bool,

    // --- Rendering (not per-entity) ---
    /// Surface presentation mode (VSync, immediate, mailbox).
    pub present_mode: PresentMode,

    // --- Structural bonds ---
    /// Structural bond visualization settings (H-bonds, disulfides).
    pub bonds: BondOptions,

    // --- Legacy ---
    /// Backbone coloring strategy (legacy field — prefer
    /// `backbone_color_scheme`).
    pub backbone_color_mode: BackboneColorMode,
}

impl Default for DisplayOptions {
    fn default() -> Self {
        Self {
            drawing_mode: DrawingMode::default(),
            backbone_color_scheme: ColorScheme::default(),
            show_sidechains: false,
            surface_kind: SurfaceKindOption::default(),
            surface_opacity: default_surface_opacity(),
            show_cavities: false,
            helix_style: HelixStyle::default(),
            sheet_style: SheetStyle::default(),
            sidechain_color_mode: SidechainColorMode::default(),
            na_color_mode: NaColorMode::default(),
            lipid_mode: LipidMode::default(),
            show_hydrogens: false,
            backbone_palette_preset: PalettePreset::default(),
            backbone_palette_mode: PaletteMode::default(),
            show_waters: false,
            show_ions: false,
            show_solvent: false,
            present_mode: PresentMode::default(),
            bonds: BondOptions::default(),
            backbone_color_mode: BackboneColorMode::default(),
        }
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

/// Default surface opacity for serde deserialization.
fn default_surface_opacity() -> f32 {
    0.35
}
