//! Unified per-entity visual appearance settings.
//!
//! [`EntityAppearance`] captures all visual settings that can be set globally
//! or per-entity. At the global level all fields have concrete values; at the
//! per-entity level `None` means "inherit from global."

use serde::{Deserialize, Serialize};

use super::display::{
    BondStyle, ColorScheme, DrawingMode, HelixStyle, LipidMode, NaColorMode,
    SheetStyle, SidechainColorMode, SurfaceKindOption,
};
use super::geometry::GeometryOptions;
use super::palette::{PaletteMode, PalettePreset};
use super::DisplayOptions;

/// Visual appearance settings that can be set globally or per-entity.
///
/// At the global level, all fields have concrete values. At the per-entity
/// level, `None` means "inherit from global."
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct EntityAppearance {
    /// Top-level drawing mode (Cartoon / Stick / BallAndStick).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drawing_mode: Option<DrawingMode>,
    /// What property drives backbone coloring.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color_scheme: Option<ColorScheme>,
    /// Whether to render amino acid sidechains.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_sidechains: Option<bool>,
    /// Molecular surface type (None / Gaussian / SES).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub surface_kind: Option<SurfaceKindOption>,
    /// Surface opacity (alpha channel, 0.0–1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub surface_opacity: Option<f32>,
    /// Whether to render internal cavity meshes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_cavities: Option<bool>,
    /// Helix rendering style within Cartoon mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub helix_style: Option<HelixStyle>,
    /// Sheet rendering style within Cartoon mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sheet_style: Option<SheetStyle>,
    /// Sidechain coloring strategy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sidechain_color_mode: Option<SidechainColorMode>,
    /// Nucleic acid coloring strategy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub na_color_mode: Option<NaColorMode>,
    /// Lipid rendering style.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lipid_mode: Option<LipidMode>,
    /// Whether to render hydrogen atoms.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_hydrogens: Option<bool>,
    /// Named color palette preset for backbone coloring.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub palette_preset: Option<PalettePreset>,
    /// How backbone palette colors are distributed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub palette_mode: Option<PaletteMode>,

    // --- Structural bond overrides ---
    /// Whether to show hydrogen bonds for this entity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_hbonds: Option<bool>,
    /// Visual style for hydrogen bonds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hbond_style: Option<BondStyle>,
    /// Whether to show disulfide bonds for this entity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_disulfides: Option<bool>,
    /// Visual style for disulfide bonds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disulfide_style: Option<BondStyle>,
}

impl EntityAppearance {
    /// Resolve this appearance against another set of overrides.
    ///
    /// Per-entity `Some` values win; `None` inherits from `global`.
    #[must_use]
    pub fn resolve(&self, global: &Self) -> Self {
        Self {
            drawing_mode: self.drawing_mode.or(global.drawing_mode),
            color_scheme: self
                .color_scheme
                .clone()
                .or_else(|| global.color_scheme.clone()),
            show_sidechains: self.show_sidechains.or(global.show_sidechains),
            surface_kind: self.surface_kind.or(global.surface_kind),
            surface_opacity: self.surface_opacity.or(global.surface_opacity),
            show_cavities: self.show_cavities.or(global.show_cavities),
            helix_style: self.helix_style.or(global.helix_style),
            sheet_style: self.sheet_style.or(global.sheet_style),
            sidechain_color_mode: self
                .sidechain_color_mode
                .clone()
                .or_else(|| global.sidechain_color_mode.clone()),
            na_color_mode: self
                .na_color_mode
                .clone()
                .or_else(|| global.na_color_mode.clone()),
            lipid_mode: self
                .lipid_mode
                .clone()
                .or_else(|| global.lipid_mode.clone()),
            show_hydrogens: self.show_hydrogens.or(global.show_hydrogens),
            palette_preset: self
                .palette_preset
                .clone()
                .or_else(|| global.palette_preset.clone()),
            palette_mode: self
                .palette_mode
                .clone()
                .or_else(|| global.palette_mode.clone()),
            show_hbonds: self.show_hbonds.or(global.show_hbonds),
            hbond_style: self.hbond_style.or(global.hbond_style),
            show_disulfides: self.show_disulfides.or(global.show_disulfides),
            disulfide_style: self.disulfide_style.or(global.disulfide_style),
        }
    }

    /// Whether all fields are `None` (no overrides).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.drawing_mode.is_none()
            && self.color_scheme.is_none()
            && self.show_sidechains.is_none()
            && self.surface_kind.is_none()
            && self.surface_opacity.is_none()
            && self.show_cavities.is_none()
            && self.helix_style.is_none()
            && self.sheet_style.is_none()
            && self.sidechain_color_mode.is_none()
            && self.na_color_mode.is_none()
            && self.lipid_mode.is_none()
            && self.show_hydrogens.is_none()
            && self.palette_preset.is_none()
            && self.palette_mode.is_none()
            && self.show_hbonds.is_none()
            && self.hbond_style.is_none()
            && self.show_disulfides.is_none()
            && self.disulfide_style.is_none()
    }

    /// Produce a [`DisplayOptions`] with per-entity overrides applied.
    ///
    /// `Some` fields override the corresponding `base` value; `None`
    /// fields pass through unchanged.
    #[must_use]
    pub fn to_display_options(&self, base: &DisplayOptions) -> DisplayOptions {
        let mut out = base.clone();
        if let Some(v) = self.drawing_mode {
            out.drawing_mode = v;
        }
        if let Some(ref v) = self.color_scheme {
            out.backbone_color_scheme = v.clone();
        }
        if let Some(v) = self.show_sidechains {
            out.show_sidechains = v;
        }
        if let Some(v) = self.surface_kind {
            out.surface_kind = v;
        }
        if let Some(v) = self.surface_opacity {
            out.surface_opacity = v;
        }
        if let Some(v) = self.show_cavities {
            out.show_cavities = v;
        }
        if let Some(v) = self.helix_style {
            out.helix_style = v;
        }
        if let Some(v) = self.sheet_style {
            out.sheet_style = v;
        }
        if let Some(ref v) = self.sidechain_color_mode {
            out.sidechain_color_mode = v.clone();
        }
        if let Some(ref v) = self.na_color_mode {
            out.na_color_mode = v.clone();
        }
        if let Some(ref v) = self.lipid_mode {
            out.lipid_mode = v.clone();
        }
        if let Some(v) = self.show_hydrogens {
            out.show_hydrogens = v;
        }
        if let Some(ref v) = self.palette_preset {
            out.backbone_palette_preset = v.clone();
        }
        if let Some(ref v) = self.palette_mode {
            out.backbone_palette_mode = v.clone();
        }
        if let Some(v) = self.show_hbonds {
            out.bonds.hydrogen_bonds.visible = v;
        }
        if let Some(v) = self.hbond_style {
            out.bonds.hydrogen_bonds.style = v;
        }
        if let Some(v) = self.show_disulfides {
            out.bonds.disulfide_bonds.visible = v;
        }
        if let Some(v) = self.disulfide_style {
            out.bonds.disulfide_bonds.style = v;
        }
        out
    }

    /// Produce a [`GeometryOptions`] by patching helix/sheet style onto
    /// `base`.
    #[must_use]
    pub fn to_geometry_options(
        &self,
        base: &GeometryOptions,
    ) -> GeometryOptions {
        let mut out = base.clone();
        if let Some(helix) = self.helix_style {
            out = out.with_helix_style(helix);
        }
        if let Some(sheet) = self.sheet_style {
            out = out.with_sheet_style(sheet);
        }
        out
    }

    /// Create the global defaults.
    ///
    /// All fields have concrete values matching the current session defaults.
    #[must_use]
    pub fn global_defaults() -> Self {
        Self {
            drawing_mode: Some(DrawingMode::Cartoon),
            color_scheme: Some(ColorScheme::Entity),
            show_sidechains: Some(true),
            surface_kind: Some(SurfaceKindOption::None),
            surface_opacity: Some(0.35),
            show_cavities: Some(false),
            helix_style: Some(HelixStyle::Ribbon),
            sheet_style: Some(SheetStyle::Ribbon),
            sidechain_color_mode: Some(SidechainColorMode::default()),
            na_color_mode: Some(NaColorMode::default()),
            lipid_mode: Some(LipidMode::default()),
            show_hydrogens: Some(false),
            palette_preset: Some(PalettePreset::default()),
            palette_mode: Some(PaletteMode::default()),
            show_hbonds: Some(true),
            hbond_style: Some(BondStyle::Solid),
            show_disulfides: Some(true),
            disulfide_style: Some(BondStyle::Solid),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn default_is_empty() {
        let a = EntityAppearance::default();
        assert!(a.is_empty());
    }

    #[test]
    fn resolve_inherits_from_global() {
        let global = EntityAppearance::global_defaults();
        let per_entity = EntityAppearance::default();
        let resolved = per_entity.resolve(&global);
        assert_eq!(resolved.drawing_mode, Some(DrawingMode::Cartoon));
        assert_eq!(resolved.show_sidechains, Some(true));
    }

    #[test]
    fn resolve_per_entity_wins() {
        let global = EntityAppearance::global_defaults();
        let per_entity = EntityAppearance {
            drawing_mode: Some(DrawingMode::BallAndStick),
            ..Default::default()
        };
        let resolved = per_entity.resolve(&global);
        assert_eq!(resolved.drawing_mode, Some(DrawingMode::BallAndStick));
        // Other fields still inherited
        assert_eq!(resolved.color_scheme, Some(ColorScheme::Entity));
    }

    #[test]
    fn round_trip_serde() {
        let a = EntityAppearance::global_defaults();
        let json = serde_json::to_string(&a).unwrap();
        let b: EntityAppearance = serde_json::from_str(&json).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn to_display_options_patches_all_fields() {
        let base = DisplayOptions::default();
        let ovr = EntityAppearance {
            drawing_mode: Some(DrawingMode::Stick),
            color_scheme: Some(ColorScheme::BFactor),
            show_sidechains: Some(false),
            ..Default::default()
        };
        let result = ovr.to_display_options(&base);
        assert_eq!(result.drawing_mode, DrawingMode::Stick);
        assert_eq!(result.backbone_color_scheme, ColorScheme::BFactor);
        assert!(!result.show_sidechains);
        // Unset fields pass through from base
        assert_eq!(result.helix_style, base.helix_style);
    }
}
