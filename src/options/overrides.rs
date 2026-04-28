//! Display override bag — optional visual settings applied at either
//! global or per-entity scope.
//!
//! [`DisplayOverrides`] holds each visual setting as `Option<T>`. `None`
//! means "inherit from the next level up": a per-entity override falls
//! back to the user's global overrides, which fall back to built-in
//! defaults. Same type is used at both scopes.

use serde::{Deserialize, Serialize};

use super::display::{
    BondStyle, ColorScheme, DrawingMode, HelixStyle, LipidMode, NaColorMode,
    SheetStyle, SidechainColorMode, SurfaceKindOption,
};
use super::geometry::GeometryOptions;
use super::palette::{PaletteMode, PalettePreset};
use super::DisplayOptions;

/// Bag of optional display overrides. Used at both global and per-entity
/// scope; `None` means "inherit from the next level up."
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default)]
pub struct DisplayOverrides {
    /// Top-level drawing mode (Cartoon / Stick / BallAndStick).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drawing_mode: Option<DrawingMode>,
    /// What property drives backbone coloring.
    #[serde(
        skip_serializing_if = "Option::is_none",
        alias = "backbone_color_scheme"
    )]
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
    #[serde(
        skip_serializing_if = "Option::is_none",
        alias = "backbone_palette_preset"
    )]
    pub palette_preset: Option<PalettePreset>,
    /// How backbone palette colors are distributed.
    #[serde(
        skip_serializing_if = "Option::is_none",
        alias = "backbone_palette_mode"
    )]
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

/// Classes of rendering work invalidated by an overrides diff.
///
/// A `DisplayOverrides::diff` projects each changed field onto a union
/// of these kinds. The dispatcher fires each kind at most once per
/// `set_options` / `set_entity_appearance` call — dedup is the structural
/// property of the type.
///
/// Bitflag-like u32 with const masks; no macro dependency. Each const
/// covers a single class of GPU work; combinations express multi-kind
/// invalidations (e.g. `drawing_mode` change ⇒ `DRAWING_MODE_RESOLVE |
/// RE_MESH`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RenderInvalidation(u32);

impl RenderInvalidation {
    /// No invalidation.
    pub const NONE: Self = Self(0);
    /// Bump entity mesh versions + resync the scene — the catch-all for
    /// any overridable change that reshapes geometry or bonds.
    pub const RE_MESH: Self = Self(1 << 0);
    /// Recompute backbone colors (palette / scheme change).
    pub const RE_COLOR: Self = Self(1 << 1);
    /// Regenerate molecular surfaces (kind / opacity / cavities).
    pub const RE_SURFACE: Self = Self(1 << 2);
    /// Per-chain LOD remesh (backbone style change).
    pub const LOD_REMESH: Self = Self(1 << 3);
    /// Re-resolve per-entity `drawing_mode` (global drawing_mode moved).
    pub const DRAWING_MODE_RESOLVE: Self = Self(1 << 4);

    /// True if no flags set.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// True if `self` contains all flags in `other`.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }
}

impl std::ops::BitOr for RenderInvalidation {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for RenderInvalidation {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl DisplayOverrides {
    /// Overlay `self` on `base`. `self`'s `Some` values win; `None`
    /// fields fall through to `base`.
    ///
    /// Used at both scopes: per-entity overlaid on global (entity's
    /// `Some` wins), and global overlaid on built-in defaults. Same
    /// operation, different layer.
    #[must_use]
    pub fn overlay(&self, base: &Self) -> Self {
        // Exhaustive destructuring — adding a field to DisplayOverrides
        // without updating this walk fails to compile.
        let Self {
            drawing_mode: _,
            color_scheme: _,
            show_sidechains: _,
            surface_kind: _,
            surface_opacity: _,
            show_cavities: _,
            helix_style: _,
            sheet_style: _,
            sidechain_color_mode: _,
            na_color_mode: _,
            lipid_mode: _,
            show_hydrogens: _,
            palette_preset: _,
            palette_mode: _,
            show_hbonds: _,
            hbond_style: _,
            show_disulfides: _,
            disulfide_style: _,
        } = self;
        Self {
            drawing_mode: self.drawing_mode.or(base.drawing_mode),
            color_scheme: self
                .color_scheme
                .clone()
                .or_else(|| base.color_scheme.clone()),
            show_sidechains: self.show_sidechains.or(base.show_sidechains),
            surface_kind: self.surface_kind.or(base.surface_kind),
            surface_opacity: self.surface_opacity.or(base.surface_opacity),
            show_cavities: self.show_cavities.or(base.show_cavities),
            helix_style: self.helix_style.or(base.helix_style),
            sheet_style: self.sheet_style.or(base.sheet_style),
            sidechain_color_mode: self
                .sidechain_color_mode
                .clone()
                .or_else(|| base.sidechain_color_mode.clone()),
            na_color_mode: self
                .na_color_mode
                .clone()
                .or_else(|| base.na_color_mode.clone()),
            lipid_mode: self
                .lipid_mode
                .clone()
                .or_else(|| base.lipid_mode.clone()),
            show_hydrogens: self.show_hydrogens.or(base.show_hydrogens),
            palette_preset: self
                .palette_preset
                .clone()
                .or_else(|| base.palette_preset.clone()),
            palette_mode: self
                .palette_mode
                .clone()
                .or_else(|| base.palette_mode.clone()),
            show_hbonds: self.show_hbonds.or(base.show_hbonds),
            hbond_style: self.hbond_style.or(base.hbond_style),
            show_disulfides: self.show_disulfides.or(base.show_disulfides),
            disulfide_style: self.disulfide_style.or(base.disulfide_style),
        }
    }

    /// Per-field invalidation diff.
    ///
    /// Projects each changed field onto a union of
    /// [`RenderInvalidation`] classes. Used by both the global path
    /// (`DisplayOptions.overrides` diff) and the per-entity path
    /// (`EntityAnnotations.appearance[eid]` diff), producing the same
    /// kind of invalidation set regardless of scope. Dispatchers fire
    /// each kind at most once per call.
    ///
    /// Exhaustive destructuring — adding a field to `DisplayOverrides`
    /// without updating this walk fails to compile.
    #[must_use]
    pub fn diff(&self, new: &Self) -> RenderInvalidation {
        // Destructure to force compile error on new fields.
        let Self {
            drawing_mode: _,
            color_scheme: _,
            show_sidechains: _,
            surface_kind: _,
            surface_opacity: _,
            show_cavities: _,
            helix_style: _,
            sheet_style: _,
            sidechain_color_mode: _,
            na_color_mode: _,
            lipid_mode: _,
            show_hydrogens: _,
            palette_preset: _,
            palette_mode: _,
            show_hbonds: _,
            hbond_style: _,
            show_disulfides: _,
            disulfide_style: _,
        } = self;

        let mut inv = RenderInvalidation::NONE;

        // Drawing mode: per-entity drawing_mode needs resolution +
        // a mesh rebuild (drawing mode can switch between Cartoon / Stick /
        // BallAndStick, each with entirely different meshes).
        if self.drawing_mode != new.drawing_mode {
            inv |= RenderInvalidation::DRAWING_MODE_RESOLVE
                | RenderInvalidation::RE_MESH;
        }

        // Color scheme / palette: mesh regenerates with new colors, and
        // backbone color buffer rebuilds separately.
        if self.color_scheme != new.color_scheme
            || self.palette_preset != new.palette_preset
            || self.palette_mode != new.palette_mode
        {
            inv |= RenderInvalidation::RE_COLOR | RenderInvalidation::RE_MESH;
        }

        // Sidechain coloring: mesh rebuild picks up new sidechain colors.
        if self.sidechain_color_mode != new.sidechain_color_mode {
            inv |= RenderInvalidation::RE_MESH;
        }

        // Sidechain/hydrogen visibility: geometry change, needs remesh.
        if self.show_sidechains != new.show_sidechains
            || self.show_hydrogens != new.show_hydrogens
        {
            inv |= RenderInvalidation::RE_MESH;
        }

        // Surface changes: regen surface mesh + sync.
        if self.surface_kind != new.surface_kind
            || self.surface_opacity != new.surface_opacity
            || self.show_cavities != new.show_cavities
        {
            inv |= RenderInvalidation::RE_SURFACE;
        }

        // Cartoon style: backbone geometry changes -> LOD remesh.
        if self.helix_style != new.helix_style
            || self.sheet_style != new.sheet_style
        {
            inv |= RenderInvalidation::LOD_REMESH | RenderInvalidation::RE_MESH;
        }

        // Nucleic acid coloring mode affects mesh attributes.
        if self.na_color_mode != new.na_color_mode {
            inv |= RenderInvalidation::RE_MESH;
        }

        // Lipid rendering style: different geometry (sphere vs ball-and-stick).
        if self.lipid_mode != new.lipid_mode {
            inv |= RenderInvalidation::RE_MESH;
        }

        // Bond visibility / style: bond geometry change.
        if self.show_hbonds != new.show_hbonds
            || self.hbond_style != new.hbond_style
            || self.show_disulfides != new.show_disulfides
            || self.disulfide_style != new.disulfide_style
        {
            inv |= RenderInvalidation::RE_MESH;
        }

        inv
    }

    /// Apply a single override field from a JSON value.
    ///
    /// `field` is the serde field name (matches a column in the entity
    /// override panel). `value` is parsed into the typed slot via
    /// `serde_json::from_value`. Unrecognised field names return `Err`.
    ///
    /// # Errors
    /// Returns `Err(field)` if the field name is not recognised.
    pub fn apply_json_field<'a>(
        &mut self,
        field: &'a str,
        value: &serde_json::Value,
    ) -> Result<(), &'a str> {
        match field {
            "backbone_color_scheme" | "color_scheme" => {
                self.color_scheme = serde_json::from_value(value.clone()).ok();
            }
            "show_sidechains" => {
                self.show_sidechains = value.as_bool();
            }
            "drawing_mode" => {
                self.drawing_mode = serde_json::from_value(value.clone()).ok();
            }
            "helix_style" => {
                self.helix_style = serde_json::from_value(value.clone()).ok();
            }
            "sheet_style" => {
                self.sheet_style = serde_json::from_value(value.clone()).ok();
            }
            "surface_kind" => {
                self.surface_kind = serde_json::from_value(value.clone()).ok();
            }
            "surface_opacity" => {
                self.surface_opacity = value.as_f64().map(|v| v as f32);
            }
            "show_hbonds" => {
                self.show_hbonds = value.as_bool();
            }
            "hbond_style" => {
                self.hbond_style = serde_json::from_value(value.clone()).ok();
            }
            "show_disulfides" => {
                self.show_disulfides = value.as_bool();
            }
            "disulfide_style" => {
                self.disulfide_style =
                    serde_json::from_value(value.clone()).ok();
            }
            _ => return Err(field),
        }
        Ok(())
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

    /// Produce a [`DisplayOptions`] with these overrides applied on top
    /// of `base`.
    ///
    /// Overlays the overridable bag and propagates the four bond-override
    /// fields into the global `bonds` config. `None` fields leave the
    /// corresponding `base` value untouched.
    #[must_use]
    pub fn to_display_options(&self, base: &DisplayOptions) -> DisplayOptions {
        let mut out = base.clone();
        out.overrides = self.overlay(&base.overrides);
        // Propagate bond-specific overrides into the global bonds config.
        // These four fields project onto a different struct shape than
        // the rest of the overlay (nested BondOptions, not DisplayOverrides).
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
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn sample_global() -> DisplayOverrides {
        DisplayOverrides {
            drawing_mode: Some(DrawingMode::Cartoon),
            color_scheme: Some(ColorScheme::Entity),
            show_sidechains: Some(true),
            ..Default::default()
        }
    }

    #[test]
    fn default_is_empty() {
        let a = DisplayOverrides::default();
        assert!(a.is_empty());
    }

    #[test]
    fn overlay_inherits_from_base() {
        let base = sample_global();
        let entity = DisplayOverrides::default();
        let overlaid = entity.overlay(&base);
        assert_eq!(overlaid.drawing_mode, Some(DrawingMode::Cartoon));
        assert_eq!(overlaid.show_sidechains, Some(true));
    }

    #[test]
    fn overlay_self_wins() {
        let base = sample_global();
        let entity = DisplayOverrides {
            drawing_mode: Some(DrawingMode::BallAndStick),
            ..Default::default()
        };
        let overlaid = entity.overlay(&base);
        assert_eq!(overlaid.drawing_mode, Some(DrawingMode::BallAndStick));
        // Other fields still inherited
        assert_eq!(overlaid.color_scheme, Some(ColorScheme::Entity));
    }

    #[test]
    fn overlay_is_associative() {
        let a = DisplayOverrides {
            drawing_mode: Some(DrawingMode::Stick),
            ..Default::default()
        };
        let b = DisplayOverrides {
            color_scheme: Some(ColorScheme::BFactor),
            ..Default::default()
        };
        let c = DisplayOverrides {
            show_sidechains: Some(true),
            ..Default::default()
        };
        let lhs = a.overlay(&b).overlay(&c);
        let rhs = a.overlay(&b.overlay(&c));
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn round_trip_serde() {
        let a = sample_global();
        let json = serde_json::to_string(&a).unwrap();
        let b: DisplayOverrides = serde_json::from_str(&json).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn legacy_toml_aliases_accepted() {
        // Existing TOML used `backbone_color_scheme` and
        // `backbone_palette_*` keys at the [display] level. After the
        // refactor these flatten into DisplayOverrides via aliases.
        let toml_str = r#"
backbone_color_scheme = "b_factor"
backbone_palette_preset = "viridis"
"#;
        let parsed: DisplayOverrides = toml::from_str(toml_str).unwrap();
        assert_eq!(parsed.color_scheme, Some(ColorScheme::BFactor));
        assert_eq!(parsed.palette_preset, Some(PalettePreset::Viridis),);
    }

    #[test]
    fn to_display_options_patches_all_fields() {
        let base = DisplayOptions::default();
        let ovr = DisplayOverrides {
            drawing_mode: Some(DrawingMode::Stick),
            color_scheme: Some(ColorScheme::BFactor),
            show_sidechains: Some(false),
            ..Default::default()
        };
        let result = ovr.to_display_options(&base);
        assert_eq!(result.drawing_mode(), DrawingMode::Stick);
        assert_eq!(result.backbone_color_scheme(), ColorScheme::BFactor);
        assert!(!result.show_sidechains());
        // Unset fields pass through from base
        assert_eq!(result.helix_style(), base.helix_style());
    }

    // ── RenderInvalidation tests ───────────────────────────────────────

    #[test]
    fn invalidation_none_is_empty() {
        assert!(RenderInvalidation::NONE.is_empty());
        assert!(!RenderInvalidation::RE_MESH.is_empty());
    }

    #[test]
    fn invalidation_bitor_and_contains() {
        let combined =
            RenderInvalidation::RE_MESH | RenderInvalidation::RE_COLOR;
        assert!(combined.contains(RenderInvalidation::RE_MESH));
        assert!(combined.contains(RenderInvalidation::RE_COLOR));
        assert!(!combined.contains(RenderInvalidation::RE_SURFACE));
    }

    #[test]
    fn diff_identical_returns_none() {
        let a = sample_global();
        assert_eq!(a.diff(&a), RenderInvalidation::NONE);
    }

    #[test]
    fn diff_drawing_mode_sets_resolve_and_mesh() {
        let a = DisplayOverrides::default();
        let b = DisplayOverrides {
            drawing_mode: Some(DrawingMode::Stick),
            ..Default::default()
        };
        let inv = a.diff(&b);
        assert!(inv.contains(RenderInvalidation::DRAWING_MODE_RESOLVE));
        assert!(inv.contains(RenderInvalidation::RE_MESH));
        assert!(!inv.contains(RenderInvalidation::RE_SURFACE));
    }

    #[test]
    fn diff_color_scheme_sets_color_and_mesh() {
        let a = DisplayOverrides::default();
        let b = DisplayOverrides {
            color_scheme: Some(ColorScheme::BFactor),
            ..Default::default()
        };
        let inv = a.diff(&b);
        assert!(inv.contains(RenderInvalidation::RE_COLOR));
        assert!(inv.contains(RenderInvalidation::RE_MESH));
        assert!(!inv.contains(RenderInvalidation::DRAWING_MODE_RESOLVE));
    }

    #[test]
    fn diff_surface_kind_sets_re_surface() {
        // Previously a bug: per-entity surface_kind changes never
        // triggered surface regeneration. RE_SURFACE must fire.
        let a = DisplayOverrides::default();
        let b = DisplayOverrides {
            surface_kind: Some(SurfaceKindOption::Gaussian),
            ..Default::default()
        };
        let inv = a.diff(&b);
        assert!(inv.contains(RenderInvalidation::RE_SURFACE));
        assert!(!inv.contains(RenderInvalidation::RE_MESH));
    }

    #[test]
    fn diff_helix_style_sets_lod_and_mesh() {
        let a = DisplayOverrides::default();
        let b = DisplayOverrides {
            helix_style: Some(HelixStyle::Cylinder),
            ..Default::default()
        };
        let inv = a.diff(&b);
        assert!(inv.contains(RenderInvalidation::LOD_REMESH));
        assert!(inv.contains(RenderInvalidation::RE_MESH));
    }

    #[test]
    fn diff_bond_style_sets_mesh_only() {
        let a = DisplayOverrides::default();
        let b = DisplayOverrides {
            show_hbonds: Some(true),
            ..Default::default()
        };
        let inv = a.diff(&b);
        assert!(inv.contains(RenderInvalidation::RE_MESH));
        assert!(!inv.contains(RenderInvalidation::RE_SURFACE));
        assert!(!inv.contains(RenderInvalidation::RE_COLOR));
    }

    #[test]
    fn diff_simultaneous_changes_union() {
        // Regression test for the historical triple-sync bug: multiple
        // concurrent field changes should OR into a single invalidation
        // set with each kind firing at most once (dedup is structural).
        let a = DisplayOverrides::default();
        let b = DisplayOverrides {
            drawing_mode: Some(DrawingMode::Stick),
            color_scheme: Some(ColorScheme::BFactor),
            helix_style: Some(HelixStyle::Tube),
            surface_kind: Some(SurfaceKindOption::Gaussian),
            ..Default::default()
        };
        let inv = a.diff(&b);
        assert!(inv.contains(RenderInvalidation::DRAWING_MODE_RESOLVE));
        assert!(inv.contains(RenderInvalidation::RE_MESH));
        assert!(inv.contains(RenderInvalidation::RE_COLOR));
        assert!(inv.contains(RenderInvalidation::LOD_REMESH));
        assert!(inv.contains(RenderInvalidation::RE_SURFACE));
    }
}
