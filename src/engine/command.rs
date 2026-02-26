//! The engine's complete interactive vocabulary.
//!
//! Every user-facing operation — whether triggered by a key press, mouse
//! gesture, GUI button, or programmatic call — is represented as an
//! `VisoCommand`.  Consumers construct commands and pass them to
//! [`VisoEngine::execute`](super::VisoEngine::execute).

use glam::{Vec2, Vec3};

// ── Command payload types ────────────────────────────────────────────────

/// Type of constraint band for color coding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BandType {
    /// Default band (purple).
    #[default]
    Default,
    /// Backbone-to-backbone band (yellow-orange).
    Backbone,
    /// Disulfide bridge band (yellow-green).
    Disulfide,
    /// Hydrogen bond band (cyan).
    HBond,
}

/// Information about a constraint band to be rendered.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct BandInfo {
    /// World-space position of first endpoint (attached to protein).
    pub endpoint_a: Vec3,
    /// World-space position of second endpoint.
    pub endpoint_b: Vec3,
    /// Whether the band is in pull mode (attracts).
    pub is_pull: bool,
    /// Whether the band is in push mode (repels).
    pub is_push: bool,
    /// Whether the band is disabled.
    pub is_disabled: bool,
    /// Band strength (affects radius and color intensity, default 1.0).
    pub strength: f32,
    /// Target length for the band (Angstroms, used for type detection if not
    /// specified).
    pub target_length: f32,
    /// Residue index for picking (typically the first residue).
    pub residue_idx: u32,
    /// Whether this is a "pull" to a point in space (vs between two atoms).
    /// When true, an anchor sphere is rendered at `endpoint_b`.
    pub is_space_pull: bool,
    /// Explicit band type (overrides auto-detection from `target_length`).
    pub band_type: Option<BandType>,
    /// Whether this band was created by a recipe/script (dimmer appearance).
    pub from_script: bool,
}

impl Default for BandInfo {
    fn default() -> Self {
        Self {
            endpoint_a: Vec3::ZERO,
            endpoint_b: Vec3::ZERO,
            is_pull: true,
            is_push: false,
            is_disabled: false,
            strength: 1.0,
            target_length: 0.0,
            residue_idx: 0,
            is_space_pull: false,
            band_type: None,
            from_script: false,
        }
    }
}

/// Information about the active pull constraint.
#[derive(Debug, Clone, PartialEq)]
pub struct PullInfo {
    /// Position of the atom being pulled.
    pub atom_pos: Vec3,
    /// Target position (mouse position in world space).
    pub target_pos: Vec3,
    /// Residue index for picking.
    pub residue_idx: u32,
}

// ── Commands ─────────────────────────────────────────────────────────────

/// A discrete or parameterized operation the engine can perform.
///
/// This is the single, centralized description of what the engine can do
/// interactively.  The engine never cares *how* a command was triggered —
/// keyboard, mouse, GUI, or API all look identical:
///
/// ```ignore
/// engine.execute(VisoCommand::ToggleWaters);
/// engine.execute(VisoCommand::Zoom { delta: 1.0 });
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum VisoCommand {
    // ── Camera ──────────────────────────────────────────────────────
    /// Animate the camera to fit the currently focused element.
    RecenterCamera,

    /// Toggle turntable auto-rotation around the current up axis.
    ToggleAutoRotate,

    /// Rotate the camera by `delta` pixels of mouse movement.
    RotateCamera {
        /// Horizontal and vertical drag delta.
        delta: Vec2,
    },

    /// Pan the camera by `delta` pixels of mouse movement.
    PanCamera {
        /// Horizontal and vertical drag delta.
        delta: Vec2,
    },

    /// Zoom the camera (positive = zoom in, negative = zoom out).
    Zoom {
        /// Scroll amount.
        delta: f32,
    },

    // ── Focus ───────────────────────────────────────────────────────
    /// Cycle focus: Session → Entity₁ → … → EntityN → Session.
    CycleFocus,

    /// Reset focus to session level (all entities).
    ResetFocus,

    // ── Playback ────────────────────────────────────────────────────
    /// Toggle trajectory playback (play / pause).
    ToggleTrajectory,

    // ── Selection ───────────────────────────────────────────────────
    /// Clear the current residue selection.
    ClearSelection,

    /// Select a single residue.
    SelectResidue {
        /// Flat residue index.
        index: i32,
        /// If true, add to / toggle in the existing selection (shift-click).
        extend: bool,
    },

    /// Select all residues in the same secondary-structure segment.
    SelectSegment {
        /// Any residue in the target segment.
        index: i32,
        /// If true, add to the existing selection.
        extend: bool,
    },

    /// Select all residues in the same chain.
    SelectChain {
        /// Any residue in the target chain.
        index: i32,
        /// If true, add to the existing selection.
        extend: bool,
    },

    // ── Constraint visualization ─────────────────────────────────────
    /// Replace the current set of constraint bands.
    UpdateBands {
        /// The complete list of bands to render.
        bands: Vec<BandInfo>,
    },

    /// Set or clear the active pull constraint.
    UpdatePull {
        /// The pull to render, or `None` to clear.
        pull: Option<PullInfo>,
    },
}
