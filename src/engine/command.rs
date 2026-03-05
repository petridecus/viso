//! The engine's complete interactive vocabulary.
//!
//! Every user-facing operation — whether triggered by a key press, mouse
//! gesture, GUI button, or programmatic call — is represented as an
//! `VisoCommand`.  Consumers construct commands and pass them to
//! [`VisoEngine::execute`](super::VisoEngine::execute).

use glam::{Vec2, Vec3};

// ── Constraint payload types ────────────────────────────────────────────

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

/// Structural reference to a specific atom.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomRef {
    /// 0-based flat residue index.
    pub residue: u32,
    /// PDB atom name ("CA", "CB", "N", etc.).
    pub atom_name: String,
}

/// One end of a band constraint.
#[derive(Debug, Clone, PartialEq)]
pub enum BandTarget {
    /// Attached to a specific atom.
    Atom(AtomRef),
    /// Anchored to a fixed world-space position (space pulls).
    Position(Vec3),
}

/// Information about a constraint band to be rendered.
///
/// Uses structural references ([`AtomRef`]) instead of world-space
/// positions. The engine resolves atom positions each frame from Scene
/// data, so bands auto-track animated atoms.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct BandInfo {
    /// First endpoint — always an atom.
    pub anchor_a: AtomRef,
    /// Second endpoint — atom or fixed position.
    pub anchor_b: BandTarget,
    /// Band strength (affects radius and color intensity, default 1.0).
    pub strength: f32,
    /// Target length for the band (Angstroms, used for type detection if
    /// not specified).
    pub target_length: f32,
    /// Explicit band type (overrides auto-detection from `target_length`).
    pub band_type: Option<BandType>,
    /// Whether the band is in pull mode (attracts).
    pub is_pull: bool,
    /// Whether the band is in push mode (repels).
    pub is_push: bool,
    /// Whether the band is disabled.
    pub is_disabled: bool,
    /// Whether this band was created by a recipe/script (dimmer appearance).
    pub from_script: bool,
}

/// Information about the active pull constraint.
///
/// Uses a structural reference ([`AtomRef`]) for the pulled atom and a
/// screen-space target. The engine resolves atom position from Scene data
/// and unprojecs `screen_target` at atom depth each frame, so the pull
/// auto-tracks the animated atom.
#[derive(Debug, Clone, PartialEq)]
pub struct PullInfo {
    /// The atom being pulled.
    pub atom: AtomRef,
    /// Screen-space drag position (physical pixels).
    pub screen_target: (f32, f32),
}

// ── Resolved types (internal, world-space) ──────────────────────────────

/// Resolved band with world-space positions, ready for the renderer.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedBand {
    /// World-space position of first endpoint.
    pub endpoint_a: Vec3,
    /// World-space position of second endpoint.
    pub endpoint_b: Vec3,
    /// Whether the band is disabled.
    pub is_disabled: bool,
    /// Band strength (affects radius and color intensity).
    pub strength: f32,
    /// Target length for the band (used for type detection).
    pub target_length: f32,
    /// Residue index for picking (from anchor_a).
    pub residue_idx: u32,
    /// Whether anchor_b is a fixed position (renders anchor sphere).
    pub is_space_pull: bool,
    /// Explicit band type.
    pub band_type: Option<BandType>,
    /// Whether this band was created by a script.
    pub from_script: bool,
}

/// Resolved pull with world-space positions, ready for the renderer.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedPull {
    /// World-space position of the atom being pulled.
    pub atom_pos: Vec3,
    /// World-space target position.
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
#[derive(Debug, Clone, Copy, PartialEq)]
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

    // ── Entity focus ──────────────────────────────────────────────
    /// Focus a specific entity by ID and fit the camera to it.
    FocusEntity {
        /// Entity identifier.
        id: u32,
    },
}
