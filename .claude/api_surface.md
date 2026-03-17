# Viso Public API Surface

## Design principles

1. **Consumers never touch internals.** Scene, renderer, GPU plumbing,
   animator — all `pub(crate)`. The public API is a thin, flat surface.
2. **Commands are actions, options are config.** If it changes how things
   look, it's a `VisoOptions` mutation. If it causes something to happen
   (camera move, selection, data load), it's a `VisoCommand`.
3. **Data goes in via the engine, texture comes out.** Consumers write
   `MoleculeEntity` updates to the input triple buffer. The pipeline
   (animator → scene → renderer → GPU) is invisible. Animation is a
   pipeline behavior, not a user-initiated action.
4. **`InputProcessor` is optional convenience.** It translates raw
   platform events into `VisoCommand`s. Consumers who handle their own
   input construct `VisoCommand`s directly.
5. **Animation behavior is per-entity config, not a command.** Callers
   assign a `Transition` preset to an entity before updating it. The
   engine uses that behavior when interpolating the entity's update.

## The public types

| Type | Role | Status |
|------|------|--------|
| `VisoEngine` | The engine. Accepts data, executes commands, produces textures. | DONE |
| `VisoCommand` | Action vocabulary. Everything the engine can *do*. | DONE |
| `VisoOptions` | Configuration. Everything about how it *looks*. | DONE |
| `VisoError` | Error type. | DONE |
| `InputProcessor` | Raw platform events → `VisoCommand` (optional convenience). | DONE |
| `Transition` | Animation behavior preset. Assigned per-entity. | DONE |

## `VisoCommand` — actions only (DONE)

```rust
pub enum VisoCommand {
    // Camera
    RecenterCamera,
    RotateCamera { delta: Vec2 },
    PanCamera { delta: Vec2 },
    Zoom { delta: f32 },
    ToggleAutoRotate,

    // Focus
    CycleFocus,
    ResetFocus,

    // Selection
    ClearSelection,
    SelectResidue { index: i32, extend: bool },
    SelectSegment { index: i32, extend: bool },
    SelectChain { index: i32, extend: bool },

    // Playback
    ToggleTrajectory,
}
```

`VisoCommand` derives `Copy` — all variants are small value types.

**What is NOT a command:**
- Display toggles (waters, ions, solvent, lipids) → `VisoOptions` mutations
- Animation to a new pose → write entities to input triple buffer
- Animation behavior assignment → `engine.set_entity_behavior(id, transition)`
- Option changes (lighting, geometry, colors) → `VisoOptions` mutations
- Band/pull updates → `engine.update_bands()` / `engine.update_pull()`
  (state, not fire-and-forget actions)

## Constraint types (DONE — structural references)

Constraint types use structural references instead of world-space
positions. The engine resolves atom positions from Scene data each frame,
so bands/pulls auto-track animated atoms.

```rust
/// Structural reference to a specific atom.
pub struct AtomRef {
    pub residue: u32,       // 0-based flat residue index
    pub atom_name: String,  // "CA", "CB", "N", etc.
}

/// One end of a band constraint.
pub enum BandTarget {
    Atom(AtomRef),          // attached to a specific atom
    Position(Vec3),         // anchored to fixed world-space position
}

/// Band constraint (stored on engine, resolved per-frame).
pub struct BandInfo {
    pub anchor_a: AtomRef,
    pub anchor_b: BandTarget,
    pub strength: f32,
    pub target_length: f32,
    pub band_type: Option<BandType>,
    pub is_pull: bool,
    pub is_push: bool,
    pub is_disabled: bool,
    pub from_script: bool,
}

/// Pull constraint (stored on engine, resolved per-frame).
pub struct PullInfo {
    pub atom: AtomRef,
    pub screen_target: (f32, f32),  // screen-space drag position
}
```

Internal `ResolvedBand` / `ResolvedPull` types carry world-space
positions for the renderer. Callers never see these.

## Picking output (DONE)

```rust
pub enum PickTarget {
    None,
    Residue(u32),
    Atom { entity_id: u32, atom_idx: u32 },
}
```

Returned by `engine.hovered_target()`.

## `Transition` — animation behavior presets (DONE — re-exported)

Data-driven via `Vec<AnimationPhase>`. No trait-based behavior system.

```rust
impl Transition {
    pub fn snap() -> Self;
    pub fn smooth() -> Self;
    pub fn backbone_then_expand(backbone: Duration, expand: Duration) -> Self;
    pub fn collapse_expand(collapse: Duration, expand: Duration) -> Self;
    pub fn cascade(base: Duration, delay_per_residue: Duration) -> Self;
    pub fn allowing_size_change(self) -> Self;
    pub fn suppressing_initial_sidechains(self) -> Self;
}
```

`AnimationPhase`, `AnimationRunner`, `StructureAnimator` — all internal.

## `VisoOptions` — configuration (DONE, renamed from `Options`)

The existing options tree (`display`, `lighting`, `post_processing`,
`camera`, `colors`, `geometry`, `debug`) stays as-is. The struct is
renamed to `VisoOptions`. TOML serialization preserved. The `pub mod
options` remains public so consumers can traverse submodules for
fine-grained config.

## Engine data API

### Entity management (DONE)

```rust
engine.load_entities(entities, fit_camera);
engine.update_entities(entities);
engine.update_entity_coords(id, coords);
engine.set_entity_visible(id, visible);
engine.set_ss_override(id, q8);
engine.set_per_residue_scores(id, scores);
engine.remove_entity(id);
```

### Constraints (DONE — structural refs, per-frame resolution)

```rust
engine.update_bands(bands);     // Vec<BandInfo> with AtomRef
engine.update_pull(pull);       // Option<PullInfo> with AtomRef
```

Engine stores constraint specs. Each frame, resolves `AtomRef` → position
from Scene data, unprojecs screen targets at atom depth. Bands/pulls
auto-track animated atoms.

### Per-entity animation behavior (DONE)

```rust
engine.set_entity_behavior(entity_id, transition);
engine.clear_entity_behavior(entity_id);
```

### Output (DONE)

```rust
engine.hovered_target() -> PickTarget  // None | Residue(u32) | Atom { .. }
engine.focused_entity() -> Option<u32> // entity ID when focused, None at session level
engine.fps() -> f32
```

### Entity updates — input triple buffer (NOT STARTED)

**Target:** Lock-free per-entity input channel replacing direct method
calls. Not yet implemented.

### Configuration (DONE)

```rust
engine.set_options(options);    // push config changes with diffing
engine.apply_options();         // force-refresh all subsystems
engine.load_preset(name, dir);
engine.save_preset(name, dir);
```

## GPU bootstrap types

For embedded consumers who already have a wgpu device:
- `RenderContext` — wraps device, queue, surface config, render scale.
  Consumers construct one and hand it to `VisoEngine`. Opaque after init.
- `RenderTarget` — "render into this texture" for embedded use.

Exposed because embedded consumers need them for construction, but they're
plumbing — not part of the core interaction model.

## Input types

- `InputProcessor` — stateful translator: raw events → `VisoCommand`
- `InputEvent` — platform-agnostic input event enum (cursor, button,
  scroll, modifiers)
- `KeyBindings` — key string → command mapping (customizable)
- `MouseButton` — left/right/middle enum

`InputProcessor` is a convenience for winit-based consumers. Web embeds
or custom hosts skip it and construct `VisoCommand`s directly.

## What is NOT public

- `Scene`, `SceneEntity`, `Focus` — internal render pipeline state
- `AnimationRunner`, `StructureAnimator`, `AnimationPhase` — internal
- `EasingFunction` — internal
- `ResolvedBand`, `ResolvedPull` — internal renderer types
- All renderer types (backbone, sidechain, impostor, picking, postprocess)
- All GPU types except `RenderContext` and `RenderTarget`

## `lib.rs` structure (current)

```rust
pub(crate) mod animation;
pub(crate) mod camera;
pub(crate) mod engine;
pub(crate) mod error;
pub(crate) mod gpu;
pub(crate) mod input;
pub(crate) mod renderer;
pub(crate) mod util;

pub mod options;

#[cfg(feature = "viewer")]
pub mod viewer;
#[cfg(feature = "gui")]
pub mod gui;

// Core
pub use engine::VisoEngine;
pub use engine::command::{AtomRef, BandInfo, BandTarget, BandType, PullInfo, VisoCommand};
pub use error::VisoError;

// Picking output
pub use renderer::picking::PickTarget;

// GPU bootstrap
pub use gpu::render_context::RenderContext;
pub use gpu::texture::RenderTarget;

// Animation
pub use animation::transition::Transition;

// Input
pub use input::{InputEvent, InputProcessor, KeyBindings, MouseButton};

// Feature-gated
#[cfg(feature = "viewer")]
pub use viewer::{Viewer, ViewerBuilder};
#[cfg(feature = "gui")]
pub use gui::webview::UiAction;
```

## Renames (all completed)

| Old | New |
|-----|-----|
| `ProteinRenderEngine` | `VisoEngine` |
| `EngineCommand` | `VisoCommand` |
| `Options` | `VisoOptions` |
| `BandRenderInfo` | `BandInfo` |
| `PullRenderInfo` | `PullInfo` |
