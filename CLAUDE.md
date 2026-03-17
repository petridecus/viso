# Viso — GPU-accelerated protein visualization engine

## Project overview

Viso is a wgpu-based rendering engine for real-time protein structure
visualization. It is designed as an embeddable library (`lib.rs`) with an
optional standalone viewer (`feature = "viewer"`).

~17,500 lines of Rust, ~2,600 lines of WGSL shaders.

## Lint policy

Strict. `#![deny(clippy::pedantic)]`, `#![deny(missing_docs)]`,
`#![deny(clippy::unwrap_used)]`. GPU casts are allowed. Lint
configuration lives in `Cargo.toml` under `[lints.clippy]` and
`[lints.rust]`, not in `lib.rs`.

## Architecture

```
foldit-rs / caller                     viso engine
    │                                      │
    ├─ engine.update_entities() ─────────►│  EntityStore (source of truth)
    │  engine.update_entity_coords()       │  Vec<MoleculeEntity> + Vec<SceneEntity>
    │                                      │
    ├─ engine.set_entity_behavior() ─────►│  per-entity behavior map
    │                                      │  HashMap<u32, Transition>
    │                                      │
    │                              engine.pre_render(dt):
    │                                poll background thread results
    │                                tick per-entity animation runners
    │                                upload GPU buffers (<1ms)
    │                                      │
    │                                      ▼
    │                              SceneTopology + VisualState
    │                              (derived metadata + interpolated positions)
    │                                      │
    │                                      ▼
    │                              renderer::geometry (background thread)
    │                              (meshes + impostors from scene data)
    │                                      │
    │                                      ▼
    │                              GPU passes
    │                              (geometry → picking → post-process)
    │                                      │
    │                                      ▼
    │                              2D texture
    │                              (winit / canvas / png / embed)
```

### VisoEngine structure (10 fields, thin dispatcher)

```
VisoEngine {
    gpu: GpuPipeline,               // renderer/gpu_pipeline.rs (9 fields, 16 methods)
    entities: EntityStore,           // engine/entity_store.rs
    topology: SceneTopology,         // engine/scene.rs
    visual: VisualState,             // engine/scene.rs
    animation: AnimationState,       // animation/state.rs (3 fields, 6 methods)
    camera_controller: CameraController,
    constraints: ConstraintSpecs,    // band_specs + pull_spec
    options: VisoOptions,
    active_preset: Option<String>,
    frame_timing: FrameTiming,
}
```

All engine methods are thin dispatchers (3-8 lines of orchestration).
Module-level types own their behavior — not passive field bags.

### Key data types

- **`EntityStore`** (`engine/entity_store.rs`): Entity ownership.
  `Vec<MoleculeEntity>` (source of truth), `Vec<SceneEntity>` (render
  state), per-entity behavior map, focus state, structural dirty
  flagging (`generation`).
- **`SceneTopology`** (`engine/scene.rs`): Derived metadata for the
  renderer. NA chains, entity residue ranges, sidechain topology, SS
  types, per-residue colors. All computed on main thread via
  `prepare_scene_metadata()` before background submission.
- **`VisualState`** (`engine/scene.rs`): Animation output buffer.
  Interpolated backbone chains, sidechain positions, backbone-sidechain
  bonds. Position-level dirty flagging (`position_generation`).
- **`GpuPipeline`** (`renderer/gpu_pipeline.rs`): Rendering entry point.
  Context, renderers, picking, scene_processor, post_process,
  shader_composer, lighting, cursor_pos, last_cull_camera_eye.
- **`AnimationState`** (`animation/state.rs`): Animation entry point.
  Animator, trajectory_player, pending_transitions.
- **`ConstraintSpecs`** (`engine/mod.rs`): Band specs + pull spec.

### Animation

Data-driven via `Transition` (re-exported from `lib.rs`). A `Transition`
holds a `Vec<AnimationPhase>`, where each phase has easing, duration,
lerp range, and a sidechain visibility flag. The `AnimationRunner`
evaluates phases sequentially — no trait-based behavior system.

Preset constructors: `snap()`, `smooth()`, `collapse_expand()`,
`backbone_then_expand()`, `cascade()`. Builder methods:
`allowing_size_change()`, `suppressing_initial_sidechains()`.

Per-entity interpolation. Each entity gets its own `AnimationRunner`
with its own `Transition` and timing. The `StructureAnimator` manages
a `HashMap<u32, EntityAnimationState>` and aggregates output each frame.

### Renderer

Consumes scene data read-only. `renderer::geometry` translates scene
data into mesh and impostor data. `renderer::pipeline` owns the
background mesh generation pipeline: `SceneProcessor` runs a background
thread producing `PreparedScene` / `PreparedAnimationFrame` (GPU-ready
byte buffers only, no metadata). Main thread does only GPU uploads and
render passes.

GPU pipeline: geometry pass → picking pass → post-process (SSAO, bloom,
fog, FXAA) → final 2D texture.

### Output

Host-agnostic texture. Consumers: winit window, HTML canvas (wasm),
PNG snapshot (headless), dioxus/egui/any framework with a texture slot.

## Secondary structure

DSSP from `molex` by default. Per-entity override via Q8 string.

## Color

Options-driven. Defaults: chain color (protein backbone), element color
(ball-and-stick), cofactor recognition. Supports `ColorRamp` per-residue.
No "score" concept in viso — callers provide ramp values.

## Feature flags

| Feature     | Description                                    |
|-------------|------------------------------------------------|
| `viewer`    | Standalone winit window                        |
| `gui`       | Native webview options panel (implies `viewer`) |
| `binary`    | CLI binary (implies `gui`)                     |

## Module map

```
src/
├── lib.rs              Public API surface (flat re-exports only)
├── main.rs             CLI binary entry point
├── engine/             Core engine struct + frame loop
│   ├── mod.rs          VisoEngine (10-field dispatcher), ConstraintSpecs, FrameTiming
│   ├── bootstrap.rs    Construction, GpuBootstrap, scene loading, assembly helpers
│   ├── command.rs      VisoCommand, BandInfo, PullInfo, AtomRef,
│   │                   BandTarget, BandType, ResolvedBand, ResolvedPull
│   ├── entity.rs       Entity management: load, update, coords, visibility,
│   │                   SS override, scores, remove. Constraints + behavior.
│   ├── constraint.rs   Constraint resolution (resolve_atom_ref, band/pull)
│   ├── entity_store.rs EntityStore: entity ownership, behaviors, focus,
│   │                   dirty tracking, SceneEntity management
│   ├── options_apply.rs Options application
│   ├── scene.rs        SceneTopology, VisualState, SidechainTopology, Focus
│   ├── scene_data.rs   SceneEntity, PerEntityData, EntityResidueRange,
│   │                   aggregation functions, bond topology tables
│   ├── sync.rs         Scene→renderer pipeline: metadata prep, GPU upload,
│   │                   animation setup, frustum culling, LOD
│   └── trajectory.rs   TrajectoryPlayer (DCD frame sequencer)
├── animation/          Structural animation
│   ├── mod.rs          Re-exports
│   ├── state.rs        AnimationState (3 fields, 6 methods — module entry point)
│   ├── animator.rs     StructureAnimator + AnimationFrame
│   ├── runner.rs       AnimationRunner + data types
│   └── transition.rs   AnimationPhase, Transition presets
├── renderer/           GPU rendering pipeline
│   ├── mod.rs          PipelineLayouts + Renderers + GeometryPassInput
│   ├── gpu_pipeline.rs GpuPipeline (9 fields, 16 methods — rendering entry point)
│   ├── draw_context.rs Bind group bundles for draw calls
│   ├── geometry/       Mesh + impostor generation from scene data
│   │   ├── backbone/   Backbone mesh generation (spline, profile, mesh, path)
│   │   ├── sidechain.rs    Sidechain impostor instances
│   │   ├── ball_and_stick.rs  Ball-and-stick impostor instances
│   │   ├── nucleic_acid.rs    Nucleic acid geometry
│   │   ├── band.rs     Constraint band geometry
│   │   ├── pull.rs     Pull arrow geometry
│   │   └── sheet_adjust.rs  Sheet-surface sidechain adjustment
│   ├── impostor/       Impostor primitives (sphere, capsule, cone, polygon)
│   ├── pipeline/       Background mesh generation pipeline
│   │   ├── prepared.rs PreparedScene, SceneRequest, byte buffer types
│   │   ├── mesh_gen.rs Per-entity + per-frame mesh generation
│   │   ├── mesh_concat.rs  Merge per-entity meshes into PreparedScene
│   │   └── processor.rs    Background thread (SceneProcessor + MeshCache)
│   ├── picking/        GPU-based object picking + PickingSystem + PickTarget
│   └── postprocess/    SSAO, bloom, fog, FXAA, composite, screen pass
├── camera/             Orbital camera (controller, core, frustum)
├── gpu/                wgpu abstractions
│   ├── render_context.rs  Device/queue/surface init
│   ├── lighting.rs     Lighting uniform + IBL cubemap generation
│   ├── shader_composer.rs  WGSL #import via naga-oil
│   ├── texture.rs      RenderTarget + texture helpers
│   ├── dynamic_buffer.rs  Resizable GPU buffer
│   ├── pipeline_helpers.rs  Pipeline creation utilities
│   └── residue_color.rs    Per-residue color GPU buffer
├── input/              Input processing (InputProcessor, InputEvent, KeyBindings)
├── options/            TOML-serializable runtime options + score_color.rs
├── viewer.rs           Standalone winit viewer
└── util/               Helpers (easing.rs, hash.rs)
```

## Public API surface

See `.claude/api_surface.md` for the full design.

**Re-exported from `lib.rs`:**
- Core: `VisoEngine`, `VisoCommand`, `BandInfo`, `BandType`, `PullInfo`,
  `AtomRef`, `BandTarget`, `VisoError`
- Picking: `PickTarget`
- Config: `VisoOptions` (pub mod)
- Input: `InputEvent`, `InputProcessor`, `KeyBindings`, `MouseButton`
- GPU bootstrap: `RenderContext`, `RenderTarget`
- Animation: `Transition`
- Feature-gated: `Viewer`, `ViewerBuilder`, `UiAction`

**NOT public:** EntityStore, SceneTopology, VisualState, AnimationRunner,
AnimationPhase, StructureAnimator, GpuPipeline, AnimationState,
ResolvedBand, ResolvedPull, renderer types, EasingFunction.

## Conventions

- No `unwrap()` or `expect()` — use `?`, `if let`, or `map_or`
- All public items must have doc comments
- Prefer `log::info/warn/error` over println
- GPU buffer labels should be descriptive
- Keep rendering code in `renderer/`, not on the engine
- Keep scene logic in `engine/scene.rs`, not spread across engine methods
- The engine is a thin coordinator/dispatcher, not a god object
- Use `&[T]` not `&Vec<T>` in function signatures
- Prefer iterators over collecting into `Vec` when possible

## Improvement plan

See `PLAN.md` for the prioritized findings roadmap (5 phases, easy wins
first). Key areas: CLI hardening, GPU init, performance hotspots,
shader deduplication, testing infrastructure, molex → molex
generalization.
