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
host application (foldit-rs / VisoApp)        viso engine
    │                                              │
    │  owns: molex::Assembly (source of truth)     │
    │  mutates freely (load, commands, backend     │
    │   writebacks from Rosetta / ML workers)      │
    │                                              │
    ├─ engine.set_assembly(Arc<Assembly>) ───────►│  scene.pending: Option<Arc<Assembly>>
    │  (called after each Assembly mutation)       │  drained next sync tick
    │                                              │
    ├─ engine.set_entity_behavior(eid, t) ───────►│  annotations: per-entity behavior map
    │                                              │  HashMap<EntityId, Transition>
    │                                              │
    │                                  engine.update(dt):
    │                                    drain scene.pending; on generation change
    │                                      rederive scene.render_state +
    │                                      per-entity EntityView + EntityPositions
    │                                    tick per-entity animation runners
    │                                    upload GPU buffers (<1ms)
    │                                              │
    │                                              ▼
    │                                  scene.render_state + scene.entity_state
    │                                  (derived per-sync; no &Assembly threading)
    │                                              │
    │                                              ▼
    │                                  renderer::geometry (background thread)
    │                                  (meshes + impostors from scene data)
    │                                              │
    │                                              ▼
    │                                  GPU passes
    │                                  (geometry → picking → post-process)
    │                                              │
    │                                              ▼
    │                                  2D texture
    │                                  (winit / canvas / png / embed)
```

### VisoEngine structure (thin dispatcher)

```
VisoEngine {
    gpu: GpuPipeline,                // renderer/gpu_pipeline.rs
    camera_controller: CameraController,
    constraints: ConstraintSpecs,    // band_specs + pull_spec
    animation: AnimationState,       // animation/state.rs
    options: VisoOptions,
    active_preset: Option<String>,
    frame_timing: FrameTiming,
    density: DensityStore,
    scene: Scene,                    // engine/scene.rs (assembly ingest +
                                     //   derived per-entity render state)
    annotations: EntityAnnotations,  // user-authored per-entity opinions
                                     //   (focus, visibility, behaviors,
                                     //   appearance, scores, ss_override)
    surface_regen: SurfaceRegen,     // background isosurface mesh worker
}
```

All engine methods are thin dispatchers (3-8 lines of orchestration).
Module-level types own their behavior — not passive field bags.

### Key data types

- **`Scene`** (`engine/scene.rs`): Assembly ingest + derived per-entity
  state. Holds `pending: Option<Arc<Assembly>>` (host's latest push,
  drained on sync), `current: Arc<Assembly>` (last applied snapshot),
  `last_seen_generation`, `render_state: SceneRenderState` (cross-entity
  rendering data), `entity_state: HashMap<EntityId, EntityView>`
  (per-entity topology + drawing mode + mesh version), and
  `positions: EntityPositions` (per-entity animator write surface).
- **`EntityView`** (`engine/entity_view.rs`): Per-entity render-ready
  view — drawing mode, ss override, topology, mesh version. Rederived
  on every Assembly sync.
- **`EntityAnnotations`** (`engine/annotations.rs`): User-authored
  per-entity opinions that ride alongside the Assembly: focus,
  visibility, behaviors, appearance overrides, scores, SS overrides,
  surfaces. All maps keyed on `EntityId`.
- **`GpuPipeline`** (`renderer/gpu_pipeline.rs`): Rendering entry point.
  Context, renderers, picking, scene_processor, post_process,
  shader_composer, lighting, cursor_pos, last_cull_camera_eye.
- **`AnimationState`** (`animation/state.rs`): Animation entry point.
  Animator, trajectory_player, pending_transitions.
- **`ConstraintSpecs`** (`engine/mod.rs`): Band specs + pull spec.

### Host integration

The library API for structural ingest is one method:
`engine.set_assembly(Arc<molex::Assembly>)`. Library consumers
(`foldit-rs`) own their own `molex::Assembly`, mutate it via molex's
APIs, and push the new snapshot to the engine after each batch of
mutations. There is no viso-defined channel, publisher, or consumer
exposed to library users.

`VisoApp` (in `src/app/mod.rs`, behind `feature = "viewer" / "web"`)
is *not* part of the library surface. It is the helper viso uses to
host its own `Assembly` when running as a standalone application
(`cargo run -p viso`, the `viewer` / `gui` / `web` features). Its
mutation methods (`load_entities`, `update_entity`, etc.) bundle a
`molex::Assembly` mutation and an `engine.set_assembly` push for the
standalone case. Library users with `default-features = false` do not
see `VisoApp` and should not look for it.

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
│   ├── mod.rs          VisoEngine dispatcher, ConstraintSpecs, set_assembly
│   ├── bootstrap.rs    Construction (RenderContext + VisoOptions → engine),
│   │                   GpuBootstrap, FrameTiming
│   ├── command.rs      VisoCommand, BandInfo, PullInfo, AtomRef,
│   │                   BandTarget, BandType, ResolvedBand, ResolvedPull
│   ├── annotations.rs  EntityAnnotations: per-entity opinions (focus,
│   │                   visibility, behaviors, appearance, scores, ss override)
│   ├── constraint.rs   Constraint resolution (resolve_atom_ref, band/pull)
│   ├── culling.rs      Frustum culling
│   ├── density.rs      Volumetric density helpers
│   ├── density_store.rs DensityStore (loaded electron density maps)
│   ├── entity_view.rs  EntityView (per-entity render-ready view)
│   ├── focus.rs        Focus enum (Session | Entity(EntityId))
│   ├── options_apply.rs Options application + preset loading
│   ├── positions.rs    EntityPositions (per-entity animator write surface)
│   ├── scene.rs        Scene: pending/current Assembly, render_state,
│   │                   entity_state, positions
│   ├── scene_state.rs  SceneRenderState (cross-entity rendering data)
│   ├── surface.rs      Per-entity surface state
│   ├── surface_regen.rs Background isosurface mesh regeneration
│   ├── sync/           Scene → renderer pipeline (poll Assembly, rederive,
│   │                   submit mesh-gen, frustum culling, LOD)
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
├── app/                Standalone-app layer (feature = "viewer"/"gui"/"web")
│   ├── mod.rs          VisoApp (owns Assembly), publish helper
│   ├── viewer.rs       Winit viewer shell
│   ├── gui/            wry-webview options panel (feature = "gui")
│   └── web/            Wasm entry (feature = "web")
└── util/               Helpers (easing.rs, hash.rs)
```

## Public API surface

See `.claude/api_surface.md` for the full design.

**Re-exported from `lib.rs`:**
- Core: `VisoEngine` (with `set_assembly`), `VisoCommand`, `BandInfo`,
  `BandType`, `PullInfo`, `AtomRef`, `BandTarget`, `CommandOutcome`,
  `VisoError`, `Focus`
- Picking: `PickTarget`
- Config: `VisoOptions` (pub mod), `DisplayOverrides`, `DrawingMode`,
  `HelixStyle`, `SheetStyle`
- Input: `InputEvent`, `InputProcessor`, `KeyBindings`, `MouseButton`
- GPU bootstrap: `RenderContext`, `RenderTarget`
- Animation: `Transition`
- Molex passthrough: `pub use molex;`
- Feature-gated: `VisoApp` (`viewer`/`web`), `Viewer`, `ViewerBuilder`
  (`viewer`), `UiAction` (`gui`)

**NOT public:** Scene, SceneRenderState, EntityView, EntityPositions,
EntityAnnotations, AnimationRunner, AnimationPhase, StructureAnimator,
GpuPipeline, AnimationState, ResolvedBand, ResolvedPull, renderer
types, EasingFunction. The host owns its own `molex::Assembly` and
pushes via `engine.set_assembly`; viso never exposes its internal
ingest channel.

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
