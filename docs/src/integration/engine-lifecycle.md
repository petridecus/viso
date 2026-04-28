# Engine Lifecycle

`VisoEngine` is the central rendering, animation, and picking
coordinator. It is **read-only with respect to structural state** —
your application owns a `molex::Assembly` and pushes the latest
snapshot to the engine via [`VisoEngine::set_assembly`]. This chapter
covers how to create the engine, what happens during initialization,
and how to manage its lifetime.

## Construction

You own your own `molex::Assembly` and hand viso the latest snapshot.
There is no viso-defined channel, publisher, or consumer in the public
API — the entire structural ingest contract is one setter on the
engine.

```rust
use std::sync::Arc;
use viso::{RenderContext, VisoEngine};
use viso::options::VisoOptions;
use molex::Assembly;

// 1. Build a wgpu RenderContext (async — use pollster or your runtime).
let context = pollster::block_on(
    RenderContext::new(window.clone(), (width, height))
)?;

// 2. Build the engine.
let mut engine = VisoEngine::new(context, VisoOptions::default())?;

// 3. Push your Assembly to the engine.
let assembly: Assembly = /* your owned Assembly */;
engine.set_assembly(Arc::new(assembly.clone()));
```

After every Assembly mutation, re-publish by calling
`engine.set_assembly(...)` again. The engine stages the snapshot in
its internal pending slot and drains it on the next `update(dt)` (or
`sync_now()`) tick — a generation check skips work if nothing changed.

> **Note for standalone deployments only.** When viso is built as its
> own standalone app via `cargo run -p viso` (features `viewer` /
> `gui` / `web`), it uses an internal helper called `VisoApp` to play
> the host role for itself. Library users **never** go through
> `VisoApp` — own your own `Assembly` and call `set_assembly`
> directly. `VisoApp` is not part of the library's public surface
> with `default-features = false`.

### What Happens During Init

1. **GPU setup** — `RenderContext` is configured with a surface, adapter,
   device, and queue.
2. **Shader compilation** — `ShaderComposer` loads and composes all WGSL
   modules using `naga_oil`.
3. **Camera** — `CameraController` is created with default orbital
   parameters (FOV 45°, fit to origin).
4. **Renderers** — backbone, sidechain, ball-and-stick, bond, band,
   pull, nucleic-acid, and isosurface renderers.
5. **Post-processing** — SSAO, bloom, composite, and FXAA passes.
6. **Picking** — GPU picking system with offscreen `R32Uint` target and
   staging buffer.
7. **Scene processor** — background thread spawned for mesh generation.
8. **Assembly slot** — `Scene` starts with an empty `current` Assembly
   and `pending: None`. The first `set_assembly` call fills `pending`;
   the next `update(dt)` consumes it.

## Initial Scene Sync

The first `engine.set_assembly(...)` call after construction pushes
your initial snapshot. The next `engine.update(dt)` drains the pending
snapshot, rederives the scene, and submits a full mesh rebuild to the
background thread. On the frame after, `apply_pending_scene` uploads
the meshes to the GPU.

If you want an explicit non-animating sync (rare — `update` does this
for you), call:

```rust
engine.sync_scene_to_renderers(std::collections::HashMap::new());
```

The `HashMap<u32, Transition>` argument lets you animate specific
entities; passing an empty map snaps everything.

## Resize and Scale Factor

Forward window resize events to the engine:

```rust
engine.resize(new_width, new_height);
```

This resizes the wgpu surface, all post-processing textures, the
picking render target, and the camera projection.

For DPI changes:

```rust
engine.set_surface_scale(scale_factor);
let inner = window.inner_size();
engine.resize(inner.width, inner.height);
```

## Shutdown

The background scene processor is joined automatically on drop. To
force shutdown earlier:

```rust
engine.shutdown();
```

This sends a `Shutdown` request to the processor thread.

## Ownership Model

`VisoEngine` owns 11 sub-systems, each in its own field:

| Field | Type | Purpose |
|-------|------|---------|
| `gpu` | `GpuPipeline` | wgpu context, all renderers, picking, post-process, lighting, shader composer, density mesh receiver |
| `camera_controller` | `CameraController` | Camera matrices, animation, frustum |
| `constraints` | `ConstraintSpecs` | Stored band/pull constraint specs |
| `animation` | `AnimationState` | Structural animator, trajectory player, pending transitions |
| `options` | `VisoOptions` | Display, lighting, post-processing, geometry, etc. |
| `active_preset` | `Option<String>` | Name of the currently-applied options preset |
| `frame_timing` | `FrameTiming` | FPS smoothing, frame pacing |
| `density` | `DensityStore` | Loaded electron density maps |
| `scene` | `Scene` | Pending/current Assembly + derived per-entity state |
| `annotations` | `EntityAnnotations` | Per-entity overrides: focus, visibility, behaviors, appearance, scores, SS, surfaces |
| `surface_regen` | `SurfaceRegen` | Background isosurface mesh regeneration |

The engine is **not thread-safe** (`!Send`, `!Sync`) because it holds
wgpu GPU resources. All engine access must happen on the main thread.
The background scene processor and surface regeneration thread
communicate via channels and triple buffers.
