# Engine Lifecycle

`VisoEngine` is the central rendering, animation, and picking
coordinator. It is **read-only with respect to structural state** —
mutations to the loaded `Assembly` always go through a `VisoApp` (or
the host application that owns the `Assembly` directly). This chapter
covers how to create the engine, what happens during initialization,
and how to manage its lifetime.

## Construction Pair

The engine reads structural snapshots from a triple buffer; the
matching `AssemblyPublisher` lives on `VisoApp`. The two are always
constructed together:

```rust
use viso::{RenderContext, VisoApp, VisoEngine};
use viso::options::VisoOptions;

// 1. Build a VisoApp + AssemblyConsumer pair.
//    Use one of:
let (app, consumer) = VisoApp::new_empty();
let (app, consumer) = VisoApp::from_entities(entities);
let (app, consumer) = VisoApp::from_bytes(bytes, "cif")?;
let (app, consumer) = VisoApp::from_file("path/to/structure.cif")?;

// 2. Build a wgpu RenderContext (async — use pollster or your runtime).
let context = pollster::block_on(
    RenderContext::new(window.clone(), (width, height))
)?;

// 3. Build the engine.
let mut engine = VisoEngine::new(context, consumer, VisoOptions::default())?;
```

Hosts that already own an `Assembly` (e.g. `foldit-rs`) build their
own `AssemblyPublisher`/`AssemblyConsumer` pair and feed the consumer
into `VisoEngine::new`. They never need a `VisoApp`.

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
8. **Assembly polling** — the consumer is wired in but the first
   `Assembly` snapshot is picked up on the first call to `update()`.

## Initial Scene Sync

`VisoApp::load_entities` / `replace_scene` / `from_file` all publish
an `Assembly` snapshot. The next `engine.update(dt)` polls the
consumer, rederives the scene, and submits a full mesh rebuild to the
background thread. On the frame after, `apply_pending_scene` uploads
the meshes to the GPU.

If you build the engine through `VisoApp::new_empty` (no entities) and
want an explicit non-animating sync (rare — `update` does this for
you), call:

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
| `scene` | `Scene` | Assembly consumer + derived per-entity state |
| `annotations` | `EntityAnnotations` | Per-entity overrides: focus, visibility, behaviors, appearance, scores, SS, surfaces |
| `surface_regen` | `SurfaceRegen` | Background isosurface mesh regeneration |

The engine is **not thread-safe** (`!Send`, `!Sync`) because it holds
wgpu GPU resources. All engine access must happen on the main thread.
The background scene processor and surface regeneration thread
communicate via channels and triple buffers.
