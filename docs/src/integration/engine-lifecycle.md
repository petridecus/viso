# Engine Lifecycle

`ProteinRenderEngine` is the central facade for all rendering, input handling, scene management, and animation in viso. This chapter covers how to create it, what happens during initialization, and how to manage its lifetime.

## Creation

The engine is created asynchronously because wgpu adapter and device initialization are async:

```rust
// With a structure file (standalone use)
let mut engine = ProteinRenderEngine::new_with_path(
    window.clone(),   // impl Into<wgpu::SurfaceTarget<'static>>
    (width, height),  // Physical pixel dimensions
    scale_factor,     // DPI scale (e.g. 2.0 on Retina)
    &cif_path,        // Path to mmCIF file
).await;

// Without a pre-loaded structure (library use)
let mut engine = ProteinRenderEngine::new(
    window.clone(),
    (width, height),
    scale_factor,
).await;
```

### What Happens During Init

1. **GPU setup** -- wgpu instance, adapter, device, queue, and surface are configured via `RenderContext`
2. **Shader compilation** -- `ShaderComposer` loads and composes all WGSL shaders using naga_oil
3. **Camera** -- `CameraController` is created with default orbital parameters (distance 150, FOV 45)
4. **Renderers** -- all molecular renderers are created (tube, ribbon, sidechain, ball-and-stick, band, pull, nucleic acid)
5. **Post-processing** -- SSAO, bloom, composite, and FXAA passes are initialized
6. **Picking** -- GPU picking system with offscreen render target and staging buffer
7. **Scene processor** -- background thread is spawned for mesh generation
8. **Structure loading** -- if a path was provided, the file is parsed and entities are added to the scene

## Initial Scene Sync

After creation, the scene has entities but no GPU meshes. You must sync:

```rust
engine.sync_scene_to_renderers(None);
```

This submits the scene to the background processor thread. The `None` argument means no animation action (the initial load uses `Snap` behavior by default). The main thread continues immediately -- mesh generation happens in the background.

On the next frame, `apply_pending_scene()` will detect the completed meshes and upload them to the GPU.

## Resize and Scale Factor

Handle window resize events by forwarding to the engine:

```rust
engine.resize(new_width, new_height);
```

This resizes:
- The wgpu surface
- All post-processing textures (SSAO, bloom, composite, FXAA)
- The picking render target
- The camera aspect ratio

For DPI changes:

```rust
engine.set_scale_factor(new_scale);
let inner = window.inner_size();
engine.resize(inner.width, inner.height);
```

## Shutdown

The engine cleans up automatically on drop. The background scene processor thread is joined:

```rust
engine.shutdown_scene_processor();
```

This sends a `SceneRequest::Shutdown` message and waits for the thread to finish. It's also called automatically in the `Drop` implementation, so explicit shutdown is only needed if you want to control timing.

## Ownership Model

`ProteinRenderEngine` owns:

| Component | Type | Purpose |
|-----------|------|---------|
| `context` | `RenderContext` | wgpu device, queue, surface |
| `scene` | `Scene` | Entity groups, focus state |
| `camera_controller` | `CameraController` | Camera matrices, animation |
| `animator` | `StructureAnimator` | Backbone/sidechain animation |
| `picking` | `Picking` | GPU picking system |
| `scene_processor` | `SceneProcessor` | Background mesh thread |
| `tube_renderer` | `TubeRenderer` | Backbone tubes/coils |
| `ribbon_renderer` | `RibbonRenderer` | Helices and sheets |
| `sidechain_renderer` | `CapsuleSidechainRenderer` | Sidechain capsules |
| `bns_renderer` | `BallAndStickRenderer` | Ligands, ions, waters |
| `band_renderer` | `BandRenderer` | Constraint bands |
| `pull_renderer` | `PullRenderer` | Active pull visualization |
| `na_renderer` | `NucleicAcidRenderer` | DNA/RNA backbones |
| `post_process` | `PostProcessStack` | SSAO, bloom, composite, FXAA |

The engine is **not thread-safe** (`!Send`, `!Sync`) because it holds wgpu GPU resources. All engine access must happen on the main thread. The background scene processor communicates via channels and triple buffers.
