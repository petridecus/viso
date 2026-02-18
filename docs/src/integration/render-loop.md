# The Render Loop

Every frame follows a specific sequence. Getting the order right matters -- applying the pending scene before rendering ensures newly generated meshes appear, and updating the camera before rendering ensures smooth animation.

## Minimal Render Loop (Standalone)

From viso's `main.rs`:

```rust
WindowEvent::RedrawRequested => {
    engine.apply_pending_scene();
    match engine.render() {
        Ok(()) => {}
        Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
            engine.resize(window.inner_size().width, window.inner_size().height);
        }
        Err(e) => log::error!("render error: {:?}", e),
    }
    window.request_redraw();
}
```

## Full Render Loop (foldit-rs)

foldit-rs has a richer per-frame sequence:

```
1. Process IPC messages (webview input)
2. Ensure surface matches window size
3. Process backend updates (Rosetta/ML triple buffers)
4. Sync scene to renderers if dirty (submits to background thread)
5. Apply pending scene from background thread (GPU uploads)
6. Update camera animation (dt)
7. Update frame visuals (bands, pulls)
8. Render
9. Push dirty UI state to webview
10. Request next frame
```

### Step-by-Step

#### 1. Apply Backend Updates

```rust
app.apply_backend_updates();
```

This drains updates from Rosetta and ML backends (delivered via triple buffers). Each update may modify entity coordinates, add groups, or change per-residue scores. See [Dynamic Structure Updates](./dynamic-updates.md).

#### 2. Sync Scene if Dirty

```rust
app.sync_engine();
```

If the scene has changed since the last sync (tracked by a generation counter), this collects `PerGroupData` for all visible groups and submits a `SceneRequest::FullRebuild` to the background processor. This is non-blocking -- the background thread generates meshes while the main thread continues rendering the previous frame's data.

#### 3. Apply Pending Scene

```rust
engine.apply_pending_scene();
```

Checks the triple buffer for a completed `PreparedScene` from the background thread. If one is available, it:

- Uploads vertex/index buffers to the GPU
- Updates sidechain instance buffers
- Updates backbone chain data for the camera and animation system
- Rebuilds picking bind groups
- Fires animation if an action was specified

This is a GPU-upload-only operation, typically <1ms.

#### 4. Update Camera Animation

```rust
let still_animating = engine.update_camera_animation(dt);
```

Advances animated camera transitions (focus point, distance, bounding radius). Returns `true` if the camera is still animating. Also handles turntable auto-rotation.

#### 5. Render

```rust
match engine.render() {
    Ok(()) => {}
    Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
        engine.resize(width, height);
    }
    Err(e) => log::error!("render error: {:?}", e),
}
```

The render method executes the full pipeline:

1. **Animation update** -- advances structure animation, generates interpolated meshes if animating
2. **GPU picking pass** -- renders to offscreen R32Uint texture, reads back hovered residue (non-blocking)
3. **Selection buffer update** -- uploads selected residue bit-array to GPU
4. **Geometry pass** -- renders all molecular geometry to HDR render targets
5. **Post-processing** -- SSAO, bloom, composite (outlines + fog + tone mapping), FXAA
6. **Present** -- submits to the wgpu surface

## Error Handling

Surface errors are expected during resize or focus changes:

- `SurfaceError::Outdated` / `SurfaceError::Lost` -- the surface needs to be reconfigured. Call `resize()` with the current window dimensions.
- Other errors are logged but non-fatal -- the next frame will attempt to render again.

## Frame Timing

The render loop uses `ControlFlow::Poll` for continuous rendering. Frame timing is tracked internally for animation interpolation. The standalone viewer runs at the display's refresh rate (vsync). foldit-rs targets 300fps with frame pacing.

## Non-Blocking Picking Readback

GPU picking uses a two-frame pipeline to avoid stalling:

1. **Frame N**: The picking pass renders to an offscreen texture and copies the pixel under the mouse to a staging buffer. `start_readback()` initiates an async buffer map.
2. **Frame N+1**: `complete_readback()` polls the device without blocking. If the map is complete, it reads the residue ID. Otherwise, it uses the cached value from the previous successful read.

This means hover feedback is one frame behind mouse movement, which is imperceptible in practice but avoids GPU pipeline stalls.
