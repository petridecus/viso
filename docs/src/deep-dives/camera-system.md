# Camera System

Viso uses an arcball camera that orbits around a focus point. The
camera supports animated transitions, auto-rotation, frustum culling,
and coordinate conversion utilities.

## Arcball Model

The camera is defined by:

- **Focus point** — the world-space point the camera orbits around
- **Distance** — how far the camera is from the focus point
- **Orientation** — a quaternion defining the camera's rotation
- **Bounding radius** — the radius of the protein being viewed
  (used for fog and culling)

All camera manipulation (rotation, pan, zoom) operates on these
parameters rather than directly on a view matrix.

## Camera Controller

`CameraController` (in `camera/controller.rs`) wraps the camera and
manages input, GPU uniforms, and animation. The type is `pub(crate)`
— consumers interact with the camera through engine methods or
`VisoCommand`s, not through the controller directly.

The controller's tunables come from `CameraOptions`:

- `rotate_speed` (default 0.5)
- `pan_speed` (default 0.5)
- `zoom_speed` (default 0.1)
- `fovy` (default 45.0°)
- `znear` (default 5.0)
- `zfar` (default 2000.0)

### Rotation

Rotation uses the arcball model — horizontal mouse movement rotates
around the up vector, vertical movement rotates around the right
vector:

```rust
engine.execute(VisoCommand::RotateCamera { delta });
```

The sensitivity is controlled by `rotate_speed`.

### Panning

Panning translates the focus point along the camera's right and up
vectors, cancelling any in-progress focus animation:

```rust
engine.execute(VisoCommand::PanCamera { delta });
```

### Zooming

Zoom adjusts the orbital distance, clamped to a sensible range:

```rust
engine.execute(VisoCommand::Zoom { delta });
```

## Camera Animation

The camera animates between states for smooth transitions when
loading structures or changing focus.

### Fitting to a Bounding Sphere

Internally, the engine computes a bounding sphere over the relevant
entities and calls one of:

- `fit_to_sphere(centroid, radius)` — instant fit (initial load)
- `fit_to_sphere_animated(centroid, radius)` — animated fit (focus
  cycle, scene replacement)

The fit accounts for both horizontal and vertical FOV so the protein
fits in the viewport.

Public entry points:

```rust
engine.fit_camera_to_focus();   // fits to current focus target
engine.execute(VisoCommand::RecenterCamera);
```

### Per-Frame Update

Inside `engine.update(dt)`, the controller's `update_animation` is
ticked, interpolating focus, distance, and bounding radius toward
their targets.

## Auto-Rotation

Toggle turntable-style auto-rotation:

```rust
engine.execute(VisoCommand::ToggleAutoRotate);
```

When enabled, the camera rotates around the up vector at a fixed
turntable speed (~29°/s). The spin axis is captured from the current
up vector when auto-rotation is enabled.

## Frustum Culling

The camera provides a frustum used for sidechain culling — sidechains
outside the view frustum (with a small Å margin) are skipped during
rendering to improve performance. The engine reuploads the
frustum-filtered sidechain instance buffer when the camera moves
enough to invalidate the previous cull.

## Coordinate Conversion

The controller exposes screen-to-world utilities for input handling:

- `screen_delta_to_world(delta_x, delta_y)` — convert mouse pixel
  movement to a world-space displacement using the camera's right
  and up vectors. The scale is proportional to the orbital distance,
  so movement feels consistent at any zoom level.
- `screen_to_world_at_depth(...)` — unproject a screen pixel onto a
  plane parallel to the camera at a reference world point's depth.
  Used for pull operations so the drag stays at the atom's depth.

These are crate-internal — they're consumed by the constraint
resolution path that produces the per-frame band/pull world-space
positions.

## Fog Derivation

Fog parameters are derived from the camera's distance and bounding
radius each frame:

- **Fog start** — based on the orbital distance.
- **Fog density** — `2.0 / max(bounding_radius, 10.0)`.

The composite post-pass uses these to apply depth-based fog, fading
distant geometry to the background color.

## GPU Uniform

The camera uniform is uploaded to the GPU each frame inside
`engine.render()`. It contains the projection matrix, view matrix,
inverse projection, camera position, hovered residue id, screen
dimensions, and an elapsed-time field. All renderers bind to this
uniform for vertex transformation and view-dependent effects.
