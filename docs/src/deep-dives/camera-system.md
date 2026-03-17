# Camera System

Viso uses an arcball camera that orbits around a focus point. The camera supports animated transitions, auto-rotation, frustum culling, and coordinate conversion utilities.

## Arcball Model

The camera is defined by:
- **Focus point** -- the world-space point the camera orbits around
- **Distance** -- how far the camera is from the focus point
- **Orientation** -- a quaternion defining the camera's rotation
- **Bounding radius** -- the radius of the protein being viewed (used for fog and culling)

All camera manipulation (rotation, pan, zoom) operates on these parameters rather than directly on a view matrix.

## Camera Controller

`CameraController` wraps the camera and manages input, GPU uniforms, and animation:

```rust
pub struct CameraController {
    pub camera: Camera,
    pub uniform: CameraUniform,
    pub buffer: wgpu::Buffer,
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub mouse_pressed: bool,
    pub shift_pressed: bool,
    pub rotate_speed: f32,  // default: 0.5
    pub pan_speed: f32,     // default: 0.5
    pub zoom_speed: f32,    // default: 0.1
}
```

### Rotation

Rotation uses the arcball model -- horizontal mouse movement rotates around the up vector, vertical movement rotates around the right vector:

```rust
controller.rotate(Vec2::new(delta_x, delta_y));
```

The sensitivity is controlled by `rotate_speed`.

### Panning

Panning translates the focus point along the camera's right and up vectors:

```rust
controller.pan(Vec2::new(delta_x, delta_y));
```

This cancels any animated focus point transition.

### Zooming

Zoom adjusts the orbital distance, clamped to [1.0, 1000.0]:

```rust
controller.zoom(scroll_delta);
```

## Camera Animation

The camera can animate between states for smooth transitions when loading structures or changing focus:

### Fitting to Positions

```rust
// Instant fit (for initial load)
controller.fit_to_positions(&all_atom_positions);

// Animated fit (for adding new structures)
controller.fit_to_positions_animated(&all_atom_positions);
```

Both methods:
1. Calculate the centroid of all positions
2. Compute a bounding sphere radius
3. Calculate the distance needed to fit the sphere in the viewport (accounting for both horizontal and vertical FOV)

The animated version sets target values that are interpolated each frame.

### Per-Frame Update

```rust
let still_animating = controller.update_animation(dt);
```

This interpolates:
- Focus point toward target focus
- Distance toward target distance
- Bounding radius toward target radius

The interpolation speed is controlled by `CAMERA_ANIMATION_SPEED` (default 3.0). Higher values mean faster convergence.

## Auto-Rotation

Toggle turntable-style auto-rotation:

```rust
let is_rotating = controller.toggle_auto_rotate();
```

When enabled, the camera rotates around the up vector at `TURNTABLE_SPEED` (approximately 29 degrees per second). The spin axis is captured from the current up vector when auto-rotation is enabled.

```rust
if controller.is_auto_rotating() {
    // Camera is spinning
}
```

## Frustum Culling

The camera provides a frustum for culling off-screen geometry:

```rust
let frustum = controller.frustum();
```

This is primarily used for sidechain culling -- sidechains outside the view frustum are skipped during rendering to improve performance. Each sidechain is tested against the frustum using a 5.0 angstrom cull radius.

## Coordinate Conversion

### Screen Delta to World

Convert mouse movement to world-space displacement:

```rust
let world_offset = controller.screen_delta_to_world(delta_x, delta_y);
```

Uses the camera's right and up vectors to map 2D screen movement to 3D space. The scale factor is proportional to the orbital distance, so movement feels consistent at any zoom level.

### Screen to World at Depth

Unproject screen coordinates to a world-space point on a plane at a specific depth:

```rust
let world_point = controller.screen_to_world_at_depth(
    screen_x, screen_y,
    screen_width, screen_height,
    reference_world_point,
);
```

This is used for pull operations -- the target position should be on a plane parallel to the camera at the residue's depth, so the pull doesn't move toward or away from the camera.

The conversion:
1. Maps screen coordinates to NDC [-1, 1] (with Y-flip since screen origin is top-left)
2. Accounts for both horizontal and vertical FOV
3. Projects onto a plane at the depth of the reference point

## Fog Derivation

Fog parameters are derived from the camera's bounding radius and distance:

- **Fog start** -- based on the distance and bounding radius
- **Fog density** -- increases with bounding radius for larger structures

The post-processing composite pass uses these to apply depth-based fog, fading distant geometry to the background color.

## GPU Uniform

The camera uniform is uploaded to the GPU each frame:

```rust
controller.update_gpu(&queue);
```

The uniform contains the projection matrix, view matrix, inverse projection, camera position, and screen dimensions. All renderers bind to this uniform for vertex transformation.
