# Quick Start

Viso ships with a standalone binary (`main.rs`) that opens a window, loads a protein structure, and renders it with full interactivity. This chapter walks through that code as a minimal integration example.

## Running the Standalone Viewer

```sh
cargo run -p viso -- 1ubq
```

This downloads PDB entry `1UBQ` from RCSB, caches it in `assets/models/`, and opens a window. You can also pass a local file path:

```sh
cargo run -p viso -- path/to/structure.cif
```

## The Code

The standalone viewer is roughly 230 lines. Here are the key pieces.

### Application State

```rust
struct RenderApp {
    window: Option<Arc<Window>>,
    engine: Option<ProteinRenderEngine>,
    last_mouse_pos: (f32, f32),
    cif_path: String,
}
```

The engine and window are `Option` because winit creates the window asynchronously in the `resumed` callback.

### Initialization

```rust
fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    if self.window.is_none() {
        // Create window at 75% of monitor size
        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        let size = window.inner_size();
        let scale = window.scale_factor();

        // Create engine (async -- uses pollster::block_on here)
        let mut engine = pollster::block_on(
            ProteinRenderEngine::new_with_path(
                window.clone(),
                (size.width, size.height),
                scale,
                &self.cif_path,
            )
        );

        // Kick off background scene processing
        engine.sync_scene_to_renderers(None);

        window.request_redraw();
        self.window = Some(window);
        self.engine = Some(engine);
    }
}
```

Key points:
- `ProteinRenderEngine::new_with_path` is async (it initializes wgpu). The standalone uses `pollster::block_on`.
- `sync_scene_to_renderers(None)` submits the loaded scene to the background mesh-generation thread. The `None` means no animation action.
- `request_redraw()` starts the render loop.

### The Render Loop

```rust
WindowEvent::RedrawRequested => {
    if let (Some(window), Some(engine)) = (&self.window, &mut self.engine) {
        engine.apply_pending_scene();
        match engine.render() {
            Ok(()) => {}
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                let inner = window.inner_size();
                engine.resize(inner.width, inner.height);
            }
            Err(e) => log::error!("render error: {:?}", e),
        }
        window.request_redraw();
    }
}
```

Each frame:
1. `apply_pending_scene()` -- checks if the background thread has finished generating meshes. If so, uploads them to the GPU (<1ms).
2. `render()` -- executes the full rendering pipeline (geometry pass, picking pass, post-processing).
3. `request_redraw()` -- schedules the next frame.

### Input Wiring

Mouse and keyboard events are forwarded to the engine:

```rust
// Rotation and panning
WindowEvent::CursorMoved { position, .. } => {
    let delta_x = position.x as f32 - self.last_mouse_pos.0;
    let delta_y = position.y as f32 - self.last_mouse_pos.1;
    engine.handle_mouse_move(delta_x, delta_y);
    engine.handle_mouse_position(position.x as f32, position.y as f32);
    self.last_mouse_pos = (position.x as f32, position.y as f32);
}

// Zoom
WindowEvent::MouseWheel { delta, .. } => {
    match delta {
        MouseScrollDelta::LineDelta(_, y) => engine.handle_mouse_wheel(y),
        MouseScrollDelta::PixelDelta(pos) => {
            engine.handle_mouse_wheel(pos.y as f32 * 0.01)
        }
    }
}

// Click selection
WindowEvent::MouseInput { button, state, .. } => {
    let pressed = state == ElementState::Pressed;
    engine.handle_mouse_button(button, pressed);
    if button == MouseButton::Left && !pressed {
        engine.handle_mouse_up();
    }
}

// Modifier keys (shift for pan and multi-select)
WindowEvent::ModifiersChanged(modifiers) => {
    engine.update_modifiers(modifiers.state());
}
```

### Resize Handling

```rust
WindowEvent::Resized(event_size) => {
    engine.resize(event_size.width, event_size.height);
}

WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
    engine.set_scale_factor(scale_factor);
    let inner = window.inner_size();
    engine.resize(inner.width, inner.height);
}
```

### PDB ID Resolution

The `resolve_structure_path` function handles both local files and 4-character PDB codes:

```rust
fn resolve_structure_path(input: &str) -> Result<String, String> {
    // If it's a file path that exists, use it directly
    if std::path::Path::new(input).exists() {
        return Ok(input.to_string());
    }

    // If it's a 4-character alphanumeric code, treat as PDB ID
    if input.len() == 4 && input.chars().all(|c| c.is_ascii_alphanumeric()) {
        let pdb_id = input.to_lowercase();
        let url = format!("https://files.rcsb.org/download/{}.cif", pdb_id);
        // Download and cache in assets/models/
        // ...
    }

    Err(format!("File not found and not a valid PDB code: {}", input))
}
```

## What's Different in foldit-rs

The standalone viewer is intentionally minimal. foldit-rs adds:

- **Multiple entity groups** with independent visibility and focus cycling
- **Backend integration** (Rosetta minimization, ML structure prediction) that streams coordinate updates
- **Animated transitions** between poses using the animation system
- **A webview UI** for controls, panels, and sequence display
- **Band and pull visualization** for constraint-based manipulation

These patterns are covered in the [Integration Guide](../integration/engine-lifecycle.md).
