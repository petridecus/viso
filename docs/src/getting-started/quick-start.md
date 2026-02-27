# Quick Start

Viso ships with a standalone viewer that opens a window, loads a protein structure, and renders it with full interactivity.

## Running the Standalone Viewer

```sh
cargo run -p viso -- 1ubq
```

This downloads PDB entry `1UBQ` from RCSB, caches it in `assets/models/`, and opens a window. You can also pass a local file path:

```sh
cargo run -p viso -- path/to/structure.cif
```

## Using the Viewer API

The simplest way to use viso is the `Viewer` builder:

```rust
use viso::Viewer;

Viewer::builder()
    .with_path("assets/models/4pnk.cif")
    .build()
    .run()
    .unwrap();
```

This handles window creation, engine initialization, input wiring, and the render loop. For customization:

```rust
use viso::{Viewer, VisoOptions};

Viewer::builder()
    .with_path("1ubq")
    .with_title("My Protein Viewer")
    .with_options(my_options)
    .build()
    .run()
    .unwrap();
```

## Under the Hood

The viewer is a thin wrapper around `VisoEngine`. Here's what it does each frame:

### Initialization

```rust
// Create engine (async wgpu init, uses pollster::block_on)
let mut engine = pollster::block_on(
    VisoEngine::new_with_path(window.clone(), (width, height), scale, &path)
)?;

// Kick off background mesh generation
engine.sync_scene_to_renderers(None);
```

### The Render Loop

```rust
// Each frame:
engine.update(dt);       // Advance animation, apply pending scene
match engine.render() {  // Full GPU pipeline
    Ok(()) => {}
    Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
        engine.resize(width, height);
    }
    Err(e) => log::error!("render error: {e:?}"),
}
window.request_redraw();
```

### Input Wiring

The viewer uses `InputProcessor` to translate winit events into commands:

```rust
let mut input = InputProcessor::new();

// On each winit event:
let event = InputEvent::CursorMoved { x, y };
if let Some(cmd) = input.handle_event(event, engine.hovered_target()) {
    let _ = engine.execute(cmd);
}
```

See [Handling Input](../integration/handling-input.md) for details.

## Embedding Without the Viewer

For embedding viso in your own application (dioxus, egui, web, etc.), use `VisoEngine` directly:

```rust
use viso::{VisoEngine, VisoCommand, InputProcessor, InputEvent};

// Create engine with your own surface
let mut engine = pollster::block_on(
    VisoEngine::new(window.clone(), (width, height), scale)
);

// Load entities
engine.load_entities(entities, "My Structure", true);
engine.sync_scene_to_renderers(None);

// Your render loop:
loop {
    engine.update(dt);
    engine.render()?;
}
```

See [Engine Lifecycle](../integration/engine-lifecycle.md) for the full integration guide.

## What's Different in foldit-rs

The standalone viewer is intentionally minimal. foldit-rs adds:

- **Multiple entity groups** with independent visibility and focus cycling
- **Backend integration** (Rosetta minimization, ML structure prediction) that streams coordinate updates
- **Animated transitions** between poses using the animation system
- **A webview UI** for controls, panels, and sequence display
- **Band and pull visualization** for constraint-based manipulation

These patterns are covered in the [Integration Guide](../integration/engine-lifecycle.md).
