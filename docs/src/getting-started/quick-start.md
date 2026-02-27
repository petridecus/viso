# Quick Start

Viso is a library first. With no feature flags enabled, it gives you `VisoEngine` — a self-contained rendering engine you embed in your own event loop. The optional `viewer` feature adds a standalone winit window for quick prototyping.

## Using Viso as a Library

Add viso to your `Cargo.toml` with no extra features:

```toml
[dependencies]
viso = { path = "../viso" }  # or git/registry
pollster = "0.4"             # for blocking on async GPU init
```

The minimal integration has three steps: create an engine, load structure data, and run a render loop.

### 1. Create the Engine

`RenderContext` initializes the wgpu device and surface. Pass it to `VisoEngine::new_empty` to get an engine with no entities loaded:

```rust
use viso::{VisoEngine, RenderContext};

let context = pollster::block_on(
    RenderContext::new(window.clone(), (width, height))
)?;
let mut engine = VisoEngine::new_empty(context)?;
```

Or load a structure file directly:

```rust
let mut engine = pollster::block_on(
    VisoEngine::new_with_path(window.clone(), (width, height), scale_factor, "path/to/structure.cif")
)?;
```

`new_with_path` accepts `.cif`, `.pdb`, and `.bcif` files. It parses the file, populates the scene, and kicks off background mesh generation.

### 2. Load Entities

If you used `new_empty`, load entities yourself:

```rust
// entities: Vec<MoleculeEntity> from foldit_conv or your own pipeline
let entity_ids = engine.load_entities(entities, true); // true = fit camera
engine.sync_scene_to_renderers(None);
```

`sync_scene_to_renderers` submits the scene to the background thread for mesh generation. Pass `None` to snap to the current state with no animation.

### 3. Render Loop

Each frame, call `update` then `render`:

```rust
engine.update(dt);       // advance animation, apply pending meshes
match engine.render() {  // full GPU pipeline → 2D texture
    Ok(()) => {}
    Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
        engine.resize(width, height);
    }
    Err(e) => log::error!("render error: {e:?}"),
}
```

That's it. The engine handles background mesh generation, animation, and the full post-processing pipeline internally. You own the event loop and the window.

### Input (Optional)

`InputProcessor` is a convenience layer that translates raw input events into `VisoCommand` values. You can use it or wire commands directly:

```rust
use viso::{InputProcessor, InputEvent};

let mut input = InputProcessor::new();

// In your event handler:
let event = InputEvent::CursorMoved { x, y };
if let Some(cmd) = input.handle_event(event, engine.hovered_target()) {
    let _ = engine.execute(cmd);
}
```

## Standalone Viewer

For quick prototyping, enable the `viewer` feature:

```toml
[dependencies]
viso = { path = "../viso", features = ["viewer"] }
```

This pulls in `winit` and `pollster` and gives you `Viewer`, which handles window creation, the event loop, input wiring, and the render loop:

```rust
use viso::Viewer;

Viewer::builder()
    .with_path("assets/models/4pnk.cif")
    .build()
    .run()?;
```

### Running the CLI

The `binary` feature (enabled by default) builds a standalone CLI that can download structures from RCSB by PDB code:

```sh
cargo run -p viso -- 1ubq
```

This downloads the CIF file, caches it in `assets/models/`, and opens a viewer window. You can also pass a local file path:

```sh
cargo run -p viso -- path/to/structure.cif
```

## What Foldit Adds

The standalone viewer is intentionally minimal. Foldit adds:

- **Multiple entity groups** with independent visibility and focus cycling
- **Backend integration** (Rosetta minimization, ML structure prediction) that streams coordinate updates
- **Animated transitions** between poses using the animation system
- **A webview UI** for controls, panels, and sequence display
- **Band and pull visualization** for constraint-based manipulation
