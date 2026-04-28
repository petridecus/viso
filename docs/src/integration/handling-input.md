# Handling Input

Viso provides an `InputProcessor` that translates raw mouse/keyboard
events into `VisoCommand` values. The engine executes commands via
`engine.execute(cmd)`.

## Architecture

```
Platform events (winit, web, etc.)
    │
    ▼
InputEvent (platform-agnostic)
    │
    ▼
InputProcessor
    │  translates to
    ▼
VisoCommand
    │
    ▼
engine.execute(cmd)
```

`InputProcessor` is optional convenience. Consumers who handle their
own input can construct `VisoCommand` values directly and skip
`InputProcessor` entirely.

## InputEvent

The platform-agnostic input enum:

```rust
pub enum InputEvent {
    CursorMoved { x: f32, y: f32 },
    MouseButton { button: MouseButton, pressed: bool },
    Scroll { delta: f32 },
    ModifiersChanged { shift: bool },
}
```

Your windowing layer converts platform events into these variants.

## Wiring Input (winit example)

From viso's standalone viewer:

```rust
struct ViewerShell {
    engine: Option<VisoEngine>,
    input: InputProcessor,
    // ...
}

impl ViewerShell {
    fn dispatch_input(&mut self, event: InputEvent) {
        let Some(engine) = &mut self.engine else { return };

        // Update cursor for GPU picking
        if let InputEvent::CursorMoved { x, y } = event {
            engine.set_cursor_pos(x, y);
        }

        // Translate event to command and execute
        if let Some(cmd) =
            self.input.handle_event(event, engine.hovered_target())
        {
            let _ = engine.execute(cmd);
        }
    }
}
```

### Mouse movement

```rust
WindowEvent::CursorMoved { position, .. } => {
    dispatch_input(InputEvent::CursorMoved {
        x: position.x as f32,
        y: position.y as f32,
    });
}
```

### Scroll (zoom)

```rust
WindowEvent::MouseWheel { delta, .. } => {
    let scroll_delta = match delta {
        MouseScrollDelta::LineDelta(_, y) => y,
        MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
    };
    dispatch_input(InputEvent::Scroll { delta: scroll_delta });
}
```

### Click (selection)

```rust
WindowEvent::MouseInput { button, state, .. } => {
    dispatch_input(InputEvent::MouseButton {
        button: MouseButton::from(button),
        pressed: state == ElementState::Pressed,
    });
}
```

### Modifier keys

```rust
WindowEvent::ModifiersChanged(modifiers) => {
    dispatch_input(InputEvent::ModifiersChanged {
        shift: modifiers.state().shift_key(),
    });
}
```

## Keyboard Input

Keyboard events go through `InputProcessor::handle_key_press`, which
looks up the physical key in the configurable `KeyBindings` map:

```rust
WindowEvent::KeyboardInput { event, .. } => {
    if event.state == ElementState::Pressed {
        if let PhysicalKey::Code(code) = event.physical_key {
            let key_str = format!("{code:?}");
            if let Some(cmd) = input.handle_key_press(&key_str) {
                let _ = engine.execute(cmd);
            }
        }
    }
}
```

Display toggles for ambient types (water/ion/solvent) and lipid mode
go through `VisoCommand::SetTypeVisibility` and
`VisoCommand::CycleLipidMode` respectively. The default key bindings
already wire these up — pressing `U`, `I`, `O`, or `L` produces the
right command without any extra code in the viewer.

## VisoCommand

The full action vocabulary:

```rust
pub enum VisoCommand {
    // Camera
    RecenterCamera,
    ToggleAutoRotate,
    RotateCamera { delta: Vec2 },
    PanCamera { delta: Vec2 },
    Zoom { delta: f32 },

    // Focus
    CycleFocus,
    ResetFocus,

    // Playback
    ToggleTrajectory,

    // Selection
    ClearSelection,
    SelectResidue { index: i32, extend: bool },
    SelectSegment { index: i32, extend: bool },
    SelectChain { index: i32, extend: bool },

    // Entity management
    FocusEntity { id: u32 },
    ToggleEntityVisibility { id: u32 },
    RemoveEntity { id: u32 },

    // Display toggles
    SetTypeVisibility { mol_type: MoleculeType, visible: Option<bool> },
    CycleLipidMode,
}
```

`engine.execute(cmd)` returns a `CommandOutcome` describing what
changed (`SelectionChanged`, `FocusChanged`, `VisibilityChanged`,
`NoEffect`, or `Unhandled`). `RemoveEntity` is `Unhandled` when sent
to the engine directly — it must be routed through `VisoApp` (or the
host that owns the `Assembly`) so the assembly is mutated and the new
snapshot is pushed via `engine.set_assembly`.

## Click Detection

`InputProcessor` tracks click timing internally to distinguish:

| Click Type | Resulting Command |
|-----------|-------------------|
| Single click on residue | `SelectResidue { index, extend: false }` |
| Shift + click on residue | `SelectResidue { index, extend: true }` |
| Double click | `SelectSegment { index, extend }` |
| Triple click | `SelectChain { index, extend }` |
| Click on background | `ClearSelection` |
| Drag (moved after press) | `RotateCamera` or `PanCamera` (no selection) |

Shift + drag produces `PanCamera` instead of `RotateCamera`.

## KeyBindings

Customizable key-to-command mapping, serde-serializable. Keys use the
`winit::keyboard::KeyCode` debug format (`"KeyQ"`, `"Tab"`,
`"Backquote"`, etc.):

```rust
let mut input = InputProcessor::new();
// Default bindings are pre-loaded.

// Or with a custom map:
let bindings: KeyBindings = toml::from_str(&toml_str)?;
let mut input = InputProcessor::with_key_bindings(bindings);
```

Default keybindings:

| Key | Action |
|-----|--------|
| `KeyQ` | Recenter camera on focus |
| `KeyT` | Toggle trajectory playback |
| `Tab` | Cycle focus |
| `KeyR` | Toggle auto-rotate |
| `Backquote` | Reset focus to session |
| `Escape` | Clear selection |
| `KeyI` | Toggle ion visibility |
| `KeyU` | Toggle water visibility |
| `KeyO` | Toggle solvent visibility |
| `KeyL` | Cycle lipid display mode |

## Skipping InputProcessor

For web embeds or custom hosts, construct commands directly:

```rust
// Rotate camera by 5 degrees worth of pixel delta
engine.execute(VisoCommand::RotateCamera {
    delta: Vec2::new(5.0, 0.0),
});

// Select residue 42
engine.execute(VisoCommand::SelectResidue {
    index: 42,
    extend: false,
});
```
