# Handling Input

Viso's input system translates mouse and keyboard events into camera manipulation and residue selection. The engine exposes simple methods that you wire to your windowing system's events.

## Mouse Events

### Movement (Rotation and Panning)

```rust
// Track mouse position yourself, then pass deltas
let delta_x = new_x - last_x;
let delta_y = new_y - last_y;

engine.handle_mouse_move(delta_x, delta_y);
engine.handle_mouse_position(new_x, new_y);
```

`handle_mouse_move` rotates or pans the camera:
- **Default drag** -- arcball rotation around the focus point
- **Shift + drag** -- panning (translates the focus point)
- Rotation and panning only activate when the mouse was pressed on the background (not on a residue), preventing accidental camera movement during selection.

`handle_mouse_position` updates the mouse coordinates for GPU picking. The picking system uses this position in the next render pass to determine which residue is under the cursor.

### Scroll (Zoom)

```rust
// Line-based scroll (most mice)
engine.handle_mouse_wheel(scroll_y);

// Pixel-based scroll (trackpads) -- scale appropriately
engine.handle_mouse_wheel(pixel_y * 0.01);
```

Zoom adjusts the orbital distance from the focus point, clamped to [1.0, 1000.0].

### Click (Selection)

Click handling is split into press and release:

```rust
// Mouse down -- records what's under the cursor
engine.handle_mouse_button(MouseButton::Left, true);

// Mouse up -- processes selection based on click type
engine.handle_mouse_button(MouseButton::Left, false);
let selection_changed = engine.handle_mouse_up();
```

The input system tracks click timing internally to distinguish:

| Click Type | Action |
|-----------|--------|
| Single click | Select the residue under cursor |
| Shift + click | Toggle residue in multi-selection |
| Double click | Select entire secondary structure segment |
| Triple click | Select entire chain |
| Click on background | Clear selection |
| Drag (moved after press) | No selection (was a camera operation) |

### How Click Detection Works

On mouse down, the engine records which residue (if any) is under the cursor via GPU picking. On mouse up, the input state machine classifies the click:

1. If the mouse moved significantly between press and release, it's a drag -- no selection action.
2. If the mouse didn't move, timing determines the click count:
   - Single click within the double-click threshold
   - Double click if two clicks occur rapidly on the same residue
   - Triple click if three clicks occur rapidly on the same residue

## Modifier Keys

```rust
// From winit ModifiersChanged event
engine.update_modifiers(modifiers.state());

// Or set shift directly (for non-winit integrations)
engine.set_shift_pressed(true);
```

Shift affects two behaviors:
- **Camera**: shift + drag pans instead of rotating
- **Selection**: shift + click adds to or toggles the selection instead of replacing it

## Selection Queries

After a click, you can query the current selection state:

```rust
// Currently hovered residue (-1 if none)
let hovered = engine.picking.hovered_residue;

// Currently selected residues
let selected: &Vec<i32> = &engine.picking.selected_residues;
```

### Clearing Selection

```rust
engine.clear_selection();
```

This clears both the selected residue list and the hover state.

## Double-Click: Secondary Structure Segments

A double-click selects all residues in the same contiguous secondary structure segment:

1. Find the SS type of the clicked residue
2. Walk backward to find the segment start (where SS type changes)
3. Walk forward to find the segment end
4. Select all residues in [start, end]

With shift held, the segment is added to the existing selection.

## Triple-Click: Chain Selection

A triple-click selects all residues in the same chain:

1. Walk the backbone chains to find which chain contains the clicked residue
2. Select all residues in that chain's range

With shift held, the chain is added to the existing selection.

## Keyboard Events

The standalone viewer handles two keys:

```rust
Key::Character("w" | "W") => engine.toggle_waters(),
Key::Named(NamedKey::Escape) => engine.clear_selection(),
```

foldit-rs extends this with configurable keybindings (see [Options and Presets](../configuration/options-and-presets.md)):

| Key | Action |
|-----|--------|
| Q | Recenter camera |
| T | Toggle trajectory playback |
| I | Toggle ions |
| U | Toggle waters |
| O | Toggle solvent |
| L | Toggle lipids |
| Tab | Cycle focus |
| R | Toggle auto-rotate |
| \` | Reset focus to session |
| Escape | Cancel / clear selection |
