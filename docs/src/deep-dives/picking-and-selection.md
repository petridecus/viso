# GPU Picking and Selection

Viso uses GPU-based picking to determine which residue is under the mouse cursor. This is faster and more accurate than CPU ray-casting, especially with complex molecular geometry.

## How Picking Works

### Offscreen Render Pass

The picking system renders all molecular geometry to an offscreen texture with format `R32Uint`. Instead of colors, each fragment writes a **residue ID** (the residue's global index + 1, where 0 means "no hit").

```
Main render: geometry → HDR color + normals + depth
Picking render: same geometry → R32Uint residue IDs + depth
```

The picking pass uses depth testing (`Less` compare with depth writes) so only the closest geometry's residue ID is stored.

### Geometry Types in Picking

The picking pass renders four types of geometry in order:

1. **Tubes** -- backbone coils (in ribbon mode, only coil segments; in tube mode, everything). Uses the `picking_mesh.wgsl` shader.
2. **Ribbons** -- helices and sheets (ribbon mode only). Uses the same mesh pipeline as tubes.
3. **Capsule sidechains** -- sidechain capsule impostors. Uses the `picking_capsule.wgsl` shader with a storage buffer of capsule instances.
4. **Ball-and-stick** -- ligand/ion sphere and bond proxies (degenerate capsules). Uses the same capsule pipeline.

### Non-Blocking Readback

Reading data back from the GPU is expensive if done synchronously. Viso uses a two-frame pipeline:

**Frame N:**
1. The picking pass renders to the offscreen texture
2. A single pixel at the mouse position is copied to a staging buffer (256 bytes minimum, aligned for wgpu)
3. `start_readback()` initiates an async buffer map without blocking

**Frame N+1:**
1. `complete_readback()` polls the wgpu device without blocking
2. If the map callback has fired (signaled via `AtomicBool`), the mapped data is read:
   - Read 4 bytes as `u32`
   - If 0: no residue hit (mouse is on background)
   - Otherwise: residue index = value - 1
3. The staging buffer is unmapped
4. Result is cached in `hovered_residue`

If the readback isn't ready yet, the previous frame's cached value is used. This means hover feedback is at most one frame behind, which is imperceptible.

```rust
// In the render method:
self.picking.render(encoder, /* geometry buffers... */, mouse_x, mouse_y);
// After queue.submit():
self.picking.start_readback();
// Next frame, before render:
self.picking.complete_readback(&device);
```

## Selection Buffer

The `SelectionBuffer` is a GPU storage buffer containing a bit-array of selected residues. It's bound to all molecular renderers so shaders can highlight selected residues.

### Bit Packing

Selection is stored as u32 words with one bit per residue:

```
Word 0: residues 0-31   (bit 0 = residue 0, bit 1 = residue 1, ...)
Word 1: residues 32-63
Word 2: residues 64-95
...
```

### Updating Selection

```rust
// Upload current selection to GPU
selection_buffer.update(&queue, &selected_residues);
```

This clears the bit array and sets bits for each selected residue index.

### Dynamic Capacity

The buffer grows as needed:

```rust
selection_buffer.ensure_capacity(&device, total_residue_count);
```

If the current buffer is too small, a new buffer and bind group are created.

## Click Handling

### Single Click

```rust
let selection_changed = picking.handle_click(shift_held);
```

- **Click on residue, no shift**: clears selection, selects clicked residue
- **Click on residue, shift held**: toggles the residue (adds if absent, removes if present)
- **Click on background**: clears all selection

### Double Click (Secondary Structure Segment)

A double-click selects all residues in the same contiguous secondary structure segment:

1. Look up the SS type of the clicked residue
2. Walk backward through `cached_ss_types` until the type changes → segment start
3. Walk forward until the type changes → segment end
4. Select all residues in [start, end]
5. If shift held, add to existing selection; otherwise replace

### Triple Click (Chain)

A triple-click selects all residues in the same chain:

1. Walk through `backbone_chains` to find which chain contains the residue
2. Calculate the global residue range for that chain
3. Select all residues in the range
4. If shift held, add to existing selection; otherwise replace

### Click Type Detection

The input state machine tracks timing between clicks:

- Clicks within a threshold on the same residue increment the click counter
- The counter determines single (1), double (2), or triple (3) click
- If the mouse moved between press and release, it's classified as a drag (no selection)

## Selection in Shaders

All molecular renderers receive the selection bind group. In the fragment shader:

```wgsl
// Check if this fragment's residue is selected
let word_idx = residue_idx / 32u;
let bit_idx = residue_idx % 32u;
let is_selected = (selection_data[word_idx] >> bit_idx) & 1u;

if is_selected == 1u {
    // Apply selection highlight (e.g., brighten color)
}
```

The hover effect uses the `hovered_residue` uniform -- the shader checks if the fragment's residue index matches the hovered residue and applies a highlight.

## Querying Selection State

```rust
// Currently hovered residue (-1 if none)
let hovered: i32 = engine.picking.hovered_residue;

// Currently selected residues
let selected: &Vec<i32> = &engine.picking.selected_residues;

// Clear everything
engine.clear_selection();
```
