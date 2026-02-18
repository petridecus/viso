# Dynamic Structure Updates

Viso is designed for live manipulation -- structures can be updated mid-session by computational backends (Rosetta energy minimization, ML structure prediction) or user actions (mutations, drag operations). This chapter covers the APIs and patterns for dynamic updates.

## Coordinate Updates

### Single Group

```rust
// Update protein coordinates in a specific group
engine.scene.update_group_protein_coords(group_id, new_coords);
engine.sync_scene_to_renderers(Some(AnimationAction::Wiggle));
```

### Combined Multi-Group Update (Rosetta)

When Rosetta operates on a multi-group session, coordinates come back as a single byte stream:

```rust
// Before sending to Rosetta: get combined coords
let result = engine.scene.combined_coords_for_backend();
// result.bytes -> send to Rosetta
// result.chain_ids_per_group -> keep for splitting the response

// After Rosetta responds: apply combined update
engine.scene.apply_combined_update(
    &response_bytes,
    &result.chain_ids_per_group,
)?;
engine.sync_scene_to_renderers(Some(AnimationAction::Wiggle));
```

### Replacing Entities

When the structure topology changes (e.g., ML prediction replaces backbone-only with full-atom):

```rust
if let Some(group) = engine.group_mut(group_id) {
    group.set_entities(new_entities);
    group.name = "Updated Structure".to_string();
    group.invalidate_render_cache();
}
engine.sync_scene_to_renderers(Some(AnimationAction::DiffusionFinalize));
```

## Animation Actions

Every update can specify an `AnimationAction` that controls the visual transition:

```rust
pub enum AnimationAction {
    Wiggle,           // Rosetta minimize -- smooth 300ms ease-out
    Shake,            // Rosetta rotamer pack -- smooth 300ms ease-out
    Mutation,         // Residue mutation -- collapse/expand effect
    Diffusion,        // ML intermediate -- fast 100ms linear
    DiffusionFinalize,// ML final result -- backbone lerp then sidechain expand
    Reveal,           // Prediction reveal -- cascading effect
    Load,             // New structure load -- instant snap
}
```

Each action maps to an animation behavior through `AnimationPreferences`. See [Animation System](../deep-dives/animation-system.md) for details.

## Backbone Animation

For updates that only change backbone coordinates:

```rust
engine.animate_to_pose(&new_backbone_chains, AnimationAction::Wiggle);
```

This sets a new animation target. The animator interpolates from the current visual position to the new target over the behavior's duration.

## Full Pose Animation (Backbone + Sidechains)

For updates that include sidechain data:

```rust
engine.animate_to_full_pose_with_action(
    &new_backbone_chains,
    &sidechain_data,        // SidechainData { positions, bonds, backbone_bonds, ... }
    &sidechain_atom_names,
    AnimationAction::Shake,
);
```

This handles:
- Capturing current visual positions as animation start (for smooth preemption)
- Setting new backbone and sidechain targets
- Coordinating backbone and sidechain animation timing
- Sheet surface offset adjustment

### Preemption

If a new update arrives while an animation is playing, the behavior's `PreemptionStrategy` determines what happens:

- **Restart** (default) -- start new animation from current visual position to new target
- **Ignore** -- ignore the new target until current animation completes
- **Blend** -- blend toward new target while maintaining velocity

Most behaviors use `Restart`, which gives responsive feedback during rapid Rosetta cycles.

## Targeted Per-Entity Animation

When only some groups should animate (e.g., a design group finalizes while the input structure stays still):

```rust
use std::collections::HashMap;

engine.sync_scene_to_renderers_targeted(
    HashMap::from([
        (design_group_id, AnimationAction::DiffusionFinalize),
    ])
);
```

Groups not in the map are snapped instantly to their current state, preventing unwanted animation of unchanged structures.

## Band and Pull Visualization

### Bands (Constraints)

Bands visualize distance constraints between atoms:

```rust
engine.update_bands(&[
    BandRenderInfo {
        endpoint_a: Vec3::new(10.0, 20.0, 30.0),
        endpoint_b: Vec3::new(15.0, 22.0, 28.0),
        strength: 1.0,
        target_length: 3.5,
        residue_idx: 42,
        band_type: BandType::Default,
        is_pull: false,
        is_push: false,
        is_disabled: false,
        is_space_pull: false,
        from_script: false,
    },
]);

// Clear all bands
engine.clear_bands();
```

Band visual properties:
- **Radius** scales with strength (0.1 to 0.4 angstroms)
- **Color** depends on band type: default (purple), backbone (yellow-orange), disulfide (yellow-green), H-bond (cyan)
- **Disabled bands** are gray

### Pulls (Active Drag)

A pull is a temporary constraint while the user drags a residue:

```rust
engine.update_pull(Some(&PullRenderInfo {
    atom_pos: atom_world_position,
    target_pos: mouse_world_position,
    residue_idx: 42,
}));

// Clear when drag ends
engine.clear_pull();
```

Pulls render as a purple cylinder from atom to mouse position with a cone/arrow at the mouse end.

## Real foldit-rs Patterns

### Rosetta Wiggle Cycle

```rust
// 1. Send coords to Rosetta
let combined = engine.scene.combined_coords_for_backend();
rosetta.minimize(combined.bytes);

// 2. Receive updated coords (via triple buffer, checked each frame)
let update = backend.try_recv();
if let BackendUpdate::RosettaCoords { coords_bytes, score, .. } = update {
    engine.scene.apply_combined_update(&coords_bytes, &chain_ids)?;

    // Cache per-residue scores for coloring
    if let Some(scores) = per_residue_scores {
        group.set_per_residue_scores(Some(scores));
    }

    engine.sync_scene_to_renderers(Some(AnimationAction::Wiggle));
}
```

### ML Structure Prediction (Streaming)

```rust
// Intermediate frames during prediction
fn on_ml_intermediate(engine: &mut Engine, entities: Vec<MoleculeEntity>, step: u32) {
    if let Some(group) = engine.group_mut(group_id) {
        group.set_entities(entities);
        group.name = format!("Predicting... ({}/{})", step, total_steps);
        group.invalidate_render_cache();
    }
    engine.sync_scene_to_renderers(Some(AnimationAction::Diffusion));
}

// Final result
fn on_ml_complete(engine: &mut Engine, entities: Vec<MoleculeEntity>) {
    if let Some(group) = engine.group_mut(group_id) {
        group.set_entities(entities);
        group.name = "Prediction Result".to_string();
        group.invalidate_render_cache();
    }
    engine.sync_scene_to_renderers_targeted(
        HashMap::from([(group_id, AnimationAction::DiffusionFinalize)])
    );
}
```

### RFDiffusion3 Design (Backbone Streaming)

RFD3 streams backbone-only intermediates, then delivers a full-atom final result:

```rust
// Intermediates: backbone only
let backbone_chains = positions_to_backbone_chains(&backbone_positions);
let coords = backbone_chains_to_coords(&backbone_chains);
let entities = split_into_entities(&coords);
engine.load_entities(entities, "Designing...", false);

// Final: full-atom with DiffusionFinalize animation
// BackboneThenExpand behavior: backbone lerps first, then sidechains expand
engine.sync_scene_to_renderers_targeted(
    HashMap::from([(design_id, AnimationAction::DiffusionFinalize)])
);
```

## Animation Control

```rust
engine.skip_animations();           // Jump to final state
engine.cancel_animations();         // Stay at current visual state
engine.set_animation_enabled(false); // Disable all animation (snap)
```
