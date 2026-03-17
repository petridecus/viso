# Dynamic Structure Updates

Viso is designed for live manipulation -- structures can be updated mid-session by computational backends (Rosetta energy minimization, ML structure prediction) or user actions (mutations, drag operations). This chapter covers the APIs and patterns for dynamic updates.

## Per-Entity Coordinate Updates

The primary API for updating structure coordinates is `update_entity_coords()`, which updates the engine's source-of-truth entities first, then propagates to the scene and animation system:

```rust
// Update a specific entity's coordinates with animation
engine.update_entity_coords(entity_id, new_coords, Transition::smooth());
```

The engine looks up the entity's assigned behavior (or uses the provided transition) and dispatches to the per-entity animation system.

### Per-Entity Behavior Control

Before updating an entity, callers can assign a specific animation behavior:

```rust
// Set behavior BEFORE updating coordinates
engine.set_entity_behavior(entity_id, Transition::collapse_expand(
    Duration::from_millis(200),
    Duration::from_millis(300),
));

// Update coordinates — engine uses CollapseExpand for this entity
engine.update_entity_coords(entity_id, new_coords, Transition::smooth());

// After the animation completes, the behavior remains set.
// To revert to default:
engine.clear_entity_behavior(entity_id);
```

## Transitions

Every update can specify a `Transition` that controls the visual animation:

```rust
// Instant snap (for loading, no animation)
Transition::snap()

// Standard smooth interpolation (300ms ease-out)
Transition::smooth()

// Two-phase: sidechains collapse to CA, backbone moves, sidechains expand
Transition::collapse_expand(
    Duration::from_millis(300),
    Duration::from_millis(300),
)

// Two-phase: backbone animates first, then sidechains expand
Transition::backbone_then_expand(
    Duration::from_millis(400),
    Duration::from_millis(600),
)

// Builder flags
Transition::collapse_expand(
    Duration::from_millis(200),
    Duration::from_millis(300),
)
    .allowing_size_change()
    .suppressing_initial_sidechains()
```

See [Animation System](../deep-dives/animation-system.md) for details on preset behaviors and phase evaluation.

### Preemption

If a new update arrives while an animation is playing, the current visual position becomes the new animation's start state and the timer resets. This provides responsive feedback during rapid update cycles (e.g., Rosetta wiggle).

## Band and Pull Visualization

### Bands (Constraints)

Bands visualize distance constraints between atoms:

```rust
engine.execute(VisoCommand::UpdateBands {
    bands: vec![
        BandInfo {
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
    ],
});
```

Band visual properties:
- **Radius** scales with strength (0.1 to 0.4 angstroms)
- **Color** depends on band type: default (purple), backbone (yellow-orange), disulfide (yellow-green), H-bond (cyan)
- **Disabled bands** are gray

### Pulls (Active Drag)

A pull is a temporary constraint while the user drags a residue:

```rust
engine.execute(VisoCommand::UpdatePull {
    pull: Some(PullInfo {
        atom_pos: atom_world_position,
        target_pos: mouse_world_position,
        residue_idx: 42,
    }),
});

// Clear when drag ends
engine.execute(VisoCommand::UpdatePull { pull: None });
```

Pulls render as a purple cylinder from atom to mouse position with a cone/arrow at the mouse end.

## Animation Control

```rust
animator.skip();     // Jump to final state
animator.cancel();   // Stay at current visual state
animator.set_enabled(false); // Disable all animation (snap)
```
