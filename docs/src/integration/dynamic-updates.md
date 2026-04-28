# Dynamic Structure Updates

Viso is designed for live manipulation — structures can be updated
mid-session by computational backends (Rosetta energy minimization,
ML structure prediction) or user actions (mutations, drag operations).

All structural mutations happen on **your** `molex::Assembly`. After
each batch of changes, push the new snapshot to the engine via
`engine.set_assembly(Arc::new(assembly.clone()))`. The engine itself
is read-only with respect to structural state.

## Per-Entity Coordinate Updates

To stream new atom positions for an existing entity, mutate the
relevant entity's coordinates on your `Assembly` (using molex's
`update_protein_entities` codec helper, or
`assembly.update_positions(eid, &coords)` for direct position updates),
then re-publish:

```rust
use std::sync::Arc;
use molex::ops::codec::update_protein_entities;

// Apply caller-provided Coords through molex's shared codec so the
// path matches the byte-format pipeline.
let mut entities = vec![assembly.entity(eid).unwrap().clone()];
update_protein_entities(&mut entities, &new_coords);
if let Some(updated) = entities.into_iter().next() {
    assembly.remove_entity(eid);
    assembly.add_entity(updated);
}

engine.set_assembly(Arc::new(assembly.clone()));
```

To make the next sync animate (instead of snapping), queue a per-entity
behavior override before re-publishing:

```rust
engine.set_entity_behavior(entity_id, Transition::smooth());
engine.set_assembly(Arc::new(assembly.clone()));
```

The engine queues the transition for the affected entity on its next
sync, regardless of whether the override was set before or after the
`set_assembly` call (transitions are picked up in `update`).

### Per-Entity Behavior Overrides

Override the default transition for a specific entity. Once set, every
subsequent sync involving that entity uses the override (until
cleared):

```rust
let eid = engine.entity_id(raw_id).expect("known entity");

engine.set_entity_behavior(eid, Transition::collapse_expand(
    Duration::from_millis(200),
    Duration::from_millis(300),
));

// Subsequent re-publishes that touch this entity will use
// collapse_expand instead of the default snap.
engine.set_assembly(Arc::new(assembly.clone()));

// Revert to default:
engine.clear_entity_behavior(eid);
```

## Transitions

Every update can specify a `Transition` controlling the visual
animation:

```rust
// Instant snap (no animation; used internally for initial loads
// and trajectory frames).
Transition::snap()

// Standard smooth interpolation (300ms cubic-hermite ease-out).
Transition::smooth()

// Two-phase: sidechains collapse to CA, then expand. For mutations.
Transition::collapse_expand(
    Duration::from_millis(300),
    Duration::from_millis(300),
)

// Two-phase: backbone moves first with sidechains hidden, then
// sidechains expand into place.
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

See [Animation System](../deep-dives/animation-system.md) for details
on the data-driven phase model.

### Preemption

If a new update arrives while an animation is playing, the current
visual position becomes the new animation's start state and the timer
resets. This provides responsive feedback during rapid update cycles
(e.g. Rosetta wiggle).

## Constraint Visualization (Bands and Pulls)

Bands and pulls are not commands — they are stored constraint specs
that the engine resolves to world-space positions every frame so they
auto-track animated atoms.

### Bands

A `BandInfo` references atoms structurally rather than by world-space
position:

```rust
use viso::{AtomRef, BandInfo, BandTarget, BandType};

let band = BandInfo {
    anchor_a: AtomRef { residue: 42, atom_name: "CA".into() },
    anchor_b: BandTarget::Atom(AtomRef {
        residue: 87,
        atom_name: "CA".into(),
    }),
    strength: 1.0,
    target_length: 3.5,
    band_type: Some(BandType::Disulfide),
    is_pull: false,
    is_push: false,
    is_disabled: false,
    from_script: false,
};
```

`BandTarget::Position(Vec3)` anchors one end to a fixed world-space
point (used for "space pulls"). `band_type` set to `None` lets the
engine auto-detect the type from `target_length`.

Visual properties:
- **Radius** scales with `strength` (0.1 to 0.4 Å)
- **Color** depends on `band_type`: default (purple), backbone
  (yellow-orange), disulfide (yellow-green), H-bond (cyan)
- **Disabled bands** are gray
- **Script-authored bands** (`from_script: true`) render dimmer

### Pulls

A `PullInfo` is a single active drag constraint. The atom is
referenced structurally; the target is given in screen-space (physical
pixels) and unprojected at the atom's depth each frame so the drag
stays parallel to the camera plane:

```rust
use viso::{AtomRef, PullInfo};

let pull = PullInfo {
    atom: AtomRef { residue: 42, atom_name: "CA".into() },
    screen_target: (mouse_x, mouse_y),
};
```

Pulls render as a purple cylinder from the atom to the target with a
cone arrow head at the target end.

Update band and pull specs through the engine. Both methods replace
the previous specs and re-resolve immediately:

```rust
engine.update_bands(vec![band1, band2]);
engine.update_pull(Some(pull));
engine.update_pull(None); // clear when drag ends
```

The engine resolves stored specs to world-space positions every frame,
so bands and pulls track animated atoms automatically.
