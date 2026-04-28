# Animation System

Viso's animation system manages smooth visual transitions when protein structures change. It is fully data-driven — a `Transition` describes the animation as a sequence of phases, and an `AnimationRunner` evaluates those phases each frame.

## Data-Driven Architecture

```
Transition              →    AnimationRunner
  (phases + flags)            (evaluates phases per frame)
```

1. **Transition** — a struct holding a `Vec<AnimationPhase>` plus metadata flags (size-change permission, sidechain suppression). Each phase has an easing function, duration, lerp range, and sidechain visibility flag.
2. **AnimationRunner** — executes a single animation from start to target states, advancing through phases sequentially.

There are no trait objects or behavior types. The consumer constructs `Transition` values using preset constructors, and the runner evaluates the phase sequence directly.

## Transition

`Transition` is the only animation type in the public API. Construct
it via preset constructors and tune with builder methods:

```rust
pub struct Transition {
    pub allows_size_change: bool,
    pub suppress_initial_sidechains: bool,
    // (phases + name are internal)
}

// Preset constructors
Transition::snap()       // Instant, allows resize
Transition::smooth()     // 300ms cubic hermite ease-out (also Default)
Transition::collapse_expand(collapse_dur, expand_dur)
Transition::backbone_then_expand(backbone_dur, expand_dur)
Transition::cascade(base_dur, delay_per_residue)

// Total duration helper (sum across phases)
let d: Duration = transition.total_duration();

// Builder methods
Transition::collapse_expand(
    Duration::from_millis(200),
    Duration::from_millis(300),
)
    .allowing_size_change()
    .suppressing_initial_sidechains()
```

## AnimationPhase (internal)

Each phase in a transition defines a segment of the animation:

```rust
pub(crate) struct AnimationPhase {
    pub(crate) easing: EasingFunction,
    pub(crate) duration: Duration,
    pub(crate) lerp_start: f32,    // e.g. 0.0
    pub(crate) lerp_end: f32,      // e.g. 0.4
    pub(crate) include_sidechains: bool,
}
```

`AnimationPhase` is `pub(crate)` — consumers don't construct it
directly; they use the preset constructors above. The runner maps raw
progress (0→1 over total duration) through the phase sequence, and
each phase applies its own easing within its lerp range.

## Preset Behaviors

### Snap

Instant transition. Duration is zero. Used for initial loads where animation would delay the first meaningful frame. Also used internally when trajectory frames are fed through the animation pipeline.

### Smooth (Default)

Standard eased lerp between start and target. 300ms with cubic hermite ease-out (`CubicHermite { c1: 0.33, c2: 1.0 }`). Good for incremental changes where start and target are close.

### Collapse/Expand

Two-phase animation for mutations:

1. **Collapse phase** — sidechain atoms collapse toward the backbone CA position (QuadraticIn easing)
2. **Expand phase** — new sidechain atoms expand outward from CA to their final positions (QuadraticOut easing)

```rust
Transition::collapse_expand(
    Duration::from_millis(300),  // Collapse duration
    Duration::from_millis(300),  // Expand duration
)
```

Collapse-to-CA is handled at animation setup time — when `allows_size_change` is true, the runner's start sidechain positions are written as CA coordinates so the lerp expands them outward into their target positions.

### Backbone Then Expand

Two-phase animation for transitions where sidechains should appear after backbone settles:

1. **Backbone phase** — backbone atoms lerp to final positions while sidechains are hidden
2. **Expand phase** — sidechain atoms expand from collapsed (at CA) to final positions

```rust
Transition::backbone_then_expand(
    Duration::from_millis(400),  // Backbone lerp duration
    Duration::from_millis(600),  // Sidechain expand duration
)
```

Uses `include_sidechains: false` on the first phase to hide sidechains during backbone movement, preventing visual artifacts when new atoms appear before the backbone has settled.

### Cascade

Staggered per-residue wave animation (QuadraticOut easing):

```rust
Transition::cascade(
    Duration::from_millis(500),  // Base duration per residue
    Duration::from_millis(5),    // Delay between residues
)
```

Note: per-residue staggering is not yet integrated into the runner — currently animates all residues with the same timing.

## Per-Entity Animation

Each entity gets its own animation runner with independent timing.
`StructureAnimator` (private) manages a `HashMap<EntityId, …>` of
per-entity runners and writes interpolated atom positions into the
engine's `EntityPositions` each frame.

The mutation surface lives on `VisoApp` (`update_entity_coords`,
`update_entity`, `update_entities`, `sync_entities`) — each call sets
the new target coordinates and queues a per-entity `Transition` for
the engine's next sync. Per-entity behavior overrides
(`engine.set_entity_behavior`) take precedence over the supplied
default transition.

### How It Works

1. The host mutates the `Assembly` and pushes the new snapshot via
   `engine.set_assembly`; pending per-entity transitions are stored on
   the engine's `AnimationState`.
2. On the next `engine.update()`, the engine rederives per-entity
   state from the new snapshot. For each entity that has a pending
   transition, an `AnimationRunner` is created with the start/target
   backbone positions and the transition's phases.
3. Each frame, the runner advances; interpolated positions are
   written into `EntityPositions`. Sidechain positions are
   interpolated with the same eased `t` as backbone.
4. When a runner completes (progress ≥ 1.0), the entity snaps to
   target and the runner is removed.

### Preemption

When a new target arrives while an entity is mid-animation:

- The current interpolated position becomes the new animation's start
  state.
- The previous animation's sidechain positions are captured for
  smooth handoff (when atom counts match).
- A new runner replaces the old one with the new target.

This provides responsive feedback during rapid update cycles (e.g.
Rosetta wiggle).

## Sidechain Animation

Sidechain positions are stored alongside backbone start/target arrays
and lerped with the same eased `t`. The animator writes interpolated
sidechain positions each frame so renderers and constraint resolution
can read them without recomputing.

Specialized sidechain behaviors:

- **Standard lerp** — for smooth transitions, sidechains lerp
  alongside backbone.
- **Collapse toward CA** — for mutations, start positions are set to
  the CA position at setup time; the runner's normal lerp handles
  the expansion.
- **Hidden during backbone phase** — multi-phase transitions use
  `include_sidechains: false` on early phases.

## Trajectory Playback

DCD trajectory frames are fed through the standard animation
pipeline. `TrajectoryPlayer` (in `engine/trajectory.rs`) is a frame
sequencer with no animation dependencies. Each frame it produces is
applied through the same path used for `Transition::snap()`, so
trajectory and structural animation share a single code path in the
engine's `tick_animation`.

Load a trajectory bound to the first visible protein entity:

```rust
engine.load_trajectory(Path::new("path/to/traj.dcd"));
engine.execute(VisoCommand::ToggleTrajectory); // play/pause
let has = engine.has_trajectory();
```

## Easing Functions

Available in `util/easing.rs`:

| Function | Description |
|----------|-------------|
| `Linear` | No easing |
| `QuadraticIn` | Slow start, fast end |
| `QuadraticOut` | Fast start, slow end |
| `SqrtOut` | Fast start, gradual slow |
| `CubicHermite { c1, c2 }` | Configurable control points (default: ease-out) |

All functions evaluate in <100ns and clamp input to [0, 1].

## Disabling Animation

Use `Transition::snap()` per-update, or set a `snap` per-entity
behavior so every subsequent update is instantaneous:

```rust
let eid = engine.entity_id(raw_id).expect("known entity");
engine.set_entity_behavior(eid, Transition::snap());
```
