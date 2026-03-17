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

```rust
pub struct Transition {
    pub phases: Vec<AnimationPhase>,
    pub allows_size_change: bool,
    pub suppress_initial_sidechains: bool,
    pub name: &'static str,
}

// Preset constructors
Transition::snap()       // Instant, allows resize
Transition::smooth()     // 300ms cubic hermite ease-out
Transition::collapse_expand(collapse_dur, expand_dur)
Transition::backbone_then_expand(backbone_dur, expand_dur)
Transition::cascade(base_dur, delay_per_residue)
Transition::default()    // Same as smooth()

// Builder methods
Transition::collapse_expand(
    Duration::from_millis(200),
    Duration::from_millis(300),
)
    .allowing_size_change()
    .suppressing_initial_sidechains()
```

## AnimationPhase

Each phase in a transition defines a segment of the animation:

```rust
pub struct AnimationPhase {
    pub easing: EasingFunction,
    pub duration: Duration,
    pub lerp_start: f32,    // e.g. 0.0
    pub lerp_end: f32,      // e.g. 0.4
    pub include_sidechains: bool,
}
```

The runner maps raw progress (0→1 over total duration) through the phase sequence. Each phase applies its own easing within its lerp range.

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

Collapse-to-CA is handled at animation setup time — the `SidechainAnimPositions` start positions are set to CA coordinates when `allows_size_change` is true.

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

Each entity gets its own `AnimationRunner` with independent timing. The `StructureAnimator` manages a `HashMap<u32, EntityAnimationState>` of per-entity runners and aggregates their interpolated output each frame.

```rust
// The engine dispatches per-entity:
animator.animate_entity(
    &range,          // EntityResidueRange (includes entity_id)
    &backbone_chains,
    &transition,
    sidechain_data,  // Option<SidechainAnimPositions>
);

// Each frame:
let still_animating = animator.update(Instant::now());
let visual_backbone = animator.get_backbone();
```

### How It Works

1. `animate_entity()` builds per-residue `ResidueAnimationData` (start/target backbone positions) for the entity's residue range
2. An `AnimationRunner` is created with those residues, the transition's phases, and optional sidechain positions
3. Each frame, `update()` calls `interpolate_residues()` on each runner, which returns an iterator of `(residue_idx, lerped_visual)` pairs
4. Sidechain positions are interpolated with the same `eased_t` as backbone
5. When a runner completes (progress >= 1.0), the entity's residues are snapped to target and the runner is removed

### Preemption

When a new target arrives while an entity is mid-animation:

- The current interpolated position becomes the new animation's start state
- The previous animation's sidechain positions are captured for smooth handoff (when atom counts match)
- A new runner replaces the old one with the new target

This provides responsive feedback during rapid update cycles (e.g., Rosetta wiggle).

## ResidueVisualState

Each residue's visual state during animation:

```rust
pub struct ResidueVisualState {
    pub backbone: [Vec3; 3],  // N, CA, C positions
}
```

Interpolation lerps backbone positions linearly, with the easing applied via the phase's easing function.

## Sidechain Animation

Sidechain positions are stored as `SidechainAnimPositions` (start + target `Vec<Vec3>`) and lerped with the same `eased_t` as backbone. The animator pre-computes interpolated sidechain positions each frame so queries can read them without recomputing.

Specialized sidechain behaviors:

- **Standard lerp** — for smooth transitions, sidechains lerp alongside backbone
- **Collapse toward CA** — for mutations, start positions are set to the CA position at setup time; the runner's normal lerp handles the expansion
- **Hidden during backbone phase** — multi-phase transitions use `include_sidechains: false` on early phases

## Trajectory Playback

DCD trajectory frames are fed through the standard animation pipeline. The `TrajectoryPlayer` (in `engine/trajectory.rs`) is a frame sequencer with no animation dependencies. Each frame it produces is fed through `animate_entity()` with `Transition::snap()`, so trajectory and structural animation share a single code path in `tick_animation()`.

## StructureState

`StructureState` (in `animator.rs`) holds the current and target visual state for the entire structure. It converts between backbone chain format (`Vec<Vec<Vec3>>`) and per-residue `ResidueVisualState` arrays, preserving chain boundaries via `chain_lengths`.

The animator owns a single `StructureState` and per-entity runners write interpolated values into it each frame.

## Easing Functions

Available in `animation/easing.rs`:

| Function | Description |
|----------|-------------|
| `Linear` | No easing |
| `QuadraticIn` | Slow start, fast end |
| `QuadraticOut` | Fast start, slow end |
| `SqrtOut` | Fast start, gradual slow |
| `CubicHermite { c1, c2 }` | Configurable control points (default: ease-out) |

All functions evaluate in <100ns and clamp input to [0, 1].

## Disabling Animation

```rust
// Disable all animation (instant snap)
animator.set_enabled(false);

// Or use Transition::snap() for individual updates
```
