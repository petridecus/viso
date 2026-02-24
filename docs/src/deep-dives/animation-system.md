# Animation System

Viso's animation system manages smooth visual transitions when protein structures change. It uses a two-layer architecture that separates *how to animate* from the application's *why*.

## Two-Layer Architecture

```
Transition              →    AnimationBehavior
  (behavior + flags)          (how to animate)
```

1. **Transition** -- a struct bundling a behavior with metadata flags (size-change permission, sidechain suppression)
2. **Behavior** -- a trait implementation that defines interpolation curves, timing, and visual effects

The consumer constructs `Transition` values directly, choosing the appropriate behavior for each update. Viso only cares about *how* to interpolate, not *why*.

## Transition

```rust
pub struct Transition {
    pub behavior: SharedBehavior,
    pub allows_size_change: bool,
    pub suppress_initial_sidechains: bool,
}

// Convenience constructors
Transition::snap()     // Instant, allows resize
Transition::smooth()   // 300ms cubic hermite ease-out
Transition::with_behavior(CollapseExpand::default())  // Custom behavior
Transition::default()  // Same as smooth()

// Builder methods
Transition::with_behavior(BackboneThenExpand::new(...))
    .allowing_size_change()
    .suppressing_initial_sidechains()
```

## Built-in Behaviors

### Snap

Instant transition. Duration is zero. Used for initial loads where animation would delay the first meaningful frame.

### SmoothInterpolation

Standard eased lerp between start and target:

```rust
SmoothInterpolation::standard()        // 300ms, cubic hermite ease-out
SmoothInterpolation::fast()            // 100ms, quadratic out
SmoothInterpolation::linear(duration)  // No easing
```

Good for incremental changes where start and target are close.

### Cascade

Staggered per-residue animation that creates a wave effect:

```rust
Cascade::new(
    Duration::from_millis(500),  // Base duration per residue
    Duration::from_millis(5),    // Delay between residues
)
```

Each residue starts its animation slightly after the previous one. The total duration is `base_duration + delay_per_residue * residue_count`. Used for dramatic reveals.

### CollapseExpand

Two-phase animation for mutations:

1. **Collapse phase** -- sidechain atoms collapse toward the backbone CA position
2. **Expand phase** -- new sidechain atoms expand outward from CA to their final positions

```rust
CollapseExpand::new(
    Duration::from_millis(300),  // Collapse duration
    Duration::from_millis(300),  // Expand duration
)
```

This provides clear visual feedback that a mutation occurred, even when the backbone barely moves.

### BackboneThenExpand

Two-phase animation for transitions where sidechains should appear after backbone settles:

1. **Backbone phase** -- backbone atoms lerp to final positions while sidechains are hidden
2. **Expand phase** -- sidechain atoms expand from collapsed (at CA) to final positions

```rust
BackboneThenExpand::new(
    Duration::from_millis(400),  // Backbone lerp duration
    Duration::from_millis(600),  // Sidechain expand duration
)
```

Uses `should_include_sidechains(raw_t)` to hide sidechains during the backbone phase, preventing visual artifacts when new atoms appear before the backbone has settled.

## The AnimationBehavior Trait

All behaviors implement this trait:

```rust
pub trait AnimationBehavior: Send + Sync {
    /// Eased time for raw progress t
    fn eased_t(&self, t: f32) -> f32;

    /// Compute visual state at time t
    fn compute_state(&self, t: f32, start: &ResidueVisualState, end: &ResidueVisualState)
        -> ResidueVisualState;

    /// Total animation duration
    fn duration(&self) -> Duration;

    /// How to handle a new target arriving mid-animation
    fn preemption(&self) -> PreemptionStrategy { PreemptionStrategy::Restart }

    /// Interpolation context (override for multi-phase behaviors)
    fn compute_context(&self, raw_t: f32) -> InterpolationContext;

    /// Whether sidechains should be visible at this progress
    fn should_include_sidechains(&self, _raw_t: f32) -> bool { true }

    /// Position interpolation with optional collapse point
    fn interpolate_position(&self, t: f32, start: Vec3, end: Vec3, collapse_point: Vec3) -> Vec3;
}
```

### Preemption Strategies

When a new target arrives while an animation is playing:

```rust
pub enum PreemptionStrategy {
    Restart,  // Start from current visual position to new target (default)
    Ignore,   // Ignore new target until animation completes
    Blend,    // Blend toward new target maintaining velocity
}
```

`Restart` is the most common -- it provides responsive feedback during rapid update cycles. The current visual state becomes the new start state, and the timer resets.

## ResidueVisualState

Each residue's visual state during animation:

```rust
pub struct ResidueVisualState {
    pub backbone: [Vec3; 3],  // N, CA, C positions
    pub chis: [f32; 4],       // Up to 4 chi (dihedral) angles
    pub num_chis: usize,      // Number of valid chi angles
}
```

Interpolation lerps backbone positions and chi angles. Chi angles use angle-aware interpolation that handles wrapping across the -180/180 boundary.

## InterpolationContext

Computed once per frame from the behavior and raw progress, then shared across all interpolation to prevent backbone/sidechain desync:

```rust
pub struct InterpolationContext {
    pub raw_t: f32,                  // 0.0 to 1.0
    pub eased_t: f32,               // After easing function
    pub phase_t: Option<f32>,       // Within current phase (multi-phase behaviors)
    pub phase_eased_t: Option<f32>, // Eased within phase
}
```

## StructureAnimator

The top-level animator that applications interact with:

```rust
let mut animator = StructureAnimator::new();

// Set animation target
animator.set_target(&new_backbone_chains, &Transition::smooth());

// Each frame:
let still_animating = animator.update(Instant::now());

// Get interpolated state
let visual_backbone = animator.get_backbone();
let sidechain_positions = animator.get_sidechain_positions();
let context = animator.interpolation_context();
```

### Per-Entity Animation

When some groups should animate and others should snap:

```rust
// Set sidechain targets with transition (call BEFORE set_target)
animator.set_sidechain_target_with_transition(
    &positions, &residue_indices, &ca_positions,
    Some(&my_transition),
);

// Set backbone target
animator.set_target(&backbone_chains, &my_transition);

// Snap non-targeted entities
animator.snap_entities_without_action(&entity_residue_ranges, &active_entities);

// Remove non-targeted residues from the runner
animator.remove_non_targeted_from_runner(&entity_residue_ranges, &active_entities);
```

## Sidechain Animation

Sidechain animation tracks positions separately from backbone and supports specialized behaviors:

- **Standard lerp** -- for smooth transitions, sidechains lerp alongside backbone
- **Collapse toward CA** -- for mutations, atoms collapse to the CA position then expand to new positions
- **Hidden during backbone phase** -- for multi-phase transitions, sidechains are invisible while backbone lerps

The animator stores start and target sidechain positions and uses the behavior's `interpolate_position` method, which receives a `collapse_point` (the CA position) for behaviors that need it.

## Disabling Animation

```rust
// Disable all animation (instant snap)
animator.set_enabled(false);

// Or use Transition::snap() for individual updates
engine.sync_scene_to_renderers(Some(Transition::snap()));
```
