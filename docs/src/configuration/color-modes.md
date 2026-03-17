# Color Modes

Viso supports several coloring schemes for backbones, sidechains, and nucleic acids. Colors are computed during background scene processing and baked into vertex data for zero-cost rendering.

## Backbone Color Modes

Set via `DisplayOptions::backbone_color_mode`:

```rust
pub enum BackboneColorMode {
    Score,              // Per-residue energy score (blue-white-red)
    ScoreRelative,      // Relative scoring within the structure
    SecondaryStructure, // Helix/sheet/coil coloring
    Chain,              // Each chain gets a distinct color (default)
}
```

### Chain (Default)

Each chain gets a distinct color interpolated from blue to red across the chain count. Single-chain proteins use a gradient along the chain. This is the most common mode for general visualization.

### Secondary Structure

Colors residues by their computed secondary structure type:
- **Alpha helix** -- distinct helix color
- **Beta sheet** -- distinct sheet color
- **Coil/Loop** -- neutral color

Secondary structure is computed from backbone geometry using dihedral angle analysis.

### Score

Colors residues by per-residue energy scores from Rosetta. Uses a blue-white-red gradient:
- **Blue** -- favorable (low) energy
- **White** -- neutral
- **Red** -- unfavorable (high) energy

Scores are cached on each `EntityGroup` via `set_per_residue_scores()`. The background processor converts scores to RGB colors during mesh generation.

### Score Relative

Similar to Score mode but normalizes within the structure, highlighting relative differences rather than absolute energy values.

## Sidechain Color Modes

Set via `DisplayOptions::sidechain_color_mode`:

```rust
pub enum SidechainColorMode {
    Hydrophobicity,  // Default -- hydrophobic vs hydrophilic
}
```

### Hydrophobicity (Default)

Sidechain atoms are colored by hydrophobicity:
- **Hydrophobic** -- blue (default: `[0.3, 0.5, 0.9]`)
- **Hydrophilic** -- orange (default: `[0.95, 0.6, 0.2]`)

These colors are configurable in `ColorOptions`.

## Nucleic Acid Color Modes

Set via `DisplayOptions::na_color_mode`:

```rust
pub enum NaColorMode {
    Uniform,  // Default -- single color for all nucleic acid backbone
}
```

### Uniform (Default)

All nucleic acid backbone uses a single color (default: light blue-violet `[0.45, 0.55, 0.85]`), configurable via `ColorOptions::nucleic_acid`.

## Non-Protein Coloring

Ligands, ions, and waters use element-based CPK coloring in the ball-and-stick renderer:

- Standard CPK colors for common elements (C, N, O, S, P, etc.)
- **Lipid carbons** use a special warm beige/tan tint (configurable via `ColorOptions::lipid_carbon_tint`)
- **Cofactors** can have per-residue-name tints via `ColorOptions::cofactor_tints`

## Color Transitions During Animation

When backbone colors change between poses (e.g., score coloring updates after minimization), the background processor caches per-residue colors in the `PreparedScene`. During animation, the tube and ribbon renderers interpolate between start and target colors using the same easing function as the backbone position interpolation.

This means color changes are smooth -- residues don't suddenly flash to new colors but transition over the animation duration.

## How Colors Flow Through the Pipeline

1. **Scene sync** -- `DisplayOptions` and `ColorOptions` are sent to the background processor as part of `SceneRequest::FullRebuild`
2. **Background thread** -- during mesh generation, colors are computed per-vertex based on the active color mode and baked into vertex data
3. **GPU upload** -- vertex buffers with embedded colors are uploaded to the GPU
4. **Rendering** -- shaders read per-vertex colors directly, with selection highlighting applied as an overlay in the fragment shader via the `SelectionBuffer` bit-array
