# Options and Presets

Viso's visual appearance is controlled by the `Options` struct, which can be loaded from and saved to TOML files. This enables presets for different visualization styles.

## Options Structure

```rust
pub struct Options {
    pub display: DisplayOptions,
    pub lighting: LightingOptions,
    pub post_processing: PostProcessingOptions,
    pub camera: CameraOptions,
    pub colors: ColorOptions,
    pub geometry: GeometryOptions,
    pub keybindings: KeybindingOptions,
}
```

All fields use `#[serde(default)]`, so TOML files can be partial -- only the fields you want to override need to be specified.

## Display Options

```rust
pub struct DisplayOptions {
    pub show_waters: bool,           // default: false
    pub show_ions: bool,             // default: true
    pub show_solvent: bool,          // default: false
    pub lipid_mode: LipidMode,       // default: Coarse
    pub show_sidechains: bool,       // default: true
    pub show_hydrogens: bool,        // default: false
    pub backbone_color_mode: BackboneColorMode,   // default: Chain
    pub sidechain_color_mode: SidechainColorMode, // default: Hydrophobicity
    pub na_color_mode: NaColorMode,  // default: Uniform
}
```

## Lighting Options

```rust
pub struct LightingOptions {
    pub light1_dir: [f32; 3],       // Primary directional light
    pub light2_dir: [f32; 3],       // Fill directional light
    pub light1_intensity: f32,      // default: 2.0
    pub light2_intensity: f32,      // default: 1.1
    pub ambient: f32,               // default: 0.45
    pub specular_intensity: f32,    // default: 0.35
    pub shininess: f32,             // default: 38.0
    pub rim_power: f32,             // default: 5.0
    pub rim_intensity: f32,         // default: 0.3
    pub rim_directionality: f32,    // default: 0.3
    pub rim_color: [f32; 3],
    pub rim_dir: [f32; 3],
    pub ibl_strength: f32,          // default: 0.6
    pub roughness: f32,             // default: 0.35
    pub metalness: f32,             // default: 0.15
}
```

## Post-Processing Options

```rust
pub struct PostProcessingOptions {
    pub outline_thickness: f32,         // default: 1.0
    pub outline_strength: f32,          // default: 0.7
    pub ao_strength: f32,               // default: 0.85
    pub ao_radius: f32,                 // default: 0.5
    pub ao_bias: f32,                   // default: 0.025
    pub ao_power: f32,                  // default: 2.0
    pub fog_start: f32,                 // default: 100.0
    pub fog_density: f32,               // default: 0.005
    pub exposure: f32,                  // default: 1.0
    pub normal_outline_strength: f32,   // default: 0.5
    pub bloom_intensity: f32,           // default: 0.0
    pub bloom_threshold: f32,           // default: 1.0
}
```

## Camera Options

```rust
pub struct CameraOptions {
    pub fovy: f32,          // Field of view in degrees, default: 45.0
    pub znear: f32,         // Near clip plane, default: 5.0
    pub zfar: f32,          // Far clip plane, default: 2000.0
    pub rotate_speed: f32,  // Mouse rotation sensitivity, default: 0.5
    pub pan_speed: f32,     // Mouse pan sensitivity, default: 0.5
    pub zoom_speed: f32,    // Scroll zoom sensitivity, default: 0.1
}
```

## Color Options

```rust
pub struct ColorOptions {
    pub lipid_carbon_tint: [f32; 3],       // Warm beige/tan for lipid carbons
    pub hydrophobic_sidechain: [f32; 3],   // Blue for hydrophobic sidechains
    pub hydrophilic_sidechain: [f32; 3],   // Orange for hydrophilic sidechains
    pub nucleic_acid: [f32; 3],            // Light blue-violet for DNA/RNA
    pub band_default: [f32; 3],            // Purple
    pub band_backbone: [f32; 3],           // Yellow-orange
    pub band_disulfide: [f32; 3],          // Yellow-green
    pub band_hbond: [f32; 3],              // Cyan
    pub solvent_color: [f32; 3],
    pub cofactor_tints: HashMap<String, [f32; 3]>,
}
```

## Geometry Options

```rust
pub struct GeometryOptions {
    pub tube_radius: f32,           // default: 0.3
    pub tube_radial_segments: u32,  // default: 8
    pub solvent_radius: f32,        // default: 0.15
    pub ligand_sphere_radius: f32,  // default: 0.3
    pub ligand_bond_radius: f32,    // default: 0.12
}
```

## Keybinding Options

```rust
pub struct KeybindingOptions {
    pub bindings: HashMap<String, String>,  // action name -> key string
}
```

Default keybindings:

| Action | Key |
|--------|-----|
| `recenter_camera` | Q |
| `toggle_trajectory` | T |
| `toggle_ions` | I |
| `toggle_waters` | U |
| `toggle_solvent` | O |
| `toggle_lipids` | L |
| `cycle_focus` | Tab |
| `toggle_auto_rotate` | R |
| `reset_focus` | \` |
| `cancel` | Escape |

## Loading and Saving

```rust
// Load from TOML file (partial files supported)
let options = Options::load(Path::new("presets/dark.toml"))?;

// Save to TOML file
options.save(Path::new("presets/my_preset.toml"))?;

// List available presets in a directory
let presets = Options::list_presets(Path::new("presets/"));
// Returns: ["dark", "publication", "presentation", ...]
```

## Example TOML Preset

```toml
[display]
show_sidechains = true
backbone_color_mode = "Chain"

[lighting]
light1_intensity = 2.5
ambient = 0.5
specular_intensity = 0.4
shininess = 50.0

[post_processing]
outline_thickness = 1.5
outline_strength = 0.8
ao_strength = 1.0
bloom_intensity = 0.15
bloom_threshold = 0.8

[camera]
fovy = 35.0
rotate_speed = 0.4

[colors]
hydrophobic_sidechain = [0.2, 0.4, 0.85]
hydrophilic_sidechain = [0.9, 0.55, 0.15]
```

## Applying Options at Runtime

Options are applied to the engine through dedicated methods. Display and color options trigger a scene re-sync (since the background processor needs them for mesh generation). Lighting and post-processing options are pushed directly to GPU uniforms.

```rust
// Apply post-processing options
engine.post_process.apply_options(&options, &queue);

// Apply lighting options
engine.update_lighting(&options.lighting);

// Display/color changes require a scene sync
engine.sync_scene_to_renderers(None);
```

Changes to display options (like `backbone_color_mode` or `show_sidechains`) invalidate the per-group mesh cache in the background processor, causing a full mesh regeneration on the next sync.
