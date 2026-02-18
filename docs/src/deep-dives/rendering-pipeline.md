# Rendering Pipeline

Viso's rendering pipeline has two main stages: a geometry pass that renders molecular structures to HDR render targets, and a post-processing stack that applies screen-space effects.

## Overview

```
Geometry Pass (7 molecular renderers)
    ↓ Color (Rgba16Float) + Normals (Rgba16Float) + Depth (Depth32Float)
    ↓
Post-Processing Stack:
    1. SSAO: depth + normals → ambient occlusion texture
    2. Bloom: color → threshold → blur → half-res bloom texture
    3. Composite: color + SSAO + depth + normals + bloom → tone-mapped result
    4. FXAA: anti-aliased final output → swapchain
```

## Geometry Pass

### Render Targets

All molecular renderers write to two HDR render targets plus a depth buffer:

| Target | Format | Contents |
|--------|--------|----------|
| Target 0 | Rgba16Float | Scene color with alpha blending |
| Target 1 | Rgba16Float | View-space normals / metadata (no blending) |
| Depth | Depth32Float | Depth buffer (Less compare, writes enabled) |

Using Rgba16Float enables HDR lighting and bloom without banding artifacts.

### Molecular Renderers

Seven renderers draw molecular geometry in order:

#### 1. Tube Renderer

Renders protein backbone as smooth cylindrical tubes.

- **Geometry**: cubic Hermite splines with rotation-minimizing frames (RMF)
- **Parameters**: radius 0.3 angstroms, 8 radial segments, 4 axial segments per CA span
- **SS filtering**: in ribbon mode, only renders coil/loop segments; tubes handle everything in tube mode
- **Vertex data**: position, normal, color, residue_idx, center_pos

#### 2. Ribbon Renderer

Renders helices and sheets as flat ribbons.

- **Helices**: ribbon normal points radially outward from the helix axis
- **Sheets**: constant width, smooth RMF-propagated normals (no pleating)
- **Parameters**: helix width 1.4, sheet width 1.6, thickness 0.25 angstroms
- **Interpolation**: B-spline with C2 continuity, 16 segments per residue
- **Sheet offsets**: sheet residues are offset from the tube centerline to separate ribbon from tube in ribbon mode

#### 3. Capsule Sidechain Renderer

Renders sidechain atoms as capsule impostors (ray-marched).

- **Technique**: storage buffer of `CapsuleInstance` structs, rendered as ray-marched impostors
- **Capsule radius**: 0.3 angstroms
- **Colors**: hydrophobic (blue), hydrophilic (orange), configurable via `ColorOptions`
- **Frustum culling**: sidechains outside the view frustum (with 5.0 angstrom margin) are skipped
- **Instance data**: two endpoints + radius, color + entity_id for picking

#### 4. Ball-and-Stick Renderer

Renders ligands, ions, waters, and non-protein entities.

- **Atoms**: ray-cast sphere impostors
- **Bonds**: capsule impostors (cylinders with hemispherical caps)
- **Atom radii**: normal atoms 0.3x van der Waals radius, ions 0.5x, water oxygen 0.3 angstroms
- **Bond radius**: 0.15 angstroms
- **Double bonds**: two parallel capsules offset by 0.2 angstroms
- **Lipid modes**: CoarseGrained (P spheres, head-group highlights, thin tail bonds) or BallAndStick (full detail)

#### 5. Band Renderer

Renders constraint bands (for Rosetta minimization).

- **Visual**: capsule impostors with variable radius (0.1 to 0.4 angstroms, scaled by constraint strength)
- **Colors by type**: default (purple), backbone (yellow-orange), disulfide (yellow-green), H-bond (cyan), disabled (gray)
- **Anchor spheres**: 0.5 angstrom radius spheres at band endpoints for pull indicators

#### 6. Pull Renderer

Renders the active drag constraint.

- **Cylinder**: capsule impostor from atom to near the mouse position (purple, 0.25 angstrom radius)
- **Arrow**: cone impostor at the mouse end pointing toward the target (0.6 angstrom radius)

#### 7. Nucleic Acid Renderer

Renders DNA/RNA backbones.

- **Geometry**: flat ribbons tracing phosphorus (P) atoms with B-spline interpolation and RMF orientation
- **Parameters**: width 1.2 angstroms (narrower than protein), thickness 0.25 angstroms, 16 segments per P-atom
- **Color**: light blue-violet (configurable)

### Shared Bind Groups

All renderers receive common bind groups via `DrawBindGroups`:

```rust
pub struct DrawBindGroups<'a> {
    pub camera: &'a wgpu::BindGroup,     // Projection/view matrices
    pub lighting: &'a wgpu::BindGroup,   // Light directions, intensities
    pub selection: &'a wgpu::BindGroup,  // Selection bit-array
    pub color: Option<&'a wgpu::BindGroup>, // Per-residue color override
}
```

## Post-Processing Stack

### 1. SSAO (Screen-Space Ambient Occlusion)

Computes local ambient occlusion from the depth and normal buffers.

- **Kernel**: 32 hemisphere samples in view-space
- **Noise**: 4x4 rotation noise texture to reduce banding
- **Parameters**: radius (0.5), bias (0.025), power (2.0)
- **Output**: single-channel AO texture
- **Blur pass**: separable blur to smooth noise patterns

### 2. Bloom

Extracts and blurs bright areas of the image.

- **Threshold**: extracts pixels above the brightness threshold to a half-resolution texture
- **Blur**: separable Gaussian blur (horizontal then vertical, ping-pong textures)
- **Mip chain**: 4 levels of downsampling
- **Upsample**: additive accumulation back to half-resolution
- **Output**: half-resolution bloom texture
- **Parameters**: threshold (1.0), intensity (0.0 by default -- disabled)

### 3. Composite

Combines all post-processing inputs into the final image.

**Inputs** (8 bind group entries):
- Scene color texture
- SSAO texture
- Depth texture
- Color/SSAO sampler (linear)
- Depth sampler (nearest)
- Params uniform buffer
- Normal G-buffer
- Bloom texture

**Effects applied:**
- SSAO as darkening multiplier on base color
- Depth-based fog (configurable start and density)
- Depth-based outlines (edge detection on depth discontinuities)
- Normal-based outlines (edge detection on normal discontinuities)
- Bloom additive blend
- HDR tone mapping (exposure control)
- Gamma correction

**Parameters** (CompositeParams uniform):
- Screen size, outline thickness/strength, AO strength
- Near/far planes, fog start/density
- Normal outline strength, exposure, gamma, bloom intensity

### 4. FXAA

Fast Approximate Anti-Aliasing as the final pass.

- Smooths jagged edges on mesh-based geometry (tubes, ribbons) that supersampling alone doesn't fully resolve
- Reads from the composite output, writes to the swapchain surface
- Uses linear filtering for edge detection

## ShaderComposer

Viso uses `naga_oil` for shader composition, enabling modular WGSL with imports:

```wgsl
#import viso::camera
#import viso::lighting

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light = calculate_lighting(in.normal, in.position);
    // ...
}
```

**Pre-loaded shared modules:**
- `camera.wgsl` -- camera matrix uniforms and transformations
- `lighting.wgsl` -- Blinn-Phong lighting, directional lights
- `sdf.wgsl` -- signed distance field utilities
- `raymarch.wgsl` -- ray marching for implicit surfaces
- `volume.wgsl` -- volume texture sampling
- `fullscreen.wgsl` -- fullscreen triangle utilities

The composer produces `naga::Module` IR directly (skipping WGSL re-parse at runtime for performance).

## Render-Scale Supersampling

The rendering resolution can differ from the display resolution via `set_scale_factor()`. All internal textures (color, depth, normal, SSAO, bloom) are sized to the render resolution. FXAA downsamples to the display resolution as the final step.
