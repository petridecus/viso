//! Band renderer
//!
//! Renders constraint bands between atoms as capsules (cylinders with
//! hemispherical caps). Bands are used to pull atoms toward each other during
//! minimization.
//!
//! Visual style matches original Foldit:
//! - Variable radius based on band strength (0.1 to 0.4)
//! - Color based on band type (default purple, backbone yellow-orange, hbond
//!   cyan, disulfide yellow-green)
//! - Color intensity varies with strength and distance
//! - Anchor sphere at endpoint for pulls (bands to a point in space)
//! - Disabled bands shown in gray
//!
//! Uses the same capsule_impostor.wgsl shader as the sidechain renderer.

use glam::Vec3;

use crate::error::VisoError;
use crate::gpu::render_context::RenderContext;
use crate::gpu::shader_composer::{Shader, ShaderComposer};
use crate::options::ColorOptions;
use crate::renderer::impostor::capsule::CapsuleInstance;
use crate::renderer::impostor::{ImpostorPass, ShaderDef};

// Foldit color constants for bands
const BAND_COLOR: [f32; 3] = [0.5, 0.0, 0.5]; // Purple - default band
const BAND_BB_COLOR: [f32; 3] = [1.0, 0.75, 0.0]; // Yellow-orange - backbone-backbone
const BAND_DISULF_COLOR: [f32; 3] = [0.5, 1.0, 0.0]; // Yellow-green - disulfide bridge
const BAND_HBOND_COLOR: [f32; 3] = [0.0, 0.75, 1.0]; // Cyan - hydrogen bond
const DISABLED_COLOR: [f32; 3] = [0.5, 0.5, 0.5]; // Gray - disabled

// Radius constants (varies with strength)
const BAND_MIN_RADIUS: f32 = 0.1;
const BAND_MID_RADIUS: f32 = 0.25;
const BAND_MAX_RADIUS: f32 = 0.4;

// Strength thresholds for radius interpolation
const BAND_MIN_STRENGTH: f32 = 0.5;
const BAND_MID_STRENGTH: f32 = 1.0;
const BAND_MAX_STRENGTH: f32 = 1.5;

// Anchor sphere radius for pulls
const BAND_ANCHOR_RADIUS: f32 = 0.5;

// Special band length thresholds (Angstroms)
const OPTIMAL_BB_BB_DIST: f32 = 3.5;
const OPTIMAL_DISULF_DIST: f32 = 2.0;
const MIN_HBOND_DIST: f32 = 1.95;
const MAX_HBOND_DIST: f32 = 3.0;
const OPTIMAL_DIST_EPSILON: f32 = 0.1;

/// Type of band for color coding
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BandType {
    /// Default band (purple)
    #[default]
    Default,
    /// Backbone-to-backbone band (yellow-orange)
    Backbone,
    /// Disulfide bridge band (yellow-green)
    Disulfide,
    /// Hydrogen bond band (cyan)
    HBond,
}

/// Information about a band to be rendered
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct BandRenderInfo {
    /// World-space position of first endpoint (attached to protein)
    pub endpoint_a: Vec3,
    /// World-space position of second endpoint
    pub endpoint_b: Vec3,
    /// Whether the band is in pull mode (attracts)
    pub is_pull: bool,
    /// Whether the band is in push mode (repels)
    pub is_push: bool,
    /// Whether the band is disabled
    pub is_disabled: bool,
    /// Band strength (affects radius and color intensity, default 1.0)
    pub strength: f32,
    /// Target length for the band (Angstroms, used for type detection if not
    /// specified)
    pub target_length: f32,
    /// Residue index for picking (typically the first residue)
    pub residue_idx: u32,
    /// Whether this is a "pull" to a point in space (vs between two atoms)
    /// When true, an anchor sphere is rendered at endpoint_b
    pub is_space_pull: bool,
    /// Explicit band type (overrides auto-detection from target_length)
    pub band_type: Option<BandType>,
    /// Whether this band was created by a recipe/script (dimmer appearance)
    pub from_script: bool,
}

impl Default for BandRenderInfo {
    fn default() -> Self {
        Self {
            endpoint_a: Vec3::ZERO,
            endpoint_b: Vec3::ZERO,
            is_pull: true,
            is_push: false,
            is_disabled: false,
            strength: 1.0,
            target_length: 0.0,
            residue_idx: 0,
            is_space_pull: false,
            band_type: None,
            from_script: false,
        }
    }
}

/// Renders constraint bands between atoms as capsule impostors.
pub struct BandRenderer {
    pass: ImpostorPass<CapsuleInstance>,
}

impl BandRenderer {
    /// Create a new band renderer with empty instance buffer.
    pub fn new(
        context: &RenderContext,
        layouts: &crate::renderer::PipelineLayouts,
        shader_composer: &mut ShaderComposer,
    ) -> Result<Self, VisoError> {
        let pass = ImpostorPass::new(
            context,
            &ShaderDef {
                label: "Band",
                shader: Shader::Capsule,
            },
            layouts,
            6,
            shader_composer,
        )?;

        Ok(Self { pass })
    }

    /// Compute band radius based on strength (matches Foldit)
    fn compute_radius(strength: f32) -> f32 {
        if strength < BAND_MID_STRENGTH {
            // Lerp from MIN to MID
            let t = (strength - BAND_MIN_STRENGTH)
                / (BAND_MID_STRENGTH - BAND_MIN_STRENGTH);
            let t = t.clamp(0.0, 1.0);
            BAND_MIN_RADIUS + t * (BAND_MID_RADIUS - BAND_MIN_RADIUS)
        } else {
            // Lerp from MID to MAX
            let t = (strength - BAND_MID_STRENGTH)
                / (BAND_MAX_STRENGTH - BAND_MID_STRENGTH);
            let t = t.clamp(0.0, 1.0);
            BAND_MID_RADIUS + t * (BAND_MAX_RADIUS - BAND_MID_RADIUS)
        }
    }

    /// Determine band type from target length if not explicitly specified
    fn detect_band_type(band: &BandRenderInfo) -> BandType {
        if let Some(band_type) = band.band_type {
            return band_type;
        }

        let length = band.target_length;
        if (length - OPTIMAL_BB_BB_DIST).abs() < OPTIMAL_DIST_EPSILON {
            BandType::Backbone
        } else if (length - OPTIMAL_DISULF_DIST).abs() < OPTIMAL_DIST_EPSILON {
            BandType::Disulfide
        } else if length > MIN_HBOND_DIST && length < MAX_HBOND_DIST {
            BandType::HBond
        } else {
            BandType::Default
        }
    }

    /// Compute band color based on type, strength, distance (matches Foldit)
    fn compute_color(
        band: &BandRenderInfo,
        band_type: BandType,
        colors: Option<&ColorOptions>,
    ) -> [f32; 3] {
        if band.is_disabled {
            return DISABLED_COLOR;
        }

        // Base color from band type (use ColorOptions if provided, else
        // fallback to constants)
        let mut color = match band_type {
            BandType::Default => colors.map_or(BAND_COLOR, |c| c.band_default),
            BandType::Backbone => {
                colors.map_or(BAND_BB_COLOR, |c| c.band_backbone)
            }
            BandType::Disulfide => {
                colors.map_or(BAND_DISULF_COLOR, |c| c.band_disulfide)
            }
            BandType::HBond => {
                colors.map_or(BAND_HBOND_COLOR, |c| c.band_hbond)
            }
        };

        // Compute current distance
        let current_length = (band.endpoint_b - band.endpoint_a).length();
        let dist_sqrt = (current_length / 30.0).sqrt().max(0.1); // Avoid division by zero

        // Modify color intensity based on strength and distance (Foldit
        // formula)
        let strength = band.strength;
        if strength <= BAND_MAX_STRENGTH {
            let color_mult = strength * strength / dist_sqrt;
            color[0] = (color_mult * color[0] / 2.0 + 0.25).min(1.0);
            color[2] = (color_mult * color[2] / 4.0 + 0.375).min(1.0);
        } else {
            let red_mult = (-1.0 / (strength - 1.0) + 4.25) / dist_sqrt;
            let blue_mult = (3.0 * (2.0 / (strength - 0.65)).sin()) / dist_sqrt;
            color[0] = (red_mult * color[0] / 2.0 + 0.25).min(1.0);
            color[2] = (blue_mult * color[2] / 4.0 + 0.375).min(1.0);
        }

        // Dim recipe/script bands
        if band.from_script {
            color[0] /= 2.0;
            color[1] /= 2.0;
            color[2] /= 2.0;
        }

        color
    }

    /// Generate capsule instances from band data
    fn generate_instances(
        bands: &[BandRenderInfo],
        colors: Option<&ColorOptions>,
    ) -> Vec<CapsuleInstance> {
        // Each band gets: 1 cylinder + 2 endpoint spheres = 3 instances
        let mut instances = Vec::with_capacity(bands.len() * 3);

        for band in bands {
            let band_type = Self::detect_band_type(band);
            let color = Self::compute_color(band, band_type, colors);
            let radius = Self::compute_radius(band.strength);

            // Main band capsule (cylinder between endpoints)
            instances.push(CapsuleInstance {
                endpoint_a: [
                    band.endpoint_a.x,
                    band.endpoint_a.y,
                    band.endpoint_a.z,
                    radius,
                ],
                endpoint_b: [
                    band.endpoint_b.x,
                    band.endpoint_b.y,
                    band.endpoint_b.z,
                    band.residue_idx as f32,
                ],
                color_a: [color[0], color[1], color[2], 0.0],
                color_b: [color[0], color[1], color[2], 0.0],
            });

            // Anchor sphere at endpoint A (atom position)
            // Rendered as a very short, fat capsule (sphere-like)
            let offset = Vec3::new(0.001, 0.0, 0.0); // Minimal offset to avoid degenerate capsule
            instances.push(CapsuleInstance {
                endpoint_a: [
                    band.endpoint_a.x,
                    band.endpoint_a.y,
                    band.endpoint_a.z,
                    BAND_ANCHOR_RADIUS,
                ],
                endpoint_b: [
                    band.endpoint_a.x + offset.x,
                    band.endpoint_a.y + offset.y,
                    band.endpoint_a.z + offset.z,
                    band.residue_idx as f32,
                ],
                color_a: [color[0], color[1], color[2], 0.0],
                color_b: [color[0], color[1], color[2], 0.0],
            });

            // Anchor sphere at endpoint B
            instances.push(CapsuleInstance {
                endpoint_a: [
                    band.endpoint_b.x,
                    band.endpoint_b.y,
                    band.endpoint_b.z,
                    BAND_ANCHOR_RADIUS,
                ],
                endpoint_b: [
                    band.endpoint_b.x + offset.x,
                    band.endpoint_b.y + offset.y,
                    band.endpoint_b.z + offset.z,
                    band.residue_idx as f32,
                ],
                color_a: [color[0], color[1], color[2], 0.0],
                color_b: [color[0], color[1], color[2], 0.0],
            });
        }

        instances
    }

    /// Update band geometry
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bands: &[BandRenderInfo],
        colors: Option<&ColorOptions>,
    ) {
        let instances = Self::generate_instances(bands, colors);
        let _ = self.pass.write_instances(device, queue, &instances);
    }

    /// Clear all bands
    pub fn clear(&mut self) {
        self.pass.instance_count = 0;
    }

    /// GPU buffer sizes: `(label, used_bytes, allocated_bytes)`.
    pub fn buffer_info(&self) -> Vec<(&'static str, usize, usize)> {
        vec![self.pass.buffer_info("Band Capsules")]
    }

    /// Draw band capsules into the given render pass.
    pub fn draw<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        bind_groups: &crate::renderer::draw_context::DrawBindGroups<'a>,
    ) {
        self.pass.draw(render_pass, bind_groups);
    }
}
