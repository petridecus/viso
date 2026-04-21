//! Engine methods for per-entity molecular surface generation.
//!
//! Surfaces are generated on a background thread and rendered through
//! the shared `IsosurfaceRenderer`. Multiple entities can each have
//! surfaces — their meshes are concatenated before upload.

use super::VisoEngine;
use crate::renderer::geometry::isosurface::IsosurfaceVertex;

/// Which kind of molecular surface to generate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum SurfaceKind {
    /// Smooth Gaussian blob surface.
    Gaussian,
    /// Solvent-excluded / Connolly surface.
    Ses,
}

/// Per-entity surface parameters.
#[derive(Debug, Clone)]
pub(crate) struct EntitySurface {
    /// Which surface type.
    pub kind: SurfaceKind,
    /// Grid resolution in Angstroms (lower = finer).
    pub resolution: f32,
    /// Probe radius for SES (Angstroms, default 1.4).
    pub probe_radius: f32,
    /// Gaussian isosurface level (only used for Gaussian kind).
    pub level: f32,
    /// Surface RGBA color.
    pub color: [f32; 4],
    /// Whether this surface is visible.
    pub visible: bool,
}

impl Default for EntitySurface {
    fn default() -> Self {
        Self {
            kind: SurfaceKind::Gaussian,
            resolution: 1.0,
            probe_radius: 1.4,
            level: 0.5,
            color: [0.7, 0.7, 0.7, 0.35],
            visible: true,
        }
    }
}

impl VisoEngine {
    /// Add a Gaussian surface for an entity.
    pub fn add_gaussian_surface(&mut self, entity_id: u32, color: [f32; 4]) {
        let surface = EntitySurface {
            kind: SurfaceKind::Gaussian,
            color,
            ..Default::default()
        };
        self.set_entity_surface(entity_id, surface);
    }

    /// Add a solvent-excluded (Connolly) surface for an entity.
    pub fn add_ses_surface(&mut self, entity_id: u32, color: [f32; 4]) {
        let surface = EntitySurface {
            kind: SurfaceKind::Ses,
            color,
            ..Default::default()
        };
        self.set_entity_surface(entity_id, surface);
    }

    /// Update a single color channel or opacity on an entity's surface.
    pub fn set_surface_color_channel(
        &mut self,
        entity_id: u32,
        channel: usize,
        value: f32,
    ) {
        if let Some(surface) = self.entity_surfaces.get_mut(&entity_id) {
            surface.color[channel] = value.clamp(0.0, 1.0);
            self.regenerate_entity_surfaces();
        }
    }

    /// Remove the molecular surface for an entity.
    ///
    /// When a global surface is active, this stores an invisible sentinel
    /// so the entity explicitly opts out instead of falling back to the
    /// global default.
    pub fn remove_entity_surface(&mut self, entity_id: u32) {
        use crate::options::SurfaceKindOption;
        let had = self.entity_surfaces.remove(&entity_id).is_some();
        // If there's a global surface, store an invisible sentinel so
        // this entity doesn't inherit the global.
        if self.options.display.surface_kind != SurfaceKindOption::None {
            let _ = self.entity_surfaces.insert(
                entity_id,
                EntitySurface {
                    visible: false,
                    ..Default::default()
                },
            );
        }
        if had || self.options.display.surface_kind != SurfaceKindOption::None {
            log::info!("removed surface for entity {entity_id}");
            self.regenerate_entity_surfaces();
        }
    }

    /// Set surface parameters for an entity and regenerate.
    fn set_entity_surface(&mut self, entity_id: u32, surface: EntitySurface) {
        log::info!("set {:?} surface for entity {entity_id}", surface.kind);
        let _ = self.entity_surfaces.insert(entity_id, surface);
        self.regenerate_entity_surfaces();
    }

    /// Regenerate all isosurface meshes (density + entity surfaces +
    /// cavities) on a background thread.
    ///
    /// Collects atom positions + radii from each entity that has a
    /// surface or cavity rendering enabled, runs the appropriate
    /// generator, concatenates all meshes, and sends the result to the
    /// density mesh channel (shared with density map rendering).
    pub(super) fn regenerate_entity_surfaces(&self) {
        use crate::options::SurfaceKindOption;
        use crate::renderer::geometry::isosurface::{
            cavity, gaussian_surface, ses,
        };

        let all_entities = self.current_assembly.entities();
        let palette = self.options.display.backbone_palette();
        let global_kind = self.options.display.surface_kind;
        let global_opacity = self.options.display.surface_opacity;
        let global_show_cavities = self.options.display.show_cavities;

        // Collect jobs: (positions, radii, surface params with color)
        let mut jobs: Vec<(Vec<glam::Vec3>, Vec<f32>, EntitySurface)> =
            Vec::new();
        // Cavity jobs: (positions, radii). Color is the fixed CAVITY_RGBA
        // constant — cavities don't pick up per-entity coloring.
        let mut cavity_jobs: Vec<(Vec<glam::Vec3>, Vec<f32>)> = Vec::new();

        for (entity_idx, se) in all_entities.iter().enumerate() {
            let eid = se.id().raw();
            if !self.is_entity_visible(eid) {
                continue;
            }

            // Per-entity surface takes priority; fall back to global
            let base_surface = self.entity_surfaces.get(&eid).map_or_else(
                || match global_kind {
                    SurfaceKindOption::Gaussian => Some(EntitySurface {
                        kind: SurfaceKind::Gaussian,
                        color: [0.7, 0.7, 0.7, global_opacity],
                        ..Default::default()
                    }),
                    SurfaceKindOption::Ses => Some(EntitySurface {
                        kind: SurfaceKind::Ses,
                        color: [0.7, 0.7, 0.7, global_opacity],
                        ..Default::default()
                    }),
                    SurfaceKindOption::None => None,
                },
                |s| if s.visible { Some(s.clone()) } else { None },
            );

            // Skip atoms gathering only when neither the surface nor cavity
            // path wants this entity — cavities want it whenever the global
            // toggle is on.
            if base_surface.is_none() && !global_show_cavities {
                continue;
            }

            let positions = se.positions();
            if positions.is_empty() {
                continue;
            }
            let radii: Vec<f32> = se
                .atom_set()
                .iter()
                .map(|a| a.element.vdw_radius())
                .collect();

            // Use the backbone palette so surface/cavity colors match the
            // backbone.
            let [r, g, b] = palette.categorical_color(entity_idx);

            if let Some(mut surface) = base_surface {
                surface.color = [r, g, b, surface.color[3]];
                // SES needs a finer grid than Gaussian to resolve atom-level
                // detail (ChimeraX default is 0.5 Å).
                if surface.kind == SurfaceKind::Ses {
                    surface.resolution = 0.5;
                }
                jobs.push((positions.clone(), radii.clone(), surface));
            }

            if global_show_cavities {
                cavity_jobs.push((positions, radii));
            }
        }

        // Also include any visible density maps
        let density_jobs: Vec<_> = self
            .density
            .visible_entries()
            .map(|(_id, entry)| {
                let [r, g, b] = entry.color;
                (entry.map.clone(), entry.threshold, [r, g, b, entry.opacity])
            })
            .collect();

        if jobs.is_empty() && density_jobs.is_empty() && cavity_jobs.is_empty()
        {
            // Nothing to generate — send empty mesh to clear renderer
            let _ = self.gpu.density_tx.send((Vec::new(), Vec::new()));
            return;
        }

        let tx = self.gpu.density_tx.clone();

        let _ = std::thread::spawn(move || {
            let mut all_verts: Vec<IsosurfaceVertex> = Vec::new();
            let mut all_idxs: Vec<u32> = Vec::new();

            // Generate density map meshes first
            for (map, threshold, color) in &density_jobs {
                let (v, i) =
                    crate::renderer::geometry::isosurface::density
                        ::generate_density_mesh(
                            map, *threshold, *color, None,
                        );
                let base = all_verts.len() as u32;
                all_verts.extend(v);
                all_idxs.extend(i.iter().map(|&idx| idx + base));
            }

            // Generate entity surface meshes
            for (positions, radii, surface) in &jobs {
                let (v, i) = match surface.kind {
                    SurfaceKind::Gaussian => {
                        gaussian_surface::generate_gaussian_surface(
                            positions,
                            radii,
                            surface.resolution,
                            surface.level,
                            surface.color,
                        )
                    }
                    SurfaceKind::Ses => ses::generate_ses(
                        positions,
                        radii,
                        Some(surface.probe_radius),
                        surface.resolution,
                        surface.color,
                    ),
                };
                let base = all_verts.len() as u32;
                all_verts.extend(v);
                all_idxs.extend(i.iter().map(|&idx| idx + base));
            }

            // Generate cavity meshes on a 0.6 Å grid — coarser than SES
            // because cavity detection is topological (flood fill from
            // grid boundary), so finer voxels can flip whether a thin
            // SES-wall separates a cavity from the exterior. 0.6 Å was
            // verified to detect the expected number of cavities on
            // benchmark structures (e.g. 1bbc has 3).
            let mut cavity_count = 0usize;
            for (positions, radii) in &cavity_jobs {
                let set =
                    cavity::generate_cavities(positions, radii, Some(1.4), 0.6);
                for mesh in &set.meshes {
                    let base = all_verts.len() as u32;
                    all_verts.extend(mesh.vertices.iter().copied());
                    all_idxs.extend(mesh.indices.iter().map(|&idx| idx + base));
                }
                cavity_count += set.meshes.len();
            }

            log::info!(
                "surface mesh: {} verts, {} triangles ({} cavities)",
                all_verts.len(),
                all_idxs.len() / 3,
                cavity_count,
            );

            if tx.send((all_verts, all_idxs)).is_err() {
                log::warn!("surface mesh channel send failed");
            }
        });
    }
}
