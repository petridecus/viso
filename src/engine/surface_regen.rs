//! Background regeneration of all isosurface meshes (density maps,
//! entity surfaces, cavities).
//!
//! Isosurface meshing is kicked off from several annotation and density
//! mutation paths; the completed mesh is shipped through an
//! `mpsc::channel` to the main thread, which drains the receiver each
//! frame and uploads the latest mesh to the GPU.
//!
//! This module owns the sender side of that channel through a thin
//! [`SurfaceRegen`] holder. Decoupling it from `GpuPipeline` (which
//! keeps the receiver) avoids exposing a worker-bound sender as a
//! crate-internal API from the renderer back to engine code.

use std::sync::mpsc;

use super::annotations::EntityAnnotations;
use super::density_store::DensityStore;
use super::scene::Scene;
use super::surface::{EntitySurface, SurfaceKind};
use crate::options::{SurfaceKindOption, VisoOptions};
use crate::renderer::geometry::isosurface::IsosurfaceVertex;

/// Worker→main message type carrying a completed isosurface mesh.
pub(crate) type MeshMessage = (Vec<IsosurfaceVertex>, Vec<u32>);

/// Owner of the sender side of the background isosurface-mesh channel.
///
/// The matching receiver lives on
/// [`crate::renderer::GpuPipeline`]; the two ends are constructed
/// together in [`crate::engine::VisoEngine::new`].
pub(crate) struct SurfaceRegen {
    /// Sender used by [`regenerate_surfaces`] to ship completed meshes
    /// back to the main thread.
    pub(crate) tx: mpsc::Sender<MeshMessage>,
}

impl SurfaceRegen {
    /// Wrap an existing sender.
    pub(crate) fn new(tx: mpsc::Sender<MeshMessage>) -> Self {
        Self { tx }
    }
}

/// Regenerate all isosurface meshes (density + entity surfaces +
/// cavities) on a background thread.
///
/// Collects atom positions + radii from each entity that has a surface
/// or cavity rendering enabled, runs the appropriate generator,
/// concatenates all meshes, and sends the result to the isosurface
/// mesh channel (shared with density map rendering).
pub(crate) fn regenerate_surfaces(
    scene: &Scene,
    annotations: &EntityAnnotations,
    density: &DensityStore,
    options: &VisoOptions,
    regen: &SurfaceRegen,
) {
    use crate::renderer::geometry::isosurface::{
        cavity, gaussian_surface, ses,
    };

    let all_entities = scene.current.entities();
    let palette = options.display.backbone_palette();
    let global_kind = options.display.surface_kind();
    let global_opacity = options.display.surface_opacity();
    let global_show_cavities = options.display.show_cavities();

    // Collect jobs: (positions, radii, surface params with color)
    let mut jobs: Vec<(Vec<glam::Vec3>, Vec<f32>, EntitySurface)> = Vec::new();
    // Cavity jobs: (positions, radii). Color is the fixed CAVITY_RGBA
    // constant — cavities don't pick up per-entity coloring.
    let mut cavity_jobs: Vec<(Vec<glam::Vec3>, Vec<f32>)> = Vec::new();

    for (entity_idx, se) in all_entities.iter().enumerate() {
        let eid = se.id();
        if !annotations.is_visible(eid) {
            continue;
        }

        // Per-entity surface takes priority; fall back to global
        let base_surface = annotations.surfaces.get(&eid).map_or_else(
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
    let density_jobs: Vec<_> = density
        .visible_entries()
        .map(|(_id, entry)| {
            let [r, g, b] = entry.color;
            (entry.map.clone(), entry.threshold, [r, g, b, entry.opacity])
        })
        .collect();

    if jobs.is_empty() && density_jobs.is_empty() && cavity_jobs.is_empty() {
        // Nothing to generate — send empty mesh to clear renderer
        let _ = regen.tx.send((Vec::new(), Vec::new()));
        return;
    }

    let tx = regen.tx.clone();

    let spawn_result = std::thread::Builder::new()
        .name("viso-surface-regen".into())
        .spawn(move || {
            let mut all_verts: Vec<IsosurfaceVertex> = Vec::new();
            let mut all_idxs: Vec<u32> = Vec::new();

            // Generate density map meshes first
            use crate::renderer::geometry::isosurface::density;
            for (map, threshold, color) in &density_jobs {
                let (v, i) = density::generate_density_mesh(
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

    if let Err(e) = spawn_result {
        log::warn!("failed to spawn surface regen thread: {e}");
    }
}
