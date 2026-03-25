//! Engine methods for loading and managing electron density maps.

use molex::entity::surface::Density;

use super::VisoEngine;

impl VisoEngine {
    /// Load a density map into the engine. Returns the assigned map ID.
    ///
    /// The map is immediately visible at the default sigma level (1.5σ).
    /// Mesh generation runs on a background thread; the result appears
    /// on the next frame after completion. The mesh is cropped to the
    /// bounding box of currently loaded entities.
    pub fn load_density_map(&mut self, map: Density) -> u32 {
        let id = self.density.add(map);
        log::info!("loaded density map id={id}");
        if let Some((centroid, radius)) = self.entities.bounding_sphere() {
            log::info!(
                "protein bounding sphere: center=[{:.1},{:.1},{:.1}], \
                 radius={:.1}, range=[{:.1},{:.1},{:.1}]→[{:.1},{:.1},{:.1}]",
                centroid.x,
                centroid.y,
                centroid.z,
                radius,
                centroid.x - radius,
                centroid.y - radius,
                centroid.z - radius,
                centroid.x + radius,
                centroid.y + radius,
                centroid.z + radius,
            );
        }
        self.regenerate_density_mesh();
        id
    }

    /// Remove a density map by ID.
    pub fn remove_density_map(&mut self, id: u32) {
        if self.density.remove(id) {
            log::info!("removed density map id={id}");
            self.regenerate_density_mesh();
        }
    }

    /// Set the raw density threshold for a density map.
    pub fn set_density_threshold(&mut self, id: u32, threshold: f32) {
        log::info!("set_density_threshold id={id} threshold={threshold:.4}");
        self.density.set_threshold(id, threshold);
        self.regenerate_density_mesh();
    }

    /// Set visibility for a density map.
    pub fn set_density_visible(&mut self, id: u32, visible: bool) {
        self.density.set_visible(id, visible);
        self.regenerate_density_mesh();
    }

    /// Set color for a density map.
    pub fn set_density_color(&mut self, id: u32, color: [f32; 3]) {
        self.density.set_color(id, color);
        self.regenerate_density_mesh();
    }

    /// Set opacity for a density map (0.0–1.0).
    pub fn set_density_opacity(&mut self, id: u32, opacity: f32) {
        self.density.set_opacity(id, opacity);
        self.regenerate_density_mesh();
    }

    /// Regenerate all isosurface meshes (density maps + entity surfaces).
    fn regenerate_density_mesh(&self) {
        self.regenerate_entity_surfaces();
    }
}
