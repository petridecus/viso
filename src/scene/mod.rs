//! Authoritative scene: flat entity storage, focus cycling, per-entity
//! render data.
//!
//! Everything is a [`MoleculeEntity`]. Each entity is wrapped in a
//! [`SceneEntity`] that pairs the core data with rendering metadata
//! (visibility, name, SS override, score cache, mesh version).

pub mod processor;

use foldit_conv::{
    coords::{
        entity::{
            merge_entities, MoleculeEntity, MoleculeType, NucleotideRing,
        },
        protein_only,
        render::extract_sequences,
        types::Coords,
        RenderBackboneResidue, RenderCoords,
    },
    secondary_structure::SSType,
    types::assembly::{
        protein_coords as assembly_protein_coords, update_protein_entities,
    },
};
use glam::Vec3;

use crate::util::bond_topology::{get_residue_bonds, is_hydrophobic};

// ---------------------------------------------------------------------------
// Focus
// ---------------------------------------------------------------------------

/// Focus state for tab cycling.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Focus {
    /// All entities.
    #[default]
    Session,
    /// A specific entity by ID.
    Entity(u32),
}

// ---------------------------------------------------------------------------
// Cached per-entity rendering data
// ---------------------------------------------------------------------------

/// Cached rendering data for a single entity's protein content.
#[derive(Debug, Clone)]
pub struct EntityRenderData {
    /// Extracted backbone and sidechain coordinates for rendering.
    pub render_coords: RenderCoords,
    /// Full amino acid sequence string.
    pub sequence: String,
    /// Per-chain sequences as (chain_id, sequence) pairs.
    pub chain_sequences: Vec<(u8, String)>,
}

// ---------------------------------------------------------------------------
// PerEntityData — per-entity data for scene processor
// ---------------------------------------------------------------------------

/// Sidechain atom data for a single entity (local indices).
#[derive(Debug, Clone)]
pub struct SidechainAtom {
    /// Atom position in world space.
    pub position: Vec3,
    /// Whether this atom is hydrophobic.
    pub is_hydrophobic: bool,
    /// Local residue index within the entity.
    pub residue_idx: u32,
    /// PDB atom name (e.g. "CB", "CG").
    pub atom_name: String,
}

/// All render data for a single entity, ready for the scene processor.
/// Indices are LOCAL (0-based within the entity).
#[derive(Debug, Clone)]
pub struct PerEntityData {
    /// Entity identifier.
    pub id: u32,
    /// Monotonic version counter for cache invalidation.
    pub mesh_version: u64,
    /// Protein backbone atom chains (N, CA, C triplets).
    pub backbone_chains: Vec<Vec<Vec3>>,
    /// Chain IDs for each backbone chain.
    pub backbone_chain_ids: Vec<u8>,
    /// Per-chain backbone residue data.
    pub backbone_residue_chains: Vec<Vec<RenderBackboneResidue>>,
    /// Sidechain atom data (local residue indices).
    pub sidechain_atoms: Vec<SidechainAtom>,
    /// Sidechain intra-residue bonds as (atom_idx, atom_idx) pairs.
    pub sidechain_bonds: Vec<(u32, u32)>,
    /// Backbone-to-sidechain bonds as (CA position, sidechain atom idx).
    pub backbone_sidechain_bonds: Vec<(Vec3, u32)>,
    /// Pre-computed secondary structure assignments.
    pub ss_override: Option<Vec<SSType>>,
    /// Per-residue energy scores for color derivation.
    pub per_residue_scores: Option<Vec<f64>>,
    /// Non-protein entities (ligands, ions, etc.).
    pub non_protein_entities: Vec<MoleculeEntity>,
    /// P-atom chains from DNA/RNA entities.
    pub nucleic_acid_chains: Vec<Vec<Vec3>>,
    /// Base ring geometry from DNA/RNA entities.
    pub nucleic_acid_rings: Vec<NucleotideRing>,
    /// Total residue count in this entity.
    pub residue_count: u32,
}

// ---------------------------------------------------------------------------
// SceneEntity
// ---------------------------------------------------------------------------

/// A scene entity wrapping a single [`MoleculeEntity`] with rendering
/// metadata.
#[derive(Debug, Clone)]
pub struct SceneEntity {
    /// Core molecule data from foldit-conv.
    pub entity: MoleculeEntity,
    /// Whether this entity is visible in the scene.
    pub visible: bool,
    /// Human-readable name (defaults to entity label).
    pub name: String,
    /// Pre-computed secondary structure assignments.
    pub ss_override: Option<Vec<SSType>>,
    /// Cached per-residue energy scores from Rosetta (raw data for viz).
    pub per_residue_scores: Option<Vec<f64>>,
    mesh_version: u64,
    render_cache: Option<EntityRenderData>,
}

impl SceneEntity {
    /// Entity identifier (delegates to `entity.entity_id`).
    #[must_use]
    pub fn id(&self) -> u32 {
        self.entity.entity_id
    }

    /// Immutable access to the underlying molecule entity.
    #[must_use]
    pub fn entity(&self) -> &MoleculeEntity {
        &self.entity
    }

    /// Replace the underlying entity data. Preserves the scene-assigned
    /// entity ID. Bumps mesh version and invalidates the render cache.
    pub fn set_entity(&mut self, entity: MoleculeEntity) {
        let id = self.entity.entity_id; // preserve scene-assigned ID
        self.entity = entity;
        self.entity.entity_id = id;
        self.mesh_version += 1;
        self.render_cache = None;
    }

    /// Current mesh version (cache key for scene processor).
    #[must_use]
    pub fn mesh_version(&self) -> u64 {
        self.mesh_version
    }

    /// Derive (or return cached) render data from protein content.
    pub fn render_data(&mut self) -> Option<&EntityRenderData> {
        if self.render_cache.is_none() {
            self.render_cache = self.compute_render_data();
        }
        self.render_cache.as_ref()
    }

    /// Invalidate cached render data (call after coord changes).
    pub fn invalidate_render_cache(&mut self) {
        self.render_cache = None;
        self.mesh_version += 1;
    }

    /// Set per-residue energy scores (raw data cached for future viz).
    /// Does NOT bump mesh_version — the scene processor derives and caches
    /// colors from these scores, and the animation path reuses cached colors.
    pub fn set_per_residue_scores(&mut self, scores: Option<Vec<f64>>) {
        self.per_residue_scores = scores;
    }

    /// Get protein-only Coords for this entity.
    #[must_use]
    pub fn protein_coords(&self) -> Option<Coords> {
        if self.entity.molecule_type != MoleculeType::Protein {
            return None;
        }
        let coords =
            assembly_protein_coords(std::slice::from_ref(&self.entity));
        if coords.num_atoms == 0 {
            None
        } else {
            Some(coords)
        }
    }

    /// Get merged Coords for this entity.
    #[must_use]
    pub fn all_coords(&self) -> Option<Coords> {
        if self.entity.coords.num_atoms == 0 {
            return None;
        }
        let coords = merge_entities(std::slice::from_ref(&self.entity));
        if coords.num_atoms == 0 {
            None
        } else {
            Some(coords)
        }
    }

    /// Build [`PerEntityData`] from this entity's current state.
    pub fn to_per_entity_data(&mut self) -> Option<PerEntityData> {
        match self.entity.molecule_type {
            MoleculeType::Protein => self.per_entity_data_protein(),
            MoleculeType::DNA | MoleculeType::RNA => {
                self.per_entity_data_nucleic_acid()
            }
            _ => self.per_entity_data_non_protein(),
        }
    }

    /// Render data for a protein entity.
    fn per_entity_data_protein(&mut self) -> Option<PerEntityData> {
        let id = self.entity.entity_id;
        let mesh_version = self.mesh_version;
        let ss_override = self.ss_override.clone();
        let per_residue_scores = self.per_residue_scores.clone();

        let render_data = self.render_data()?;
        let rc = &render_data.render_coords;
        let res_count: u32 = rc
            .backbone_chains
            .iter()
            .map(|c| (c.len() / 3) as u32)
            .sum();

        let sc_atoms: Vec<SidechainAtom> = rc
            .sidechain_atoms
            .iter()
            .map(|a| SidechainAtom {
                position: a.position,
                is_hydrophobic: a.is_hydrophobic,
                residue_idx: a.residue_idx,
                atom_name: a.atom_name.clone(),
            })
            .collect();

        Some(PerEntityData {
            id,
            mesh_version,
            backbone_chains: rc.backbone_chains.clone(),
            backbone_chain_ids: rc.backbone_chain_ids.clone(),
            backbone_residue_chains: rc.backbone_residue_chains.clone(),
            sidechain_atoms: sc_atoms,
            sidechain_bonds: rc.sidechain_bonds.clone(),
            backbone_sidechain_bonds: rc.backbone_sidechain_bonds.clone(),
            ss_override,
            per_residue_scores,
            non_protein_entities: vec![],
            nucleic_acid_chains: vec![],
            nucleic_acid_rings: vec![],
            residue_count: res_count,
        })
    }

    /// Render data for a DNA/RNA entity.
    fn per_entity_data_nucleic_acid(&self) -> Option<PerEntityData> {
        let chains = self.entity.extract_p_atom_chains();
        let rings = self.entity.extract_base_rings();
        if chains.is_empty() && rings.is_empty() {
            return None;
        }
        Some(PerEntityData {
            id: self.entity.entity_id,
            mesh_version: self.mesh_version,
            backbone_chains: vec![],
            backbone_chain_ids: vec![],
            backbone_residue_chains: vec![],
            sidechain_atoms: vec![],
            sidechain_bonds: vec![],
            backbone_sidechain_bonds: vec![],
            ss_override: None,
            per_residue_scores: None,
            non_protein_entities: vec![self.entity.clone()],
            nucleic_acid_chains: chains,
            nucleic_acid_rings: rings,
            residue_count: 0,
        })
    }

    /// Render data for a non-protein, non-nucleic-acid entity.
    fn per_entity_data_non_protein(&self) -> Option<PerEntityData> {
        if self.entity.coords.num_atoms == 0 {
            return None;
        }
        Some(PerEntityData {
            id: self.entity.entity_id,
            mesh_version: self.mesh_version,
            backbone_chains: vec![],
            backbone_chain_ids: vec![],
            backbone_residue_chains: vec![],
            sidechain_atoms: vec![],
            sidechain_bonds: vec![],
            backbone_sidechain_bonds: vec![],
            ss_override: None,
            per_residue_scores: None,
            non_protein_entities: vec![self.entity.clone()],
            nucleic_acid_chains: vec![],
            nucleic_acid_rings: vec![],
            residue_count: 0,
        })
    }

    fn compute_render_data(&self) -> Option<EntityRenderData> {
        let protein_coords = self.protein_coords()?;
        let coords = protein_only(&protein_coords);
        if coords.num_atoms == 0 {
            return None;
        }

        let render_coords = RenderCoords::from_coords_with_topology(
            &coords,
            is_hydrophobic,
            |name| get_residue_bonds(name).map(|b| b.to_vec()),
        );

        let (sequence, chain_sequences) = extract_sequences(&coords);

        Some(EntityRenderData {
            render_coords,
            sequence,
            chain_sequences,
        })
    }
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

/// The authoritative scene. Owns all entities in a flat list.
pub struct Scene {
    /// Entities in insertion order.
    entities: Vec<SceneEntity>,
    focus: Focus,
    next_entity_id: u32,
    /// Monotonically increasing generation; bumped on any mutation.
    generation: u64,
    /// Generation that was last consumed by the renderer.
    rendered_generation: u64,
}

impl Scene {
    /// Create an empty scene with no entities and session-level focus.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            focus: Focus::Session,
            next_entity_id: 0,
            generation: 0,
            rendered_generation: 0,
        }
    }

    // -- Mutation helpers --

    fn invalidate(&mut self) {
        self.generation += 1;
    }

    /// Whether scene data changed since last `mark_rendered()`.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.generation != self.rendered_generation
    }

    /// Force the scene dirty (e.g. when display options change but scene data
    /// hasn't).
    pub fn force_dirty(&mut self) {
        self.invalidate();
    }

    /// Mark current generation as rendered (call after updating renderers).
    pub fn mark_rendered(&mut self) {
        self.rendered_generation = self.generation;
    }

    // -- Entity management --

    /// Add entities to the scene. Entity IDs are reassigned to be globally
    /// unique. Returns the assigned entity IDs.
    pub fn add_entities(&mut self, entities: Vec<MoleculeEntity>) -> Vec<u32> {
        let mut ids = Vec::with_capacity(entities.len());
        for mut entity in entities {
            let id = self.next_entity_id;
            self.next_entity_id += 1;
            entity.entity_id = id;
            let name = entity.label();
            self.entities.push(SceneEntity {
                entity,
                visible: true,
                name,
                ss_override: None,
                per_residue_scores: None,
                mesh_version: 0,
                render_cache: None,
            });
            ids.push(id);
        }
        self.invalidate();
        ids
    }

    /// Remove an entity by ID. Returns the removed entity, if any.
    pub fn remove_entity(&mut self, id: u32) -> Option<SceneEntity> {
        let idx = self.entities.iter().position(|e| e.id() == id)?;
        let entity = self.entities.remove(idx);
        self.invalidate();
        Some(entity)
    }

    /// Remove multiple entities by ID.
    pub fn remove_entities(&mut self, ids: &[u32]) {
        self.entities.retain(|e| !ids.contains(&e.id()));
        self.invalidate();
    }

    /// Replace all entities whose IDs are in `old_ids` with new entities.
    /// Returns the new entity IDs.
    pub fn replace_entities(
        &mut self,
        old_ids: &[u32],
        new_entities: Vec<MoleculeEntity>,
    ) -> Vec<u32> {
        self.entities.retain(|e| !old_ids.contains(&e.id()));
        self.add_entities(new_entities)
    }

    /// Read access to an entity.
    #[must_use]
    pub fn entity(&self, id: u32) -> Option<&SceneEntity> {
        self.entities.iter().find(|e| e.id() == id)
    }

    /// Write access (invalidates cache).
    pub fn entity_mut(&mut self, id: u32) -> Option<&mut SceneEntity> {
        self.invalidate();
        self.entities.iter_mut().find(|e| e.id() == id)
    }

    /// Read access to all entities (insertion order).
    #[must_use]
    pub fn entities(&self) -> &[SceneEntity] {
        &self.entities
    }

    /// Number of entities.
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Toggle visibility.
    pub fn set_visible(&mut self, id: u32, visible: bool) {
        if let Some(e) = self.entities.iter_mut().find(|e| e.id() == id) {
            if e.visible != visible {
                e.visible = visible;
                self.invalidate();
            }
        }
    }

    /// Remove all entities and reset.
    pub fn clear(&mut self) {
        self.entities.clear();
        self.focus = Focus::Session;
        self.invalidate();
    }

    /// Check if an entity exists.
    #[must_use]
    pub fn contains(&self, id: u32) -> bool {
        self.entities.iter().any(|e| e.id() == id)
    }

    // -- Focus / tab cycling --

    /// Current focus state.
    #[must_use]
    pub fn focus(&self) -> &Focus {
        &self.focus
    }

    /// Set the focus state directly.
    pub fn set_focus(&mut self, focus: Focus) {
        self.focus = focus;
    }

    /// Cycle: Session -> Entity1 -> ... -> EntityN -> Session.
    pub fn cycle_focus(&mut self) -> Focus {
        let focusable: Vec<u32> = self
            .entities
            .iter()
            .filter(|e| e.visible)
            .map(|e| e.id())
            .collect();

        self.focus = match self.focus {
            Focus::Session => focusable
                .first()
                .map(|&id| Focus::Entity(id))
                .unwrap_or(Focus::Session),
            Focus::Entity(current_id) => {
                let idx = focusable.iter().position(|&id| id == current_id);
                match idx {
                    Some(i) if i + 1 < focusable.len() => {
                        Focus::Entity(focusable[i + 1])
                    }
                    _ => Focus::Session,
                }
            }
        };
        self.focus
    }

    /// Revert to Session if focused entity was removed.
    pub fn validate_focus(&mut self) {
        if let Focus::Entity(eid) = self.focus {
            if !self.contains(eid) {
                self.focus = Focus::Session;
            }
        }
    }

    /// Human-readable description of current focus.
    #[must_use]
    pub fn focus_description(&self) -> String {
        match self.focus {
            Focus::Session => "Session (all structures)".into(),
            Focus::Entity(eid) => self
                .entity(eid)
                .map(|e| e.name.clone())
                .unwrap_or_else(|| "Entity (unknown)".into()),
        }
    }

    // -- Per-entity data for scene processor --

    /// Collect per-entity render data for all visible entities.
    pub fn per_entity_data(&mut self) -> Vec<PerEntityData> {
        self.entities
            .iter_mut()
            .filter(|e| e.visible)
            .filter_map(|e| e.to_per_entity_data())
            .collect()
    }

    /// All atom positions across all visible entities (for camera fitting).
    #[must_use]
    pub fn all_positions(&self) -> Vec<Vec3> {
        self.entities
            .iter()
            .filter(|e| e.visible)
            .flat_map(|e| {
                e.entity
                    .coords
                    .atoms
                    .iter()
                    .map(|a| Vec3::new(a.x, a.y, a.z))
            })
            .collect()
    }

    /// Update protein entity coords in a specific entity.
    pub fn update_entity_protein_coords(&mut self, id: u32, coords: Coords) {
        if let Some(se) =
            self.entities.iter_mut().find(|e| e.entity.entity_id == id)
        {
            // Wrap in a temporary Vec for the assembly update function
            let mut entities = vec![se.entity.clone()];
            update_protein_entities(&mut entities, coords);
            if let Some(updated) = entities.into_iter().next() {
                se.entity = updated;
            }
            se.invalidate_render_cache();
        }
        self.invalidate();
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}
