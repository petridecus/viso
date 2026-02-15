//! Authoritative scene: entity groups, focus cycling, aggregated render data.
//!
//! Everything is a `MoleculeEntity`. They're organized into **groups**
//! (entities loaded/created together from one file or operation).

use crate::bond_topology::{get_residue_bonds, is_hydrophobic};
use foldit_conv::coords::entity::{MoleculeEntity, MoleculeType, NucleotideRing, merge_entities};
use foldit_conv::coords::render::extract_sequences;
use foldit_conv::coords::types::Coords;
use foldit_conv::coords::{protein_only, RenderCoords, RenderBackboneResidue};
use foldit_conv::secondary_structure::SSType;
use foldit_conv::types::assembly::{
    prepare_combined_assembly, split_combined_result,
    update_entities_from_backend, update_protein_entities,
    residue_count as assembly_residue_count,
    protein_coords as assembly_protein_coords,
};
use glam::Vec3;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// IDs
// ---------------------------------------------------------------------------

/// Unique group identifier (atomic counter, assigned by Scene).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GroupId(pub u64);

impl GroupId {
    fn next() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

// ---------------------------------------------------------------------------
// Focus
// ---------------------------------------------------------------------------

/// Focus state for tab cycling.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Focus {
    /// All groups.
    #[default]
    Session,
    /// A specific loaded group (protein + ligands + etc.)
    Group(GroupId),
    /// A specific non-protein entity by globally-unique entity_id.
    Entity(u32),
}

// ---------------------------------------------------------------------------
// Per-residue render data (forwarded from controllers)
// ---------------------------------------------------------------------------

/// Per-residue render data aggregated from controllers.
#[derive(Debug, Clone, Default)]
pub struct ResidueRenderData {
    pub rama_colors: Option<Vec<[f32; 3]>>,
    pub blueprint_colors: Option<Vec<[f32; 3]>>,
    pub selection: Vec<u32>,
    pub color_mode: ResidueColorMode,
}

/// Which controller provides the current residue coloring.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum ResidueColorMode {
    #[default]
    Default,
    Rama,
    Blueprint,
    Alignment,
    Score,
}

// ---------------------------------------------------------------------------
// Cached per-group rendering data
// ---------------------------------------------------------------------------

/// Cached rendering data for a single group's protein entities.
#[derive(Debug, Clone)]
pub struct GroupRenderData {
    pub render_coords: RenderCoords,
    pub sequence: String,
    pub chain_sequences: Vec<(u8, String)>,
}

// ---------------------------------------------------------------------------
// PerGroupData — per-group data for scene processor
// ---------------------------------------------------------------------------

/// Sidechain atom data for a single group (local indices).
#[derive(Debug, Clone)]
pub struct SidechainAtom {
    pub position: Vec3,
    pub is_hydrophobic: bool,
    pub residue_idx: u32,
    pub atom_name: String,
}

/// All render data for a single group, ready for the scene processor.
/// Indices are LOCAL (0-based within the group).
#[derive(Debug, Clone)]
pub struct PerGroupData {
    pub id: GroupId,
    pub mesh_version: u64,
    // Protein backbone
    pub backbone_chains: Vec<Vec<Vec3>>,
    pub backbone_chain_ids: Vec<u8>,
    pub backbone_residue_chains: Vec<Vec<RenderBackboneResidue>>,
    // Sidechain (local residue indices)
    pub sidechain_atoms: Vec<SidechainAtom>,
    pub sidechain_bonds: Vec<(u32, u32)>,
    pub backbone_sidechain_bonds: Vec<(Vec3, u32)>,
    // Secondary structure
    pub ss_override: Option<Vec<SSType>>,
    // Per-residue energy scores (scene processor derives colors from these)
    pub per_residue_scores: Option<Vec<f64>>,
    // Non-protein
    pub non_protein_entities: Vec<MoleculeEntity>,
    // Nucleic acid
    pub nucleic_acid_chains: Vec<Vec<Vec3>>,
    pub nucleic_acid_rings: Vec<NucleotideRing>,
    // Counts
    pub residue_count: u32,
}

// ---------------------------------------------------------------------------
// EntityGroup
// ---------------------------------------------------------------------------

/// A group of entities loaded together (from one file or one backend operation).
#[derive(Debug, Clone)]
pub struct EntityGroup {
    pub id: GroupId,
    pub visible: bool,
    pub name: String,
    entities: Vec<MoleculeEntity>,
    mesh_version: u64,
    pub ss_override: Option<Vec<SSType>>,
    /// Cached per-residue energy scores from Rosetta (raw data for viz).
    pub per_residue_scores: Option<Vec<f64>>,
    /// Cached per-group rendering data (derived from protein entities).
    render_cache: Option<GroupRenderData>,
}

impl EntityGroup {
    /// Human-readable name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Access entities (immutable).
    pub fn entities(&self) -> &[MoleculeEntity] {
        &self.entities
    }

    /// Mutable access to entities. Bumps mesh_version.
    pub fn entities_mut(&mut self) -> &mut Vec<MoleculeEntity> {
        self.mesh_version += 1;
        &mut self.entities
    }

    /// Replace entities. Bumps mesh_version.
    pub fn set_entities(&mut self, entities: Vec<MoleculeEntity>) {
        self.entities = entities;
        self.mesh_version += 1;
    }

    /// Current mesh version (cache key for scene processor).
    pub fn mesh_version(&self) -> u64 {
        self.mesh_version
    }

    /// Derive (or return cached) render data from protein entities.
    pub fn render_data(&mut self) -> Option<&GroupRenderData> {
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

    /// Get protein-only Coords for this group.
    pub fn protein_coords(&self) -> Option<Coords> {
        let coords = assembly_protein_coords(&self.entities);
        if coords.num_atoms == 0 { None } else { Some(coords) }
    }

    /// Get merged Coords for ALL entities.
    pub fn all_coords(&self) -> Option<Coords> {
        if self.entities.is_empty() { return None; }
        let coords = merge_entities(&self.entities);
        if coords.num_atoms == 0 { None } else { Some(coords) }
    }

    /// Build PerGroupData from this group's current state.
    pub fn to_per_group_data(&mut self) -> Option<PerGroupData> {
        // Collect non-protein entities and nucleic acid data
        let mut non_protein_entities = Vec::new();
        let mut nucleic_acid_chains = Vec::new();
        let mut nucleic_acid_rings = Vec::new();

        for entity in &self.entities {
            if entity.molecule_type != MoleculeType::Protein {
                non_protein_entities.push(entity.clone());
            }
            if matches!(entity.molecule_type, MoleculeType::DNA | MoleculeType::RNA) {
                nucleic_acid_chains.extend(entity.extract_p_atom_chains());
                nucleic_acid_rings.extend(entity.extract_base_rings());
            }
        }

        let ss_override = self.ss_override.clone();

        // Get protein render data
        let render_data = self.render_data();
        let (backbone_chains, backbone_chain_ids, backbone_residue_chains,
             sidechain_atoms, sidechain_bonds, backbone_sidechain_bonds, residue_count_val) =
            if let Some(rd) = render_data {
                let rc = &rd.render_coords;
                let res_count: u32 = rc.backbone_chains.iter()
                    .map(|c| (c.len() / 3) as u32)
                    .sum();

                let sc_atoms: Vec<SidechainAtom> = rc.sidechain_atoms.iter()
                    .map(|a| SidechainAtom {
                        position: a.position,
                        is_hydrophobic: a.is_hydrophobic,
                        residue_idx: a.residue_idx,
                        atom_name: a.atom_name.clone(),
                    })
                    .collect();

                (
                    rc.backbone_chains.clone(),
                    rc.backbone_chain_ids.clone(),
                    rc.backbone_residue_chains.clone(),
                    sc_atoms,
                    rc.sidechain_bonds.clone(),
                    rc.backbone_sidechain_bonds.clone(),
                    res_count,
                )
            } else {
                (vec![], vec![], vec![], vec![], vec![], vec![], 0)
            };

        // Skip groups with no geometry at all
        if backbone_chains.is_empty() && non_protein_entities.is_empty()
            && nucleic_acid_chains.is_empty()
        {
            return None;
        }

        Some(PerGroupData {
            id: self.id,
            mesh_version: self.mesh_version,
            backbone_chains,
            backbone_chain_ids,
            backbone_residue_chains,
            sidechain_atoms,
            sidechain_bonds,
            backbone_sidechain_bonds,
            ss_override,
            per_residue_scores: self.per_residue_scores.clone(),
            non_protein_entities,
            nucleic_acid_chains,
            nucleic_acid_rings,
            residue_count: residue_count_val,
        })
    }

    fn compute_render_data(&self) -> Option<GroupRenderData> {
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

        Some(GroupRenderData {
            render_coords,
            sequence,
            chain_sequences,
        })
    }
}

// ---------------------------------------------------------------------------
// Aggregated render data (kept for backward compat with animation etc.)
// ---------------------------------------------------------------------------

/// Pre-computed aggregated data for efficient rendering across all visible groups.
#[derive(Debug, Clone, Default)]
pub struct AggregatedRenderData {
    pub backbone_chains: Vec<Vec<Vec3>>,
    pub backbone_chain_ids: Vec<u8>,
    pub backbone_residue_chains: Vec<Vec<RenderBackboneResidue>>,
    pub sidechain_positions: Vec<Vec3>,
    pub sidechain_hydrophobicity: Vec<bool>,
    pub sidechain_residue_indices: Vec<u32>,
    pub sidechain_atom_names: Vec<String>,
    pub sidechain_bonds: Vec<(u32, u32)>,
    pub backbone_sidechain_bonds: Vec<(Vec3, u32)>,
    pub all_positions: Vec<Vec3>,
    pub ss_types: Option<Vec<SSType>>,
    pub residue_render_data: ResidueRenderData,
    /// All non-protein entities across all visible groups (for ball-and-stick).
    pub non_protein_entities: Vec<MoleculeEntity>,
    /// P-atom chains from DNA/RNA entities (for nucleic acid ribbon rendering).
    pub nucleic_acid_chains: Vec<Vec<Vec3>>,
    /// Base ring geometry from DNA/RNA entities (for filled polygon rendering).
    pub nucleic_acid_rings: Vec<NucleotideRing>,
}

// ---------------------------------------------------------------------------
// CombinedCoordsResult
// ---------------------------------------------------------------------------

/// Result of combining coords from all visible groups for Rosetta.
#[derive(Debug, Clone)]
pub struct CombinedCoordsResult {
    pub bytes: Vec<u8>,
    /// Chain IDs assigned to each group (for splitting Rosetta exports by chain).
    pub chain_ids_per_group: Vec<(GroupId, Vec<u8>)>,
    /// Residue ranges per group: GroupId -> (start_residue, end_residue) - 1-indexed, inclusive.
    pub residue_ranges: HashMap<GroupId, (usize, usize)>,
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

/// The authoritative scene. Owns all entity groups.
pub struct Scene {
    /// Groups in insertion order.
    groups: Vec<EntityGroup>,
    focus: Focus,
    next_entity_id: u32,
    agg_cache: Option<Arc<AggregatedRenderData>>,
    /// Monotonically increasing generation; bumped on any mutation.
    generation: u64,
    /// Generation that was last consumed by the renderer.
    rendered_generation: u64,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            groups: Vec::new(),
            focus: Focus::Session,
            next_entity_id: 0,
            agg_cache: None,
            generation: 0,
            rendered_generation: 0,
        }
    }

    // -- Mutation helpers --

    fn invalidate(&mut self) {
        self.agg_cache = None;
        self.generation += 1;
    }

    /// Whether scene data changed since last `mark_rendered()`.
    pub fn is_dirty(&self) -> bool {
        self.generation != self.rendered_generation
    }

    /// Force the scene dirty (e.g. when display options change but scene data hasn't).
    pub fn force_dirty(&mut self) {
        self.invalidate();
    }

    /// Mark current generation as rendered (call after updating renderers).
    pub fn mark_rendered(&mut self) {
        self.rendered_generation = self.generation;
    }

    // -- Group management --

    /// Add a group of entities. Entity IDs are reassigned to be globally unique.
    pub fn add_group(&mut self, mut entities: Vec<MoleculeEntity>, name: impl Into<String>) -> GroupId {
        // Reassign globally unique entity IDs
        for entity in &mut entities {
            entity.entity_id = self.next_entity_id;
            self.next_entity_id += 1;
        }

        let id = GroupId::next();
        self.groups.push(EntityGroup {
            id,
            visible: true,
            name: name.into(),
            entities,
            mesh_version: 0,
            ss_override: None,
            per_residue_scores: None,
            render_cache: None,
        });
        self.invalidate();
        id
    }

    /// Remove a group by ID. Returns the removed group, if any.
    pub fn remove_group(&mut self, id: GroupId) -> Option<EntityGroup> {
        let idx = self.groups.iter().position(|g| g.id == id)?;
        let group = self.groups.remove(idx);
        self.invalidate();
        Some(group)
    }

    /// Read access to a group.
    pub fn group(&self, id: GroupId) -> Option<&EntityGroup> {
        self.groups.iter().find(|g| g.id == id)
    }

    /// Write access (invalidates cache).
    pub fn group_mut(&mut self, id: GroupId) -> Option<&mut EntityGroup> {
        self.invalidate();
        self.groups.iter_mut().find(|g| g.id == id)
    }

    /// Ordered group IDs.
    pub fn group_ids(&self) -> Vec<GroupId> {
        self.groups.iter().map(|g| g.id).collect()
    }

    /// Number of groups.
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Toggle visibility.
    pub fn set_visible(&mut self, id: GroupId, visible: bool) {
        if let Some(g) = self.groups.iter_mut().find(|g| g.id == id) {
            if g.visible != visible {
                g.visible = visible;
                self.invalidate();
            }
        }
    }

    /// Remove all groups and reset.
    pub fn clear(&mut self) {
        self.groups.clear();
        self.focus = Focus::Session;
        self.invalidate();
    }

    /// Check if a group exists.
    pub fn contains(&self, id: GroupId) -> bool {
        self.groups.iter().any(|g| g.id == id)
    }

    /// Iterate over all groups (insertion order).
    pub fn iter(&self) -> impl Iterator<Item = &EntityGroup> {
        self.groups.iter()
    }

    // -- Focus / tab cycling --

    pub fn focus(&self) -> &Focus {
        &self.focus
    }

    pub fn set_focus(&mut self, focus: Focus) {
        self.focus = focus;
    }

    /// Cycle: Session -> Group1 -> ... -> GroupN -> focusable entities -> Session.
    pub fn cycle_focus(&mut self) -> Focus {
        let focusable_entities: Vec<u32> = self.groups.iter()
            .filter(|g| g.visible)
            .flat_map(|g| g.entities().iter())
            .filter(|e| e.is_focusable())
            .map(|e| e.entity_id)
            .collect();

        let group_ids = self.group_ids();

        self.focus = match self.focus {
            Focus::Session => {
                group_ids.first()
                    .map(|&id| Focus::Group(id))
                    .or_else(|| focusable_entities.first().map(|&id| Focus::Entity(id)))
                    .unwrap_or(Focus::Session)
            }
            Focus::Group(current_id) => {
                let idx = group_ids.iter().position(|&id| id == current_id);
                match idx {
                    Some(i) if i + 1 < group_ids.len() => Focus::Group(group_ids[i + 1]),
                    _ => {
                        focusable_entities.first()
                            .map(|&id| Focus::Entity(id))
                            .unwrap_or(Focus::Session)
                    }
                }
            }
            Focus::Entity(current_id) => {
                let idx = focusable_entities.iter().position(|&id| id == current_id);
                match idx {
                    Some(i) if i + 1 < focusable_entities.len() => {
                        Focus::Entity(focusable_entities[i + 1])
                    }
                    _ => Focus::Session,
                }
            }
        };
        self.focus
    }

    /// Revert to Session if focused group/entity was removed.
    pub fn validate_focus(&mut self) {
        match self.focus {
            Focus::Session => {}
            Focus::Group(id) => {
                if !self.contains(id) {
                    self.focus = Focus::Session;
                }
            }
            Focus::Entity(eid) => {
                let exists = self.groups.iter()
                    .flat_map(|g| g.entities().iter())
                    .any(|e| e.entity_id == eid);
                if !exists {
                    self.focus = Focus::Session;
                }
            }
        }
    }

    /// Human-readable description of current focus.
    pub fn focus_description(&self) -> String {
        match self.focus {
            Focus::Session => "Session (all structures)".to_string(),
            Focus::Group(id) => {
                let idx = self.groups.iter().position(|g| g.id == id).unwrap_or(0) + 1;
                let name = self.group(id).map(|g| g.name().to_string()).unwrap_or_default();
                format!("Structure {} ({})", idx, name)
            }
            Focus::Entity(eid) => {
                for g in &self.groups {
                    for e in g.entities() {
                        if e.entity_id == eid {
                            return e.label();
                        }
                    }
                }
                "Entity (unknown)".to_string()
            }
        }
    }

    // -- Per-group data for scene processor --

    /// Collect per-group render data for all visible groups.
    pub fn per_group_data(&mut self) -> Vec<PerGroupData> {
        self.groups.iter_mut()
            .filter(|g| g.visible)
            .filter_map(|g| g.to_per_group_data())
            .collect()
    }

    /// All atom positions across all visible groups (for camera fitting).
    pub fn all_positions(&self) -> Vec<Vec3> {
        self.groups.iter()
            .filter(|g| g.visible)
            .flat_map(|g| g.entities().iter())
            .flat_map(|e| e.coords.atoms.iter().map(|a| Vec3::new(a.x, a.y, a.z)))
            .collect()
    }

    // -- Aggregated data (lazy cached, for animation/passthrough) --

    /// Get aggregated render data. Computed lazily and cached.
    pub fn aggregated(&mut self) -> Arc<AggregatedRenderData> {
        if self.agg_cache.is_none() {
            self.agg_cache = Some(Arc::new(self.compute_aggregated()));
        }
        Arc::clone(self.agg_cache.as_ref().unwrap())
    }

    fn compute_aggregated(&mut self) -> AggregatedRenderData {
        let mut data = AggregatedRenderData::default();
        let mut global_residue_offset: u32 = 0;
        let mut has_any_ss_override = false;
        let mut ss_parts: Vec<(u32, Option<Vec<SSType>>, u32)> = Vec::new();

        for group in &mut self.groups {
            if !group.visible {
                continue;
            }

            // Collect non-protein entities for ball-and-stick, and NA chains for ribbon
            for entity in group.entities() {
                if entity.molecule_type != MoleculeType::Protein {
                    data.non_protein_entities.push(entity.clone());
                }
                if matches!(entity.molecule_type, MoleculeType::DNA | MoleculeType::RNA) {
                    let p_chains = entity.extract_p_atom_chains();
                    for chain in &p_chains {
                        data.all_positions.extend(chain);
                    }
                    data.nucleic_acid_chains.extend(p_chains);
                    data.nucleic_acid_rings.extend(entity.extract_base_rings());
                }
            }

            // Capture ss_override before the mutable borrow from render_data()
            let ss_override = group.ss_override.clone();

            let group_name = group.name().to_string();
            let render_data = match group.render_data() {
                Some(rd) => {
                    eprintln!("[scene::aggregated] group '{}': {} backbone chains, {} residues, {} sidechain atoms",
                        group_name,
                        rd.render_coords.backbone_chains.len(),
                        rd.render_coords.backbone_chains.iter().map(|c| c.len() / 3).sum::<usize>(),
                        rd.render_coords.sidechain_atoms.len(),
                    );
                    rd
                }
                None => {
                    eprintln!("[scene::aggregated] group '{}': no protein render data", group_name);
                    continue;
                }
            };

            let atom_offset = data.sidechain_positions.len() as u32;

            // Count residues
            let structure_residue_count: u32 = render_data.render_coords.backbone_chains
                .iter()
                .map(|c| (c.len() / 3) as u32)
                .sum();

            // Track SS override
            if ss_override.is_some() {
                has_any_ss_override = true;
            }
            ss_parts.push((global_residue_offset, ss_override, structure_residue_count));

            // Aggregate backbone
            for chain in &render_data.render_coords.backbone_chains {
                data.backbone_chains.push(chain.clone());
                data.all_positions.extend(chain);
            }
            for chain_id in &render_data.render_coords.backbone_chain_ids {
                data.backbone_chain_ids.push(*chain_id);
            }
            for residue_chain in &render_data.render_coords.backbone_residue_chains {
                data.backbone_residue_chains.push(residue_chain.clone());
            }

            // Aggregate sidechain atoms with global residue indices
            for atom in &render_data.render_coords.sidechain_atoms {
                data.sidechain_positions.push(atom.position);
                data.sidechain_hydrophobicity.push(atom.is_hydrophobic);
                data.sidechain_residue_indices.push(atom.residue_idx + global_residue_offset);
                data.sidechain_atom_names.push(atom.atom_name.clone());
                data.all_positions.push(atom.position);
            }

            // Aggregate bonds (adjust indices by offset)
            for &(a, b) in &render_data.render_coords.sidechain_bonds {
                data.sidechain_bonds.push((a + atom_offset, b + atom_offset));
            }

            // Aggregate backbone-sidechain bonds
            for &(ca_pos, cb_idx) in &render_data.render_coords.backbone_sidechain_bonds {
                data.backbone_sidechain_bonds.push((ca_pos, cb_idx + atom_offset));
            }

            global_residue_offset += structure_residue_count;
        }

        // Build flat ss_types if any group has an override
        if has_any_ss_override {
            let total_residues = global_residue_offset as usize;
            let mut ss_types = vec![SSType::Coil; total_residues];
            for (offset, ss_override, count) in &ss_parts {
                if let Some(overrides) = ss_override {
                    let start = *offset as usize;
                    let end = (start + *count as usize).min(total_residues);
                    for (i, &ss) in overrides.iter().enumerate() {
                        if start + i < end {
                            ss_types[start + i] = ss;
                        }
                    }
                }
            }
            data.ss_types = Some(ss_types);
        }

        data
    }

    // -- Backend support (combined coords for Rosetta) --

    /// Get combined ASSEM01 bytes from all visible groups for Rosetta operations.
    pub fn combined_coords_for_backend(&self) -> Option<CombinedCoordsResult> {
        let visible_groups: Vec<&EntityGroup> = self.groups.iter()
            .filter(|g| g.visible && assembly_residue_count(g.entities()) > 0)
            .collect();

        if visible_groups.is_empty() {
            return None;
        }

        let entity_slices: Vec<&[MoleculeEntity]> = visible_groups.iter()
            .map(|g| g.entities())
            .collect();
        let group_ids: Vec<GroupId> = visible_groups.iter()
            .map(|g| g.id)
            .collect();

        let combined = prepare_combined_assembly(&entity_slices)?;

        let chain_ids_per_group: Vec<(GroupId, Vec<u8>)> = group_ids.iter()
            .zip(combined.chain_ids.iter())
            .map(|(&gid, chains)| (gid, chains.clone()))
            .collect();

        let residue_ranges: HashMap<GroupId, (usize, usize)> = group_ids.iter()
            .zip(combined.residue_ranges.iter())
            .map(|(&gid, &range)| (gid, range))
            .collect();

        Some(CombinedCoordsResult {
            bytes: combined.bytes,
            chain_ids_per_group,
            residue_ranges,
        })
    }

    /// Apply combined Rosetta update to all groups in the session.
    pub fn apply_combined_update(
        &mut self,
        coords_bytes: &[u8],
        chain_ids_per_group: &[(GroupId, Vec<u8>)],
    ) -> Result<(), String> {
        let chain_ids_list: Vec<Vec<u8>> = chain_ids_per_group.iter()
            .map(|(_, chains)| chains.clone())
            .collect();

        let split_coords = split_combined_result(coords_bytes, &chain_ids_list)?;

        for (i, (group_id, _)) in chain_ids_per_group.iter().enumerate() {
            let structure_coords = &split_coords[i];
            if structure_coords.num_atoms == 0 {
                log::warn!("No atoms found for group {:?}", group_id);
                continue;
            }

            if let Some(group) = self.groups.iter_mut().find(|g| g.id == *group_id) {
                update_entities_from_backend(group.entities_mut(), structure_coords.clone());
                group.invalidate_render_cache();
                log::info!("Updated group {:?} from backend ({} atoms)",
                    group_id, structure_coords.num_atoms);
            }
        }

        self.invalidate();
        Ok(())
    }

    /// Update protein entities' coords in a specific group.
    pub fn update_group_protein_coords(&mut self, id: GroupId, coords: Coords) {
        if let Some(group) = self.groups.iter_mut().find(|g| g.id == id) {
            update_protein_entities(group.entities_mut(), coords);
            group.invalidate_render_cache();
        }
        self.invalidate();
    }

    /// Get visible group IDs and their residue counts (for Rosetta topology check).
    pub fn visible_residue_counts(&self) -> Vec<(GroupId, usize)> {
        self.groups.iter()
            .filter(|g| g.visible)
            .map(|g| (g.id, assembly_residue_count(g.entities())))
            .collect()
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}
