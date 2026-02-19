//! Queries & Backend methods for ProteinRenderEngine

use foldit_conv::coords::get_ca_position_from_chains;
use glam::Vec3;

use super::ProteinRenderEngine;

impl ProteinRenderEngine {
    /// Get the CA position of a residue by index.
    /// Returns None if the residue index is out of bounds.
    pub fn get_residue_ca_position(&self, residue_idx: usize) -> Option<Vec3> {
        // First try animator (has interpolated positions during animation)
        if let Some(pos) = self.animator.get_ca_position(residue_idx) {
            return Some(pos);
        }

        // Fall back to tube_renderer's cached backbone chains
        get_ca_position_from_chains(
            self.tube_renderer.cached_chains(),
            residue_idx,
        )
    }

    /// Get the current visual backbone chains (interpolated during animation).
    pub fn get_current_backbone_chains(&self) -> Vec<Vec<Vec3>> {
        if self.animator.is_animating() {
            self.animator.get_backbone()
        } else {
            self.tube_renderer.cached_chains().to_vec()
        }
    }

    /// Get the current visual sidechain positions (interpolated during
    /// animation).
    pub fn get_current_sidechain_positions(&self) -> Vec<Vec3> {
        if self.animator.is_animating() && self.animator.has_sidechain_data() {
            self.animator.get_sidechain_positions()
        } else {
            self.sc.target_sidechain_positions.clone()
        }
    }

    /// Get the current visual CA positions for all residues (interpolated
    /// during animation).
    pub fn get_current_ca_positions(&self) -> Vec<Vec3> {
        let chains = self.get_current_backbone_chains();
        foldit_conv::coords::extract_ca_from_chains(&chains)
    }

    /// Get a single interpolated CA position by residue index.
    pub fn get_current_ca_position(&self, residue_idx: usize) -> Option<Vec3> {
        if let Some(pos) = self.animator.get_ca_position(residue_idx) {
            return Some(pos);
        }
        get_ca_position_from_chains(
            self.tube_renderer.cached_chains(),
            residue_idx,
        )
    }

    /// Get the interpolated position of the closest atom to a reference point
    /// for a given residue.
    pub fn get_closest_atom_for_residue(
        &self,
        residue_idx: usize,
        reference_point: Vec3,
    ) -> Option<Vec3> {
        let backbone_chains = self.get_current_backbone_chains();
        let sidechain_positions = self.get_current_sidechain_positions();

        foldit_conv::coords::get_closest_atom_for_residue(
            &backbone_chains,
            &sidechain_positions,
            &self.sc.cached_sidechain_residue_indices,
            residue_idx,
            reference_point,
        )
    }

    /// Get the closest atom position and name for a given residue relative to
    /// a reference point. Returns both backbone and sidechain atoms.
    pub fn get_closest_atom_with_name(
        &self,
        residue_idx: usize,
        reference_point: Vec3,
    ) -> Option<(Vec3, String)> {
        let backbone_chains = self.get_current_backbone_chains();
        let sidechain_positions = self.get_current_sidechain_positions();

        foldit_conv::coords::get_closest_atom_with_name(
            &backbone_chains,
            &sidechain_positions,
            &self.sc.cached_sidechain_residue_indices,
            &self.sc.cached_sidechain_atom_names,
            residue_idx,
            reference_point,
        )
    }

    /// Get the interpolated position of a specific atom by residue index and
    /// atom name.
    pub fn get_atom_position_by_name(
        &self,
        residue_idx: usize,
        atom_name: &str,
    ) -> Option<Vec3> {
        // Check backbone atoms first (N, CA, C)
        if atom_name == "N" || atom_name == "CA" || atom_name == "C" {
            let backbone_chains = self.get_current_backbone_chains();
            let offset = match atom_name {
                "N" => 0,
                "CA" => 1,
                "C" => 2,
                _ => return None,
            };

            let mut current_idx = 0;
            for chain in &backbone_chains {
                let residues_in_chain = chain.len() / 3;
                if residue_idx < current_idx + residues_in_chain {
                    let local_idx = residue_idx - current_idx;
                    let atom_idx = local_idx * 3 + offset;
                    return chain.get(atom_idx).copied();
                }
                current_idx += residues_in_chain;
            }
            return None;
        }

        // Check sidechain atoms
        let sidechain_positions = self.get_current_sidechain_positions();
        for (i, (res_idx, name)) in self
            .sc
            .cached_sidechain_residue_indices
            .iter()
            .zip(self.sc.cached_sidechain_atom_names.iter())
            .enumerate()
        {
            if *res_idx as usize == residue_idx && name == atom_name {
                return sidechain_positions.get(i).copied();
            }
        }

        None
    }
}
