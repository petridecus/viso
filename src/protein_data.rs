use glam::Vec3;
use pdbtbx::{Format, PDB, ReadOptions};
use std::collections::HashMap;
use std::path::Path;

use crate::bond_topology::{get_residue_bonds, is_hydrophobic};

/// Full backbone atom positions for a single residue
/// These are the raw PDB coordinates used for ribbon geometry
#[derive(Debug, Clone, Copy)]
pub struct BackboneResidue {
    pub n_pos: Vec3,
    pub ca_pos: Vec3,
    pub c_pos: Vec3,
    pub o_pos: Vec3,
}

/// A chain of backbone residues (continuous, no breaks)
#[derive(Debug, Clone)]
pub struct BackboneChain {
    pub residues: Vec<BackboneResidue>,
}

/// A sidechain atom with its position and residue context
#[derive(Debug, Clone)]
pub struct SidechainAtom {
    pub position: Vec3,
    pub residue_idx: usize,
    pub atom_name: String,
    pub chain_id: String,
    pub is_hydrophobic: bool,
}

/// A backbone-to-sidechain bond (CA to CB)
#[derive(Debug, Clone)]
pub struct BackboneSidechainBond {
    pub ca_position: Vec3,
    pub cb_index: u32, // Index into sidechain_atoms
}

/// Maximum distance (Angstroms) for a valid peptide bond (C to N)
/// Normal C-N peptide bond is ~1.33Å, we use 2.5Å as threshold to detect breaks
const MAX_PEPTIDE_BOND_DISTANCE: f32 = 2.5;

/// Extracted protein data for rendering
pub struct ProteinData {
    /// Backbone segments (split at chain breaks). Each segment is a continuous run of N, CA, C atoms
    /// Legacy format for compatibility - use backbone_residue_chains for new geometry
    pub backbone_chains: Vec<Vec<Vec3>>,
    /// Full backbone residue data with N, CA, C, O positions per residue
    /// This is the primary data source for ribbon geometry (Foldit-style)
    pub backbone_residue_chains: Vec<BackboneChain>,
    /// All sidechain atoms (non-backbone: not N, CA, C, O)
    pub sidechain_atoms: Vec<SidechainAtom>,
    /// Bond pairs as indices into sidechain_atoms (internal sidechain bonds)
    pub sidechain_bonds: Vec<(u32, u32)>,
    /// Bonds connecting backbone (CA) to sidechain (CB)
    pub backbone_sidechain_bonds: Vec<BackboneSidechainBond>,
    /// All atom positions (for compatibility with existing code)
    pub all_positions: Vec<Vec3>,
}

impl ProteinData {
    /// Load protein data from an mmCIF file
    pub fn from_mmcif<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path_str = path.as_ref().to_string_lossy();
        let (pdb, _errors) = ReadOptions::default()
            .set_format(Format::Mmcif)
            .set_level(pdbtbx::StrictnessLevel::Loose)
            .read(&*path_str)
            .map_err(|e| format!("Failed to parse mmCIF: {:?}", e))?;

        Self::from_pdb(&pdb)
    }

    /// Extract protein data from a parsed PDB structure
    fn from_pdb(pdb: &PDB) -> Result<Self, String> {
        let mut backbone_chains: Vec<Vec<Vec3>> = Vec::new();
        let mut backbone_residue_chains: Vec<BackboneChain> = Vec::new();
        let mut sidechain_atoms: Vec<SidechainAtom> = Vec::new();
        let mut all_positions: Vec<Vec3> = Vec::new();
        let mut backbone_sidechain_bonds: Vec<BackboneSidechainBond> = Vec::new();

        // Track atoms by (chain_id, residue_serial, atom_name) for bond lookup
        let mut atom_index_map: HashMap<(String, isize, String), usize> = HashMap::new();

        // Process each chain
        for chain in pdb.chains() {
            let chain_id = chain.id().to_string();
            let mut current_segment: Vec<Vec3> = Vec::new();
            let mut current_residues: Vec<BackboneResidue> = Vec::new();
            let mut prev_c_pos: Option<Vec3> = None;
            let mut prev_res_serial: Option<isize> = None;

            for residue in chain.residues() {
                let res_serial = residue.serial_number();
                let res_name = residue.name().unwrap_or("UNK");
                let hydrophobic = is_hydrophobic(res_name);

                // Collect all backbone atoms: N, CA, C, O
                let mut n_pos: Option<Vec3> = None;
                let mut ca_pos: Option<Vec3> = None;
                let mut c_pos: Option<Vec3> = None;
                let mut o_pos: Option<Vec3> = None;
                let mut cb_idx: Option<usize> = None;

                for atom in residue.atoms() {
                    let atom_name = atom.name().trim().to_string();
                    let pos = Vec3::new(atom.x() as f32, atom.y() as f32, atom.z() as f32);

                    all_positions.push(pos);

                    match atom_name.as_str() {
                        "N" => n_pos = Some(pos),
                        "CA" => ca_pos = Some(pos),
                        "C" => c_pos = Some(pos),
                        "O" => o_pos = Some(pos), // Now we capture O for peptide plane orientation
                        _ => {
                            // Sidechain atom
                            let sidechain_idx = sidechain_atoms.len();
                            atom_index_map.insert(
                                (chain_id.clone(), res_serial, atom_name.clone()),
                                sidechain_idx,
                            );

                            if atom_name == "CB" {
                                cb_idx = Some(sidechain_idx);
                            }

                            sidechain_atoms.push(SidechainAtom {
                                position: pos,
                                residue_idx: backbone_chains.len(), // Will be updated
                                atom_name,
                                chain_id: chain_id.clone(),
                                is_hydrophobic: hydrophobic,
                            });
                        }
                    }
                }

                // Check for chain break before adding this residue
                let is_chain_break = if let (Some(prev_c), Some(n)) = (prev_c_pos, n_pos) {
                    let distance = (n - prev_c).length();
                    distance > MAX_PEPTIDE_BOND_DISTANCE
                } else {
                    false
                };

                // Also check for sequence gap (missing residues)
                let has_sequence_gap = if let Some(prev_serial) = prev_res_serial {
                    // Allow for insertion codes by checking if gap is > 1
                    (res_serial - prev_serial).abs() > 1
                } else {
                    false
                };

                // If chain break detected, save current segment and start new one
                if (is_chain_break || has_sequence_gap) && !current_segment.is_empty() {
                    backbone_chains.push(std::mem::take(&mut current_segment));
                    backbone_residue_chains.push(BackboneChain {
                        residues: std::mem::take(&mut current_residues),
                    });
                }

                // Add backbone atoms in order for legacy spline format (N, CA, C)
                if let Some(n) = n_pos {
                    current_segment.push(n);
                }
                if let Some(ca) = ca_pos {
                    current_segment.push(ca);

                    // Add CA-CB bond if CB exists
                    if let Some(cb_i) = cb_idx {
                        backbone_sidechain_bonds.push(BackboneSidechainBond {
                            ca_position: ca,
                            cb_index: cb_i as u32,
                        });
                    }
                }
                if let Some(c) = c_pos {
                    current_segment.push(c);
                    prev_c_pos = Some(c);
                } else {
                    prev_c_pos = None; // No C atom means we can't check continuity
                }

                // Add full residue data if all backbone atoms present
                if let (Some(n), Some(ca), Some(c), Some(o)) = (n_pos, ca_pos, c_pos, o_pos) {
                    current_residues.push(BackboneResidue {
                        n_pos: n,
                        ca_pos: ca,
                        c_pos: c,
                        o_pos: o,
                    });
                }

                prev_res_serial = Some(res_serial);
            }

            // Don't forget the last segment
            if !current_segment.is_empty() {
                backbone_chains.push(current_segment);
            }
            if !current_residues.is_empty() {
                backbone_residue_chains.push(BackboneChain {
                    residues: current_residues,
                });
            }
        }

        // Generate sidechain bonds from topology tables
        let sidechain_bonds = Self::generate_sidechain_bonds(pdb, &atom_index_map);

        Ok(Self {
            backbone_chains,
            backbone_residue_chains,
            sidechain_atoms,
            sidechain_bonds,
            backbone_sidechain_bonds,
            all_positions,
        })
    }

    /// Generate bonds between sidechain atoms using residue topology
    fn generate_sidechain_bonds(
        pdb: &PDB,
        atom_index_map: &HashMap<(String, isize, String), usize>,
    ) -> Vec<(u32, u32)> {
        let mut bonds: Vec<(u32, u32)> = Vec::new();

        for chain in pdb.chains() {
            let chain_id = chain.id().to_string();

            for residue in chain.residues() {
                let res_name = residue.name().unwrap_or("UNK");
                let res_serial = residue.serial_number();

                // Get bond topology for this residue type
                if let Some(residue_bonds) = get_residue_bonds(res_name) {
                    for (atom1, atom2) in residue_bonds {
                        // Only include bonds where both atoms are sidechain atoms
                        let key1 = (chain_id.clone(), res_serial, atom1.to_string());
                        let key2 = (chain_id.clone(), res_serial, atom2.to_string());

                        if let (Some(&idx1), Some(&idx2)) =
                            (atom_index_map.get(&key1), atom_index_map.get(&key2))
                        {
                            bonds.push((idx1 as u32, idx2 as u32));
                        }
                    }
                }
            }
        }

        bonds
    }

    /// Get all sidechain positions as a flat Vec<Vec3>
    pub fn sidechain_positions(&self) -> Vec<Vec3> {
        self.sidechain_atoms.iter().map(|a| a.position).collect()
    }
}

impl BackboneChain {
    /// Get CA positions for secondary structure detection
    pub fn ca_positions(&self) -> Vec<Vec3> {
        self.residues.iter().map(|r| r.ca_pos).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_hydrophobic() {
        use crate::bond_topology::is_hydrophobic;
        assert!(is_hydrophobic("ALA"));
        assert!(is_hydrophobic("VAL"));
        assert!(!is_hydrophobic("SER"));
        assert!(!is_hydrophobic("LYS"));
    }
}
