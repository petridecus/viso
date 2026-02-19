use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ColorOptions {
    pub lipid_carbon_tint: [f32; 3],
    pub hydrophobic_sidechain: [f32; 3],
    pub hydrophilic_sidechain: [f32; 3],
    pub nucleic_acid: [f32; 3],
    pub band_default: [f32; 3],
    pub band_backbone: [f32; 3],
    pub band_disulfide: [f32; 3],
    pub band_hbond: [f32; 3],
    pub solvent_color: [f32; 3],
    pub cofactor_tints: HashMap<String, [f32; 3]>,
}

impl Default for ColorOptions {
    fn default() -> Self {
        let mut cofactor_tints = HashMap::new();
        cofactor_tints.insert("CLA".to_string(), [0.2, 0.7, 0.3]);
        cofactor_tints.insert("CHL".to_string(), [0.2, 0.6, 0.35]);
        cofactor_tints.insert("BCR".to_string(), [0.9, 0.5, 0.1]);
        cofactor_tints.insert("BCB".to_string(), [0.9, 0.5, 0.1]);
        cofactor_tints.insert("HEM".to_string(), [0.7, 0.15, 0.15]);
        cofactor_tints.insert("HEC".to_string(), [0.7, 0.15, 0.15]);
        cofactor_tints.insert("HEA".to_string(), [0.7, 0.15, 0.15]);
        cofactor_tints.insert("HEB".to_string(), [0.7, 0.15, 0.15]);
        cofactor_tints.insert("PHO".to_string(), [0.5, 0.7, 0.3]);
        cofactor_tints.insert("PL9".to_string(), [0.6, 0.5, 0.2]);
        cofactor_tints.insert("PLQ".to_string(), [0.6, 0.5, 0.2]);

        Self {
            lipid_carbon_tint: [0.76, 0.70, 0.50],
            hydrophobic_sidechain: [0.3, 0.5, 0.9],
            hydrophilic_sidechain: [0.95, 0.6, 0.2],
            nucleic_acid: [0.45, 0.55, 0.85],
            band_default: [0.5, 0.0, 0.5],
            band_backbone: [1.0, 0.75, 0.0],
            band_disulfide: [0.5, 1.0, 0.0],
            band_hbond: [0.0, 0.75, 1.0],
            solvent_color: [0.6, 0.6, 0.6],
            cofactor_tints,
        }
    }
}

impl ColorOptions {
    /// Look up cofactor carbon tint by 3-letter residue name. Falls back to
    /// neutral gray.
    pub fn cofactor_tint(&self, res_name: &str) -> [f32; 3] {
        self.cofactor_tints
            .get(res_name.trim())
            .copied()
            .unwrap_or([0.5, 0.5, 0.5])
    }
}
