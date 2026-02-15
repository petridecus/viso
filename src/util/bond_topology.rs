/// Bond topology tables for the 20 standard amino acids.
/// Returns all covalent bonds within a residue (both backbone and sidechain).
/// Bonds are represented as pairs of atom names.

/// Get the bond pairs for a residue type (sidechain internal bonds only).
/// Returns None for unknown residue types.
pub fn get_residue_bonds(residue_name: &str) -> Option<&'static [(&'static str, &'static str)]> {
    match residue_name.to_uppercase().as_str() {
        "ALA" => Some(ALANINE_BONDS),
        "ARG" => Some(ARGININE_BONDS),
        "ASN" => Some(ASPARAGINE_BONDS),
        "ASP" => Some(ASPARTATE_BONDS),
        "CYS" => Some(CYSTEINE_BONDS),
        "GLN" => Some(GLUTAMINE_BONDS),
        "GLU" => Some(GLUTAMATE_BONDS),
        "GLY" => Some(GLYCINE_BONDS),
        "HIS" => Some(HISTIDINE_BONDS),
        "ILE" => Some(ISOLEUCINE_BONDS),
        "LEU" => Some(LEUCINE_BONDS),
        "LYS" => Some(LYSINE_BONDS),
        "MET" => Some(METHIONINE_BONDS),
        "PHE" => Some(PHENYLALANINE_BONDS),
        "PRO" => Some(PROLINE_BONDS),
        "SER" => Some(SERINE_BONDS),
        "THR" => Some(THREONINE_BONDS),
        "TRP" => Some(TRYPTOPHAN_BONDS),
        "TYR" => Some(TYROSINE_BONDS),
        "VAL" => Some(VALINE_BONDS),
        _ => None,
    }
}

/// Check if an amino acid is hydrophobic
pub fn is_hydrophobic(residue_name: &str) -> bool {
    matches!(
        residue_name.to_uppercase().as_str(),
        "ALA" | "VAL" | "ILE" | "LEU" | "MET" | "PHE" | "TRP" | "PRO" | "GLY"
    )
}


// Glycine has no sidechain atoms (only backbone N, CA, C, O)
const GLYCINE_BONDS: &[(&str, &str)] = &[];

// Alanine: CA-CB (CB is the only sidechain atom)
const ALANINE_BONDS: &[(&str, &str)] = &[];

// Valine: CA-CB-CG1, CB-CG2
const VALINE_BONDS: &[(&str, &str)] = &[
    ("CB", "CG1"),
    ("CB", "CG2"),
];

// Leucine: CA-CB-CG-CD1, CG-CD2
const LEUCINE_BONDS: &[(&str, &str)] = &[
    ("CB", "CG"),
    ("CG", "CD1"),
    ("CG", "CD2"),
];

// Isoleucine: CA-CB-CG1-CD1, CB-CG2
const ISOLEUCINE_BONDS: &[(&str, &str)] = &[
    ("CB", "CG1"),
    ("CG1", "CD1"),
    ("CB", "CG2"),
];

// Proline: CA-CB-CG-CD-N (ring)
const PROLINE_BONDS: &[(&str, &str)] = &[
    ("CB", "CG"),
    ("CG", "CD"),
];

// Serine: CA-CB-OG
const SERINE_BONDS: &[(&str, &str)] = &[
    ("CB", "OG"),
];

// Threonine: CA-CB-OG1, CB-CG2
const THREONINE_BONDS: &[(&str, &str)] = &[
    ("CB", "OG1"),
    ("CB", "CG2"),
];

// Cysteine: CA-CB-SG
const CYSTEINE_BONDS: &[(&str, &str)] = &[
    ("CB", "SG"),
];

// Methionine: CA-CB-CG-SD-CE
const METHIONINE_BONDS: &[(&str, &str)] = &[
    ("CB", "CG"),
    ("CG", "SD"),
    ("SD", "CE"),
];

// Asparagine: CA-CB-CG-OD1, CG-ND2
const ASPARAGINE_BONDS: &[(&str, &str)] = &[
    ("CB", "CG"),
    ("CG", "OD1"),
    ("CG", "ND2"),
];

// Aspartate: CA-CB-CG-OD1, CG-OD2
const ASPARTATE_BONDS: &[(&str, &str)] = &[
    ("CB", "CG"),
    ("CG", "OD1"),
    ("CG", "OD2"),
];

// Glutamine: CA-CB-CG-CD-OE1, CD-NE2
const GLUTAMINE_BONDS: &[(&str, &str)] = &[
    ("CB", "CG"),
    ("CG", "CD"),
    ("CD", "OE1"),
    ("CD", "NE2"),
];

// Glutamate: CA-CB-CG-CD-OE1, CD-OE2
const GLUTAMATE_BONDS: &[(&str, &str)] = &[
    ("CB", "CG"),
    ("CG", "CD"),
    ("CD", "OE1"),
    ("CD", "OE2"),
];

// Lysine: CA-CB-CG-CD-CE-NZ
const LYSINE_BONDS: &[(&str, &str)] = &[
    ("CB", "CG"),
    ("CG", "CD"),
    ("CD", "CE"),
    ("CE", "NZ"),
];

// Arginine: CA-CB-CG-CD-NE-CZ-NH1, CZ-NH2
const ARGININE_BONDS: &[(&str, &str)] = &[
    ("CB", "CG"),
    ("CG", "CD"),
    ("CD", "NE"),
    ("NE", "CZ"),
    ("CZ", "NH1"),
    ("CZ", "NH2"),
];

// Histidine: CA-CB-CG-ND1-CE1-NE2-CD2-CG (imidazole ring)
const HISTIDINE_BONDS: &[(&str, &str)] = &[
    ("CB", "CG"),
    ("CG", "ND1"),
    ("ND1", "CE1"),
    ("CE1", "NE2"),
    ("NE2", "CD2"),
    ("CD2", "CG"),
];

// Phenylalanine: CA-CB-CG benzene ring
const PHENYLALANINE_BONDS: &[(&str, &str)] = &[
    ("CB", "CG"),
    ("CG", "CD1"),
    ("CD1", "CE1"),
    ("CE1", "CZ"),
    ("CZ", "CE2"),
    ("CE2", "CD2"),
    ("CD2", "CG"),
];

// Tyrosine: Like Phe + OH on CZ
const TYROSINE_BONDS: &[(&str, &str)] = &[
    ("CB", "CG"),
    ("CG", "CD1"),
    ("CD1", "CE1"),
    ("CE1", "CZ"),
    ("CZ", "OH"),
    ("CZ", "CE2"),
    ("CE2", "CD2"),
    ("CD2", "CG"),
];

// Tryptophan: indole ring system
const TRYPTOPHAN_BONDS: &[(&str, &str)] = &[
    ("CB", "CG"),
    ("CG", "CD1"),
    ("CD1", "NE1"),
    ("NE1", "CE2"),
    ("CE2", "CD2"),
    ("CD2", "CG"),
    // Benzene ring of indole
    ("CE2", "CZ2"),
    ("CZ2", "CH2"),
    ("CH2", "CZ3"),
    ("CZ3", "CE3"),
    ("CE3", "CD2"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_residue_bonds() {
        assert!(get_residue_bonds("ALA").is_some());
        assert!(get_residue_bonds("ala").is_some()); // case insensitive
        assert!(get_residue_bonds("GLY").is_some());
        assert!(get_residue_bonds("TRP").is_some());
        assert!(get_residue_bonds("XXX").is_none());
    }

    #[test]
    fn test_phenylalanine_ring() {
        let bonds = get_residue_bonds("PHE").unwrap();
        // PHE should have 7 bonds in the sidechain
        assert_eq!(bonds.len(), 7);
    }
}
