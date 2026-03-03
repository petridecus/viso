// Bit-packed selection lookup — shared by all geometry shaders.
//
// The selection buffer binding lives in each consuming shader (at varying
// group indices), so this module provides a pure function that takes the
// relevant word by value.

#define_import_path viso::selection

/// Check whether `residue_idx` is marked as selected.
///
/// `word_count` — `arrayLength(&selection)` from the caller.
/// `word`       — `selection[residue_idx / 32u]` from the caller.
///
/// With robust buffer access the caller's word read is always safe; this
/// function performs the bounds check on `word_count` so an out-of-range
/// index still returns `false`.
fn check_selection(residue_idx: u32, word_count: u32, word: u32) -> bool {
    let word_idx = residue_idx / 32u;
    if (word_idx >= word_count) {
        return false;
    }
    let bit_idx = residue_idx % 32u;
    return (word & (1u << bit_idx)) != 0u;
}
