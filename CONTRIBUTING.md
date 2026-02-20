# Contributing to Viso

Thanks for your interest in contributing! This document covers the conventions
and tooling you need to know.

## Lint policy

All lint rules live as crate-level attributes in `src/lib.rs`. That file is the
single source of truth. There is no `[lints]` section in `Cargo.toml`.

Highlights:

- `clippy::all`, `clippy::pedantic`, and `clippy::nursery` are all **denied**.
- `missing_docs` is denied: every public item needs a doc comment.
- `clippy::unwrap_used` and `clippy::expect_used` are denied in library code.
  Tests and `main.rs` may use `#[allow]` where appropriate.
- Complexity lints (`cognitive_complexity`, `too_many_lines`,
  `excessive_nesting`) are denied with thresholds set in `clippy.toml`.

## Running checks locally

If you have [just](https://github.com/casey/just) installed:

```sh
just check        # fmt + clippy + test + doc (mirrors CI)
just check-all    # above + cargo-deny + machete + file-lengths
```

Or run the commands individually:

```sh
cargo +nightly fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items
```

## Formatting

We use **nightly rustfmt** (config in `rustfmt.toml`). Run `just fmt` (or
`cargo +nightly fmt`) before pushing.

## Documentation

Every public struct, enum, trait, function, method, field, and variant must have
a doc comment. Use backticks for code references in doc comments (rustdoc will
warn otherwise). Functions that return `Result` should include an `# Errors`
section.

## File length limit

Source files should stay under **500 lines**. A few legacy files are above this
today (CI currently enforces 800 as a transitional limit). If you are adding to
a file that is already long, consider splitting it.

## Pull requests

- Use a descriptive title that summarizes the change.
- Link any related issues.
- Keep PRs focused on a single concern. Avoid bundling unrelated changes.
- All CI checks must pass before merge.

## Architecture

For a deeper look at the rendering pipeline, scene graph, and animation system,
see the companion [mdBook documentation](docs/).
