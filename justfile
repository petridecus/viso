# Run all checks (what CI runs)
check: fmt-check clippy test doc

# Format check (nightly)
fmt-check:
    cargo +nightly fmt --check

# Format (nightly)
fmt:
    cargo +nightly fmt

# Clippy with all targets
clippy:
    cargo clippy --all-targets --all-features -- -D warnings

# Run tests
test:
    cargo test --all-features

# Build docs and check for warnings
doc:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items

# Dependency audit
deny:
    cargo deny check

# Check for unused dependencies
machete:
    cargo machete

# Check file lengths (max 800 lines)
# TODO: lower to 500 once large files are split
file-lengths:
    #!/usr/bin/env bash
    failed=0
    while IFS= read -r file; do
        lines=$(wc -l < "$file")
        if [ "$lines" -gt 800 ]; then
            echo "ERROR: $file has $lines lines (max 800)"
            failed=1
        fi
    done < <(find src -name '*.rs' -not -path '*/target/*')
    exit $failed

# Run everything including optional tools
check-all: check deny machete file-lengths
