# Building and Running

## Prerequisites

- **Rust** (stable, 1.80+)
- **A GPU with WebGPU support** -- Metal (macOS), Vulkan (Linux/Windows), or DX12 (Windows)
- **Internet access** (optional, for RCSB downloads)

## Building

From the repository root:

```sh
# Build the standalone viewer
cargo build -p viso

# Build with optimizations (recommended for real use)
cargo build -p viso --release
```

## Running

### With a PDB ID

Pass a 4-character PDB code to auto-download from RCSB:

```sh
cargo run -p viso --release -- 1ubq
```

The file is downloaded as mmCIF and cached in `assets/models/1ubq.cif`. Subsequent runs with the same ID load from cache.

### With a Local File

```sh
cargo run -p viso --release -- path/to/structure.cif
```

Viso supports mmCIF (`.cif`) files.

## Logging

Viso uses `env_logger`. Control verbosity with `RUST_LOG`:

```sh
# Errors only (default)
cargo run -p viso -- 1ubq

# Info-level (see download progress, frame counts, etc.)
RUST_LOG=info cargo run -p viso -- 1ubq

# Debug-level (animation frames, picking results, mesh timing)
RUST_LOG=debug cargo run -p viso -- 1ubq

# Module-specific filtering
RUST_LOG=viso::scene::processor=debug cargo run -p viso -- 1ubq
```

## Platform Notes

### macOS (Metal)

Metal is the default backend. No extra setup needed. Ensure your macOS version is 10.15+ (Catalina) or later.

### Linux (Vulkan)

Requires Vulkan drivers. Install:

```sh
# Ubuntu/Debian
sudo apt install libvulkan-dev vulkan-tools

# Fedora
sudo dnf install vulkan-loader-devel vulkan-tools
```

### Windows (DX12 / Vulkan)

DX12 is the default backend on Windows 10+. Vulkan is also supported if drivers are installed.

## Controls

| Input | Action |
|-------|--------|
| Left drag | Rotate camera |
| Shift + left drag | Pan camera |
| Scroll wheel | Zoom |
| Click residue | Select residue |
| Shift + click | Add/remove from selection |
| Double-click | Select secondary structure segment |
| Triple-click | Select entire chain |
| Click background | Clear selection |
| Escape | Clear selection |
| W | Toggle water visibility |
