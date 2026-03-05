//! Build script: ensures the viso-ui dist directory exists for rust-embed.
//!
//! On first build (when no trunk output exists), this automatically runs
//! `trunk build` so that a fresh clone can `cargo run` without manual steps.
#![allow(clippy::expect_used)]

use std::path::{Path, PathBuf};

fn main() {
    // Only needed when the gui feature is enabled.
    if std::env::var("CARGO_FEATURE_GUI").is_err() {
        return;
    }

    let ui_dir: &Path = Path::new("crates/viso-ui");
    let dist: PathBuf = ui_dir.join("dist");

    // Check whether trunk has already produced real output by looking
    // for any .wasm file in the dist directory.
    let has_wasm = dist.is_dir()
        && std::fs::read_dir(&dist).ok().is_some_and(|entries| {
            entries
                .flatten()
                .any(|e| e.path().extension().is_some_and(|ext| ext == "wasm"))
        });

    if !has_wasm {
        // First build – run trunk automatically.
        println!(
            "cargo:warning=viso-ui has not been built yet, running `trunk \
             build`…"
        );

        let mut cmd = std::process::Command::new("trunk");
        let _ = cmd.arg("build").current_dir(ui_dir);

        // Match the cargo profile: use --release for release builds.
        if std::env::var("PROFILE").as_deref() == Ok("release") {
            let _ = cmd.arg("--release");
        }

        let status = cmd.status();
        match status {
            Ok(s) if s.success() => {
                println!("cargo:warning=viso-ui built successfully");
            }
            Ok(s) => {
                // trunk ran but failed – fall back to placeholder so
                // compilation can still succeed (the UI just won't
                // work).
                println!(
                    "cargo:warning=trunk build exited with {s}; falling back \
                     to placeholder"
                );
                write_placeholder(&dist);
            }
            Err(e) => {
                println!(
                    "cargo:warning=failed to run `trunk`: {e}; falling back \
                     to placeholder"
                );
                write_placeholder(&dist);
            }
        }
    }

    // Re-run when the dist contents change (after trunk build).
    println!("cargo:rerun-if-changed=crates/viso-ui/dist");
}

/// Create the dist directory with a placeholder `index.html` so
/// rust-embed compiles even when trunk is unavailable.
fn write_placeholder(dist: &Path) {
    std::fs::create_dir_all(dist).expect("failed to create dist dir");
    std::fs::write(
        dist.join("index.html"),
        "<!DOCTYPE html><html><body>viso-ui not built – install trunk and \
         rebuild</body></html>",
    )
    .expect("failed to write placeholder index.html");
}
