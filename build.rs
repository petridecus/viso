use std::path::Path;

fn main() {
    // Only needed when the gui feature is enabled.
    if std::env::var("CARGO_FEATURE_GUI").is_err() {
        return;
    }

    // Ensure the viso-ui dist directory exists so rust-embed compiles even
    // before `trunk build` has been run.  A placeholder index.html is
    // created when the real build output is absent.
    let dist = Path::new("crates/viso-ui/dist");
    if !dist.exists() {
        std::fs::create_dir_all(dist).expect("failed to create dist dir");
    }

    let index = dist.join("index.html");
    if !index.exists() {
        std::fs::write(
            &index,
            "<!DOCTYPE html><html><body>viso-ui not built</body></html>",
        )
        .expect("failed to write placeholder index.html");
    }

    // Re-run when the dist contents change (after trunk build).
    println!("cargo:rerun-if-changed=crates/viso-ui/dist");
}
