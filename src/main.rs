//! CLI binary for the Viso protein visualization engine.

use std::process::ExitCode;

fn resolve_structure_path(input: &str) -> Result<String, String> {
    if std::path::Path::new(input).exists() {
        return Ok(input.to_owned());
    }

    if input.len() == 4 && input.chars().all(|c| c.is_ascii_alphanumeric()) {
        let pdb_id = input.to_lowercase();
        let models_dir = std::path::Path::new("assets/models");
        let local_path = models_dir.join(format!("{pdb_id}.cif"));

        if local_path.exists() {
            return Ok(local_path.to_string_lossy().into_owned());
        }

        if !models_dir.exists() {
            std::fs::create_dir_all(models_dir).map_err(|e| {
                format!("Failed to create models directory: {e}")
            })?;
        }

        let url = format!("https://files.rcsb.org/download/{pdb_id}.cif");
        log::info!("Downloading {} from RCSB...", pdb_id.to_uppercase());

        let agent = ureq::Agent::new_with_config(
            ureq::config::Config::builder()
                .timeout_global(Some(std::time::Duration::from_secs(30)))
                .build(),
        );
        let content = agent
            .get(&url)
            .call()
            .map_err(|e| format!("Network error downloading {pdb_id}: {e}"))?
            .into_body()
            .with_config()
            .limit(50 * 1024 * 1024)
            .read_to_string()
            .map_err(|e| format!("Failed to read response body: {e}"))?;

        std::fs::write(&local_path, &content)
            .map_err(|e| format!("I/O error saving CIF file: {e}"))?;

        log::info!("Downloaded to {}", local_path.display());
        return Ok(local_path.to_string_lossy().into_owned());
    }

    Err(format!("File not found and not a valid PDB code: {input}"))
}

fn main() -> ExitCode {
    env_logger::init();

    let Some(input) = std::env::args().nth(1) else {
        log::error!("Usage: viso <PDB_ID or path>");
        return ExitCode::from(2);
    };

    let cif_path = match resolve_structure_path(&input) {
        Ok(path) => path,
        Err(e) => {
            log::error!("{e}");
            return ExitCode::FAILURE;
        }
    };

    if let Err(e) = viso::Viewer::builder().with_path(cif_path).build().run() {
        log::error!("Viewer error: {e}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}
