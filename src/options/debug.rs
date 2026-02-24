use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Debug visualization toggles.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Default, JsonSchema,
)]
#[schemars(title = "Debug", inline)]
#[serde(default)]
pub struct DebugOptions {
    /// Visualize surface normals as RGB colors.
    #[schemars(title = "Show Normals")]
    pub show_normals: bool,
}
