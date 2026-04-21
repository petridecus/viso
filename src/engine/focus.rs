//! Focus state for tab cycling between entities.

/// Focus state for tab cycling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Focus {
    /// All entities.
    #[default]
    Session,
    /// A specific entity by ID.
    Entity(u32),
}
