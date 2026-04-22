//! Host → viso assembly channel primitives.
//!
//! The standalone-application layer ([`super::VisoApp`]) owns the
//! writer side of a triple buffer; the engine holds the matching
//! [`AssemblyConsumer`]
//! reader. Snapshots are published as `Arc<Assembly>` and consumed
//! lock-free from the render loop.

use std::sync::Arc;

use molex::Assembly;

use crate::engine::assembly_consumer::AssemblyConsumer;

/// Writer side of the host → viso assembly channel.
///
/// Produced together with an [`AssemblyConsumer`] by
/// [`assembly_channel`]. Commits an `Arc<Assembly>` snapshot to the
/// triple buffer each time the standalone app mutates the assembly.
pub(crate) struct AssemblyPublisher {
    tx: triple_buffer::Input<Option<Arc<Assembly>>>,
}

impl AssemblyPublisher {
    /// Publish a new `Assembly` snapshot to the consumer side.
    pub(crate) fn commit(&mut self, assembly: Arc<Assembly>) {
        self.tx.write(Some(assembly));
    }
}

/// Construct a new host → viso assembly channel.
pub(crate) fn assembly_channel() -> (AssemblyPublisher, AssemblyConsumer) {
    let (tx, rx) = triple_buffer::triple_buffer(&None);
    (AssemblyPublisher { tx }, AssemblyConsumer { rx })
}
