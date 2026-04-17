//! Triple-buffered host → viso assembly channel.
//!
//! The host application mutates an [`Assembly`] and publishes immutable
//! `Arc<Assembly>` snapshots via [`AssemblyPublisher::commit`]. Viso reads
//! the latest snapshot from its matching [`AssemblyConsumer::latest`] on
//! sync. Lockless; producer and consumer cadences decouple.
//!
//! For the standalone viewer, viso constructs both ends internally. For
//! host integrations (e.g. foldit-rs), the host owns the publisher and
//! passes the consumer handle into the engine.

use std::sync::Arc;

use molex::Assembly;

/// Writer side of the host → viso assembly channel.
///
/// The host mutates its owned `Assembly`, wraps it in an `Arc`, and calls
/// [`commit`](Self::commit) to publish the new snapshot. Viso's matching
/// consumer picks it up on the next sync.
pub struct AssemblyPublisher {
    tx: triple_buffer::Input<Option<Arc<Assembly>>>,
}

impl AssemblyPublisher {
    /// Publish a new assembly snapshot. Non-blocking.
    pub fn commit(&mut self, assembly: Arc<Assembly>) {
        self.tx.write(Some(assembly));
    }
}

/// Reader side of the host → viso assembly channel.
///
/// [`latest`](Self::latest) returns the most recently committed snapshot
/// if one has not yet been observed, otherwise `None`. Intended to be
/// polled once per frame at sync time.
pub struct AssemblyConsumer {
    rx: triple_buffer::Output<Option<Arc<Assembly>>>,
}

impl AssemblyConsumer {
    /// Return the latest published snapshot if one is waiting.
    ///
    /// Always returns `None` until the host has committed at least once.
    /// After a snapshot is consumed, subsequent calls return `None` until
    /// the host commits a newer snapshot.
    pub fn latest(&mut self) -> Option<Arc<Assembly>> {
        let _ = self.rx.update();
        self.rx.output_buffer_mut().take()
    }
}

/// Create a linked publisher/consumer pair.
#[must_use]
pub fn assembly_channel() -> (AssemblyPublisher, AssemblyConsumer) {
    let (tx, rx) = triple_buffer::triple_buffer(&None);
    (AssemblyPublisher { tx }, AssemblyConsumer { rx })
}
