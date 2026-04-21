//! Reader side of the host → viso `Assembly` triple buffer.
//!
//! The host owns the matching publisher and commits `Arc<Assembly>`
//! snapshots on mutation. Viso holds only this consumer handle: the
//! engine polls [`AssemblyConsumer::latest`] each frame and, on a
//! generation change, rederives its local view of the assembly.

use std::sync::Arc;

use molex::Assembly;

/// Reader side of the host → viso assembly channel.
///
/// [`latest`](Self::latest) returns the most recently committed snapshot
/// once per publish, otherwise `None`. Constructed by
/// [`crate::app::VisoApp`] (standalone deployments) or by the real host
/// application; passed into [`crate::VisoEngine::new`] at construction
/// time.
pub struct AssemblyConsumer {
    pub(crate) rx: triple_buffer::Output<Option<Arc<Assembly>>>,
}

impl AssemblyConsumer {
    /// Return the latest published snapshot if one is waiting.
    ///
    /// Returns `None` until the host has committed at least once, and
    /// `None` between publishes after the most recent snapshot has been
    /// consumed.
    pub fn latest(&mut self) -> Option<Arc<Assembly>> {
        let _ = self.rx.update();
        self.rx.output_buffer_mut().take()
    }
}
