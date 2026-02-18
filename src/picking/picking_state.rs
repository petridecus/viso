use crate::picking::Picking;
use crate::renderer::molecular::ball_and_stick::BallAndStickRenderer;
use crate::renderer::molecular::capsule_sidechain::CapsuleSidechainRenderer;

/// Owns the GPU bind groups used for picking ray-tests against capsule geometry
/// (sidechains and ball-and-stick atoms).
pub(crate) struct PickingState {
    pub capsule_picking_bind_group: Option<wgpu::BindGroup>,
    pub bns_picking_bind_group: Option<wgpu::BindGroup>,
}

impl PickingState {
    pub fn new() -> Self {
        Self {
            capsule_picking_bind_group: None,
            bns_picking_bind_group: None,
        }
    }

    /// Rebuild the capsule (sidechain) picking bind group from current buffer.
    pub fn rebuild_capsule(
        &mut self,
        picking: &Picking,
        device: &wgpu::Device,
        sidechain_renderer: &CapsuleSidechainRenderer,
    ) {
        self.capsule_picking_bind_group =
            Some(picking.create_capsule_bind_group(
                device,
                sidechain_renderer.capsule_buffer(),
            ));
    }

    /// Rebuild the ball-and-stick picking bind group from current buffer.
    pub fn rebuild_bns(
        &mut self,
        picking: &Picking,
        device: &wgpu::Device,
        bns_renderer: &BallAndStickRenderer,
    ) {
        self.bns_picking_bind_group = if bns_renderer.picking_count() > 0 {
            Some(picking.create_capsule_bind_group(
                device,
                bns_renderer.picking_buffer(),
            ))
        } else {
            None
        };
    }

    /// Rebuild both picking bind groups.
    pub fn rebuild_all(
        &mut self,
        picking: &Picking,
        device: &wgpu::Device,
        sidechain_renderer: &CapsuleSidechainRenderer,
        bns_renderer: &BallAndStickRenderer,
    ) {
        self.rebuild_capsule(picking, device, sidechain_renderer);
        self.rebuild_bns(picking, device, bns_renderer);
    }
}
