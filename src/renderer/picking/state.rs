use super::Picking;
use crate::renderer::geometry::ball_and_stick::BallAndStickRenderer;
use crate::renderer::geometry::sidechain::SidechainRenderer;

/// Owns the GPU bind groups used for picking ray-tests against capsule geometry
/// (sidechains and ball-and-stick atoms).
pub(crate) struct PickingState {
    pub capsule: Option<wgpu::BindGroup>,
    pub bns_bond: Option<wgpu::BindGroup>,
    pub bns_sphere: Option<wgpu::BindGroup>,
}

impl PickingState {
    /// Create a new picking state with no bind groups allocated.
    pub fn new() -> Self {
        Self {
            capsule: None,
            bns_bond: None,
            bns_sphere: None,
        }
    }

    /// Rebuild the capsule (sidechain) picking bind group from current buffer.
    pub fn rebuild_capsule(
        &mut self,
        picking: &Picking,
        device: &wgpu::Device,
        sidechain_renderer: &SidechainRenderer,
    ) {
        self.capsule = Some(picking.create_capsule_bind_group(
            device,
            sidechain_renderer.capsule_buffer(),
        ));
    }

    /// Rebuild the ball-and-stick bond picking bind group from the visual bond
    /// buffer.
    pub fn rebuild_bns_bond(
        &mut self,
        picking: &Picking,
        device: &wgpu::Device,
        bns_renderer: &BallAndStickRenderer,
    ) {
        self.bns_bond =
            if bns_renderer.bond_count() > 0 {
                Some(picking.create_capsule_bind_group(
                    device,
                    bns_renderer.bond_buffer(),
                ))
            } else {
                None
            };
    }

    /// Rebuild the ball-and-stick sphere picking bind group from the visual
    /// sphere buffer.
    pub fn rebuild_bns_sphere(
        &mut self,
        picking: &Picking,
        device: &wgpu::Device,
        bns_renderer: &BallAndStickRenderer,
    ) {
        self.bns_sphere =
            if bns_renderer.sphere_count() > 0 {
                Some(picking.create_sphere_bind_group(
                    device,
                    bns_renderer.sphere_buffer(),
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
        sidechain_renderer: &SidechainRenderer,
        bns_renderer: &BallAndStickRenderer,
    ) {
        self.rebuild_capsule(picking, device, sidechain_renderer);
        self.rebuild_bns_bond(picking, device, bns_renderer);
        self.rebuild_bns_sphere(picking, device, bns_renderer);
    }
}
