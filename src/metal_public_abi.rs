//! Metal Driver Track M2 — typed public ABI surface for guest-facing ops.
//!
//! Capability enums and operation dispatch exist so callers can name Metal /
//! acceleration requests without inventing device success. Every op that would
//! imply a guest Metal stack returns an explicit [`MetalAbiReject`]. Device
//! enumeration is rejected (not an empty success list) so callers cannot treat
//! "ABI present" as "Metal works".
//!
//! Display-path freeze (M1) and Design D10 honesty (M0) remain unchanged.
//! Guest Metal remains unverified.

use crate::display_path_freeze::{freeze_under_metal_track, DisplayPathFreeze};
use crate::metal_acceptance::MetalAcceptanceReport;

/// Machine-checkable marker for M2 public ABI (research doc lock).
pub const M2_PUBLIC_ABI_DOC_MARKER: &str = "M2_PUBLIC_ABI:typed-reject-no-enumeration";

/// Guest-facing capability bits. At M2 every bit is unsupported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetalGuestCapability {
    /// Guest can enumerate a Metal-class device.
    DeviceEnumeration,
    /// Guest can allocate GPU-visible resources.
    ResourceAllocation,
    /// Guest can create and submit a command queue.
    CommandQueueSubmit,
    /// Guest can create/wait fences for GPU work.
    FenceSynchronization,
    /// Guest can load a Metal library / metallib image.
    MetallibLoad,
    /// Guest can dispatch compute kernels.
    ComputeDispatch,
    /// Guest can encode render passes.
    RenderEncode,
    /// Guest can present a drawable to the scanout path via Metal.
    PresentDrawable,
}

impl MetalGuestCapability {
    /// Stable iteration order for contract tests and capability maps.
    pub const ALL: [Self; 8] = [
        Self::DeviceEnumeration,
        Self::ResourceAllocation,
        Self::CommandQueueSubmit,
        Self::FenceSynchronization,
        Self::MetallibLoad,
        Self::ComputeDispatch,
        Self::RenderEncode,
        Self::PresentDrawable,
    ];
}

/// Guest-facing operations accepted as typed requests on the public ABI.
///
/// Dispatch always rejects at M2. Success variants are reserved for later
/// phases and are never returned here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalGuestOp {
    /// Would advertise Metal devices to the guest. Must reject (no empty OK).
    EnumerateDevices,
    CreateBuffer { size: u64 },
    CreateCommandQueue,
    SubmitCommandBuffer,
    CreateFence,
    WaitFence,
    LoadMetallib,
    DispatchCompute,
    EncodeRender,
    PresentDrawable,
}

impl MetalGuestOp {
    /// Primary capability this op would require if Metal were implemented.
    pub fn required_capability(self) -> MetalGuestCapability {
        match self {
            Self::EnumerateDevices => MetalGuestCapability::DeviceEnumeration,
            Self::CreateBuffer { .. } => MetalGuestCapability::ResourceAllocation,
            Self::CreateCommandQueue | Self::SubmitCommandBuffer => {
                MetalGuestCapability::CommandQueueSubmit
            }
            Self::CreateFence | Self::WaitFence => MetalGuestCapability::FenceSynchronization,
            Self::LoadMetallib => MetalGuestCapability::MetallibLoad,
            Self::DispatchCompute => MetalGuestCapability::ComputeDispatch,
            Self::EncodeRender => MetalGuestCapability::RenderEncode,
            Self::PresentDrawable => MetalGuestCapability::PresentDrawable,
        }
    }
}

/// Explicit reject path. Prefer these over `Ok` with empty/fake payloads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetalAbiReject {
    /// Named capability is not available on this host/guest path.
    CapabilityUnsupported(MetalGuestCapability),
    /// No guest Metal device is registered; refuse to invent one.
    NoGuestMetalDevice,
    /// Enumeration must not succeed (even with zero devices) at M2.
    EnumerationNotAdvertised,
    /// Design D10 / `metal_verified` still unmet.
    AcceptanceUnmet,
}

impl std::fmt::Display for MetalAbiReject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CapabilityUnsupported(cap) => {
                write!(f, "Metal capability unsupported: {cap:?}")
            }
            Self::NoGuestMetalDevice => {
                write!(f, "no guest Metal device; refusing fake device success")
            }
            Self::EnumerationNotAdvertised => write!(
                f,
                "device enumeration is not advertised; empty success would imply Metal"
            ),
            Self::AcceptanceUnmet => {
                write!(f, "metal acceptance unmet; metal_verified remains false")
            }
        }
    }
}

impl std::error::Error for MetalAbiReject {}

/// Bitmap of guest Metal capabilities. Honest default: all bits clear.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MetalCapabilityBitmap {
    bits: u32,
}

impl MetalCapabilityBitmap {
    pub const fn empty() -> Self {
        Self { bits: 0 }
    }

    pub fn is_supported(self, cap: MetalGuestCapability) -> bool {
        (self.bits & (1u32 << (cap as u32))) != 0
    }

    pub fn any_supported(self) -> bool {
        self.bits != 0
    }

    /// Contract helper: M2 surface must advertise nothing.
    pub fn asserts_all_unsupported(self) -> bool {
        !self.any_supported()
            && MetalGuestCapability::ALL
                .iter()
                .all(|&cap| !self.is_supported(cap))
    }
}

/// Public ABI handle. Does not register devices or promote acceptance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalPublicAbi {
    acceptance: MetalAcceptanceReport,
    capabilities: MetalCapabilityBitmap,
}

impl MetalPublicAbi {
    /// Honest M2 surface: unmet acceptance, zero capabilities.
    pub fn current() -> Self {
        Self {
            acceptance: MetalAcceptanceReport::unmet(),
            capabilities: MetalCapabilityBitmap::empty(),
        }
    }

    pub fn acceptance(&self) -> &MetalAcceptanceReport {
        &self.acceptance
    }

    pub fn capabilities(&self) -> MetalCapabilityBitmap {
        self.capabilities
    }

    /// Capability query. Never returns true for Metal-implying bits at M2.
    pub fn capability_supported(&self, cap: MetalGuestCapability) -> bool {
        self.capabilities.is_supported(cap)
    }

    /// Display-path freeze under this ABI (must remain M1 lock).
    pub fn display_path_freeze(&self) -> DisplayPathFreeze {
        freeze_under_metal_track().with_metal_acceptance(&self.acceptance)
    }

    /// Dispatch a typed guest op. Always rejects unsupported / Metal-implying work.
    pub fn execute(&self, op: MetalGuestOp) -> Result<(), MetalAbiReject> {
        if self.acceptance.metal_verified || self.acceptance.d10_complete() {
            // M2 must not observe a verified report; refuse rather than succeed.
            return Err(MetalAbiReject::AcceptanceUnmet);
        }

        // Enumeration must never succeed at M2 (empty Ok would imply Metal).
        if matches!(op, MetalGuestOp::EnumerateDevices) {
            return Err(MetalAbiReject::EnumerationNotAdvertised);
        }

        let cap = op.required_capability();
        if !self.capability_supported(cap) {
            return Err(MetalAbiReject::CapabilityUnsupported(cap));
        }

        // Even if a future bitmap bit were set, refuse invented devices.
        Err(MetalAbiReject::NoGuestMetalDevice)
    }

    /// Convenience: reject path for enumeration without implying Metal.
    pub fn reject_enumerate_devices(&self) -> MetalAbiReject {
        match self.execute(MetalGuestOp::EnumerateDevices) {
            Err(e) => e,
            Ok(()) => MetalAbiReject::EnumerationNotAdvertised,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_abi_advertises_no_capabilities() {
        let abi = MetalPublicAbi::current();
        assert!(abi.capabilities().asserts_all_unsupported());
        assert!(!abi.acceptance().metal_verified);
        for cap in MetalGuestCapability::ALL {
            assert!(!abi.capability_supported(cap));
        }
    }

    #[test]
    fn enumerate_devices_is_explicit_reject() {
        let abi = MetalPublicAbi::current();
        let err = abi
            .execute(MetalGuestOp::EnumerateDevices)
            .expect_err("enumeration must not succeed");
        assert_eq!(err, MetalAbiReject::EnumerationNotAdvertised);
        assert_eq!(abi.reject_enumerate_devices(), MetalAbiReject::EnumerationNotAdvertised);
    }

    #[test]
    fn accel_ops_reject_without_fake_device() {
        let abi = MetalPublicAbi::current();
        let cases = [
            MetalGuestOp::CreateBuffer { size: 64 },
            MetalGuestOp::CreateCommandQueue,
            MetalGuestOp::SubmitCommandBuffer,
            MetalGuestOp::CreateFence,
            MetalGuestOp::WaitFence,
            MetalGuestOp::LoadMetallib,
            MetalGuestOp::DispatchCompute,
            MetalGuestOp::EncodeRender,
            MetalGuestOp::PresentDrawable,
        ];
        for op in cases {
            let err = abi.execute(op).expect_err("unsupported op must reject");
            assert_eq!(
                err,
                MetalAbiReject::CapabilityUnsupported(op.required_capability())
            );
        }
    }

    #[test]
    fn abi_preserves_m1_display_path_freeze() {
        let abi = MetalPublicAbi::current();
        let freeze = abi.display_path_freeze();
        assert!(freeze.matches_m1_lock());
        assert_eq!(freeze.adp_mmio_window_id, 0x6);
        assert_eq!(freeze.adp_aic_irq_line, 2);
    }
}
