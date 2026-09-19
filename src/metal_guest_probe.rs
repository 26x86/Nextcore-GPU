//! Metal Driver Track M5 — guest Metal probe harness.
//!
//! Recovery / installed-OS oriented automation that **attempts** a guest Metal
//! probe and records `actual_guest_metal_probe_executed` plus structured failure
//! reasons. On hosts without a guest Metal device (the default lab case), the
//! attempt is expected to fail honestly.
//!
//! This module never passes without a real guest Metal device, never sets
//! `metal_verified`, and does not invent enumeration success. Guest Metal remains unverified.

use serde::{Deserialize, Serialize};

use crate::canonical_spec::GpuCapabilities;
use crate::display_path_freeze::{freeze_under_metal_track, DisplayPathFreeze};
use crate::metal_accel_probe::AccelProbe;
use crate::metal_acceptance::MetalAcceptanceReport;
use crate::metal_public_abi::{MetalAbiReject, MetalGuestOp, MetalPublicAbi};
use crate::metal_transport::MetalTransport;
use crate::virtual_device::VirtualMetalDevice;

/// Machine-checkable marker for M5 guest Metal probe harness (research doc lock).
pub const M5_GUEST_METAL_PROBE_DOC_MARKER: &str = "M5_GUEST_METAL_PROBE:executed-fail-reasons";

/// Where the harness believes a guest Metal runtime could exist.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GuestMetalRuntimePresence {
    /// In-process host is not a macOS guest; Metal.framework is unavailable here.
    HostNotGuestMacOs {
        /// Short host hint (`windows`, `linux`, `macos`, `unknown`).
        host_os: String,
    },
    /// A guest session / Recovery Terminal context is not attached to this process.
    GuestSessionUnavailable { reason: String },
    /// Guest Metal stack would be reachable, but no Metal-class device is present.
    /// Reserved for in-guest runs; not claimed by the default host harness.
    FrameworkPresentNoDevice,
}

/// Structured failure reason recorded when a probe attempt does not satisfy D10.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GuestMetalProbeFailureReason {
    /// No guest Metal device is registered / discovered.
    NoGuestMetalDevice,
    /// Public ABI refuses enumeration (empty success would imply Metal).
    EnumerationNotAdvertised,
    /// Named guest capability is unsupported on this surface.
    CapabilityUnsupported { capability: String },
    /// In-process / attached guest Metal runtime is unavailable.
    GuestRuntimeUnavailable { detail: String },
    /// Software VirtualMetalDevice still reports `supports_metal() == false`.
    VirtualDeviceDoesNotSupportMetal,
    /// Design D10 / acceptance remains unmet.
    AcceptanceUnmet,
    /// Probe attempt completed but must not pass without a real guest device.
    PassForbiddenWithoutGuestDevice,
}

impl GuestMetalProbeFailureReason {
    /// Stable machine id (recovery-style tokens where applicable).
    pub fn code(&self) -> &'static str {
        match self {
            Self::NoGuestMetalDevice => "no-metal-device",
            Self::EnumerationNotAdvertised => "enumeration-not-advertised",
            Self::CapabilityUnsupported { .. } => "capability-unsupported",
            Self::GuestRuntimeUnavailable { .. } => "guest-runtime-unavailable",
            Self::VirtualDeviceDoesNotSupportMetal => "virtual-device-no-metal",
            Self::AcceptanceUnmet => "acceptance-unmet",
            Self::PassForbiddenWithoutGuestDevice => "pass-forbidden-without-guest-device",
        }
    }
}

/// Immutable receipt from one guest Metal probe attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuestMetalProbeReceipt {
    /// Doc/CI lock marker echoed into the receipt.
    pub marker: String,
    /// True when the harness actually ran the probe attempt (even on failure).
    pub actual_guest_metal_probe_executed: bool,
    /// Soft presence of a guest Metal runtime context.
    pub runtime_presence: GuestMetalRuntimePresence,
    /// Ordered structured failure reasons (non-empty when `probe_passed` is false).
    pub failure_reasons: Vec<GuestMetalProbeFailureReason>,
    /// Machine ids mirroring `failure_reasons` (e.g. `no-metal-device`).
    pub failure_codes: Vec<String>,
    /// Always false at M5 unless a real guest Metal device exists (never invented).
    pub probe_passed: bool,
    /// Always false at M5 — Design D10 unmet.
    pub metal_verified: bool,
    /// Always false — no guest Metal device registration at M5.
    pub guest_metal_device: bool,
    /// M1 freeze still holds under this probe.
    pub display_path_freeze_green: bool,
}

impl GuestMetalProbeReceipt {
    /// Contract helper: M5 must never promote Metal acceptance or fake pass.
    pub fn asserts_honest_fail_without_device(&self) -> bool {
        self.actual_guest_metal_probe_executed
            && !self.probe_passed
            && !self.metal_verified
            && !self.guest_metal_device
            && !self.failure_reasons.is_empty()
            && self.failure_codes.iter().any(|c| c == "no-metal-device")
            && self
                .failure_codes
                .iter()
                .any(|c| c == "pass-forbidden-without-guest-device")
    }
}

/// Guest Metal probe harness. Composes M0–M4 surfaces read-only.
#[derive(Debug, Clone)]
pub struct GuestMetalProbe {
    abi: MetalPublicAbi,
}

impl GuestMetalProbe {
    /// Honest M5 handle: current ABI / unmet acceptance / frozen display path.
    pub fn current() -> Self {
        Self {
            abi: MetalPublicAbi::current(),
        }
    }

    pub fn abi(&self) -> &MetalPublicAbi {
        &self.abi
    }

    pub fn acceptance(&self) -> &MetalAcceptanceReport {
        self.abi.acceptance()
    }

    pub fn display_path_freeze(&self) -> DisplayPathFreeze {
        self.abi.display_path_freeze()
    }

    pub fn transport(&self) -> MetalTransport {
        MetalTransport::from_abi(&self.abi)
    }

    pub fn accel_probe(&self) -> AccelProbe {
        AccelProbe::current()
    }

    /// Soft-detect whether this process is a macOS guest Metal context.
    pub fn soft_detect_runtime(&self) -> GuestMetalRuntimePresence {
        soft_detect_runtime_inner()
    }

    /// Run the guest Metal probe **attempt** and emit a receipt.
    ///
    /// Always sets `actual_guest_metal_probe_executed=true` when this function
    /// returns. On hosts without guest Metal (default), records structured
    /// failure reasons and keeps `probe_passed` / `metal_verified` /
    /// `guest_metal_device` false. Never invents a device or D10 pass.
    pub fn run_probe_attempt(&self) -> GuestMetalProbeReceipt {
        let runtime_presence = self.soft_detect_runtime();
        let mut failure_reasons: Vec<GuestMetalProbeFailureReason> = Vec::new();

        // Runtime context (honest absence on Windows/WSL lab hosts).
        match &runtime_presence {
            GuestMetalRuntimePresence::HostNotGuestMacOs { host_os } => {
                failure_reasons.push(GuestMetalProbeFailureReason::GuestRuntimeUnavailable {
                    detail: format!(
                        "host_os={host_os}; in-process Metal.framework / Recovery Terminal unavailable"
                    ),
                });
            }
            GuestMetalRuntimePresence::GuestSessionUnavailable { reason } => {
                failure_reasons.push(GuestMetalProbeFailureReason::GuestRuntimeUnavailable {
                    detail: reason.clone(),
                });
            }
            GuestMetalRuntimePresence::FrameworkPresentNoDevice => {
                // Still no device — continue collecting reject reasons below.
            }
        }

        // Public ABI enumeration must reject (not empty success).
        match self.abi.execute(MetalGuestOp::EnumerateDevices) {
            Ok(()) => {
                // Defensive: M2/M5 surfaces must never succeed here.
                failure_reasons.push(GuestMetalProbeFailureReason::PassForbiddenWithoutGuestDevice);
            }
            Err(MetalAbiReject::EnumerationNotAdvertised) => {
                failure_reasons.push(GuestMetalProbeFailureReason::EnumerationNotAdvertised);
            }
            Err(MetalAbiReject::NoGuestMetalDevice) => {
                failure_reasons.push(GuestMetalProbeFailureReason::NoGuestMetalDevice);
            }
            Err(MetalAbiReject::CapabilityUnsupported(cap)) => {
                failure_reasons.push(GuestMetalProbeFailureReason::CapabilityUnsupported {
                    capability: format!("{cap:?}"),
                });
            }
            Err(MetalAbiReject::AcceptanceUnmet) => {
                failure_reasons.push(GuestMetalProbeFailureReason::AcceptanceUnmet);
            }
        }

        // Sample a resource-class op: must reject without inventing a device.
        match self.abi.execute(MetalGuestOp::CreateBuffer { size: 64 }) {
            Ok(()) => {
                failure_reasons.push(GuestMetalProbeFailureReason::PassForbiddenWithoutGuestDevice);
            }
            Err(MetalAbiReject::CapabilityUnsupported(cap)) => {
                failure_reasons.push(GuestMetalProbeFailureReason::CapabilityUnsupported {
                    capability: format!("{cap:?}"),
                });
            }
            Err(MetalAbiReject::NoGuestMetalDevice) => {
                if !failure_reasons
                    .iter()
                    .any(|r| matches!(r, GuestMetalProbeFailureReason::NoGuestMetalDevice))
                {
                    failure_reasons.push(GuestMetalProbeFailureReason::NoGuestMetalDevice);
                }
            }
            Err(MetalAbiReject::EnumerationNotAdvertised) => {
                failure_reasons.push(GuestMetalProbeFailureReason::EnumerationNotAdvertised);
            }
            Err(MetalAbiReject::AcceptanceUnmet) => {
                failure_reasons.push(GuestMetalProbeFailureReason::AcceptanceUnmet);
            }
        }

        // Software VirtualMetalDevice remains non-Metal (host path ≠ guest Metal).
        let vdev = VirtualMetalDevice::new(GpuCapabilities::default_amd());
        if vdev.supports_metal() {
            // Unexpected: do not treat as guest Metal pass.
            failure_reasons.push(GuestMetalProbeFailureReason::PassForbiddenWithoutGuestDevice);
        } else {
            failure_reasons.push(GuestMetalProbeFailureReason::VirtualDeviceDoesNotSupportMetal);
            if !failure_reasons
                .iter()
                .any(|r| matches!(r, GuestMetalProbeFailureReason::NoGuestMetalDevice))
            {
                failure_reasons.push(GuestMetalProbeFailureReason::NoGuestMetalDevice);
            }
        }

        if !self.acceptance().d10_complete() || self.acceptance().metal_verified {
            // Unmet is expected; verified-without-D10 must not appear from current().
            if !self.acceptance().d10_complete() {
                failure_reasons.push(GuestMetalProbeFailureReason::AcceptanceUnmet);
            }
        }

        // Hard lock: M5 cannot pass without a real guest Metal device.
        failure_reasons.push(GuestMetalProbeFailureReason::PassForbiddenWithoutGuestDevice);

        let failure_codes: Vec<String> = failure_reasons
            .iter()
            .map(|r| r.code().to_string())
            .collect();

        let freeze = freeze_under_metal_track().with_metal_acceptance(self.acceptance());

        GuestMetalProbeReceipt {
            marker: M5_GUEST_METAL_PROBE_DOC_MARKER.to_string(),
            actual_guest_metal_probe_executed: true,
            runtime_presence,
            failure_reasons,
            failure_codes,
            probe_passed: false,
            metal_verified: false,
            guest_metal_device: false,
            display_path_freeze_green: freeze.matches_m1_lock(),
        }
    }
}

fn soft_detect_runtime_inner() -> GuestMetalRuntimePresence {
    // M5 harness runs in the nextcore-gpu host process. Unless we are a macOS
    // guest with an attached Recovery/installed-OS Metal session (not modeled
    // here), report honest absence. Do not invent FrameworkPresentNoDevice.
    #[cfg(target_os = "macos")]
    {
        GuestMetalRuntimePresence::GuestSessionUnavailable {
            reason: "macos host process is not an attached Recovery/installed-OS guest Metal session"
                .to_string(),
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        let host_os = if cfg!(target_os = "windows") {
            "windows"
        } else if cfg!(target_os = "linux") {
            "linux"
        } else {
            "unknown"
        };
        GuestMetalRuntimePresence::HostNotGuestMacOs {
            host_os: host_os.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal_public_abi::MetalGuestCapability;

    #[test]
    fn probe_attempt_executes_and_fails_honestly() {
        let harness = GuestMetalProbe::current();
        let receipt = harness.run_probe_attempt();
        assert_eq!(receipt.marker, M5_GUEST_METAL_PROBE_DOC_MARKER);
        assert!(receipt.actual_guest_metal_probe_executed);
        assert!(receipt.asserts_honest_fail_without_device());
        assert!(receipt.display_path_freeze_green);
        assert!(!harness.acceptance().metal_verified);
        assert!(!harness.acceptance().d10_complete());
        assert!(receipt
            .failure_codes
            .iter()
            .any(|c| c == "enumeration-not-advertised"));
        assert!(receipt
            .failure_codes
            .iter()
            .any(|c| c == "guest-runtime-unavailable"));
    }

    #[test]
    fn m5_preserves_m0_m4_surfaces() {
        let harness = GuestMetalProbe::current();
        assert!(harness.abi().capabilities().asserts_all_unsupported());
        assert!(harness.display_path_freeze().matches_m1_lock());
        assert!(!harness.transport().acceptance().metal_verified);
        let accel = harness.accel_probe().run_software_probe();
        assert!(accel.asserts_no_metal_claim());
        let _ = harness.run_probe_attempt();
        assert!(!harness.acceptance().metal_verified);
        assert!(!harness.abi().capability_supported(MetalGuestCapability::DeviceEnumeration));
    }

    #[test]
    fn failure_reason_codes_are_stable() {
        assert_eq!(
            GuestMetalProbeFailureReason::NoGuestMetalDevice.code(),
            "no-metal-device"
        );
        assert_eq!(
            GuestMetalProbeFailureReason::EnumerationNotAdvertised.code(),
            "enumeration-not-advertised"
        );
    }
}
