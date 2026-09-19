//! Metal Driver Track M4 — non-Metal acceleration probe backend (x86-oriented).
//!
//! Runs an honest host compute/render soft probe and emits a receipt. The probe
//! uses the software CPU path that this crate already owns (buffer copy +
//! framebuffer fill). Optional host Vulkan presence is soft-detected when the
//! `vulkan` feature is compiled in; absence is reported without inventing
//! devices.
//!
//! This module never claims guest Metal, never sets `metal_verified`, and does
//! not register a guest Metal device. Guest Metal remains unverified.

use serde::{Deserialize, Serialize};

use crate::canonical_spec::GpuCapabilities;
use crate::display_path_freeze::{freeze_under_metal_track, DisplayPathFreeze};
use crate::framebuffer::{LinearFramebuffer, PixelFormat};
use crate::metal_acceptance::MetalAcceptanceReport;
use crate::metal_public_abi::MetalPublicAbi;
use crate::metal_transport::MetalTransport;
use crate::virtual_device::{CommandBuffer, GpuCommand, VirtualMetalDevice};

/// Machine-checkable marker for M4 accel probe (research doc lock).
pub const M4_ACCEL_PROBE_DOC_MARKER: &str = "M4_ACCEL_PROBE:non-metal-host-receipt";

/// Which non-Metal path the probe exercised or detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccelProbeBackend {
    /// Host CPU software path (buffer copy + framebuffer fill). Always usable.
    SoftwareCpu,
    /// Soft host-Vulkan detection path (not Metal; may report absent).
    HostVulkanSoft,
}

/// Honest host acceleration presence (x86 lab / Windows-WSL oriented).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HostAccelPresence {
    /// Crate built without `feature = "vulkan"`; host Vulkan soft probe skipped.
    VulkanFeatureDisabled,
    /// `vulkan` feature present but loader/device soft probe found nothing usable.
    VulkanUnavailable { reason: String },
    /// Soft probe observed at least one host Vulkan physical device.
    /// Does **not** imply guest Metal or `metal_verified`.
    VulkanPresent {
        device_count: usize,
        primary_name: String,
    },
}

/// Outcome of one compute or render soft probe step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeStepOutcome {
    pub ok: bool,
    pub detail: String,
}

/// Immutable receipt from a non-Metal accel probe run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccelProbeReceipt {
    /// Doc/CI lock marker echoed into the receipt.
    pub marker: String,
    /// Backend used for the executable compute/render steps.
    pub backend_used: AccelProbeBackend,
    /// Soft-detected host accel presence (may differ from `backend_used`).
    pub host_accel: HostAccelPresence,
    pub compute: ProbeStepOutcome,
    pub render: ProbeStepOutcome,
    /// Always false at M4 — probe is explicitly non-Metal.
    pub metal_claimed: bool,
    /// Always false at M4 — Design D10 unmet.
    pub metal_verified: bool,
    /// Always false — no guest Metal device registration.
    pub guest_metal_device: bool,
    /// M1 freeze still holds under this probe.
    pub display_path_freeze_green: bool,
}

impl AccelProbeReceipt {
    /// Contract helper: M4 must never promote Metal acceptance.
    pub fn asserts_no_metal_claim(&self) -> bool {
        !self.metal_claimed && !self.metal_verified && !self.guest_metal_device
    }
}

/// Non-Metal acceleration probe handle. Composes M0–M3 surfaces read-only.
#[derive(Debug, Clone)]
pub struct AccelProbe {
    abi: MetalPublicAbi,
}

impl AccelProbe {
    /// Honest M4 handle: current ABI / unmet acceptance / frozen display path.
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

    /// Soft-detect host Vulkan / accel availability without claiming Metal.
    ///
    /// Soft-detect belongs to the [`AccelProbeBackend::HostVulkanSoft`] family;
    /// the return value is presence/absence only and never implies Metal.
    pub fn soft_detect_host_accel(&self) -> HostAccelPresence {
        soft_detect_host_accel_inner()
    }

    /// Backend family used for soft host-accel detection (not Metal).
    pub fn soft_detect_backend_family(&self) -> AccelProbeBackend {
        AccelProbeBackend::HostVulkanSoft
    }

    /// Run the non-Metal software compute + render probe and emit a receipt.
    ///
    /// Compute: host `VirtualMetalDevice` buffer copy (software; `supports_metal`
    /// remains false). Render: host framebuffer clear + fill (software scanout
    /// helper; not ADP guest Metal). Host Vulkan soft detection is recorded
    /// separately and never flips `metal_verified`.
    pub fn run_software_probe(&self) -> AccelProbeReceipt {
        let host_accel = self.soft_detect_host_accel();
        let compute = run_software_compute_probe();
        let render = run_software_render_probe();
        let freeze = freeze_under_metal_track().with_metal_acceptance(self.acceptance());

        AccelProbeReceipt {
            marker: M4_ACCEL_PROBE_DOC_MARKER.to_string(),
            backend_used: AccelProbeBackend::SoftwareCpu,
            host_accel,
            compute,
            render,
            metal_claimed: false,
            metal_verified: false,
            guest_metal_device: false,
            display_path_freeze_green: freeze.matches_m1_lock(),
        }
    }
}

fn soft_detect_host_accel_inner() -> HostAccelPresence {
    #[cfg(not(feature = "vulkan"))]
    {
        HostAccelPresence::VulkanFeatureDisabled
    }

    #[cfg(feature = "vulkan")]
    {
        soft_detect_vulkan_devices()
    }
}

#[cfg(feature = "vulkan")]
fn soft_detect_vulkan_devices() -> HostAccelPresence {
    // Soft probe: load the host Vulkan entry and enumerate physical devices.
    // Errors become honest absence — never invent Metal success.
    // SAFETY: use the platform Vulkan loader and retain it until the probe's
    // instance is destroyed; no Vulkan handles or function pointers escape.
    match unsafe { ash::Entry::load() } {
        Err(e) => HostAccelPresence::VulkanUnavailable {
            reason: format!("vulkan entry load failed: {e}"),
        },
        Ok(entry) => match unsafe { entry.enumerate_instance_extension_properties(None) } {
            Err(e) => HostAccelPresence::VulkanUnavailable {
                reason: format!("instance extension query failed: {e}"),
            },
            Ok(_) => {
                let app_info = ash::vk::ApplicationInfo::default()
                    .application_name(c"nextcore-gpu-m4-soft-probe")
                    .application_version(ash::vk::make_api_version(0, 0, 0, 1))
                    .engine_name(c"nextcore-gpu")
                    .engine_version(ash::vk::make_api_version(0, 0, 0, 1))
                    .api_version(ash::vk::API_VERSION_1_0);
                let create_info = ash::vk::InstanceCreateInfo::default().application_info(&app_info);
                match unsafe { entry.create_instance(&create_info, None) } {
                    Err(e) => HostAccelPresence::VulkanUnavailable {
                        reason: format!("instance create failed: {e}"),
                    },
                    Ok(instance) => {
                        let result = match unsafe { instance.enumerate_physical_devices() } {
                            Err(e) => HostAccelPresence::VulkanUnavailable {
                                reason: format!("physical device enumerate failed: {e}"),
                            },
                            Ok(devices) if devices.is_empty() => {
                                HostAccelPresence::VulkanUnavailable {
                                    reason: "no physical devices".to_string(),
                                }
                            }
                            Ok(devices) => {
                                let count = devices.len();
                                let primary_name = {
                                    let props =
                                        unsafe { instance.get_physical_device_properties(devices[0]) };
                                    let raw = props.device_name;
                                    let bytes: Vec<u8> = raw
                                        .iter()
                                        .take_while(|&&c| c != 0)
                                        .map(|&c| c as u8)
                                        .collect();
                                    String::from_utf8_lossy(&bytes).into_owned()
                                };
                                HostAccelPresence::VulkanPresent {
                                    device_count: count,
                                    primary_name,
                                }
                            }
                        };
                        unsafe {
                            instance.destroy_instance(None);
                        }
                        result
                    }
                }
            }
        },
    }
}

fn run_software_compute_probe() -> ProbeStepOutcome {
    let mut device = VirtualMetalDevice::new(GpuCapabilities::default_amd());
    if device.supports_metal() {
        return ProbeStepOutcome {
            ok: false,
            detail: "VirtualMetalDevice unexpectedly claimed Metal".to_string(),
        };
    }
    let src = device.allocate_buffer(32);
    let dst = device.allocate_buffer(32);
    // Seed pattern so the copy is observable.
    let pattern: Vec<u8> = (0u8..32).collect();
    if device.memory.write(src.handle, 0, &pattern).is_err() {
        return ProbeStepOutcome {
            ok: false,
            detail: "software compute seed write failed".to_string(),
        };
    }
    let ok = device.write_command_buffer(CommandBuffer {
        cmds: vec![GpuCommand::CopyBuffer {
            src: src.handle,
            dst: dst.handle,
            size: 32,
        }],
    });
    match ok {
        Ok(()) => match device.memory.read(dst.handle, 0, 32) {
            Ok(got) if got == pattern => ProbeStepOutcome {
                ok: true,
                detail: "software buffer-copy compute probe passed (non-Metal)".to_string(),
            },
            Ok(_) => ProbeStepOutcome {
                ok: false,
                detail: "software buffer-copy readback mismatch".to_string(),
            },
            Err(e) => ProbeStepOutcome {
                ok: false,
                detail: format!("software buffer-copy readback failed: {e}"),
            },
        },
        Err(e) => ProbeStepOutcome {
            ok: false,
            detail: format!("software buffer-copy failed: {e}"),
        },
    }
}

fn run_software_render_probe() -> ProbeStepOutcome {
    let mut fb = LinearFramebuffer::new(8, 8, PixelFormat::Rgba8);
    fb.clear([0, 0, 0, 255]);
    if let Err(e) = fb.fill_rect(2, 2, 4, 4, [255, 0, 0, 255]) {
        return ProbeStepOutcome {
            ok: false,
            detail: format!("software fill_rect failed: {e}"),
        };
    }
    match fb.get_pixel(3, 3) {
        Ok([255, 0, 0, 255]) => ProbeStepOutcome {
            ok: true,
            detail: "software framebuffer fill render probe passed (non-Metal)".to_string(),
        },
        Ok(other) => ProbeStepOutcome {
            ok: false,
            detail: format!("software render pixel mismatch: {other:?}"),
        },
        Err(e) => ProbeStepOutcome {
            ok: false,
            detail: format!("software render get_pixel failed: {e}"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn software_probe_receipt_never_claims_metal() {
        let probe = AccelProbe::current();
        let receipt = probe.run_software_probe();
        assert_eq!(receipt.marker, M4_ACCEL_PROBE_DOC_MARKER);
        assert_eq!(receipt.backend_used, AccelProbeBackend::SoftwareCpu);
        assert!(receipt.compute.ok);
        assert!(receipt.render.ok);
        assert!(receipt.asserts_no_metal_claim());
        assert!(receipt.display_path_freeze_green);
        assert!(!probe.acceptance().metal_verified);
        assert!(!probe.acceptance().d10_complete());
    }

    #[test]
    fn soft_detect_is_honest_without_metal() {
        let probe = AccelProbe::current();
        let presence = probe.soft_detect_host_accel();
        match &presence {
            HostAccelPresence::VulkanFeatureDisabled
            | HostAccelPresence::VulkanUnavailable { .. }
            | HostAccelPresence::VulkanPresent { .. } => {}
        }
        let receipt = probe.run_software_probe();
        assert!(receipt.asserts_no_metal_claim());
        assert_eq!(receipt.host_accel, presence);
    }

    #[test]
    fn m4_preserves_m0_m3_surfaces() {
        let probe = AccelProbe::current();
        assert!(probe.abi().capabilities().asserts_all_unsupported());
        assert!(probe.display_path_freeze().matches_m1_lock());
        let transport = probe.transport();
        assert!(!transport.acceptance().metal_verified);
        // Soft probe must not flip acceptance.
        let _ = probe.run_software_probe();
        assert!(!probe.acceptance().metal_verified);
    }

    #[test]
    fn vulkan_feature_disabled_reports_absence_path() {
        #[cfg(not(feature = "vulkan"))]
        {
            let presence = AccelProbe::current().soft_detect_host_accel();
            assert_eq!(presence, HostAccelPresence::VulkanFeatureDisabled);
        }
    }
}
