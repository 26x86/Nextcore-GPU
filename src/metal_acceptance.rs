//! Guest Metal acceptance honesty contract (feature `metal_track`).
//!
//! Design D10 requires guest device enumeration, resource creation, command
//! queue execution, and fence synchronization. This module only records whether
//! those gates are met. It never synthesizes success from host Vulkan,
//! framebuffer scanout, PCI capability metadata, or feature flags.
//!
//! Guest Metal remains unverified.

use serde::{Deserialize, Serialize};

/// Named gates from Design D10. All start unmet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetalAcceptanceGate {
    DeviceEnumeration,
    ResourceCreation,
    CommandQueueExecution,
    FenceSynchronization,
}

/// Immutable snapshot of Metal acceptance. Construct via [`MetalAcceptanceReport::unmet`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetalAcceptanceReport {
    pub device_enumerated: bool,
    pub resources_created: bool,
    pub command_queue_executed: bool,
    pub fences_synchronized: bool,
    /// Project L6 flag. Remains false until every D10 gate is true with evidence.
    pub metal_verified: bool,
}

impl MetalAcceptanceReport {
    /// Honest default: no guest Metal evidence.
    pub fn unmet() -> Self {
        Self {
            device_enumerated: false,
            resources_created: false,
            command_queue_executed: false,
            fences_synchronized: false,
            metal_verified: false,
        }
    }

    pub fn gate_met(&self, gate: MetalAcceptanceGate) -> bool {
        match gate {
            MetalAcceptanceGate::DeviceEnumeration => self.device_enumerated,
            MetalAcceptanceGate::ResourceCreation => self.resources_created,
            MetalAcceptanceGate::CommandQueueExecution => self.command_queue_executed,
            MetalAcceptanceGate::FenceSynchronization => self.fences_synchronized,
        }
    }

    /// True only when all four D10 gates are set. Does not imply boot success.
    pub fn d10_complete(&self) -> bool {
        self.device_enumerated
            && self.resources_created
            && self.command_queue_executed
            && self.fences_synchronized
    }

    /// `metal_verified` may be true only when D10 is complete. Callers that set
    /// the flag without evidence are rejected.
    pub fn with_metal_verified(mut self, verified: bool) -> Result<Self, MetalAcceptanceError> {
        if verified && !self.d10_complete() {
            return Err(MetalAcceptanceError::VerifiedWithoutD10);
        }
        self.metal_verified = verified;
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetalAcceptanceError {
    /// Refuses `metal_verified=true` when any D10 gate is still unmet.
    VerifiedWithoutD10,
}

impl std::fmt::Display for MetalAcceptanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VerifiedWithoutD10 => write!(
                f,
                "metal_verified requires all Design D10 gates; refusing synthetic success"
            ),
        }
    }
}

impl std::error::Error for MetalAcceptanceError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unmet_report_locks_all_gates_false() {
        let report = MetalAcceptanceReport::unmet();
        assert!(!report.device_enumerated);
        assert!(!report.resources_created);
        assert!(!report.command_queue_executed);
        assert!(!report.fences_synchronized);
        assert!(!report.metal_verified);
        assert!(!report.d10_complete());
        assert!(!report.gate_met(MetalAcceptanceGate::DeviceEnumeration));
    }

    #[test]
    fn rejects_metal_verified_without_d10() {
        let err = MetalAcceptanceReport::unmet()
            .with_metal_verified(true)
            .expect_err("synthetic metal_verified must fail");
        assert_eq!(err, MetalAcceptanceError::VerifiedWithoutD10);
    }
}
