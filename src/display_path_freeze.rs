//! Preserve project-defined logical display wiring while evaluating Metal contracts.
//! These authored test identifiers are not Apple physical register addresses.
//! This contract does not establish guest Metal support.

use crate::metal_acceptance::MetalAcceptanceReport;

/// Graph-local ADP MMIO window id (`VF_M1_MMIO_WINDOW_ADP`).
pub const ADP_MMIO_WINDOW_ID: u32 = 0x6;

/// Graph-local ADP vblank / present AIC line (`VF_M1_ADP_IRQ_LINE` / `DISPLAY_SOURCE`).
pub const ADP_AIC_IRQ_LINE: u32 = 2;

/// Machine-checkable identifier for the project-defined display contract.
pub const M1_FREEZE_DOC_MARKER: &str = "M1_DISPLAY_PATH_FREEZE:window=0x6,irq=2";

/// Immutable freeze snapshot. Metal acceptance state cannot alter these ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayPathFreeze {
    pub adp_mmio_window_id: u32,
    pub adp_aic_irq_line: u32,
}

impl DisplayPathFreeze {
    /// Canonical frozen scanout wiring for Metal track M1+.
    pub const fn locked() -> Self {
        Self {
            adp_mmio_window_id: ADP_MMIO_WINDOW_ID,
            adp_aic_irq_line: ADP_AIC_IRQ_LINE,
        }
    }

    /// Metal honesty reports never own or clear the display IRQ path.
    pub fn with_metal_acceptance(self, report: &MetalAcceptanceReport) -> Self {
        let _ = report;
        self
    }

    /// True when values still match the M1 lock (window `0x6`, IRQ `2`).
    pub fn matches_m1_lock(&self) -> bool {
        self.adp_mmio_window_id == ADP_MMIO_WINDOW_ID
            && self.adp_aic_irq_line == ADP_AIC_IRQ_LINE
    }
}

/// Metal track feature / unmet acceptance must leave the freeze unchanged.
pub fn freeze_under_metal_track() -> DisplayPathFreeze {
    DisplayPathFreeze::locked().with_metal_acceptance(&MetalAcceptanceReport::unmet())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal_acceptance::MetalAcceptanceReport;

    #[test]
    fn locked_constants_are_window_6_irq_2() {
        let freeze = DisplayPathFreeze::locked();
        assert_eq!(freeze.adp_mmio_window_id, 0x6);
        assert_eq!(freeze.adp_aic_irq_line, 2);
        assert!(freeze.matches_m1_lock());
    }

    #[test]
    fn metal_acceptance_cannot_remap_display_irq() {
        let report = MetalAcceptanceReport::unmet();
        let freeze = DisplayPathFreeze::locked().with_metal_acceptance(&report);
        assert!(freeze.matches_m1_lock());
        assert!(!report.metal_verified);
        assert_eq!(freeze_under_metal_track(), DisplayPathFreeze::locked());
    }
}
