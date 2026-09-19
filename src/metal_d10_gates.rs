//! Metal Driver Track M6 — Design D10 acceptance gate evaluator.
//!
//! Evaluates a **single** guest transcript for all four Design D10 gates
//! (enumeration, resources, queue, fences). Always runs to completion and
//! emits a structured report; absence of guest Metal is an honest fail, not a
//! halt. `metal_verified` flips true only when provenance is a live guest
//! session **and** every D10 evidence token appears in that one transcript.
//! Fixtures and host-process transcripts stay `metal_verified=false`.
//!
//! Guest Metal remains unverified.

use serde::{Deserialize, Serialize};

use crate::display_path_freeze::{freeze_under_metal_track, DisplayPathFreeze};
use crate::metal_acceptance::{MetalAcceptanceGate, MetalAcceptanceReport};
use crate::metal_guest_probe::GuestMetalProbe;
use crate::metal_public_abi::MetalPublicAbi;
use crate::metal_transport::MetalTransport;

/// Machine-checkable marker for M6 D10 gates evaluator (research doc lock).
pub const M6_D10_GATES_DOC_MARKER: &str = "M6_D10_GATES:evaluator-acceptance-pending";

/// Clean-room evidence tokens a real guest Metal probe must emit (one transcript).
pub const D10_EVIDENCE_ENUMERATE: &str = "NCMETAL_D10:enumerate:ok";
pub const D10_EVIDENCE_RESOURCE: &str = "NCMETAL_D10:resource:ok";
pub const D10_EVIDENCE_QUEUE: &str = "NCMETAL_D10:queue:ok";
pub const D10_EVIDENCE_FENCE: &str = "NCMETAL_D10:fence:ok";

/// Where a transcript came from. Only [`TranscriptProvenance::GuestLive`] may
/// promote `metal_verified` when all D10 tokens are present.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TranscriptProvenance {
    /// No guest transcript attached (default lab / auto-overcome path).
    AbsentGuestTranscript { reason: String },
    /// Host-process or labeled fixture input; never promotes Metal acceptance.
    HostProcessOrFixture { label: String },
    /// Live guest session transcript (Recovery / installed-OS Metal probe).
    GuestLive { session_id: String },
}

impl TranscriptProvenance {
    pub fn may_promote_metal_verified(&self) -> bool {
        matches!(self, Self::GuestLive { .. })
    }

    pub fn code(&self) -> &'static str {
        match self {
            Self::AbsentGuestTranscript { .. } => "absent-guest-transcript",
            Self::HostProcessOrFixture { .. } => "host-process-or-fixture",
            Self::GuestLive { .. } => "guest-live",
        }
    }
}

/// One Design D10 gate evaluation row.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct D10GateChecklistRow {
    pub gate: MetalAcceptanceGate,
    pub met: bool,
    pub evidence_token: String,
    pub fail_reason: Option<String>,
}

/// Immutable structured report from one D10 evaluation run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetalD10EvaluationReport {
    /// Doc/CI lock marker echoed into the report.
    pub marker: String,
    /// True when the evaluator actually ran (always true on return).
    pub evaluator_executed: bool,
    /// Provenance of the evaluated transcript.
    pub provenance: TranscriptProvenance,
    /// Per-gate checklist (enumeration / resources / queue / fences).
    pub gate_checklist: Vec<D10GateChecklistRow>,
    /// Acceptance snapshot derived from the checklist.
    pub acceptance: MetalAcceptanceReport,
    /// True only for GuestLive + all four D10 tokens in one transcript.
    pub metal_verified: bool,
    /// True when the four D10 checklist rows are all met (may still be fixture).
    pub d10_complete_in_transcript: bool,
    /// M1 freeze still holds under this evaluation.
    pub display_path_freeze_green: bool,
    /// Auto-overcome: evaluation continued despite missing guest Metal.
    pub continued_without_guest_metal: bool,
    /// Stable fail codes when acceptance is unmet.
    pub fail_codes: Vec<String>,
    /// Human-readable notes (English).
    pub notes: String,
}

impl MetalD10EvaluationReport {
    /// Contract helper: default host / absent-guest path must not claim Metal.
    pub fn asserts_honest_fail_without_guest(&self) -> bool {
        self.evaluator_executed
            && !self.metal_verified
            && !self.acceptance.metal_verified
            && !self.acceptance.d10_complete()
            && self.continued_without_guest_metal
            && self.gate_checklist.iter().all(|row| !row.met)
            && !self.fail_codes.is_empty()
    }
}

/// Input transcript: one guest log body evaluated as a whole.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuestMetalTranscript {
    pub provenance: TranscriptProvenance,
    /// Full guest log / receipt body (UTF-8 lines or free-form text).
    pub body: String,
}

impl GuestMetalTranscript {
    /// Default lab transcript: absent guest Metal (auto-overcome continue path).
    pub fn absent_on_host() -> Self {
        Self {
            provenance: TranscriptProvenance::AbsentGuestTranscript {
                reason: "no guest Metal transcript attached; host nextcore-gpu evaluator only"
                    .to_string(),
            },
            body: String::new(),
        }
    }

    /// Labeled fixture / host-process input (never promotes `metal_verified`).
    pub fn fixture(label: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            provenance: TranscriptProvenance::HostProcessOrFixture {
                label: label.into(),
            },
            body: body.into(),
        }
    }

    /// Live guest session transcript (only path that may set `metal_verified`).
    pub fn guest_live(session_id: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            provenance: TranscriptProvenance::GuestLive {
                session_id: session_id.into(),
            },
            body: body.into(),
        }
    }
}

/// Design D10 gate evaluator. Composes M0–M5 surfaces read-only.
#[derive(Debug, Clone)]
pub struct MetalD10Gates {
    abi: MetalPublicAbi,
}

impl MetalD10Gates {
    /// Honest M6 handle: current ABI / unmet acceptance / frozen display path.
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

    pub fn guest_probe(&self) -> GuestMetalProbe {
        GuestMetalProbe::current()
    }

    /// Evaluate the default absent-guest transcript and continue (auto-overcome).
    ///
    /// Always returns a structured report with all gates unmet and
    /// `metal_verified=false`. Does not halt the Metal track.
    pub fn evaluate_absent_guest_and_continue(&self) -> MetalD10EvaluationReport {
        self.evaluate_transcript(&GuestMetalTranscript::absent_on_host())
    }

    /// Evaluate one guest transcript for all four Design D10 gates.
    ///
    /// Always completes. `metal_verified` is true only when provenance is
    /// [`TranscriptProvenance::GuestLive`] and every D10 evidence token appears
    /// in this single transcript body. Framebuffer-only or host Vulkan text is
    /// ignored. Fixture provenance keeps `metal_verified=false` even if tokens
    /// are present.
    pub fn evaluate_transcript(&self, transcript: &GuestMetalTranscript) -> MetalD10EvaluationReport {
        let body = &transcript.body;
        let enum_met = body.contains(D10_EVIDENCE_ENUMERATE);
        let resource_met = body.contains(D10_EVIDENCE_RESOURCE);
        let queue_met = body.contains(D10_EVIDENCE_QUEUE);
        let fence_met = body.contains(D10_EVIDENCE_FENCE);

        let gate_checklist = vec![
            checklist_row(
                MetalAcceptanceGate::DeviceEnumeration,
                enum_met,
                D10_EVIDENCE_ENUMERATE,
                "device enumeration evidence token missing from transcript",
            ),
            checklist_row(
                MetalAcceptanceGate::ResourceCreation,
                resource_met,
                D10_EVIDENCE_RESOURCE,
                "resource creation evidence token missing from transcript",
            ),
            checklist_row(
                MetalAcceptanceGate::CommandQueueExecution,
                queue_met,
                D10_EVIDENCE_QUEUE,
                "command queue execution evidence token missing from transcript",
            ),
            checklist_row(
                MetalAcceptanceGate::FenceSynchronization,
                fence_met,
                D10_EVIDENCE_FENCE,
                "fence synchronization evidence token missing from transcript",
            ),
        ];

        let d10_complete_in_transcript = enum_met && resource_met && queue_met && fence_met;
        let may_promote = transcript.provenance.may_promote_metal_verified();
        let metal_verified = d10_complete_in_transcript && may_promote;

        // Fixtures may contain tokens for parser tests; acceptance flags that
        // feed product `metal_verified` stay false unless GuestLive.
        let acceptance = if metal_verified {
            MetalAcceptanceReport {
                device_enumerated: true,
                resources_created: true,
                command_queue_executed: true,
                fences_synchronized: true,
                metal_verified: true,
            }
        } else {
            MetalAcceptanceReport::unmet()
        };

        let mut fail_codes: Vec<String> = Vec::new();
        if !may_promote {
            fail_codes.push(transcript.provenance.code().to_string());
        }
        if !enum_met {
            fail_codes.push("d10-enumerate-missing".to_string());
        }
        if !resource_met {
            fail_codes.push("d10-resource-missing".to_string());
        }
        if !queue_met {
            fail_codes.push("d10-queue-missing".to_string());
        }
        if !fence_met {
            fail_codes.push("d10-fence-missing".to_string());
        }
        if d10_complete_in_transcript && !may_promote {
            fail_codes.push("fixture-or-non-guest-cannot-verify".to_string());
        }
        if metal_verified {
            fail_codes.clear();
        } else if fail_codes.is_empty() {
            fail_codes.push("acceptance-unmet".to_string());
        }

        let continued_without_guest_metal = !metal_verified;
        let freeze = freeze_under_metal_track().with_metal_acceptance(self.acceptance());

        let notes = if metal_verified {
            "D10 complete in live guest transcript; metal_verified=true".to_string()
        } else if d10_complete_in_transcript && !may_promote {
            format!(
                "transcript contains all D10 tokens but provenance={} cannot promote metal_verified; evaluator continued",
                transcript.provenance.code()
            )
        } else {
            format!(
                "D10 gates unmet (provenance={}); evaluator executed and continued without guest Metal; metal_verified=false",
                transcript.provenance.code()
            )
        };

        MetalD10EvaluationReport {
            marker: M6_D10_GATES_DOC_MARKER.to_string(),
            evaluator_executed: true,
            provenance: transcript.provenance.clone(),
            gate_checklist,
            acceptance,
            metal_verified,
            d10_complete_in_transcript,
            display_path_freeze_green: freeze.matches_m1_lock(),
            continued_without_guest_metal,
            fail_codes,
            notes,
        }
    }
}

fn checklist_row(
    gate: MetalAcceptanceGate,
    met: bool,
    token: &str,
    missing_reason: &str,
) -> D10GateChecklistRow {
    D10GateChecklistRow {
        gate,
        met,
        evidence_token: token.to_string(),
        fail_reason: if met {
            None
        } else {
            Some(missing_reason.to_string())
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_guest_all_gates_fail_and_continues() {
        let eval = MetalD10Gates::current().evaluate_absent_guest_and_continue();
        assert_eq!(eval.marker, M6_D10_GATES_DOC_MARKER);
        assert!(eval.evaluator_executed);
        assert!(eval.asserts_honest_fail_without_guest());
        assert!(eval.display_path_freeze_green);
        assert!(!eval.metal_verified);
        assert!(!eval.acceptance.metal_verified);
        assert!(!eval.d10_complete_in_transcript);
        assert_eq!(eval.gate_checklist.len(), 4);
        assert!(eval
            .fail_codes
            .iter()
            .any(|c| c == "absent-guest-transcript"));
    }

    #[test]
    fn fixture_with_all_tokens_stays_unverified() {
        let body = format!(
            "{D10_EVIDENCE_ENUMERATE}\n{D10_EVIDENCE_RESOURCE}\n{D10_EVIDENCE_QUEUE}\n{D10_EVIDENCE_FENCE}\n"
        );
        let report = MetalD10Gates::current()
            .evaluate_transcript(&GuestMetalTranscript::fixture("unit-fixture", body));
        assert!(report.d10_complete_in_transcript);
        assert!(!report.metal_verified);
        assert!(!report.acceptance.metal_verified);
        assert!(report
            .fail_codes
            .iter()
            .any(|c| c == "fixture-or-non-guest-cannot-verify"));
        assert!(report.continued_without_guest_metal);
    }

    #[test]
    fn guest_live_partial_transcript_does_not_verify() {
        let body = format!("{D10_EVIDENCE_ENUMERATE}\n{D10_EVIDENCE_RESOURCE}\n");
        let report = MetalD10Gates::current()
            .evaluate_transcript(&GuestMetalTranscript::guest_live("session-test", body));
        assert!(!report.d10_complete_in_transcript);
        assert!(!report.metal_verified);
        assert!(report.fail_codes.iter().any(|c| c == "d10-queue-missing"));
        assert!(report.fail_codes.iter().any(|c| c == "d10-fence-missing"));
    }

    #[test]
    fn guest_live_complete_transcript_verifies() {
        let body = format!(
            "guest boot\n{D10_EVIDENCE_ENUMERATE}\n{D10_EVIDENCE_RESOURCE}\n{D10_EVIDENCE_QUEUE}\n{D10_EVIDENCE_FENCE}\n"
        );
        let report = MetalD10Gates::current()
            .evaluate_transcript(&GuestMetalTranscript::guest_live("session-ok", body));
        assert!(report.d10_complete_in_transcript);
        assert!(report.metal_verified);
        assert!(report.acceptance.metal_verified);
        assert!(report.acceptance.d10_complete());
        assert!(!report.continued_without_guest_metal);
        assert!(report.fail_codes.is_empty());
        assert!(report.gate_checklist.iter().all(|r| r.met));
    }

    #[test]
    fn framebuffer_only_text_is_not_d10() {
        let body = "framebuffer scanout ok\nhost vulkan present\n";
        let report = MetalD10Gates::current()
            .evaluate_transcript(&GuestMetalTranscript::guest_live("fb-only", body));
        assert!(!report.metal_verified);
        assert!(!report.d10_complete_in_transcript);
        assert!(report.gate_checklist.iter().all(|r| !r.met));
    }

    #[test]
    fn m6_preserves_m0_m5_surfaces() {
        let gates = MetalD10Gates::current();
        let _ = gates.evaluate_absent_guest_and_continue();
        assert!(!gates.acceptance().metal_verified);
        assert!(gates.display_path_freeze().matches_m1_lock());
        assert!(gates.abi().capabilities().asserts_all_unsupported());
        assert!(!gates.transport().acceptance().metal_verified);
        let probe = gates.guest_probe().run_probe_attempt();
        assert!(probe.asserts_honest_fail_without_device());
    }
}
