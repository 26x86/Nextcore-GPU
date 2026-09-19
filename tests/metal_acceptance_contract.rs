//! Contract tests for Metal Driver Track M0–M6.
//!
//! Enabling `metal_track` must not promote Metal acceptance, must not change
//! VirtualMetalDevice software limits, must not remap ADP window `0x6` /
//! AIC IRQ line 2, must reject guest Metal ABI ops without fake device
//! success or enumeration that implies Metal works, must validate
//! host↔guest buffer/copy transport shapes without queue execution, must
//! run a non-Metal host accel probe with receipt, must run a guest Metal
//! probe attempt that records `actual_guest_metal_probe_executed` plus
//! structured failure reasons, and must run the M6 D10 transcript evaluator
//! (absent guest → all gates fail honestly; fixtures stay unverified) while
//! keeping default-host `metal_verified=false`.


use nextcore_gpu::canonical_spec::GpuCapabilities;
use nextcore_gpu::display_path_freeze::{
    freeze_under_metal_track, DisplayPathFreeze, ADP_AIC_IRQ_LINE, ADP_MMIO_WINDOW_ID,
};
use nextcore_gpu::metal_acceptance::{
    MetalAcceptanceError, MetalAcceptanceGate, MetalAcceptanceReport,
};
use nextcore_gpu::metal_accel_probe::{
    AccelProbe, AccelProbeBackend, HostAccelPresence, M4_ACCEL_PROBE_DOC_MARKER,
};
use nextcore_gpu::metal_d10_gates::{
    GuestMetalTranscript, MetalD10Gates, D10_EVIDENCE_ENUMERATE, D10_EVIDENCE_FENCE,
    D10_EVIDENCE_QUEUE, D10_EVIDENCE_RESOURCE, M6_D10_GATES_DOC_MARKER,
};
use nextcore_gpu::metal_guest_probe::{
    GuestMetalProbe, GuestMetalRuntimePresence, M5_GUEST_METAL_PROBE_DOC_MARKER,
};
use nextcore_gpu::metal_public_abi::{
    MetalAbiReject, MetalGuestCapability, MetalGuestOp, MetalPublicAbi,
};
use nextcore_gpu::metal_transport::{
    BufferCopyShape, BufferCreateShape, BufferDestroyShape, MetalTransport, MetalTransportHeader,
    MetalTransportOpcode, MetalTransportReject, MetalTransportShape, METAL_TRANSPORT_MAGIC,
    METAL_TRANSPORT_VERSION,
};
use nextcore_gpu::virtual_device::{CommandBuffer, GpuCommand, VirtualMetalDevice};

#[test]
fn metal_track_default_report_is_unmet() {
    let report = MetalAcceptanceReport::unmet();
    assert!(!report.metal_verified);
    assert!(!report.d10_complete());
    for gate in [
        MetalAcceptanceGate::DeviceEnumeration,
        MetalAcceptanceGate::ResourceCreation,
        MetalAcceptanceGate::CommandQueueExecution,
        MetalAcceptanceGate::FenceSynchronization,
    ] {
        assert!(!report.gate_met(gate));
    }
}

#[test]
fn metal_track_rejects_synthetic_verified_flag() {
    let err = MetalAcceptanceReport::unmet()
        .with_metal_verified(true)
        .expect_err("must refuse metal_verified without D10");
    assert_eq!(err, MetalAcceptanceError::VerifiedWithoutD10);
}

#[test]
fn metal_track_does_not_promote_virtual_metal_device() {
    let mut device = VirtualMetalDevice::new(GpuCapabilities::default_amd());
    assert!(!device.supports_metal());
    assert!(device.is_software_accelerated());

    let src = device.allocate_buffer(16);
    let dst = device.allocate_buffer(16);
    let ok = device.write_command_buffer(CommandBuffer {
        cmds: vec![GpuCommand::CopyBuffer {
            src: src.handle,
            dst: dst.handle,
            size: 16,
        }],
    });
    assert!(ok.is_ok());

    let rejected = device.write_command_buffer(CommandBuffer {
        cmds: vec![GpuCommand::ComputeKernel {
            kernel_id: 0,
            grid: (1, 1, 1),
            block: (1, 1, 1),
        }],
    });
    assert!(rejected.is_err());
}

#[test]
fn metal_track_cannot_regress_adp_display_path_freeze() {
    let freeze = freeze_under_metal_track();
    assert_eq!(freeze.adp_mmio_window_id, ADP_MMIO_WINDOW_ID);
    assert_eq!(freeze.adp_aic_irq_line, ADP_AIC_IRQ_LINE);
    assert_eq!(ADP_MMIO_WINDOW_ID, 0x6);
    assert_eq!(ADP_AIC_IRQ_LINE, 2);
    assert!(freeze.matches_m1_lock());

    let report = MetalAcceptanceReport::unmet();
    let after = freeze.with_metal_acceptance(&report);
    assert_eq!(after, DisplayPathFreeze::locked());
    assert!(!report.metal_verified);
    assert!(!report.d10_complete());
}

#[test]
fn metal_track_m2_public_abi_rejects_without_fake_device() {
    let abi = MetalPublicAbi::current();
    assert!(abi.capabilities().asserts_all_unsupported());
    assert!(!abi.acceptance().metal_verified);
    assert!(!abi.acceptance().d10_complete());

    for cap in MetalGuestCapability::ALL {
        assert!(!abi.capability_supported(cap));
    }

    let enum_err = abi
        .execute(MetalGuestOp::EnumerateDevices)
        .expect_err("enumeration must not succeed or imply Metal");
    assert_eq!(enum_err, MetalAbiReject::EnumerationNotAdvertised);

    let ops = [
        MetalGuestOp::CreateBuffer { size: 256 },
        MetalGuestOp::CreateCommandQueue,
        MetalGuestOp::SubmitCommandBuffer,
        MetalGuestOp::CreateFence,
        MetalGuestOp::WaitFence,
        MetalGuestOp::LoadMetallib,
        MetalGuestOp::DispatchCompute,
        MetalGuestOp::EncodeRender,
        MetalGuestOp::PresentDrawable,
    ];
    for op in ops {
        let err = abi.execute(op).expect_err("unsupported Metal op must reject");
        assert_eq!(
            err,
            MetalAbiReject::CapabilityUnsupported(op.required_capability())
        );
    }

    // M1 freeze stays green under the M2 ABI surface.
    let freeze = abi.display_path_freeze();
    assert!(freeze.matches_m1_lock());
    assert_eq!(freeze, freeze_under_metal_track());
}

#[test]
fn metal_track_m3_transport_shapes_reject_without_queue() {
    let abi = MetalPublicAbi::current();
    let transport = MetalTransport::from_abi(&abi);
    assert!(!transport.acceptance().metal_verified);

    let shapes = [
        MetalTransportShape::BufferCreate(BufferCreateShape {
            size: 128,
            alignment: 8,
            flags: 0,
        }),
        MetalTransportShape::BufferDestroy(BufferDestroyShape { handle: 3 }),
        MetalTransportShape::BufferCopy(BufferCopyShape {
            src_handle: 1,
            dst_handle: 2,
            src_offset: 0,
            dst_offset: 16,
            size: 32,
        }),
    ];
    for shape in shapes {
        let frame = MetalTransport::encode_shape(shape);
        let parsed = transport
            .parse_frame(&frame)
            .expect("well-formed buffer/copy shape must parse");
        assert_eq!(parsed, shape);
        assert_eq!(
            transport.execute_shape(parsed),
            Err(MetalTransportReject::TransportNotExecutable)
        );
        assert_eq!(
            transport.ingest(&frame),
            Err(MetalTransportReject::TransportNotExecutable)
        );
    }

    // Queue-class opcodes never parse into executable work.
    for op in [
        MetalTransportOpcode::QueueSubmit,
        MetalTransportOpcode::FenceSignal,
    ] {
        let frame = MetalTransportHeader {
            magic: METAL_TRANSPORT_MAGIC,
            version: METAL_TRANSPORT_VERSION,
            opcode: op as u16,
            payload_len: 0,
            flags: 0,
        }
        .encode()
        .to_vec();
        assert_eq!(
            transport.parse_frame(&frame),
            Err(MetalTransportReject::QueueExecutionForbidden(op))
        );
    }

    // Fuzz / reject: truncated and mutated frames never succeed.
    let good = MetalTransport::encode_shape(MetalTransportShape::BufferCopy(BufferCopyShape {
        src_handle: 4,
        dst_handle: 5,
        src_offset: 0,
        dst_offset: 0,
        size: 8,
    }));
    assert!(transport.ingest(&good[..8]).is_err());
    let mut mutant = good.clone();
    mutant[0] ^= 0xFF;
    assert!(transport.ingest(&mutant).is_err());

    // M0–M2 still hold under M3.
    assert!(abi.capabilities().asserts_all_unsupported());
    assert!(abi.display_path_freeze().matches_m1_lock());
    assert!(!abi.acceptance().metal_verified);
}

#[test]
fn metal_track_m4_accel_probe_receipt_without_metal() {
    let probe = AccelProbe::current();
    let receipt = probe.run_software_probe();

    assert_eq!(receipt.marker, M4_ACCEL_PROBE_DOC_MARKER);
    assert_eq!(receipt.backend_used, AccelProbeBackend::SoftwareCpu);
    assert!(receipt.compute.ok, "compute: {}", receipt.compute.detail);
    assert!(receipt.render.ok, "render: {}", receipt.render.detail);
    assert!(receipt.asserts_no_metal_claim());
    assert!(receipt.display_path_freeze_green);
    assert!(!receipt.metal_verified);
    assert!(!receipt.metal_claimed);
    assert!(!receipt.guest_metal_device);

    match &receipt.host_accel {
        HostAccelPresence::VulkanFeatureDisabled
        | HostAccelPresence::VulkanUnavailable { .. }
        | HostAccelPresence::VulkanPresent { .. } => {}
    }

    // Soft detect must not promote acceptance or guest Metal ABI.
    assert!(!probe.acceptance().metal_verified);
    assert!(!probe.acceptance().d10_complete());
    assert!(probe.abi().capabilities().asserts_all_unsupported());
    assert!(probe.display_path_freeze().matches_m1_lock());
    assert!(!probe.transport().acceptance().metal_verified);

    // VirtualMetalDevice remains non-Metal after the probe path.
    let device = VirtualMetalDevice::new(GpuCapabilities::default_amd());
    assert!(!device.supports_metal());
    assert!(device.is_software_accelerated());
    assert_eq!(
        probe.soft_detect_backend_family(),
        AccelProbeBackend::HostVulkanSoft
    );
}

#[test]
fn metal_track_m5_guest_probe_executed_with_fail_reasons() {
    let harness = GuestMetalProbe::current();
    let receipt = harness.run_probe_attempt();

    assert_eq!(receipt.marker, M5_GUEST_METAL_PROBE_DOC_MARKER);
    assert!(
        receipt.actual_guest_metal_probe_executed,
        "M5 must record that the guest Metal probe attempt ran"
    );
    assert!(receipt.asserts_honest_fail_without_device());
    assert!(!receipt.probe_passed);
    assert!(!receipt.metal_verified);
    assert!(!receipt.guest_metal_device);
    assert!(receipt.display_path_freeze_green);
    assert!(
        !receipt.failure_reasons.is_empty(),
        "expected structured failure reasons without guest Metal"
    );
    assert!(receipt
        .failure_codes
        .iter()
        .any(|c| c == "no-metal-device"));
    assert!(receipt
        .failure_codes
        .iter()
        .any(|c| c == "enumeration-not-advertised"));
    assert!(receipt
        .failure_codes
        .iter()
        .any(|c| c == "pass-forbidden-without-guest-device"));

    match &receipt.runtime_presence {
        GuestMetalRuntimePresence::HostNotGuestMacOs { .. }
        | GuestMetalRuntimePresence::GuestSessionUnavailable { .. } => {}
        GuestMetalRuntimePresence::FrameworkPresentNoDevice => {
            panic!("default host harness must not invent FrameworkPresentNoDevice");
        }
    }

    // M0–M4 stay green under M5; acceptance never flips.
    assert!(!harness.acceptance().metal_verified);
    assert!(!harness.acceptance().d10_complete());
    assert!(harness.abi().capabilities().asserts_all_unsupported());
    assert!(harness.display_path_freeze().matches_m1_lock());
    assert!(!harness.transport().acceptance().metal_verified);
    let accel = harness.accel_probe().run_software_probe();
    assert!(accel.asserts_no_metal_claim());
    assert!(!VirtualMetalDevice::new(GpuCapabilities::default_amd()).supports_metal());
}

#[test]
fn metal_track_m6_d10_gates_evaluator_absent_guest() {
    let gates = MetalD10Gates::current();
    let report = gates.evaluate_absent_guest_and_continue();

    assert_eq!(report.marker, M6_D10_GATES_DOC_MARKER);
    assert!(report.evaluator_executed);
    assert!(report.asserts_honest_fail_without_guest());
    assert!(!report.metal_verified);
    assert!(!report.acceptance.metal_verified);
    assert!(!report.d10_complete_in_transcript);
    assert!(report.continued_without_guest_metal);
    assert!(report.display_path_freeze_green);
    assert_eq!(report.gate_checklist.len(), 4);
    for row in &report.gate_checklist {
        assert!(!row.met, "absent guest must leave {:?} unmet", row.gate);
        assert!(row.fail_reason.is_some());
    }
    assert!(report
        .fail_codes
        .iter()
        .any(|c| c == "absent-guest-transcript"));
    assert!(report.fail_codes.iter().any(|c| c == "d10-enumerate-missing"));
    assert!(report.fail_codes.iter().any(|c| c == "d10-resource-missing"));
    assert!(report.fail_codes.iter().any(|c| c == "d10-queue-missing"));
    assert!(report.fail_codes.iter().any(|c| c == "d10-fence-missing"));

    // Fixture with all tokens still cannot promote metal_verified.
    let fixture_body = format!(
        "{D10_EVIDENCE_ENUMERATE}\n{D10_EVIDENCE_RESOURCE}\n{D10_EVIDENCE_QUEUE}\n{D10_EVIDENCE_FENCE}\n"
    );
    let fixture = gates.evaluate_transcript(&GuestMetalTranscript::fixture(
        "m6-contract-fixture",
        fixture_body,
    ));
    assert!(fixture.d10_complete_in_transcript);
    assert!(!fixture.metal_verified);
    assert!(!fixture.acceptance.metal_verified);

    // M0–M5 stay green under M6; default acceptance never flips.
    assert!(!gates.acceptance().metal_verified);
    assert!(!gates.acceptance().d10_complete());
    assert!(gates.abi().capabilities().asserts_all_unsupported());
    assert!(gates.display_path_freeze().matches_m1_lock());
    assert!(!gates.transport().acceptance().metal_verified);
    let accel = gates.guest_probe().accel_probe().run_software_probe();
    assert!(accel.asserts_no_metal_claim());
    let probe = gates.guest_probe().run_probe_attempt();
    assert!(probe.asserts_honest_fail_without_device());
    assert!(!VirtualMetalDevice::new(GpuCapabilities::default_amd()).supports_metal());
}
