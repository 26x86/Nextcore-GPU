use nextcore_gpu::{
    command_executor::ExecutorError,
    compute::{
        BufferBinding, ComputeError, ComputePipelineDescriptor, ComputePipelineManager,
        ComputeShader,
    },
    sgpu_command::{
        SgpuCommand, SgpuCommandList, SgpuWireError, SGPU_MAX_COMMANDS, SGPU_MAX_COMMAND_BYTES,
    },
    sgpu_compute::{SgpuComputeError, SgpuComputeSession, SGPU_MAX_RESOURCES},
};

const SRC: u64 = 0x1234_5678_0000_0001;
const DST: u64 = 0xfedc_ba98_0000_0002;
const KERNEL: u32 = 0xf000_0001;

fn dispatch() -> SgpuCommand {
    SgpuCommand::ComputeDispatch {
        kernel_id: KERNEL,
        grid: (1, 1, 1),
        block: (1, 1, 1),
    }
}
fn setup() -> (SgpuComputeSession, u32, u64) {
    let mut manager = ComputePipelineManager::new();
    let src = manager.create_storage_buffer(16);
    let dst = manager.create_storage_buffer(16);
    manager.write_storage_buffer(src, 0, &[0xa5; 16]).unwrap();
    manager.write_storage_buffer(dst, 0, &[0x5a; 16]).unwrap();
    let pipeline = manager.create_pipeline(ComputePipelineDescriptor {
        shader: ComputeShader {
            id: 1,
            entry_point: "main".into(),
            bytecode: vec![],
        },
        workgroup_size: [1, 1, 1],
        buffer_bindings: vec![BufferBinding {
            binding: 0,
            buffer_id: dst,
            offset: 0,
            size: 16,
        }],
    });
    let mut session = SgpuComputeSession::new(manager);
    session.register_resource(SRC, src).unwrap();
    session.register_resource(DST, dst).unwrap();
    (session, pipeline, dst)
}

#[test]
fn independently_authored_little_endian_fixture_keeps_full_width_handles() {
    let bytes = [
        1, 0, 0, 0, 1, 1, 0, 0, 0, 0x78, 0x56, 0x34, 0x12, 2, 0, 0, 0, 0x98, 0xba, 0xdc, 0xfe, 16,
        0, 0, 0, 0, 0, 0, 0,
    ];
    let list = SgpuCommandList::decode(&bytes).unwrap();
    assert_eq!(
        list.cmds,
        vec![SgpuCommand::CopyBuffer {
            src: SRC,
            dst: DST,
            size: 16
        }]
    );
    let mut unaligned = vec![0xcc];
    unaligned.extend_from_slice(&bytes);
    unaligned.push(0xdd);
    let (mut session, _, _) = setup();
    assert_eq!(
        session
            .submit_wire(&unaligned, 1, bytes.len() as u64)
            .unwrap(),
        1
    );
    assert_eq!(session.read_resource(DST, 0, 16).unwrap(), [0xa5; 16]);
    assert_eq!(session.completed_command_count(), 1);
}

#[test]
fn all_truncations_and_trailing_bytes_are_rejected() {
    let commands = vec![
        SgpuCommand::CopyBuffer {
            src: SRC,
            dst: DST,
            size: 16,
        },
        SgpuCommand::RenderClear {
            color: [0.0, 1.0, 0.5, 1.0],
        },
        dispatch(),
        SgpuCommand::Present { buffer: DST },
    ];
    for command in &commands {
        let bytes = command.encode();
        for length in 0..bytes.len() {
            assert!(SgpuCommand::decode(&bytes[..length]).is_err());
        }
        let mut trailing = bytes;
        trailing.push(0);
        assert_eq!(
            SgpuCommand::decode(&trailing).unwrap_err(),
            SgpuWireError::Malformed
        );
    }
    let bytes = SgpuCommandList { cmds: commands }.encode();
    for length in 0..bytes.len() {
        assert!(SgpuCommandList::decode(&bytes[..length]).is_err());
    }
    let mut trailing = bytes;
    trailing.push(0);
    assert!(SgpuCommandList::decode(&trailing).is_err());
}

#[test]
fn count_and_byte_attacks_fail_before_reservation() {
    for count in [u32::MAX, SGPU_MAX_COMMANDS as u32 + 1] {
        assert_eq!(
            SgpuCommandList::decode(&count.to_le_bytes()).unwrap_err(),
            SgpuWireError::Limit
        );
    }
    assert_eq!(
        SgpuCommandList::decode(&[2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap_err(),
        SgpuWireError::Malformed
    );
    assert_eq!(
        SgpuCommandList::decode(&vec![0; SGPU_MAX_COMMAND_BYTES + 1]).unwrap_err(),
        SgpuWireError::Limit
    );
    assert!(SgpuCommandList::decode(&[1, 0, 0, 0, 0xff, 0, 0, 0, 0, 0, 0, 0, 0]).is_err());
    let maximum = SgpuCommandList {
        cmds: vec![dispatch(); SGPU_MAX_COMMANDS],
    }
    .encode();
    assert_eq!(maximum.len(), SGPU_MAX_COMMAND_BYTES);
    assert_eq!(
        SgpuCommandList::decode(&maximum).unwrap().cmds.len(),
        SGPU_MAX_COMMANDS
    );
    assert!(SgpuCommandList::decode(&[0, 0, 0, 0])
        .unwrap()
        .cmds
        .is_empty());
    assert!(SgpuCommandList::decode(&[0, 0, 0, 0, 0]).is_err());
}

#[test]
fn snapshot_windows_check_u64_overflow_and_32_bit_boundary() {
    for (offset, length) in [
        (u64::MAX, 4),
        (u64::MAX - 1, 4),
        (1u64 << 32, 4),
        (1, 4),
        (0, u64::MAX),
    ] {
        assert!(SgpuCommandList::decode_window(&[0; 4], offset, length).is_err());
    }
    let (mut session, _, _) = setup();
    assert!(session.submit_wire(&[0; 4], u64::MAX, 4).is_err());
    assert_eq!(session.completed_command_count(), 0);
}

#[test]
fn unsupported_command_preflight_preserves_earlier_copy() {
    for unsupported in [
        SgpuCommand::RenderClear { color: [0.0; 4] },
        SgpuCommand::Present { buffer: DST },
    ] {
        let (mut session, _, _) = setup();
        let error = session
            .submit(&SgpuCommandList {
                cmds: vec![
                    SgpuCommand::CopyBuffer {
                        src: SRC,
                        dst: DST,
                        size: 16,
                    },
                    unsupported,
                ],
            })
            .unwrap_err();
        assert_eq!(error.completed, 0);
        assert_eq!(error.command_index, Some(1));
        assert!(matches!(
            error.error,
            SgpuComputeError::UnsupportedCommand(_)
        ));
        assert_eq!(session.read_resource(DST, 0, 16).unwrap(), [0x5a; 16]);
        assert_eq!(session.completed_command_count(), 0);
    }
}

#[test]
fn backend_failure_reports_committed_prefix_and_stops_later_work() {
    let (mut session, pipeline, _) = setup();
    session.register_kernel(KERNEL, pipeline).unwrap();
    let error = session
        .submit(&SgpuCommandList {
            cmds: vec![
                SgpuCommand::CopyBuffer {
                    src: SRC,
                    dst: DST,
                    size: 8,
                },
                dispatch(),
                SgpuCommand::CopyBuffer {
                    src: DST,
                    dst: SRC,
                    size: 16,
                },
            ],
        })
        .unwrap_err();
    assert_eq!(error.command_index, Some(1));
    assert_eq!(error.completed, 1);
    assert_eq!(session.completed_command_count(), 1);
    assert!(matches!(
        error.error,
        SgpuComputeError::Executor(ExecutorError::Compute(ComputeError::BackendUnavailable))
    ));
    assert_eq!(session.read_resource(DST, 0, 8).unwrap(), [0xa5; 8]);
    assert_eq!(session.read_resource(DST, 8, 8).unwrap(), [0x5a; 8]);
    assert_eq!(session.read_resource(SRC, 0, 16).unwrap(), [0xa5; 16]);
}

#[test]
fn mapping_isolation_and_registered_kernel_lifetime_are_enforced() {
    let (mut session, pipeline, backing) = setup();
    assert!(matches!(
        session.register_resource(99, backing),
        Err(SgpuComputeError::DuplicateResource)
    ));
    session.register_kernel(KERNEL, pipeline).unwrap();
    assert!(matches!(
        session.unregister_resource(DST),
        Err(SgpuComputeError::ResourceInUse(DST))
    ));
    assert!(matches!(
        session.register_kernel(KERNEL, pipeline),
        Err(SgpuComputeError::DuplicateKernel(KERNEL))
    ));
    let other = SgpuComputeSession::new(ComputePipelineManager::new());
    assert!(matches!(
        other.read_resource(DST, 0, 1),
        Err(SgpuComputeError::ResourceNotFound(DST))
    ));
    session.unregister_kernel(KERNEL).unwrap();
    session.unregister_resource(DST).unwrap();
    assert!(matches!(
        session.register_kernel(KERNEL, pipeline),
        Err(SgpuComputeError::UnregisteredBinding(_))
    ));
    assert!(session.read_resource(DST, 0, 1).is_err());
}

#[test]
fn copy_and_io_ranges_do_not_wrap_or_truncate() {
    let (mut session, _, _) = setup();
    for (offset, length) in [(u64::MAX, 1), (16, 1), (1 << 32, 1), (0, u64::MAX)] {
        assert!(matches!(
            session.read_resource(DST, offset, length),
            Err(SgpuComputeError::Range)
        ));
    }
    assert!(matches!(
        session.write_resource(DST, u64::MAX, &[1]),
        Err(SgpuComputeError::Range)
    ));
    for size in [17, u64::MAX, 1 << 32] {
        let error = session
            .submit(&SgpuCommandList {
                cmds: vec![SgpuCommand::CopyBuffer {
                    src: SRC,
                    dst: DST,
                    size,
                }],
            })
            .unwrap_err();
        assert_eq!(error.completed, 0);
        assert!(matches!(error.error, SgpuComputeError::Range));
    }
    assert_eq!(session.read_resource(DST, 0, 16).unwrap(), [0x5a; 16]);
    assert_eq!(session.read_resource(DST, 16, 0).unwrap(), Vec::<u8>::new());
}

#[test]
fn kernel_shape_and_invocation_limits_are_checked_before_any_work() {
    let (mut session, pipeline, _) = setup();
    let missing = session
        .submit(&SgpuCommandList {
            cmds: vec![dispatch()],
        })
        .unwrap_err();
    assert!(matches!(
        missing.error,
        SgpuComputeError::KernelNotFound(KERNEL)
    ));
    session.register_kernel(KERNEL, pipeline).unwrap();
    for (grid, block) in [
        ((1, 1, 1), (2, 1, 1)),
        ((0, 1, 1), (1, 1, 1)),
        ((u32::MAX, u32::MAX, u32::MAX), (1, 1, 1)),
    ] {
        let error = session
            .submit(&SgpuCommandList {
                cmds: vec![SgpuCommand::ComputeDispatch {
                    kernel_id: KERNEL,
                    grid,
                    block,
                }],
            })
            .unwrap_err();
        assert_eq!(error.completed, 0);
        assert!(matches!(
            error.error,
            SgpuComputeError::BlockMismatch
                | SgpuComputeError::Compute(ComputeError::InvalidDispatch(_))
        ));
    }
    assert_eq!(session.completed_command_count(), 0);
}

#[test]
fn host_registrations_are_bounded_and_reject_zero_handles() {
    let mut manager = ComputePipelineManager::new();
    let buffers: Vec<_> = (0..=SGPU_MAX_RESOURCES)
        .map(|_| manager.create_storage_buffer(4))
        .collect();
    let mut session = SgpuComputeSession::new(manager);
    assert!(matches!(
        session.register_resource(0, buffers[0]),
        Err(SgpuComputeError::ZeroHandle)
    ));
    for (index, buffer) in buffers[..SGPU_MAX_RESOURCES].iter().enumerate() {
        session
            .register_resource(index as u64 + 1, *buffer)
            .unwrap();
    }
    assert!(matches!(
        session.register_resource(999, buffers[SGPU_MAX_RESOURCES]),
        Err(SgpuComputeError::Limit)
    ));
    assert_eq!(session.resource_count(), SGPU_MAX_RESOURCES);
    assert_eq!(session.kernel_count(), 0);
}
