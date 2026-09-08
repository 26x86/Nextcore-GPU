use nextcore_gpu::{
    command_executor::{CommandExecutor, ExecutorError, GpuCommand, RecordedCommandBuffer},
    compute::{
        BufferBinding, ComputeError, ComputePipelineDescriptor, ComputePipelineManager,
        ComputeShader,
    },
    framebuffer::{LinearFramebuffer, PixelFormat},
    rasterizer::SoftwareRasterizer,
    texture::{TextureDescriptor, TextureFormat, TextureManager},
};

fn dispatch(pipeline_id: u32, x: u32) -> GpuCommand {
    GpuCommand::DispatchCompute {
        pipeline_id,
        x,
        y: 1,
        z: 1,
    }
}
fn manager() -> (ComputePipelineManager, u32, u64) {
    let mut manager = ComputePipelineManager::new();
    let buffer = manager.create_storage_buffer(16);
    manager
        .write_storage_buffer(buffer, 0, &[0xa5; 16])
        .unwrap();
    let pipeline = manager.create_pipeline(ComputePipelineDescriptor {
        shader: ComputeShader {
            id: 1,
            entry_point: "main".into(),
            bytecode: vec![],
        },
        workgroup_size: [1, 1, 1],
        buffer_bindings: vec![BufferBinding {
            binding: 0,
            buffer_id: buffer,
            offset: 0,
            size: 16,
        }],
    });
    (manager, pipeline, buffer)
}

#[test]
fn recorded_compute_requires_a_backend_and_never_counts_a_noop() {
    let mut executor = CommandExecutor::new();
    let mut textures = TextureManager::new();
    let commands = RecordedCommandBuffer {
        commands: vec![dispatch(1, 1)],
    };
    assert!(matches!(
        executor.execute(&commands, &mut [], &mut textures),
        Err(ExecutorError::Compute(ComputeError::BackendUnavailable))
    ));
    assert_eq!(executor.executed_command_count(), 0);
}

#[test]
fn recorded_compute_forwards_typed_manager_errors_and_preserves_buffers() {
    let (mut manager, pipeline, buffer) = manager();
    let mut executor = CommandExecutor::new();
    let mut textures = TextureManager::new();
    for (id, x, expected) in [
        (u32::MAX, 1, "pipeline"),
        (pipeline, 0, "dimensions"),
        (pipeline, 1, "backend"),
    ] {
        let commands = RecordedCommandBuffer {
            commands: vec![dispatch(id, x)],
        };
        let error = executor
            .execute_with_compute(&commands, &mut [], &mut textures, &mut manager)
            .unwrap_err();
        assert!(matches!(
            (expected, error),
            (
                "pipeline",
                ExecutorError::Compute(ComputeError::PipelineNotFound(u32::MAX))
            ) | (
                "dimensions",
                ExecutorError::Compute(ComputeError::InvalidDispatch([0, 1, 1]))
            ) | (
                "backend",
                ExecutorError::Compute(ComputeError::BackendUnavailable)
            )
        ));
        assert_eq!(executor.executed_command_count(), 0);
        assert_eq!(
            manager.read_storage_buffer(buffer, 0, 16).unwrap(),
            [0xa5; 16]
        );
    }
}

#[test]
fn failed_dispatch_stops_later_commands_and_retains_prior_work() {
    let (mut manager, pipeline, _) = manager();
    let mut executor = CommandExecutor::new();
    let mut textures = TextureManager::new();
    let texture = textures
        .create_texture(TextureDescriptor::new_2d(2, 2, TextureFormat::Rgba8Unorm))
        .unwrap();
    let commands = RecordedCommandBuffer {
        commands: vec![
            GpuCommand::ClearTexture {
                texture_id: texture,
                color: [1.0, 0.0, 0.0, 1.0],
            },
            dispatch(pipeline, 1),
            GpuCommand::ClearTexture {
                texture_id: texture,
                color: [0.0, 1.0, 0.0, 1.0],
            },
        ],
    };
    assert!(executor
        .execute_with_compute(&commands, &mut [], &mut textures, &mut manager)
        .is_err());
    assert_eq!(
        textures.download_texture(texture).unwrap(),
        [255, 0, 0, 255].repeat(4)
    );
    assert_eq!(executor.executed_command_count(), 0);
}

#[test]
fn compute_rejects_active_render_pass_across_command_lists() {
    let (mut manager, pipeline, _) = manager();
    let mut executor = CommandExecutor::new();
    let mut textures = TextureManager::new();
    let mut framebuffers = [LinearFramebuffer::new(2, 2, PixelFormat::Rgba8)];
    let begin = RecordedCommandBuffer {
        commands: vec![GpuCommand::BeginRenderPass {
            clear_color: [1.0, 0.0, 0.0, 1.0],
            clear_depth: 1.0,
        }],
    };
    executor
        .execute(&begin, &mut framebuffers, &mut textures)
        .unwrap();
    let commands = RecordedCommandBuffer {
        commands: vec![dispatch(pipeline, 1)],
    };
    assert!(matches!(
        executor.execute_with_compute(&commands, &mut framebuffers, &mut textures, &mut manager),
        Err(ExecutorError::ComputeInsideRenderPass)
    ));
    assert_eq!(executor.executed_command_count(), 0);
    let end = RecordedCommandBuffer {
        commands: vec![GpuCommand::EndRenderPass],
    };
    executor
        .execute(&end, &mut framebuffers, &mut textures)
        .unwrap();
    assert!(matches!(
        executor.execute_with_compute(&commands, &mut [], &mut textures, &mut manager),
        Err(ExecutorError::Compute(ComputeError::BackendUnavailable))
    ));
}

#[test]
fn rasterizer_entry_point_does_not_silently_accept_compute() {
    let mut executor = CommandExecutor::new();
    let mut textures = TextureManager::new();
    let mut rasterizer = SoftwareRasterizer::new();
    let commands = RecordedCommandBuffer {
        commands: vec![dispatch(1, 1)],
    };
    assert!(matches!(
        executor.execute_with_rasterizer(&commands, &mut [], &mut textures, &mut rasterizer),
        Err(ExecutorError::Compute(ComputeError::BackendUnavailable))
    ));
    let (mut manager, pipeline, _) = manager();
    let commands = RecordedCommandBuffer {
        commands: vec![dispatch(pipeline, 0)],
    };
    assert!(matches!(
        executor.execute_with_backends(
            &commands,
            &mut [],
            &mut textures,
            &mut rasterizer,
            &mut manager
        ),
        Err(ExecutorError::Compute(ComputeError::InvalidDispatch([
            0, 1, 1
        ])))
    ));
}
