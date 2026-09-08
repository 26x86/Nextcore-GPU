//! Actual host GPU acceptance through ComputePipelineManager; never guest Metal.
use nextcore_gpu::{
    command_executor::{CommandExecutor, ExecutorError, GpuCommand, RecordedCommandBuffer},
    compute::{
        BufferBinding, ComputePipelineDescriptor, ComputePipelineManager, ComputeShader,
        DispatchWorkgroup,
    },
    sync::GpuSyncManager,
    texture::TextureManager,
    vulkan_compute::{VulkanComputeConfig, VulkanDeviceSelector},
};
use std::{error::Error, path::PathBuf, time::Duration};

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: vulkan_compute <vendor-hex> <device-hex> <spirv-val-path>".into());
    }
    let hexadecimal = |arg: &std::ffi::OsStr| -> Result<u32, Box<dyn Error>> {
        Ok(u32::from_str_radix(
            arg.to_str()
                .ok_or("non-UTF8 selector")?
                .trim_start_matches("0x"),
            16,
        )?)
    };
    let mut manager = ComputePipelineManager::with_vulkan(VulkanComputeConfig {
        selector: VulkanDeviceSelector {
            vendor_id: hexadecimal(&args[0])?,
            device_id: hexadecimal(&args[1])?,
        },
        spirv_validator: PathBuf::from(&args[2]),
        fence_timeout: Duration::from_secs(3),
    })?;
    let device = manager
        .vulkan_device_info()
        .ok_or("missing device info")?
        .clone();
    let buffer = manager.create_storage_buffer(16 + 256 * 4 + 16);
    let template = ComputePipelineDescriptor {
        shader: ComputeShader {
            id: 1,
            entry_point: "main".into(),
            bytecode: include_bytes!("../tests/fixtures/storage_transform.spv").to_vec(),
        },
        workgroup_size: [64, 1, 1],
        buffer_bindings: vec![BufferBinding {
            binding: 0,
            buffer_id: buffer,
            offset: 16,
            size: 1024,
        }],
    };
    let pipeline = manager.create_pipeline(template.clone());
    let mut sync = GpuSyncManager::new();
    let mut runs = Vec::new();
    for run_index in 0..2u32 {
        let inputs: Vec<u32> = (0..256u32)
            .map(|i| {
                if run_index == 0 {
                    i
                } else {
                    (i * 13 + 5) ^ 0x55aa
                }
            })
            .collect();
        let bytes: Vec<u8> = inputs.iter().flat_map(|v| v.to_le_bytes()).collect();
        manager.write_storage_buffer(buffer, 0, &[0xa5; 16])?;
        manager.write_storage_buffer(buffer, 16, &bytes)?;
        manager.write_storage_buffer(buffer, 1040, &[0x5a; 16])?;
        let fence = sync.create_fence();
        if sync.get_fence(fence).unwrap().is_signaled() {
            return Err("fence already signaled".into());
        }
        manager.dispatch(
            pipeline,
            DispatchWorkgroup { x: 4, y: 1, z: 1 },
            &sync,
            fence,
        )?;
        let fence_completed = sync.get_fence(fence).unwrap().is_signaled();
        if !fence_completed {
            return Err("GPU-complete fence was not signaled".into());
        }
        let output = manager.read_storage_buffer(buffer, 16, 1024)?;
        let values: Vec<u32> = output
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let expected: Vec<_> = inputs
            .iter()
            .map(|v| v.wrapping_mul(3).wrapping_add(7))
            .collect();
        if values != expected || values == inputs {
            return Err("GPU readback differs from independent expected output".into());
        }
        if manager.read_storage_buffer(buffer, 0, 16)? != [0xa5; 16]
            || manager.read_storage_buffer(buffer, 1040, 16)? != [0x5a; 16]
        {
            return Err("binding offset guard changed".into());
        }
        runs.push(serde_json::json!({"run": run_index, "input": inputs, "readback": values, "all_256_matched": true, "guards_unchanged": true, "fence_completed": fence_completed}));
    }
    let mut rejected = Vec::new();
    for kind in [
        "binding-mismatch",
        "local-size-mismatch",
        "invalid-spirv-store",
    ] {
        let mut invalid = template.clone();
        match kind {
            "binding-mismatch" => invalid.buffer_bindings[0].binding = 1,
            "local-size-mismatch" => invalid.workgroup_size = [32, 1, 1],
            _ => {
                // Preserve reflection metadata but give OpStore an undefined
                // pointer ID. SPIRV-Tools must reject this before Vulkan sees it.
                let code = &mut invalid.shader.bytecode;
                let mut at = 20;
                while at < code.len() {
                    let instruction = u32::from_le_bytes(code[at..at + 4].try_into().unwrap());
                    if instruction & 0xffff == 62 {
                        code[at + 4..at + 8].copy_from_slice(&u32::MAX.to_le_bytes());
                        break;
                    }
                    at += (instruction >> 16) as usize * 4;
                }
            }
        }
        let before = manager.read_storage_buffer(buffer, 0, 1056)?;
        let rejected_pipeline = manager.create_pipeline(invalid);
        let fence = sync.create_fence();
        let error = manager
            .dispatch(
                rejected_pipeline,
                DispatchWorkgroup { x: 4, y: 1, z: 1 },
                &sync,
                fence,
            )
            .err()
            .ok_or("invalid shader input unexpectedly executed")?;
        if sync.get_fence(fence).unwrap().is_signaled()
            || manager.read_storage_buffer(buffer, 0, 1056)? != before
        {
            return Err("rejected dispatch changed host buffers or signaled completion".into());
        }
        if kind == "invalid-spirv-store" && !error.to_string().contains("spirv-val rejected") {
            return Err("semantic SPIR-V corruption was not rejected by the validator".into());
        }
        rejected.push(serde_json::json!({"kind": kind, "error": error.to_string(), "fence_unsignaled": true, "host_buffer_unchanged": true}));
    }
    // Exercise the recorded command path, including the dependency between two
    // dispatches. The second upload must contain the first GPU's readback.
    let mut executor = CommandExecutor::new();
    let mut textures = TextureManager::new();
    let inputs: Vec<u32> = (0..256u32).map(|i| (i * 37 + 11) ^ 0xaa55).collect();
    let bytes: Vec<u8> = inputs.iter().flat_map(|v| v.to_le_bytes()).collect();
    manager.write_storage_buffer(buffer, 16, &bytes)?;
    let command = GpuCommand::DispatchCompute {
        pipeline_id: pipeline,
        x: 4,
        y: 1,
        z: 1,
    };
    executor.execute_with_compute(
        &RecordedCommandBuffer {
            commands: vec![command.clone(), command],
        },
        &mut [],
        &mut textures,
        &mut manager,
    )?;
    let actual: Vec<u32> = manager
        .read_storage_buffer(buffer, 16, 1024)?
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let expected: Vec<u32> = inputs
        .iter()
        .map(|v| {
            v.wrapping_mul(3)
                .wrapping_add(7)
                .wrapping_mul(3)
                .wrapping_add(7)
        })
        .collect();
    if actual != expected || executor.executed_command_count() != 2 {
        return Err("recorded dispatch dependency/readback/count mismatch".into());
    }
    if manager.read_storage_buffer(buffer, 0, 16)? != [0xa5; 16]
        || manager.read_storage_buffer(buffer, 1040, 16)? != [0x5a; 16]
    {
        return Err("recorded dispatch changed offset guards".into());
    }
    let before = manager.read_storage_buffer(buffer, 0, 1056)?;
    let error = executor
        .execute_with_compute(
            &RecordedCommandBuffer {
                commands: vec![
                    GpuCommand::DispatchCompute {
                        pipeline_id: u32::MAX,
                        x: 4,
                        y: 1,
                        z: 1,
                    },
                    GpuCommand::DispatchCompute {
                        pipeline_id: pipeline,
                        x: 4,
                        y: 1,
                        z: 1,
                    },
                ],
            },
            &mut [],
            &mut textures,
            &mut manager,
        )
        .err()
        .ok_or("recorded invalid pipeline was accepted")?;
    if !matches!(
        error,
        ExecutorError::Compute(nextcore_gpu::compute::ComputeError::PipelineNotFound(
            u32::MAX
        ))
    ) || manager.read_storage_buffer(buffer, 0, 1056)? != before
        || executor.executed_command_count() != 2
    {
        return Err("recorded command failure was not propagated or did not stop the list".into());
    }
    if manager.vulkan_pipeline_build_count() != Some(1) {
        return Err("same-shader recorded work recompiled the native pipeline".into());
    }
    // Change only an authored shader constant, retaining the public shader ID.
    // A cache keyed by the caller's ID would incorrectly execute the old shader.
    let mut changed = template.clone();
    let code = &mut changed.shader.bytecode;
    let mut at = 20;
    let mut changes = 0;
    while at < code.len() {
        let instruction = u32::from_le_bytes(code[at..at + 4].try_into().unwrap());
        if instruction & 0xffff == 43
            && instruction >> 16 == 4
            && u32::from_le_bytes(code[at + 12..at + 16].try_into().unwrap()) == 7
        {
            code[at + 12..at + 16].copy_from_slice(&9u32.to_le_bytes());
            changes += 1;
        }
        at += (instruction >> 16) as usize * 4;
    }
    if changes != 1 {
        return Err("authored fixture no longer has one additive constant".into());
    }
    let changed_pipeline = manager.create_pipeline(changed);
    for (pipeline_id, addend, compilations) in
        [(changed_pipeline, 9u32, 2u64), (pipeline, 7u32, 3u64)]
    {
        manager.write_storage_buffer(buffer, 16, &bytes)?;
        executor.execute_with_compute(
            &RecordedCommandBuffer {
                commands: vec![GpuCommand::DispatchCompute {
                    pipeline_id,
                    x: 4,
                    y: 1,
                    z: 1,
                }],
            },
            &mut [],
            &mut textures,
            &mut manager,
        )?;
        let readback: Vec<u32> = manager
            .read_storage_buffer(buffer, 16, 1024)?
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let wanted: Vec<u32> = inputs
            .iter()
            .map(|v| v.wrapping_mul(3).wrapping_add(addend))
            .collect();
        if readback != wanted || manager.vulkan_pipeline_build_count() != Some(compilations) {
            return Err(
                "native shader cache replacement reused stale code or compiled incorrectly".into(),
            );
        }
    }
    if manager.read_storage_buffer(buffer, 0, 16)? != [0xa5; 16]
        || manager.read_storage_buffer(buffer, 1040, 16)? != [0x5a; 16]
    {
        return Err("replacement shader changed guards".into());
    }
    let recorded = serde_json::json!({"ordered_dispatches": 2, "input": inputs, "readback": actual,
        "all_256_matched": true, "guards_unchanged": true, "failure_stopped_remaining_work": true,
        "completed_dispatch_count": executor.executed_command_count(), "cache_replacement_dispatches": 2, "same_shader_reused": true,
        "same_id_different_shader_recompiled": true, "native_pipeline_compilations": manager.vulkan_pipeline_build_count()});
    println!(
        "{}",
        serde_json::json!({"device": device, "runs": runs, "recorded_commands": recorded, "rejected_dispatches": rejected, "host_vulkan_compute_verified": true,
        "cpu_fallback": false, "guest_metal_verified": false, "macos_boot_verified": false})
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("vulkan_compute: {error}");
        std::process::exit(1);
    }
}
