//! Explicit SGPU command adapter over a caller-configured compute manager.
//! Host registration does not implement a guest Metal driver or EFI GPU backend.
use std::collections::HashMap;
use thiserror::Error;

use crate::{
    command_executor::{CommandExecutor, ExecutorError, GpuCommand, RecordedCommandBuffer},
    compute::{validate_dispatch, ComputeError, ComputePipelineManager},
    sgpu_command::{SgpuCommand, SgpuCommandList, SgpuWireError, SGPU_MAX_COMMANDS},
    texture::TextureManager,
};

pub const SGPU_MAX_RESOURCES: usize = 64;
pub const SGPU_MAX_KERNELS: usize = 256;
pub const SGPU_MAX_RESOURCE_BYTES: u64 = 16 * 1024 * 1024;
pub const SGPU_MAX_TOTAL_RESOURCE_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum SgpuComputeError {
    #[error(transparent)]
    Wire(#[from] SgpuWireError),
    #[error(transparent)]
    Compute(#[from] ComputeError),
    #[error(transparent)]
    Executor(#[from] ExecutorError),
    #[error("SGPU resource not registered: {0}")]
    ResourceNotFound(u64),
    #[error("SGPU resource handle or backing buffer already registered")]
    DuplicateResource,
    #[error("SGPU kernel not registered: {0}")]
    KernelNotFound(u32),
    #[error("SGPU kernel handle already registered: {0}")]
    DuplicateKernel(u32),
    #[error("SGPU kernel references an unregistered backing buffer: {0}")]
    UnregisteredBinding(u64),
    #[error("SGPU resource is referenced by a registered kernel: {0}")]
    ResourceInUse(u64),
    #[error("SGPU resource byte range overflow or out of bounds")]
    Range,
    #[error("SGPU registration, command, or completion limit exceeded")]
    Limit,
    #[error("SGPU handles must be nonzero")]
    ZeroHandle,
    #[error("SGPU dispatch block differs from its registered kernel")]
    BlockMismatch,
    #[error("SGPU command tag {0} is unsupported by the compute adapter")]
    UnsupportedCommand(u8),
}

/// The committed prefix is explicit even when a later backend operation fails.
#[derive(Debug, Error)]
#[error("SGPU submission failed at {command_index:?}, after {completed} commands: {error}")]
pub struct SgpuSubmissionError {
    pub command_index: Option<usize>,
    pub completed: usize,
    #[source]
    pub error: SgpuComputeError,
}

#[derive(Clone, Copy)]
struct Resource {
    buffer: u64,
    size: u64,
}

pub struct SgpuComputeSession {
    manager: ComputePipelineManager,
    executor: CommandExecutor,
    textures: TextureManager,
    resources: HashMap<u64, Resource>,
    kernels: HashMap<u32, u32>,
    resource_bytes: u64,
    completed: u64,
}

impl SgpuComputeSession {
    /// The manager and its objects are owned by this session. Registration is a
    /// trusted host operation; the wire accepts neither pointers nor shader bytes.
    pub fn new(manager: ComputePipelineManager) -> Self {
        Self {
            manager,
            executor: CommandExecutor::new(),
            textures: TextureManager::new(),
            resources: HashMap::new(),
            kernels: HashMap::new(),
            resource_bytes: 0,
            completed: 0,
        }
    }

    pub fn register_resource(
        &mut self,
        guest_handle: u64,
        buffer: u64,
    ) -> Result<(), SgpuComputeError> {
        if guest_handle == 0 || buffer == 0 {
            return Err(SgpuComputeError::ZeroHandle);
        }
        if self.resources.contains_key(&guest_handle)
            || self
                .resources
                .values()
                .any(|resource| resource.buffer == buffer)
        {
            return Err(SgpuComputeError::DuplicateResource);
        }
        let size = self
            .manager
            .storage_buffer_size(buffer)
            .ok_or(SgpuComputeError::ResourceNotFound(guest_handle))? as u64;
        let total = self
            .resource_bytes
            .checked_add(size)
            .ok_or(SgpuComputeError::Limit)?;
        if size == 0
            || size > SGPU_MAX_RESOURCE_BYTES
            || total > SGPU_MAX_TOTAL_RESOURCE_BYTES
            || self.resources.len() >= SGPU_MAX_RESOURCES
        {
            return Err(SgpuComputeError::Limit);
        }
        self.resources
            .insert(guest_handle, Resource { buffer, size });
        self.resource_bytes = total;
        Ok(())
    }

    pub fn unregister_resource(&mut self, guest_handle: u64) -> Result<(), SgpuComputeError> {
        let resource = self.resource(guest_handle)?;
        if self.kernels.values().any(|id| {
            self.manager.get_pipeline(*id).is_some_and(|pipeline| {
                pipeline
                    .descriptor
                    .buffer_bindings
                    .iter()
                    .any(|binding| binding.buffer_id == resource.buffer)
            })
        }) {
            return Err(SgpuComputeError::ResourceInUse(guest_handle));
        }
        self.resources.remove(&guest_handle);
        self.resource_bytes -= resource.size;
        Ok(())
    }

    pub fn register_kernel(
        &mut self,
        guest_id: u32,
        pipeline_id: u32,
    ) -> Result<(), SgpuComputeError> {
        if guest_id == 0 {
            return Err(SgpuComputeError::ZeroHandle);
        }
        if self.kernels.contains_key(&guest_id) {
            return Err(SgpuComputeError::DuplicateKernel(guest_id));
        }
        if self.kernels.len() >= SGPU_MAX_KERNELS {
            return Err(SgpuComputeError::Limit);
        }
        let pipeline = self
            .manager
            .get_pipeline(pipeline_id)
            .ok_or(ComputeError::PipelineNotFound(pipeline_id))?;
        validate_dispatch(&pipeline.descriptor, [1, 1, 1])?;
        for binding in &pipeline.descriptor.buffer_bindings {
            let resource = self
                .resources
                .values()
                .find(|resource| resource.buffer == binding.buffer_id)
                .ok_or(SgpuComputeError::UnregisteredBinding(binding.buffer_id))?;
            checked_range(resource.size, binding.offset, binding.size)?;
        }
        self.kernels.insert(guest_id, pipeline_id);
        Ok(())
    }

    pub fn unregister_kernel(&mut self, guest_id: u32) -> Result<(), SgpuComputeError> {
        self.kernels
            .remove(&guest_id)
            .ok_or(SgpuComputeError::KernelNotFound(guest_id))?;
        Ok(())
    }

    pub fn write_resource(
        &mut self,
        handle: u64,
        offset: u64,
        data: &[u8],
    ) -> Result<(), SgpuComputeError> {
        let resource = self.resource(handle)?;
        let (offset, _) = checked_range(resource.size, offset, data.len() as u64)?;
        self.manager
            .write_storage_buffer(resource.buffer, offset, data)?;
        Ok(())
    }

    pub fn read_resource(
        &self,
        handle: u64,
        offset: u64,
        length: u64,
    ) -> Result<Vec<u8>, SgpuComputeError> {
        let resource = self.resource(handle)?;
        let (offset, length) = checked_range(resource.size, offset, length)?;
        Ok(self
            .manager
            .read_storage_buffer(resource.buffer, offset, length)?)
    }

    /// Exposes host-side observation only; registrations cannot be invalidated
    /// through a mutable manager reference while the session is live.
    pub fn manager(&self) -> &ComputePipelineManager {
        &self.manager
    }
    pub fn into_manager(self) -> ComputePipelineManager {
        self.manager
    }
    pub fn completed_command_count(&self) -> u64 {
        self.completed
    }
    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }
    pub fn kernel_count(&self) -> usize {
        self.kernels.len()
    }

    pub fn submit_wire(
        &mut self,
        snapshot: &[u8],
        offset: u64,
        length: u64,
    ) -> Result<usize, SgpuSubmissionError> {
        let list = SgpuCommandList::decode_window(snapshot, offset, length).map_err(|error| {
            SgpuSubmissionError {
                command_index: None,
                completed: 0,
                error: error.into(),
            }
        })?;
        self.submit(&list)
    }

    pub fn submit(&mut self, list: &SgpuCommandList) -> Result<usize, SgpuSubmissionError> {
        if list.cmds.len() > SGPU_MAX_COMMANDS
            || self.completed.checked_add(list.cmds.len() as u64).is_none()
        {
            return Err(SgpuSubmissionError {
                command_index: None,
                completed: 0,
                error: SgpuComputeError::Limit,
            });
        }
        // Whole-list preflight keeps malformed, unsupported, and unmapped work
        // from changing buffers. Driver failures can still occur during execution.
        for (index, command) in list.cmds.iter().enumerate() {
            self.validate_command(command)
                .map_err(|error| SgpuSubmissionError {
                    command_index: Some(index),
                    completed: 0,
                    error,
                })?;
        }
        for (index, command) in list.cmds.iter().enumerate() {
            self.execute_command(command)
                .map_err(|error| SgpuSubmissionError {
                    command_index: Some(index),
                    completed: index,
                    error,
                })?;
            self.completed += 1;
        }
        Ok(list.cmds.len())
    }

    fn resource(&self, handle: u64) -> Result<Resource, SgpuComputeError> {
        self.resources
            .get(&handle)
            .copied()
            .ok_or(SgpuComputeError::ResourceNotFound(handle))
    }

    fn validate_command(&self, command: &SgpuCommand) -> Result<(), SgpuComputeError> {
        match command {
            SgpuCommand::CopyBuffer { src, dst, size } => {
                checked_range(self.resource(*src)?.size, 0, *size)?;
                checked_range(self.resource(*dst)?.size, 0, *size)?;
                Ok(())
            }
            SgpuCommand::ComputeDispatch {
                kernel_id,
                grid,
                block,
            } => {
                let pipeline_id = self
                    .kernels
                    .get(kernel_id)
                    .ok_or(SgpuComputeError::KernelNotFound(*kernel_id))?;
                let pipeline = self
                    .manager
                    .get_pipeline(*pipeline_id)
                    .ok_or(ComputeError::PipelineNotFound(*pipeline_id))?;
                if pipeline.descriptor.workgroup_size != [block.0, block.1, block.2] {
                    return Err(SgpuComputeError::BlockMismatch);
                }
                validate_dispatch(&pipeline.descriptor, [grid.0, grid.1, grid.2])?;
                Ok(())
            }
            _ => Err(SgpuComputeError::UnsupportedCommand(command.tag())),
        }
    }

    fn execute_command(&mut self, command: &SgpuCommand) -> Result<(), SgpuComputeError> {
        match command {
            SgpuCommand::CopyBuffer { src, dst, size } => {
                let data = self.read_resource(*src, 0, *size)?;
                self.write_resource(*dst, 0, &data)
            }
            SgpuCommand::ComputeDispatch {
                kernel_id, grid, ..
            } => {
                let pipeline_id = *self
                    .kernels
                    .get(kernel_id)
                    .ok_or(SgpuComputeError::KernelNotFound(*kernel_id))?;
                self.executor.execute_with_compute(
                    &RecordedCommandBuffer {
                        commands: vec![GpuCommand::DispatchCompute {
                            pipeline_id,
                            x: grid.0,
                            y: grid.1,
                            z: grid.2,
                        }],
                    },
                    &mut [],
                    &mut self.textures,
                    &mut self.manager,
                )?;
                Ok(())
            }
            _ => Err(SgpuComputeError::UnsupportedCommand(command.tag())),
        }
    }
}

fn checked_range(size: u64, offset: u64, length: u64) -> Result<(usize, usize), SgpuComputeError> {
    let end = offset.checked_add(length).ok_or(SgpuComputeError::Range)?;
    if end > size {
        return Err(SgpuComputeError::Range);
    }
    Ok((
        usize::try_from(offset).map_err(|_| SgpuComputeError::Range)?,
        usize::try_from(length).map_err(|_| SgpuComputeError::Range)?,
    ))
}
