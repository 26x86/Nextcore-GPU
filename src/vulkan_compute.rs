//! Explicit host Vulkan 1.1 compute with storage-buffer readback.
//!
//! Supported shader profile: SPIR-V 1.0..1.3, Shader capability only, Logical/
//! GLSL450 memory model, one GLCompute entry, literal LocalSize, set 0 scalar
//! StorageBuffer descriptors. No images, descriptor arrays, push constants,
//! shared memory, specialization constants, or optional device features.
//! A trusted Khronos spirv-val executable validates the complete module too.
//!
//! Fence wait errors poison the backend and retain all possibly in-flight Vulkan
//! objects/loader lifetime. They are not destroyed while commands may use them.
//! An external process deadline is still required for a stuck driver call.

use crate::compute::{ComputeError, ComputePipelineDescriptor};
use ash::{vk, Entry};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::{CStr, CString},
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Copy)]
pub struct VulkanDeviceSelector {
    pub vendor_id: u32,
    pub device_id: u32,
}

#[derive(Debug, Clone)]
pub struct VulkanComputeConfig {
    pub selector: VulkanDeviceSelector,
    /// Trusted, public Khronos SPIRV-Tools executable; not fetched by this crate.
    pub spirv_validator: PathBuf,
    /// 1 ns..5 s. This bounds fence waiting, not every driver function.
    pub fence_timeout: Duration,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct VulkanDeviceInfo {
    pub vendor_id: u32,
    pub device_id: u32,
    pub name: String,
    pub device_type: String,
    pub api_version: u32,
    pub queue_family: u32,
    pub robust_buffer_access: bool,
    /// Actual physical-device report, not an assertion of guest Metal support.
    pub capabilities: VulkanDeviceCandidate,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct VulkanQueueFamilyInfo {
    pub index: u32,
    pub queue_count: u32,
    pub flags: u32,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct VulkanMemoryTypeInfo {
    pub index: u32,
    pub property_flags: u32,
    pub heap_index: u32,
    pub heap_size: u64,
    pub heap_flags: u32,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct VulkanComputeLimits {
    pub max_workgroup_count: [u32; 3],
    pub max_workgroup_size: [u32; 3],
    pub max_workgroup_invocations: u32,
    pub max_storage_buffer_range: u32,
    pub max_per_stage_storage_buffers: u32,
    pub max_descriptor_set_storage_buffers: u32,
    pub max_per_stage_resources: u32,
    pub max_bound_descriptor_sets: u32,
    pub max_memory_allocation_count: u32,
    pub non_coherent_atom_size: u64,
}

/// Driver-reported baseline capabilities. Empty unsupported_reasons does not
/// guarantee per-buffer memory compatibility, shader compilation or execution.
#[derive(Debug, Clone, Default, Serialize)]
pub struct VulkanDeviceCandidate {
    pub vendor_id: u32,
    pub device_id: u32,
    pub name: String,
    pub device_type: String,
    pub api_version: u32,
    pub driver_version: u32,
    /// None when the physical device does not provide the required Vulkan 1.1 ABI.
    pub device_uuid: Option<[u8; 16]>,
    pub driver_uuid: Option<[u8; 16]>,
    pub robust_buffer_access: bool,
    pub queue_families: Vec<VulkanQueueFamilyInfo>,
    pub memory_types: Vec<VulkanMemoryTypeInfo>,
    pub limits: VulkanComputeLimits,
    pub unsupported_reasons: Vec<String>,
}

/// Inspect all actual loader-visible devices without creating a logical device.
/// This never silently selects one or enables a software fallback.
pub fn enumerate_vulkan_devices() -> Result<Vec<VulkanDeviceCandidate>, ComputeError> {
    let session = Session::empty()?;
    Ok(session
        .probe_devices()?
        .into_iter()
        .map(|p| p.report)
        .collect())
}

pub struct VulkanComputeBackend {
    config: VulkanComputeConfig,
    info: VulkanDeviceInfo,
    session: Option<Session>,
}

fn invalid(message: impl Into<String>) -> ComputeError {
    ComputeError::Unsupported(message.into())
}
fn driver(error: vk::Result) -> ComputeError {
    ComputeError::Generic(format!("Vulkan {error:?}"))
}

impl VulkanComputeBackend {
    pub fn new(config: VulkanComputeConfig) -> Result<Self, ComputeError> {
        Self::new_selected(config, None)
    }

    /// Match both existing IDs and the explicitly selected physical device UUID.
    /// Duplicate reports of the same UUID are still rejected as ambiguous.
    pub fn new_for_device_uuid(
        config: VulkanComputeConfig,
        device_uuid: [u8; 16],
    ) -> Result<Self, ComputeError> {
        Self::new_selected(config, Some(device_uuid))
    }

    fn new_selected(
        mut config: VulkanComputeConfig,
        uuid: Option<[u8; 16]>,
    ) -> Result<Self, ComputeError> {
        if config.selector.vendor_id == 0
            || config.selector.device_id == 0
            || config.fence_timeout.is_zero()
            || config.fence_timeout > Duration::from_secs(5)
        {
            return Err(invalid(
                "explicit nonzero device IDs and fence timeout in (0, 5s] required",
            ));
        }
        config.spirv_validator = config
            .spirv_validator
            .canonicalize()
            .map_err(|e| invalid(format!("spirv-val path: {e}")))?;
        if !config.spirv_validator.is_file() {
            return Err(invalid("spirv-val must be a regular executable file"));
        }
        let session = Session::new_selected(config.selector, uuid, hardware_type)?;
        let info = session.info.clone();
        Ok(Self {
            config,
            info,
            session: Some(session),
        })
    }

    pub fn device_info(&self) -> &VulkanDeviceInfo {
        &self.info
    }

    /// Number of successful native pipeline compilations in this live session.
    pub fn pipeline_build_count(&self) -> Option<u64> {
        self.session
            .as_ref()
            .map(|session| session.pipeline_build_count)
    }

    pub(crate) fn dispatch(
        &mut self,
        descriptor: &ComputePipelineDescriptor,
        groups: [u32; 3],
        views: &[(u32, Vec<u8>)],
    ) -> Result<Vec<Vec<u8>>, ComputeError> {
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| invalid("Vulkan backend poisoned by unresolved submission"))?;
        let shader = inspect_spirv(&descriptor.shader.bytecode, &descriptor.shader.entry_point)?;
        if shader.local_size != descriptor.workgroup_size {
            return Err(invalid(
                "descriptor workgroup size differs from SPIR-V LocalSize",
            ));
        }
        let declared: BTreeSet<_> = descriptor
            .buffer_bindings
            .iter()
            .map(|binding| binding.binding)
            .collect();
        if shader.bindings != declared {
            return Err(invalid(
                "descriptor bindings differ from SPIR-V storage variables",
            ));
        }
        session.validate_limits(descriptor, groups)?;
        validate_module(&self.config.spirv_validator, &descriptor.shader.bytecode)?;
        let result = session.execute(
            &shader.words,
            &descriptor.shader.entry_point,
            groups,
            views,
            self.config.fence_timeout,
        );
        if session.in_flight {
            // Session::drop intentionally retains device objects and the loader.
            self.session.take();
        }
        result
    }
}

struct ShaderProfile {
    words: Vec<u32>,
    local_size: [u32; 3],
    bindings: BTreeSet<u32>,
}

/// Contract reflection, not a replacement for SPIRV-Tools semantic validation.
/// Opcodes/enumerants follow Khronos SPIRV-Headers unified1 grammar.
fn inspect_spirv(bytes: &[u8], requested_entry: &str) -> Result<ShaderProfile, ComputeError> {
    if !(20..=64 * 1024).contains(&bytes.len())
        || bytes.len() % 4 != 0
        || requested_entry.is_empty()
        || requested_entry.len() > 64
        || !requested_entry
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        return Err(invalid("SPIR-V byte/entry-point bounds"));
    }
    let words: Vec<_> = bytes
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    if words[0] != 0x0723_0203
        || !matches!(
            words[1],
            0x0001_0000 | 0x0001_0100 | 0x0001_0200 | 0x0001_0300
        )
        || words[3] == 0
        || words[3] > 1_000_000
        || words[4] != 0
    {
        return Err(invalid("SPIR-V header/version"));
    }
    let mut entry = None;
    let mut local = None;
    let mut model = false;
    let mut capability = false;
    let (mut bindings, mut sets, mut pointers, mut structures, mut block_types) = (
        BTreeMap::new(),
        BTreeMap::new(),
        BTreeMap::new(),
        BTreeSet::new(),
        BTreeSet::new(),
    );
    let mut variables = Vec::new();
    let (mut constants, mut composites) = (BTreeMap::new(), BTreeMap::new());
    let mut workgroup_builtin = None;
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        let opcode = words[at] & 0xffff;
        if count == 0 || count > words.len() - at {
            return Err(invalid("truncated SPIR-V instruction"));
        }
        let instruction = &words[at..at + count];
        match opcode {
            10 | 25..=27 | 48..=52 | 331 => {
                return Err(invalid("unsupported SPIR-V extension/image/specialization"))
            }
            14 => {
                if instruction != [0x0003_000e, 0, 1] || model {
                    return Err(invalid("require one Logical/GLSL450 memory model"));
                }
                model = true;
            }
            15 => {
                if count < 4 || instruction[1] != 5 || entry.is_some() {
                    return Err(invalid("require one GLCompute entry point"));
                }
                let encoded: Vec<u8> = instruction[3..]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                let end = encoded
                    .iter()
                    .position(|b| *b == 0)
                    .ok_or_else(|| invalid("unterminated entry point"))?;
                if &encoded[..end] != requested_entry.as_bytes() {
                    return Err(invalid("SPIR-V entry point mismatch"));
                }
                entry = Some(instruction[2]);
            }
            16 => {
                if count != 6 || instruction[2] != 17 || local.is_some() {
                    return Err(invalid("only literal LocalSize execution mode supported"));
                }
                local = Some((
                    instruction[1],
                    [instruction[3], instruction[4], instruction[5]],
                ));
            }
            17 => {
                if count != 2 || instruction[1] != 1 || capability {
                    return Err(invalid("only Shader capability supported"));
                }
                capability = true;
            }
            21 | 22 => {
                if count < 3 || instruction[2] != 32 {
                    return Err(invalid("only 32-bit shader numeric types supported"));
                }
            }
            30 if count >= 2 => {
                structures.insert(instruction[1]);
            }
            32 if count == 4 => {
                pointers.insert(instruction[1], (instruction[2], instruction[3]));
            }
            43 if count == 4 => {
                constants.insert(instruction[2], instruction[3]);
            }
            44 if count == 6 => {
                composites.insert(
                    instruction[2],
                    [instruction[3], instruction[4], instruction[5]],
                );
            }
            59 => {
                if count < 4 {
                    return Err(invalid("truncated OpVariable"));
                }
                match instruction[3] {
                    12 => variables.push((instruction[1], instruction[2])),
                    1 | 6 | 7 => {} // Input/Private/Function: no descriptors or shared storage.
                    _ => return Err(invalid("unsupported shader storage class")),
                }
            }
            71 => {
                if count < 3 {
                    return Err(invalid("truncated OpDecorate"));
                }
                match instruction[2] {
                    2 => {
                        block_types.insert(instruction[1]);
                    }
                    11 if count == 4 && instruction[3] == 25 => {
                        if workgroup_builtin.replace(instruction[1]).is_some() {
                            return Err(invalid("duplicate WorkgroupSize builtin"));
                        }
                    }
                    33 | 34 => {
                        if count != 4 {
                            return Err(invalid("invalid descriptor decoration"));
                        }
                        let map = if instruction[2] == 33 {
                            &mut bindings
                        } else {
                            &mut sets
                        };
                        if map.insert(instruction[1], instruction[3]).is_some() {
                            return Err(invalid("duplicate descriptor decoration"));
                        }
                    }
                    _ => {}
                }
            }
            // Decoration groups could hide resource bindings from this profile.
            73..=75 => return Err(invalid("SPIR-V decoration groups unsupported")),
            _ => {}
        }
        at += count;
    }
    let (entry_id, local_size) = local.ok_or_else(|| invalid("missing literal LocalSize"))?;
    if !model || !capability || entry != Some(entry_id) || local_size.contains(&0) {
        return Err(invalid("incomplete SPIR-V compute profile"));
    }
    // WorkgroupSize can override LocalSize in SPIR-V. Accept only a literal
    // constant vector equal to LocalSize, so device-limit checks remain exact.
    if let Some(builtin) = workgroup_builtin {
        let ids = composites
            .get(&builtin)
            .ok_or_else(|| invalid("WorkgroupSize must be a literal constant vector"))?;
        for i in 0..3 {
            if constants.get(&ids[i]) != Some(&local_size[i]) {
                return Err(invalid("WorkgroupSize builtin differs from LocalSize"));
            }
        }
    }
    let mut found = BTreeSet::new();
    for (pointer_type, id) in variables {
        let (storage, structure) = pointers
            .get(&pointer_type)
            .ok_or_else(|| invalid("missing storage pointer type"))?;
        if *storage != 12 || !structures.contains(structure) || !block_types.contains(structure) {
            return Err(invalid(
                "storage descriptor must point to a Block struct, not an array",
            ));
        }
        let binding = *bindings
            .get(&id)
            .ok_or_else(|| invalid("missing storage Binding"))?;
        if sets.get(&id) != Some(&0) || !found.insert(binding) {
            return Err(invalid(
                "storage descriptors require unique bindings in set zero",
            ));
        }
    }
    if found.is_empty() || found.len() != bindings.len() || found.len() != sets.len() {
        return Err(invalid("unreflected/empty descriptor bindings"));
    }
    Ok(ShaderProfile {
        words,
        local_size,
        bindings: found,
    })
}

struct TemporaryModule(PathBuf);
impl Drop for TemporaryModule {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn validate_module(validator: &Path, bytes: &[u8]) -> Result<(), ComputeError> {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let path = std::env::temp_dir().join(format!(
        "nextcore-spv-{}-{}.spv",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&path)
        .map_err(|e| invalid(format!("validator input: {e}")))?;
    let temporary = TemporaryModule(path);
    file.write_all(bytes)
        .map_err(|e| invalid(format!("validator input write: {e}")))?;
    drop(file);
    // No pipe can fill and stall the parent; shader and diagnostic sizes are not
    // used as an unbounded host allocation. The executable is a trusted tool.
    let mut child = Command::new(validator)
        .args(["--target-env", "vulkan1.1"])
        .arg(&temporary.0)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| invalid(format!("spirv-val spawn: {e}")))?;
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return if status.success() {
                    Ok(())
                } else {
                    Err(invalid(format!("spirv-val rejected module ({status})")))
                }
            }
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(5)),
            result => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(invalid(format!(
                    "spirv-val deadline/wait failure: {result:?}"
                )));
            }
        }
    }
}

#[derive(Default)]
struct Resources {
    buffers: Vec<(vk::Buffer, vk::DeviceMemory, bool)>,
    descriptor_pool: vk::DescriptorPool,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
}

#[derive(Debug, PartialEq, Eq)]
struct PipelineKey {
    words: Vec<u32>,
    entry: String,
    bindings: Vec<u32>,
}

impl PipelineKey {
    fn matches(&self, words: &[u32], entry: &str, views: &[(u32, Vec<u8>)]) -> bool {
        self.words == words
            && self.entry == entry
            && self
                .bindings
                .iter()
                .copied()
                .eq(views.iter().map(|(binding, _)| *binding))
    }
}

#[derive(Default)]
struct CachedPipeline {
    key: Option<PipelineKey>,
    shader: vk::ShaderModule,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

struct Session {
    entry: Option<Entry>,
    instance: ash::Instance,
    device: Option<ash::Device>,
    memory: vk::PhysicalDeviceMemoryProperties,
    limits: vk::PhysicalDeviceLimits,
    queue: vk::Queue,
    info: VulkanDeviceInfo,
    resources: Resources,
    pipeline: CachedPipeline,
    pipeline_build_count: u64,
    in_flight: bool,
}

struct DeviceProbe {
    physical: vk::PhysicalDevice,
    properties: vk::PhysicalDeviceProperties,
    memory: vk::PhysicalDeviceMemoryProperties,
    report: VulkanDeviceCandidate,
}

fn hardware_type(kind: &str) -> bool {
    matches!(kind, "INTEGRATED_GPU" | "DISCRETE_GPU")
}

fn compute_family(report: &VulkanDeviceCandidate) -> Option<u32> {
    report
        .queue_families
        .iter()
        .find(|f| f.queue_count > 0 && f.flags & vk::QueueFlags::COMPUTE.as_raw() != 0)
        .map(|f| f.index)
}

// No optional memory features are enabled on this logical device. In
// particular DEVICE_COHERENT_AMD requires deviceCoherentMemory (VUID 02790).
fn host_memory_flags(flags: vk::MemoryPropertyFlags) -> bool {
    flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        && !flags.intersects(
            vk::MemoryPropertyFlags::DEVICE_COHERENT_AMD
                | vk::MemoryPropertyFlags::PROTECTED
                | vk::MemoryPropertyFlags::LAZILY_ALLOCATED,
        )
}

fn baseline_rejections(
    report: &VulkanDeviceCandidate,
    allowed_type: impl Fn(&str) -> bool,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if !allowed_type(&report.device_type) {
        reasons.push(format!(
            "device type {} is not an integrated/discrete GPU; CPU/virtual fallback denied",
            report.device_type
        ));
    }
    if vk::api_version_variant(report.api_version) != 0 || report.api_version < vk::API_VERSION_1_1
    {
        reasons.push("Vulkan 1.1 physical device required".into());
    }
    if !report.robust_buffer_access {
        reasons.push("robustBufferAccess is required".into());
    }
    if compute_family(report).is_none() {
        reasons.push("no nonempty compute queue".into());
    }
    if !report.memory_types.iter().any(|m| {
        host_memory_flags(vk::MemoryPropertyFlags::from_raw(m.property_flags)) && m.heap_size >= 4
    }) {
        reasons.push(
            "no feature-compatible HOST_VISIBLE memory heap large enough for a storage word".into(),
        );
    }
    let limits = &report.limits;
    if limits.max_workgroup_count.contains(&0)
        || limits.max_workgroup_size.contains(&0)
        || limits.max_workgroup_invocations == 0
    {
        reasons.push("compute workgroup limits cannot execute one invocation".into());
    }
    if limits.max_storage_buffer_range < 4
        || limits.max_per_stage_storage_buffers == 0
        || limits.max_descriptor_set_storage_buffers == 0
        || limits.max_per_stage_resources == 0
        || limits.max_bound_descriptor_sets == 0
        || limits.max_memory_allocation_count == 0
    {
        reasons.push("storage/descriptor/allocation limits cannot bind one storage word".into());
    }
    if !limits.non_coherent_atom_size.is_power_of_two() {
        reasons.push("invalid nonCoherentAtomSize reported by driver".into());
    }
    reasons
}

fn select_candidate(
    reports: &[VulkanDeviceCandidate],
    selector: VulkanDeviceSelector,
    uuid: Option<[u8; 16]>,
) -> Result<usize, ComputeError> {
    let matching: Vec<_> = reports
        .iter()
        .enumerate()
        .filter(|(_, r)| {
            r.vendor_id == selector.vendor_id
                && r.device_id == selector.device_id
                && uuid.is_none_or(|id| r.device_uuid == Some(id))
        })
        .collect();
    match matching.as_slice() {
        [] => Err(invalid("exact Vulkan device IDs/UUID not found; fallback denied; enumerate_vulkan_devices reports available devices")),
        [(index, report)] => {
            if report.unsupported_reasons.is_empty() { Ok(*index) }
            else { Err(invalid(format!("selected Vulkan device {} unsupported: {}", report.name, report.unsupported_reasons.join("; ")))) }
        }
        _ => Err(invalid("device selector is ambiguous; select a unique deviceUUID from enumerate_vulkan_devices (duplicate UUID reports are also rejected)")),
    }
}

/// memoryTypeBits describes this buffer, not every memory type in the device.
fn host_memory_type(
    memory: &vk::PhysicalDeviceMemoryProperties,
    bits: u32,
    size: u64,
) -> Result<(u32, bool), ComputeError> {
    if size == 0 || memory.memory_type_count > 32 || memory.memory_heap_count > 16 {
        return Err(invalid(
            "invalid Vulkan allocation size or memory property counts",
        ));
    }
    (0..memory.memory_type_count).filter_map(|index| {
        let kind = memory.memory_types[index as usize];
        if bits & (1u32 << index) == 0 || !host_memory_flags(kind.property_flags) || kind.heap_index >= memory.memory_heap_count || size > memory.memory_heaps[kind.heap_index as usize].size {
            return None;
        }
        let coherent = kind.property_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT);
        let cached = kind.property_flags.contains(vk::MemoryPropertyFlags::HOST_CACHED);
        Some((index, coherent, u8::from(coherent) * 2 + u8::from(cached)))
    }).max_by_key(|(index, _, rank)| (*rank, std::cmp::Reverse(*index)))
      .map(|(index, coherent, _)| (index, coherent))
      .ok_or_else(|| invalid(format!("no compatible HOST_VISIBLE memory: memoryTypeBits={bits:#010x}, allocation_size={size}; device-local-only memory is not host mappable; protected/device-coherent memory features are not enabled")))
}

impl Session {
    fn empty() -> Result<Self, ComputeError> {
        let entry = unsafe { Entry::load() }.map_err(|e| invalid(format!("Vulkan loader: {e}")))?;
        let version = unsafe { entry.try_enumerate_instance_version() }
            .map_err(driver)?
            .unwrap_or(vk::API_VERSION_1_0);
        if version < vk::API_VERSION_1_1 {
            return Err(invalid("Vulkan loader 1.1 required"));
        }
        let application = vk::ApplicationInfo::default()
            .application_name(c"Nextcore host compute")
            .api_version(vk::API_VERSION_1_1);
        let create = vk::InstanceCreateInfo::default().application_info(&application);
        let instance = unsafe { entry.create_instance(&create, None) }.map_err(driver)?;
        Ok(Self {
            entry: Some(entry),
            instance,
            device: None,
            memory: Default::default(),
            limits: Default::default(),
            queue: vk::Queue::null(),
            info: Default::default(),
            resources: Resources::default(),
            pipeline: CachedPipeline::default(),
            pipeline_build_count: 0,
            in_flight: false,
        })
    }

    fn probe_devices(&self) -> Result<Vec<DeviceProbe>, ComputeError> {
        let devices = unsafe { self.instance.enumerate_physical_devices() }.map_err(driver)?;
        let mut probes = Vec::new();
        for physical in devices {
            let properties = unsafe { self.instance.get_physical_device_properties(physical) };
            let mut ids = vk::PhysicalDeviceIDProperties::default();
            let has_ids = properties.api_version >= vk::API_VERSION_1_1;
            if has_ids {
                let mut extended = vk::PhysicalDeviceProperties2::default().push_next(&mut ids);
                unsafe {
                    self.instance
                        .get_physical_device_properties2(physical, &mut extended)
                };
            }
            let features = unsafe { self.instance.get_physical_device_features(physical) };
            let families = unsafe {
                self.instance
                    .get_physical_device_queue_family_properties(physical)
            };
            let memory = unsafe {
                self.instance
                    .get_physical_device_memory_properties(physical)
            };
            if memory.memory_type_count > 32 || memory.memory_heap_count > 16 {
                return Err(invalid(
                    "driver memory property counts exceed Vulkan arrays",
                ));
            }
            let mut types = Vec::new();
            for index in 0..memory.memory_type_count {
                let kind = memory.memory_types[index as usize];
                if kind.heap_index >= memory.memory_heap_count {
                    return Err(invalid("driver memory type references missing heap"));
                }
                let heap = memory.memory_heaps[kind.heap_index as usize];
                types.push(VulkanMemoryTypeInfo {
                    index,
                    property_flags: kind.property_flags.as_raw(),
                    heap_index: kind.heap_index,
                    heap_size: heap.size,
                    heap_flags: heap.flags.as_raw(),
                });
            }
            let limits = properties.limits;
            let mut report = VulkanDeviceCandidate {
                vendor_id: properties.vendor_id,
                device_id: properties.device_id,
                name: unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned(),
                device_type: format!("{:?}", properties.device_type),
                api_version: properties.api_version,
                driver_version: properties.driver_version,
                device_uuid: has_ids.then_some(ids.device_uuid),
                driver_uuid: has_ids.then_some(ids.driver_uuid),
                robust_buffer_access: features.robust_buffer_access != 0,
                queue_families: families
                    .iter()
                    .enumerate()
                    .map(|(i, f)| VulkanQueueFamilyInfo {
                        index: i as u32,
                        queue_count: f.queue_count,
                        flags: f.queue_flags.as_raw(),
                    })
                    .collect(),
                memory_types: types,
                limits: VulkanComputeLimits {
                    max_workgroup_count: limits.max_compute_work_group_count,
                    max_workgroup_size: limits.max_compute_work_group_size,
                    max_workgroup_invocations: limits.max_compute_work_group_invocations,
                    max_storage_buffer_range: limits.max_storage_buffer_range,
                    max_per_stage_storage_buffers: limits.max_per_stage_descriptor_storage_buffers,
                    max_descriptor_set_storage_buffers: limits.max_descriptor_set_storage_buffers,
                    max_per_stage_resources: limits.max_per_stage_resources,
                    max_bound_descriptor_sets: limits.max_bound_descriptor_sets,
                    max_memory_allocation_count: limits.max_memory_allocation_count,
                    non_coherent_atom_size: limits.non_coherent_atom_size,
                },
                unsupported_reasons: Vec::new(),
            };
            report.unsupported_reasons = baseline_rejections(&report, hardware_type);
            probes.push(DeviceProbe {
                physical,
                properties,
                memory,
                report,
            });
        }
        Ok(probes)
    }

    #[cfg(all(test, target_os = "linux"))]
    fn new(selector: VulkanDeviceSelector) -> Result<Self, ComputeError> {
        Self::new_selected(selector, None, hardware_type)
    }

    // Private injection permits an explicit software ICD lifecycle test only.
    #[cfg(all(test, target_os = "linux"))]
    fn new_matching_type(
        selector: VulkanDeviceSelector,
        allowed_type: impl Fn(vk::PhysicalDeviceType) -> bool,
    ) -> Result<Self, ComputeError> {
        Self::new_selected(selector, None, |kind| {
            allowed_type(match kind {
                "CPU" => vk::PhysicalDeviceType::CPU,
                "INTEGRATED_GPU" => vk::PhysicalDeviceType::INTEGRATED_GPU,
                "DISCRETE_GPU" => vk::PhysicalDeviceType::DISCRETE_GPU,
                _ => vk::PhysicalDeviceType::OTHER,
            })
        })
    }

    fn new_selected(
        selector: VulkanDeviceSelector,
        uuid: Option<[u8; 16]>,
        allowed_type: impl Fn(&str) -> bool,
    ) -> Result<Self, ComputeError> {
        let mut session = Self::empty()?;
        let mut probes = session.probe_devices()?;
        for probe in &mut probes {
            probe.report.unsupported_reasons = baseline_rejections(&probe.report, &allowed_type);
        }
        let reports: Vec<_> = probes.iter().map(|p| p.report.clone()).collect();
        let index = select_candidate(&reports, selector, uuid)?;
        let selected = probes.swap_remove(index);
        let family = compute_family(&selected.report).ok_or_else(|| invalid("no compute queue"))?;
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let enabled = vk::PhysicalDeviceFeatures::default().robust_buffer_access(true);
        let create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_features(&enabled);
        let device = unsafe {
            session
                .instance
                .create_device(selected.physical, &create, None)
        }
        .map_err(driver)?;
        session.queue = unsafe { device.get_device_queue(family, 0) };
        session.device = Some(device);
        session.memory = selected.memory;
        session.limits = selected.properties.limits;
        session.info = VulkanDeviceInfo {
            vendor_id: selected.report.vendor_id,
            device_id: selected.report.device_id,
            name: selected.report.name.clone(),
            device_type: selected.report.device_type.clone(),
            api_version: selected.report.api_version,
            queue_family: family,
            robust_buffer_access: true,
            capabilities: selected.report,
        };
        Ok(session)
    }

    fn validate_limits(
        &self,
        descriptor: &ComputePipelineDescriptor,
        groups: [u32; 3],
    ) -> Result<(), ComputeError> {
        for i in 0..3 {
            if groups[i] > self.limits.max_compute_work_group_count[i]
                || descriptor.workgroup_size[i] > self.limits.max_compute_work_group_size[i]
            {
                return Err(invalid("compute dimensions exceed device limits"));
            }
        }
        let invocations: u64 = descriptor
            .workgroup_size
            .iter()
            .map(|v| *v as u64)
            .product();
        if invocations > self.limits.max_compute_work_group_invocations as u64
            || descriptor.buffer_bindings.len()
                > self.limits.max_per_stage_descriptor_storage_buffers as usize
            || descriptor.buffer_bindings.len()
                > self.limits.max_descriptor_set_storage_buffers as usize
            || descriptor.buffer_bindings.len() > self.limits.max_per_stage_resources as usize
            || descriptor.buffer_bindings.len() > self.limits.max_memory_allocation_count as usize
            || self.limits.max_bound_descriptor_sets == 0
            || descriptor
                .buffer_bindings
                .iter()
                .any(|b| b.size > self.limits.max_storage_buffer_range as u64)
        {
            return Err(invalid("compute invocation/storage limits exceeded"));
        }
        Ok(())
    }

    fn execute(
        &mut self,
        words: &[u32],
        entry_name: &str,
        groups: [u32; 3],
        views: &[(u32, Vec<u8>)],
        timeout: Duration,
    ) -> Result<Vec<Vec<u8>>, ComputeError> {
        self.cleanup();
        let result = self.execute_inner(words, entry_name, groups, views, timeout);
        if !self.in_flight {
            self.cleanup();
        }
        result
    }

    fn prepare_pipeline(
        &mut self,
        words: &[u32],
        entry_name: &str,
        views: &[(u32, Vec<u8>)],
    ) -> Result<(), ComputeError> {
        if self
            .pipeline
            .key
            .as_ref()
            .is_some_and(|key| key.matches(words, entry_name, views))
        {
            return Ok(());
        }
        // execute() cleaned all completed command buffers/descriptors first.
        // A poisoned session never enters this function again.
        self.cleanup_pipeline();
        let device = self.device.as_ref().unwrap();
        unsafe {
            let bindings: Vec<_> = views
                .iter()
                .map(|(binding, _)| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(*binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            self.pipeline.descriptor_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(driver)?;
            let layouts = [self.pipeline.descriptor_layout];
            self.pipeline.pipeline_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                    None,
                )
                .map_err(driver)?;
            self.pipeline.shader = device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)
                .map_err(driver)?;
            let entry = CString::new(entry_name).map_err(|_| invalid("entry contains NUL"))?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(self.pipeline.shader)
                .name(&entry);
            let pipeline = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(self.pipeline.pipeline_layout);
            match device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline], None) {
                Ok(pipelines) => self.pipeline.pipeline = pipelines[0],
                Err((pipelines, error)) => {
                    for pipeline in pipelines {
                        if pipeline != vk::Pipeline::null() {
                            device.destroy_pipeline(pipeline, None);
                        }
                    }
                    return Err(driver(error));
                }
            }
        }
        self.pipeline.key = Some(PipelineKey {
            words: words.to_vec(),
            entry: entry_name.into(),
            bindings: views.iter().map(|(binding, _)| *binding).collect(),
        });
        self.pipeline_build_count += 1;
        Ok(())
    }

    fn execute_inner(
        &mut self,
        words: &[u32],
        entry_name: &str,
        groups: [u32; 3],
        views: &[(u32, Vec<u8>)],
        timeout: Duration,
    ) -> Result<Vec<Vec<u8>>, ComputeError> {
        self.prepare_pipeline(words, entry_name, views)?;
        let device = self.device.as_ref().unwrap();
        // SAFETY throughout: objects share this device, commands are externally
        // synchronized by &mut self, bounds/profile were checked before entry.
        unsafe {
            for (_, bytes) in views {
                let create = vk::BufferCreateInfo::default()
                    .size(bytes.len() as u64)
                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE);
                let buffer = device.create_buffer(&create, None).map_err(driver)?;
                self.resources
                    .buffers
                    .push((buffer, vk::DeviceMemory::null(), false));
                let requirements = device.get_buffer_memory_requirements(buffer);
                let (memory_type, coherent) = host_memory_type(
                    &self.memory,
                    requirements.memory_type_bits,
                    requirements.size,
                )?;
                self.resources.buffers.last_mut().unwrap().2 = coherent;
                let allocate = vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type);
                let memory = device.allocate_memory(&allocate, None).map_err(driver)?;
                self.resources.buffers.last_mut().unwrap().1 = memory;
                device
                    .bind_buffer_memory(buffer, memory, 0)
                    .map_err(driver)?;
                let mapped = device
                    .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                    .map_err(driver)?;
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped.cast::<u8>(), bytes.len());
                let flushed = if coherent {
                    Ok(())
                } else {
                    device.flush_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                        .memory(memory)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)])
                };
                device.unmap_memory(memory);
                flushed.map_err(driver)?;
            }
            let layouts = [self.pipeline.descriptor_layout];
            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: views.len() as u32,
            }];
            self.resources.descriptor_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&pool_sizes),
                    None,
                )
                .map_err(driver)?;
            let descriptor = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.resources.descriptor_pool)
                        .set_layouts(&layouts),
                )
                .map_err(driver)?[0];
            let infos: Vec<_> = self
                .resources
                .buffers
                .iter()
                .zip(views)
                .map(|((buffer, _, _), (_, bytes))| {
                    [vk::DescriptorBufferInfo {
                        buffer: *buffer,
                        offset: 0,
                        range: bytes.len() as u64,
                    }]
                })
                .collect();
            let writes: Vec<_> = views
                .iter()
                .zip(&infos)
                .map(|((binding, _), info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor)
                        .dst_binding(*binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                })
                .collect();
            device.update_descriptor_sets(&writes, &[]);
            self.resources.command_pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(self.info.queue_family),
                    None,
                )
                .map_err(driver)?;
            let command = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.resources.command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(driver)?[0];
            device
                .begin_command_buffer(
                    command,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(driver)?;
            device.cmd_bind_pipeline(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.pipeline_layout,
                0,
                &[descriptor],
                &[],
            );
            device.cmd_dispatch(command, groups[0], groups[1], groups[2]);
            let barriers: Vec<_> = self
                .resources
                .buffers
                .iter()
                .map(|(buffer, _, _)| {
                    vk::BufferMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::HOST_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .buffer(*buffer)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)
                })
                .collect();
            // Queue submission makes coherent/flushed host upload available to the GPU;
            // this barrier makes compute writes available to the host domain.
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &barriers,
                &[],
            );
            device.end_command_buffer(command).map_err(driver)?;
            self.resources.fence = device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(driver)?;
            let commands = [command];
            let submit = [vk::SubmitInfo::default().command_buffers(&commands)];
            self.in_flight = true; // includes uncertain queue-submit error paths.
            device
                .queue_submit(self.queue, &submit, self.resources.fence)
                .map_err(driver)?;
            device.wait_for_fences(&[self.resources.fence], true, timeout.as_nanos() as u64).map_err(|error|
                ComputeError::Generic(format!("GPU completion unresolved ({error:?}); backend poisoned; in-flight resources retained")))?;
            self.in_flight = false;
            let mut results = Vec::with_capacity(views.len());
            for ((_, memory, coherent), (_, source)) in self.resources.buffers.iter().zip(views) {
                let mut bytes = vec![0u8; source.len()];
                let mapped = device
                    .map_memory(*memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                    .map_err(driver)?;
                if !coherent {
                    if let Err(error) =
                        device.invalidate_mapped_memory_ranges(&[vk::MappedMemoryRange::default()
                            .memory(*memory)
                            .offset(0)
                            .size(vk::WHOLE_SIZE)])
                    {
                        device.unmap_memory(*memory);
                        return Err(driver(error));
                    }
                }
                std::ptr::copy_nonoverlapping(mapped.cast::<u8>(), bytes.as_mut_ptr(), bytes.len());
                device.unmap_memory(*memory);
                results.push(bytes);
            }
            Ok(results)
        }
    }

    fn cleanup_pipeline(&mut self) {
        if self.in_flight {
            return;
        }
        let Some(device) = &self.device else {
            return;
        };
        let pipeline = std::mem::take(&mut self.pipeline);
        unsafe {
            if pipeline.pipeline != vk::Pipeline::null() {
                device.destroy_pipeline(pipeline.pipeline, None);
            }
            if pipeline.pipeline_layout != vk::PipelineLayout::null() {
                device.destroy_pipeline_layout(pipeline.pipeline_layout, None);
            }
            if pipeline.descriptor_layout != vk::DescriptorSetLayout::null() {
                device.destroy_descriptor_set_layout(pipeline.descriptor_layout, None);
            }
            if pipeline.shader != vk::ShaderModule::null() {
                device.destroy_shader_module(pipeline.shader, None);
            }
        }
    }

    fn cleanup(&mut self) {
        if self.in_flight {
            return;
        }
        let Some(device) = &self.device else {
            return;
        };
        let resources = std::mem::take(&mut self.resources);
        unsafe {
            if resources.fence != vk::Fence::null() {
                device.destroy_fence(resources.fence, None);
            }
            if resources.command_pool != vk::CommandPool::null() {
                device.destroy_command_pool(resources.command_pool, None);
            }
            if resources.descriptor_pool != vk::DescriptorPool::null() {
                device.destroy_descriptor_pool(resources.descriptor_pool, None);
            }
            for (buffer, memory, _) in resources.buffers {
                device.destroy_buffer(buffer, None);
                if memory != vk::DeviceMemory::null() {
                    device.free_memory(memory, None);
                }
            }
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        if self.in_flight {
            // Skip all Vulkan destruction and retain the dynamic loader. Releasing
            // the outer Rust vectors only releases handle values, not GPU objects.
            if let Some(entry) = self.entry.take() {
                std::mem::forget(entry);
            }
            log::error!("unresolved Vulkan submission: GPU objects retained until process exit");
            return;
        }
        self.cleanup();
        self.cleanup_pipeline();
        unsafe {
            if let Some(device) = &self.device {
                device.destroy_device(None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const SHADER: &[u8] = include_bytes!("../tests/fixtures/storage_transform.spv");

    // Independently authored capability fixtures; these do not emulate hardware
    // execution or establish support on an untested vendor's physical device.
    fn candidate(vendor: u32, id: u8) -> VulkanDeviceCandidate {
        let mut report = VulkanDeviceCandidate {
            vendor_id: vendor,
            device_id: 7,
            name: "capability fixture".into(),
            device_type: "INTEGRATED_GPU".into(),
            api_version: vk::API_VERSION_1_1,
            device_uuid: Some([id; 16]),
            robust_buffer_access: true,
            queue_families: vec![
                VulkanQueueFamilyInfo {
                    index: 0,
                    queue_count: 1,
                    flags: vk::QueueFlags::GRAPHICS.as_raw(),
                },
                VulkanQueueFamilyInfo {
                    index: 2,
                    queue_count: 0,
                    flags: vk::QueueFlags::COMPUTE.as_raw(),
                },
                VulkanQueueFamilyInfo {
                    index: 3,
                    queue_count: 1,
                    flags: vk::QueueFlags::COMPUTE.as_raw(),
                },
            ],
            memory_types: vec![VulkanMemoryTypeInfo {
                property_flags: vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw(),
                heap_size: 4096,
                ..Default::default()
            }],
            limits: VulkanComputeLimits {
                max_workgroup_count: [8; 3],
                max_workgroup_size: [32; 3],
                max_workgroup_invocations: 64,
                max_storage_buffer_range: 4096,
                max_per_stage_storage_buffers: 4,
                max_descriptor_set_storage_buffers: 4,
                max_per_stage_resources: 4,
                max_bound_descriptor_sets: 1,
                max_memory_allocation_count: 8,
                non_coherent_atom_size: 256,
            },
            ..Default::default()
        };
        report.unsupported_reasons = baseline_rejections(&report, hardware_type);
        report
    }

    #[test]
    fn uuid_selects_same_model_without_first_device_or_vendor_policy() {
        for vendor_id in [0x1002, 0x10de, 0x8086, 0xabcd] {
            let reports = [candidate(vendor_id, 1), candidate(vendor_id, 2)];
            let selector = VulkanDeviceSelector {
                vendor_id,
                device_id: 7,
            };
            assert!(select_candidate(&reports, selector, None)
                .unwrap_err()
                .to_string()
                .contains("ambiguous"));
            assert_eq!(
                select_candidate(&reports, selector, Some([2; 16])).unwrap(),
                1
            );
            assert_eq!(
                select_candidate(
                    &[reports[1].clone(), reports[0].clone()],
                    selector,
                    Some([2; 16])
                )
                .unwrap(),
                0
            );
            assert!(select_candidate(&reports, selector, Some([3; 16]))
                .unwrap_err()
                .to_string()
                .contains("not found"));
            assert_eq!(compute_family(&reports[0]), Some(3));
        }
    }

    #[test]
    fn duplicate_uuid_and_wrong_ids_never_choose_a_candidate() {
        let reports = [candidate(0x8086, 1), candidate(0x8086, 1)];
        let selector = VulkanDeviceSelector {
            vendor_id: 0x8086,
            device_id: 7,
        };
        assert!(select_candidate(&reports, selector, Some([1; 16]))
            .unwrap_err()
            .to_string()
            .contains("ambiguous"));
        assert!(select_candidate(
            &reports,
            VulkanDeviceSelector {
                vendor_id: 0x1002,
                ..selector
            },
            Some([1; 16])
        )
        .is_err());
        assert!(select_candidate(&[], selector, None).is_err());
    }

    #[test]
    fn unsupported_diagnostics_report_all_failed_baseline_requirements() {
        let mut report = candidate(0x10de, 1);
        report.device_type = "CPU".into();
        report.api_version = vk::API_VERSION_1_0;
        report.robust_buffer_access = false;
        report.queue_families[2].queue_count = 0;
        report.memory_types[0].property_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL.as_raw();
        report.limits.max_workgroup_size[1] = 0;
        report.limits.max_storage_buffer_range = 3;
        report.limits.non_coherent_atom_size = 0;
        report.unsupported_reasons = baseline_rejections(&report, hardware_type);
        assert_eq!(report.unsupported_reasons.len(), 8);
        let error = select_candidate(
            &[report],
            VulkanDeviceSelector {
                vendor_id: 0x10de,
                device_id: 7,
            },
            None,
        )
        .unwrap_err()
        .to_string();
        for reason in [
            "CPU/virtual fallback denied",
            "Vulkan 1.1",
            "robustBufferAccess",
            "compute queue",
            "HOST_VISIBLE",
            "workgroup",
            "storage/descriptor",
            "nonCoherentAtomSize",
        ] {
            assert!(error.contains(reason), "{error}");
        }
    }

    #[test]
    fn virtual_device_remains_unsupported_and_low_capacity_is_explicit() {
        let mut report = candidate(0x8086, 1);
        report.device_type = "VIRTUAL_GPU".into();
        assert_eq!(baseline_rejections(&report, hardware_type).len(), 1);
        report.device_type = "DISCRETE_GPU".into();
        report.limits.max_bound_descriptor_sets = 0;
        assert_eq!(baseline_rejections(&report, hardware_type).len(), 1);
        report.limits.max_bound_descriptor_sets = 1;
        report.memory_types[0].heap_size = 3;
        assert_eq!(baseline_rejections(&report, hardware_type).len(), 1);
    }

    fn memory_fixture() -> vk::PhysicalDeviceMemoryProperties {
        let mut memory = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 4,
            memory_heap_count: 2,
            ..Default::default()
        };
        memory.memory_heaps[0].size = 4096;
        memory.memory_heaps[1].size = 65536;
        memory.memory_types[0] = vk::MemoryType {
            property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            heap_index: 1,
        };
        memory.memory_types[1] = vk::MemoryType {
            property_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_CACHED,
            heap_index: 0,
        };
        memory.memory_types[2] = vk::MemoryType {
            property_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::DEVICE_LOCAL,
            heap_index: 1,
        };
        memory.memory_types[3] = vk::MemoryType {
            property_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT
                | vk::MemoryPropertyFlags::HOST_CACHED,
            heap_index: 0,
        };
        memory
    }

    #[test]
    fn host_memory_selection_respects_buffer_bits_coherency_and_shared_heaps() {
        let memory = memory_fixture();
        assert_eq!(host_memory_type(&memory, 0b1111, 4096).unwrap(), (3, true));
        assert_eq!(host_memory_type(&memory, 0b0111, 4096).unwrap(), (2, true));
        assert_eq!(host_memory_type(&memory, 0b0011, 4096).unwrap(), (1, false));
        assert!(host_memory_type(&memory, 0b0001, 4).is_err());
        assert_eq!(host_memory_type(&memory, 0b1111, 4097).unwrap(), (2, true));
        assert!(host_memory_type(&memory, 0b0011, 4097).is_err());
        assert!(host_memory_type(&memory, 0, 4).is_err());
        assert!(host_memory_type(&memory, u32::MAX, u64::MAX).is_err());
        assert!(host_memory_type(&memory, u32::MAX, 0).is_err());
    }

    #[test]
    fn optional_memory_features_are_not_silently_used() {
        let mut memory = memory_fixture();
        memory.memory_types[3].property_flags |= vk::MemoryPropertyFlags::DEVICE_COHERENT_AMD;
        assert_eq!(host_memory_type(&memory, 0b1111, 4).unwrap(), (2, true));
        assert!(host_memory_type(&memory, 0b1000, 4).is_err());
        memory.memory_types[2].property_flags |= vk::MemoryPropertyFlags::PROTECTED;
        assert_eq!(host_memory_type(&memory, 0b1111, 4).unwrap(), (1, false));
        let mut report = candidate(0x1002, 1);
        report.memory_types[0].property_flags |=
            vk::MemoryPropertyFlags::DEVICE_COHERENT_AMD.as_raw();
        assert_eq!(baseline_rejections(&report, hardware_type).len(), 1);
    }

    #[test]
    fn memory_type_index_31_and_malformed_counts_are_checked() {
        let mut memory = memory_fixture();
        memory.memory_type_count = 32;
        memory.memory_types[31] = memory.memory_types[1];
        assert_eq!(host_memory_type(&memory, 1 << 31, 4).unwrap(), (31, false));
        memory.memory_types[31].heap_index = 2;
        assert!(host_memory_type(&memory, 1 << 31, 4).is_err());
        memory.memory_type_count = 33;
        assert!(host_memory_type(&memory, u32::MAX, 4).is_err());
        memory.memory_type_count = 4;
        memory.memory_heap_count = 17;
        assert!(host_memory_type(&memory, u32::MAX, 4).is_err());
    }

    /// Explicit CPU Vulkan execution tests API/resource lifetime only. This
    /// entry is Linux-only and ignored by ordinary builds/hardware acceptance.
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "requires Mesa llvmpipe Vulkan ICD and /usr/bin/spirv-val"]
    fn software_vulkan_pipeline_reuse_and_replacement() {
        let selector = VulkanDeviceSelector {
            vendor_id: 0x10005,
            device_id: 0,
        };
        let session =
            Session::new_matching_type(selector, |kind| kind == vk::PhysicalDeviceType::CPU)
                .unwrap();
        assert_eq!(session.info.device_type, "CPU");
        let mut backend = VulkanComputeBackend {
            config: VulkanComputeConfig {
                selector,
                spirv_validator: PathBuf::from("/usr/bin/spirv-val"),
                fence_timeout: Duration::from_secs(3),
            },
            info: session.info.clone(),
            session: Some(session),
        };
        let mut changed = SHADER.to_vec();
        let mut at = 20;
        let mut modifications = 0;
        while at < changed.len() {
            let instruction = u32::from_le_bytes(changed[at..at + 4].try_into().unwrap());
            if instruction & 0xffff == 43
                && instruction >> 16 == 4
                && u32::from_le_bytes(changed[at + 12..at + 16].try_into().unwrap()) == 7
            {
                changed[at + 12..at + 16].copy_from_slice(&9u32.to_le_bytes());
                modifications += 1;
            }
            at += (instruction >> 16) as usize * 4;
        }
        assert_eq!(modifications, 1);
        let mut inputs: Vec<u32> = (0..256).map(|i| i * 17 + 5).collect();
        for (shader, addend, compilations) in [
            (SHADER, 7u32, 1u64),
            (SHADER, 7, 1),
            (changed.as_slice(), 9, 2),
            (changed.as_slice(), 9, 2),
            (SHADER, 7, 3),
        ] {
            let descriptor = ComputePipelineDescriptor {
                shader: crate::compute::ComputeShader {
                    id: 1,
                    entry_point: "main".into(),
                    bytecode: shader.to_vec(),
                },
                workgroup_size: [64, 1, 1],
                buffer_bindings: vec![crate::compute::BufferBinding {
                    binding: 0,
                    buffer_id: 1,
                    offset: 0,
                    size: 1024,
                }],
            };
            let uploaded: Vec<u8> = inputs.iter().flat_map(|v| v.to_le_bytes()).collect();
            let outputs = backend
                .dispatch(&descriptor, [4, 1, 1], &[(0, uploaded)])
                .unwrap();
            let actual: Vec<u32> = outputs[0]
                .chunks_exact(4)
                .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            let expected: Vec<u32> = inputs
                .iter()
                .map(|v| v.wrapping_mul(3).wrapping_add(addend))
                .collect();
            assert_eq!(actual, expected);
            assert_ne!(actual, inputs);
            assert_eq!(backend.pipeline_build_count(), Some(compilations));
            inputs = actual;
        }
        // Normal production selection still rejects this exact CPU device.
        assert!(Session::new(selector).is_err());
        std::println!("{{\"software_vulkan_dispatches\":5,\"values_verified\":1280,\"native_pipeline_compilations\":3,\"hardware_verified\":false,\"metal_verified\":false}}");
    }

    #[test]
    fn authored_fixture_profile() {
        let profile = inspect_spirv(SHADER, "main").unwrap();
        assert_eq!(profile.local_size, [64, 1, 1]);
        assert_eq!(profile.bindings, BTreeSet::from([0]));
    }

    #[test]
    fn rejects_bad_header_truncation_entry_and_optional_capabilities() {
        assert!(inspect_spirv(&[], "main").is_err());
        assert!(inspect_spirv(&SHADER[..SHADER.len() - 1], "main").is_err());
        assert!(inspect_spirv(SHADER, "missing").is_err());
        let mut broken = SHADER.to_vec();
        broken[..4].fill(0);
        assert!(inspect_spirv(&broken, "main").is_err());
        let mut optional = SHADER.to_vec();
        optional[24..28].copy_from_slice(&11u32.to_le_bytes()); // first OpCapability Shader -> Int64
        assert!(inspect_spirv(&optional, "main").is_err());
    }

    #[test]
    fn rejects_nonzero_descriptor_set_and_conflicting_workgroup_builtin() {
        for modification in 0..2 {
            let mut words: Vec<_> = SHADER
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let mut at = 5;
            let mut changed = false;
            while at < words.len() {
                let count = (words[at] >> 16) as usize;
                let opcode = words[at] & 0xffff;
                if modification == 0 && opcode == 71 && count == 4 && words[at + 2] == 34 {
                    words[at + 3] = 1;
                    changed = true;
                    break;
                }
                if modification == 1 && opcode == 43 && count == 4 && words[at + 3] == 64 {
                    words[at + 3] = 32;
                    changed = true;
                    break;
                }
                at += count;
            }
            assert!(changed);
            let bytes: Vec<u8> = words.into_iter().flat_map(u32::to_le_bytes).collect();
            assert!(inspect_spirv(&bytes, "main").is_err());
        }
    }
}
