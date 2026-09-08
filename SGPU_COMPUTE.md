# SGPU command submission contract — BP26

Current state: the APLS SGPU adapter owns a count-prefixed command list codec and
maps it to VirtualMetalDevice, whose execution is limited to software copies.
Host Vulkan compute and recorded CommandExecutor dispatch already execute real
work. The missing boundary is an architecture-independent decoded SGPU list
connected to an explicitly configured compute manager.

Decision: move the existing SgpuCommand/SgpuCommandList codec into this GPU
module and re-export those same types from APLS. Preserve existing tag numbers
and payload layouts. Add no transport envelope, opcode, Metal shader format,
background queue, or EFI Vulkan dependency.

## Wire and bounds

A list is a little-endian u32 count followed by packed tagged records, with no
native pointers or padding. Tags remain: 1=CopyBuffer (25 bytes), 2=RenderClear
(17), 3=ComputeDispatch (29), 4=Present (9). Handles are u64, kernel IDs and
geometry components are u32, and color components are IEEE binary32. Exact
record lengths and full-list consumption are required. Decode accepts at most
1,024 commands and 29,700 bytes; reject impossible counts before reserving any
command capacity. Empty lists contain exactly the four-byte zero count.

A checked snapshot-window entry accepts u64 offset and length, checks addition
and usize conversion, then decodes copied values from the bounded immutable
slice. The caller owns guest memory/grant validation and supplies an immutable
snapshot; this module never dereferences a guest address. The same bytes have
identical meaning for 32/64-bit ARM and x86 callers, including unaligned input.

## Execution and ownership

SgpuComputeSession owns one existing ComputePipelineManager and CommandExecutor.
Host setup explicitly maps nonzero guest resource/kernel IDs to existing manager
buffers/pipelines. Resource handles remain opaque u64 identifiers. Registered
buffers are capped at 16 MiB each, 64 MiB total and 64 resources; kernels at 256.
A kernel may bind only registered session resources, and its buffer spans and
workgroup shape must pass the existing compute contract. This API supplies no
new guest pipeline-registration wire payload and does not interpret Metal bytes
as SPIR-V. Shader validation remains the existing backend's responsibility.

Guest read/write/copy accesses use checked u64 ranges. Unregistering a referenced
resource is rejected until its registered kernels are removed. Unregistration
revokes the guest mapping; the manager retains ownership of underlying objects
until it is returned or dropped. No manager-mutating escape is exposed while
registrations are live. Sessions do not share registries.

Every list is structurally and semantically preflighted before any command:
resource/kernel lookup, copy range, matching block size, dimensions, and command
support. CopyBuffer preserves existing host-buffer copy semantics. ComputeDispatch
uses the mapped pipeline through CommandExecutor::execute_with_compute and waits
for backend readback. RenderClear and Present are rejected explicitly. Completion
counts include only operations that finished. A backend error stops execution;
previously completed commands remain committed and the error reports the failed
index and completed prefix. Unsupported/malformed lists commit nothing.

No success implies guest Metal registration, physical EFI command submission,
shader translation, or macOS boot. Vulkan remains an optional OS-hosted reference
backend. The default manager cannot execute compute.

## Validation plan

Exercise legacy APLS codec callers, explicit little-endian/high-bit handle
fixtures, unaligned windows, every truncation, trailing bytes, maximal count and
length attacks, resource bounds/lifetime/isolation, unknown kernels, block and
invocation limits, unsupported-command preflight, and partial completion when
an unavailable backend follows a valid copy. Extend the existing hardware
acceptance example with authored SGPU bytes, actual mapped compute dispatch,
readback, guards and rejection; cross-build in WSL and run against the existing
Windows RX 6800 XT without driver replacement or reboot.

## Validation result

The implementation passed 109 default GPU tests, 112 Vulkan GPU tests (one
ignored), and 48 APLS tests. The existing APLS public codec names now re-export
this module's implementation. Its default software session behavior is unchanged.
The physical RX 6800 XT run successfully decoded an authored 87-byte list at
snapshot offset one, copied one registered buffer, completed two real compute
dispatches, and matched 256 independent results. Rejected lists preserved prior
readback, and kernel-bound resource revocation was refused. Full source hashes,
raw payload, result arrays, tool provenance and reproduction instructions are in
[the hardware evidence](validation/sgpu-wire-rx6800xt-20260908/README.md).
