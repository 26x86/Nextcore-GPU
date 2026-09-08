# Explicit host Vulkan compute

Current state: `ComputePipelineManager::new()` stores host buffers/descriptors but
cannot execute shaders. Its `dispatch` returns `BackendUnavailable` without
signaling a fence. The optional `vulkan` feature provides a real host Vulkan 1.1
compute backend; it does not attach guest Metal, SGPU, or a VirtualMetalDevice.
The latter model supports software buffer copies only. Its Metal support query
returns false, and its command-list API rejects compute, clear, and present
before executing any copy in a mixed list. SGPU propagates that failure without
a success response and does not advertise compute execution support.

Create a manager with `ComputePipelineManager::with_vulkan(VulkanComputeConfig)`.
The config requires exact nonzero vendor/device IDs, an explicitly trusted
Khronos `spirv-val` executable, and a fence timeout in `(0, 5 seconds]`. Only one
matching integrated/discrete GPU is accepted. CPU/virtual devices and ambiguous
selectors are rejected. `robustBufferAccess` must be supported and is enabled.

Supported inputs are SPIR-V 1.0–1.3 with only Shader capability, Logical/GLSL450
memory model, one compute entry point and literal LocalSize. A WorkgroupSize
builtin must be a matching literal constant vector. The entire module must pass
`spirv-val --target-env vulkan1.1`, in addition to the crate's contract reflection.
No optional shader features, images, push constants, specialization constants,
descriptor arrays, decoration groups, or workgroup storage are supported.

Use 1–8 distinct storage buffers with unique bindings in descriptor set zero.
Every source view has a four-byte-aligned offset and size; each view is at most
16 MiB (minimum four bytes) and the combined cap is 64 MiB. Aliasing even disjoint views
of the same host buffer is explicitly unsupported. The view is uploaded into a
separate Vulkan buffer, so its Vulkan descriptor offset is zero. Actual device
storage/compute limits and the 16,777,216 total-invocation cap are also checked.

Dispatch uploads to HOST_VISIBLE|HOST_COHERENT memory, records compute and a
COMPUTE_SHADER/SHADER_WRITE → HOST/HOST_READ buffer barrier, submits with a real
Vulkan fence, waits, and reads every buffer back. Only then are host buffers
updated together and the manager fence signaled. The destination fence must
exist and be unsignaled. Failed validation/dispatch leaves host data uncommitted.

If submission completion is uncertain (including a fence timeout), the backend
is poisoned. Its potentially referenced Vulkan objects and dynamic loader are
retained until process exit; it cannot dispatch again. It never destroys those
resources while commands may still use them. Fence timeouts do not bound driver
initialization, shader compilation, memory mapping, or destruction calls. Run
the execution example under an external process deadline too:

```sh
cargo build -p nextcore-gpu --features vulkan --example vulkan_compute
timeout --signal=TERM --kill-after=2s 15s \
  target/debug/examples/vulkan_compute 8086 7d41 /path/to/spirv-val
```

The host must supply its public Vulkan loader/ICD. Driver environment variables
are runtime configuration, never crate build inputs. This example uploads two
different 256-value inputs, dispatches `input * 3 + 7`, verifies every readback
value against independent host arithmetic, checks nonzero-offset prefix/suffix
guards, and tests rejection without fake fence completion. It prints one JSON
receipt. Host Vulkan success does not establish guest graphics or Metal support.

`tests/fixtures/storage_transform.comp` is independently authored in this crate.
Rebuild its owned SPIR-V fixture with public Khronos tools:

```sh
glslangValidator -V --target-env vulkan1.1 \
  tests/fixtures/storage_transform.comp -o tests/fixtures/storage_transform.spv
spirv-val --target-env vulkan1.1 tests/fixtures/storage_transform.spv
```

Public contracts: [Vulkan synchronization](https://docs.vulkan.org/spec/latest/chapters/synchronization.html),
[Khronos synchronization examples](https://docs.vulkan.org/guide/latest/synchronization_examples.html),
[SPIR-V grammar](https://github.com/KhronosGroup/SPIRV-Headers/blob/main/include/spirv/unified1/spirv.core.grammar.json).

## Recorded compute command execution

`CommandExecutor::execute_with_compute` connects recorded `DispatchCompute`
commands to an explicitly supplied `ComputePipelineManager`. Pipeline IDs refer
to that manager, including its registered buffer bindings. Each dispatch finishes
GPU readback before the next command; ordered dispatches can therefore consume
previous results. Only successful compute completions increment the executor
count. A failed command stops the remaining list and preserves prior completed
work. Compute inside an active render pass is rejected. `execute_with_backends`
combines this compute path with the existing software rasterizer.

The older `execute` and `execute_with_rasterizer` entry points now return
`ComputeError::BackendUnavailable` for compute commands instead of counting a
logged no-op. The manager's default instance also has no executing backend.
A temporary host fence manager is used per synchronous dispatch, so recorded
submissions do not accumulate fence entries. Vulkan timeout ownership remains in
the Vulkan backend itself.

The example also runs two ordered recorded dispatches over 256 fresh values,
compares with the independent double transform, checks prefix/suffix guards and
verifies that a bad recorded pipeline stops a subsequent valid command. These
checks validate the host command adapter; guest Metal registration and transport
remain unimplemented.

The backend retains one native pipeline for byte-identical SPIR-V, entry name and
ordered storage binding numbers. Its cache is bounded and replaced only when no
submission is in flight. Per-dispatch buffers/descriptors/command buffers remain
short-lived; uncertain submissions retain both sets of objects. Caller shader IDs
are not cache keys. `vulkan_pipeline_build_count` exposes successful native driver
compilations for a live backend. The example verifies reuse, replacement by changed
shader code with the same caller ID, and rebuilding the original after eviction.

A Linux-only ignored lifecycle test can run with the explicitly selected Mesa
llvmpipe CPU ICD. It validates Vulkan execution and resource/cache management;
it does not validate a hardware GPU or Metal. The module-private device-type
predicate used for this test is not exposed by the public constructor, whose
hardware-only policy remains unchanged:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json \
  cargo test -p nextcore-gpu --features vulkan --lib \
  software_vulkan_pipeline_reuse_and_replacement -- --ignored --nocapture
```

The test verifies five ordered Vulkan dispatches / 1,280 output values, reuse of
identical shader code, recompilation for changed code with the same caller ID,
eviction/rebuilding, and rejection of the same CPU by production device selection.

## Recorded validation and deployment scope

On 2026-09-08, the WSL2 Ubuntu 24.04 development environment passed 99 tests with
`--no-default-features`, and 102 with `--features vulkan` (one ignored lifecycle
test). The ignored test was then run explicitly with Mesa 25.2.8 llvmpipe,
LLVM 20.1.2, and Rust 1.98.1: five actual dispatches completed, 1,280 output values
matched independent arithmetic, and the native pipeline compilation count was
three. The public constructor also rejected this CPU device as expected.
`cargo check --features vulkan --example vulkan_compute` also passed.

The unchanged hardware acceptance example was subsequently built from WSL2 for
`x86_64-pc-windows-msvc` and executed natively on Windows 11 build 26200. Vulkan
selected an AMD Radeon RX 6800 XT (`DISCRETE_GPU`, vendor `1002`, device `73bf`,
Windows driver `32.0.21043.10005`) with robust buffer access enabled and no CPU
fallback. Six actual dispatches completed; 1,280 readback values matched
independent arithmetic, guards and failure/fence checks passed, and three native
pipeline compilations established cache reuse, changed-code replacement, and
rebuilding after eviction. The two ordered dispatches are checked through their
final 256-value result; the intermediate result is not separately persisted.
The 768 values included in stdout were independently recomputed afterward.

[The public hardware receipt](validation/windows-rx6800xt-20260908/receipt.json)
and [raw stdout arrays](validation/windows-rx6800xt-20260908/hardware-example.json)
preserve device selection and source/tool hashes. The run used the installed
Windows driver without replacement or reboot. Its
[reproduction notes](validation/windows-rx6800xt-20260908/README.md) include the
standalone WSL cross-build environment and exact Cargo dependency lock.

The x86 EFI product path cannot directly use this OS-hosted Vulkan loader. It
still needs physical GPU command submission and a guest Metal driver/transport
implementation. The separate ISE/EFI GOP path performs framebuffer scanout and
readback; it does not establish Metal acceleration. Standalone build instructions
and preserved release provenance are in [README.md](README.md).
