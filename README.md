# NextCore GPU

`nextcore-gpu` provides graphics resource contracts, a software rasterizer, and an
optional host Vulkan compute backend. This module is developed in its own
repository and included by the [26x86 integration repository](https://github.com/26x86/26x86)
as a Git submodule. It has no sibling NextCore crate dependencies.

Recorded compute commands now execute through an explicitly supplied
`ComputePipelineManager`: each submission waits for Vulkan completion and commits
readback before the next command. An unavailable backend or failed command returns
an error; completed earlier commands remain committed. The Vulkan backend reuses
one native pipeline for identical shader code, entry point, and ordered buffer
bindings, and recompiles when that key changes.

The product target is an x86 EFI host executing an ARM64E macOS guest. The Vulkan
backend runs under a host operating system as development and reference code.
The EFI GOP scanout implementation belongs to
[Nextcore-ISE](https://github.com/26x86/Nextcore-ISE), with its caller in
[Nextcore-EFI](https://github.com/26x86/Nextcore-EFI).

## Build and verify

From this standalone repository, with Rust stable installed:

```sh
cargo test --no-default-features
cargo test --features vulkan
cargo check --features vulkan --example vulkan_compute
```

These commands do not require a working hardware GPU. The ignored Linux Vulkan
lifecycle test requires a Vulkan loader, the Mesa llvmpipe ICD, and Khronos
`spirv-val`. On the validated Ubuntu 24.04 setup:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json \
  timeout --kill-after=2s 30s \
  cargo test --features vulkan --lib \
  software_vulkan_pipeline_reuse_and_replacement -- --ignored --nocapture
```

The 2026-09-08 WSL2 validation used Rust 1.98.1 and Mesa 25.2.8 llvmpipe
(LLVM 20.1.2). The default suite passed 99 tests; the Vulkan suite passed 102 tests
with one ignored test. Running that ignored test explicitly passed five actual
software Vulkan dispatches, independently checked 1,280 output values, and
observed three native pipeline compilations.

A subsequent run on the same date cross-built the unchanged hardware acceptance
example from WSL2 and executed it natively on Windows with an AMD Radeon RX 6800
XT (`1002:73bf`, driver `32.0.21043.10005`). Six actual hardware dispatches passed
1,280 independent readback comparisons and guard/fence/failure checks. Three
native pipeline compilations established shader reuse and replacement;
`cpu_fallback=false`. The [hardware validation evidence](validation/windows-rx6800xt-20260908/README.md)
includes the selected device, stdout arrays, source hashes, and reproduction
instructions. No driver replacement or reboot was needed.

See [VULKAN_COMPUTE.md](VULKAN_COMPUTE.md) for the supported shader/buffer contract,
recorded command API, hardware acceptance example, and failure semantics.

## Current limits

- The default compute manager has no execution backend. Public Vulkan construction
  requires an exact integrated/discrete device match and rejects CPU/virtual GPUs.
- Guest Metal registration, driver transport, Metal shader translation, and a
  physical PCI GPU command backend for EFI are not implemented by this module.
- `VirtualMetalDevice` supports software buffer copies only and reports Metal
  support as false. Its command API rejects unsupported compute, clear, and
  present operations.
- The hardware result establishes Windows-host Vulkan compute on the recorded
  GPU and driver. macOS boot, guest Metal execution, and EFI GPU command submission
  remain unverified.

## Provenance

The historical `26x86-Nextcore-GPU-v0.1.1` snapshot was exported from integration
commit `dcc90013109eac694ccbf997b1e44a7018480f78`. The current development tree
includes subsequent command execution and pipeline lifecycle changes; it is not
an unchanged copy of that release. [repository.json](repository.json) preserves
the original release metadata and describes the development state.
[repository-files.json](repository-files.json) records current public file sizes
and SHA256 hashes, excluding itself, Git metadata, ignored files, build outputs,
and private `_isolated` paths.

This repository contains public source and an independently authored shader test
fixture. It includes no Apple firmware, operating-system binaries, or private
research inputs. The [repository license](LICENSE.txt) applies.
