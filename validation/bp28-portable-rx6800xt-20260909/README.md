# BP28 portable Vulkan validation

The final GPU worktree sources were cross-built in WSL2 and executed natively on
Windows against the installed AMD Radeon RX 6800 XT driver. Both the existing
1002:73bf selection and the freshly enumerated device UUID completed eight real
GPU dispatches, buffer readback, shader-cache replacement and SGPU command checks.
Each run checked 1,536 output values internally; an independent Python check
recomputed all 1,024 values present in each stdout. The ordered two-dispatch pairs
are checked by their final result. An absent UUID returned failure without output
or fallback. No driver replacement or reboot was performed.

The Vulkan suite passed 119 tests (one explicitly ignored Linux software test),
default features passed 109, and native Windows passed all 10 library tests. The
separate llvmpipe lifecycle test passed five software dispatches/1,280 values.
It does not establish physical acceleration. Seven new authored capability tests
cover duplicate IDs/UUIDs, vendor-independent selection, unsupported reasons,
queue availability, memory type masks/heaps, index 31, malformed counts, and
unenabled optional memory features. These fixtures are not fabricated hardware
acceptance results.

The public JSON/stdout and build logs are preserved without content redaction.
[receipt.json](receipt.json) records final source, executable, validator and raw
artifact hashes. Cargo dependency resolution is preserved in cargo-lock.toml.
The Windows driver identity is recorded independently in windows-devices.json.

To reproduce, build with the crate's Vulkan feature and installed Vulkan loader,
then run under an external process deadline (45 seconds here):

```powershell
.\vulkan_compute.exe --list
.\vulkan_compute.exe 1002 73bf C:\trusted-tools\spirv-val.exe
.\vulkan_compute.exe 1002 73bf C:\trusted-tools\spirv-val.exe OBSERVED_32_DIGIT_UUID
```

Use a UUID from that machine's current inventory; the recorded UUID is not a
portable machine setting. The WSL cross-build uses lld-link plus Windows SDK
10.0.26100.0 um/x64 and ucrt/x64 libraries and MSVC 14.44.35207 lib/x64, passed as
linker /libpath arguments. A native MSVC build can use the normal Windows setup.

All observed host-visible types on this GPU were coherent. Actual noncoherent
memory operations, NVIDIA/Intel integrated hardware, duplicate physical GPUs,
EFI GPU submission, macOS guest Metal and macOS boot are not established by this
receipt. The backend uses no OS-hosted Vulkan library from the EFI product.
