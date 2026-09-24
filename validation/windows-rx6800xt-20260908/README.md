# RX 6800 XT Windows Vulkan validation — 2026-09-08

Current result: the unchanged hardware acceptance example from module commit
`1b5b56f3041b60c169cf6020c0a57359b113b91d` executed successfully on an AMD Radeon
RX 6800 XT (`1002:73bf`) through the installed Windows Vulkan driver. The goal of
this run was physical host GPU compute validation; EFI GPU command submission
and guest Metal remain separate, unimplemented product work.

[receipt.json](receipt.json) records source and tool hashes, device information,
and checks. [hardware-example.json](hardware-example.json) is the program's raw
stdout with CRLF line endings normalized to LF, containing input/output arrays
and the actual selected Vulkan device. Both native and normalized stdout hashes
are preserved in the receipt.
[cross-build.log](cross-build.log) records the successful standalone build; its
local checkout path is normalized. [cargo-lock.toml](cargo-lock.toml) preserves
the exact Cargo dependency resolution used by that build.

## Observed checks

- Six hardware dispatches completed: two direct, two ordered recorded commands,
  and two shader-cache replacement commands.
- The executable independently compared 1,280 readback values. The two ordered
  commands are checked together through their final 256-value result; their
  intermediate result is not separately persisted in stdout.
- An additional Python check independently recomputed all 768 readback values
  present in the raw stdout arrays.
- Three native pipeline compilations established identical-shader reuse,
  replacement by changed shader code with the same caller ID, and rebuilding
  the original shader after cache eviction.
- Offset buffer guards, fence completion, three rejected invalid dispatches,
  and stopping the command list after an invalid pipeline all passed.
- The selected device was `DISCRETE_GPU`, with robust buffer access enabled and
  `cpu_fallback=false`.

## Reproduce

Use the module sources matching the receipt's source hashes. A standalone Cargo
build on Windows can use MSVC normally. This run cross-built from WSL2 with Rust
1.98.1 and its `x86_64-pc-windows-msvc` standard library, `lld-link`, Windows SDK
10.0.26100.0 `um/x64` and `ucrt/x64` libraries, and MSVC 14.44.35207 `lib/x64`.
The three library directories were supplied as linker `/libpath:` arguments.
Copy `cargo-lock.toml` to the build checkout's `Cargo.lock` for the recorded
resolution, then run:

```sh
cargo build --target x86_64-pc-windows-msvc --features vulkan --example vulkan_compute
```

Execute the resulting program on Windows, with the existing Vulkan driver and
an explicitly trusted Khronos SPIRV-Tools validator:

```powershell
.\vulkan_compute.exe 1002 73bf C:\trusted-tools\spirv-val.exe
```

This validation used a 45-second external process deadline and the example's
three-second Vulkan fence timeout. The validator came from the
[Khronos SPIRV-Tools project's official continuous Windows build](https://github.com/KhronosGroup/SPIRV-Tools);
its exact archive URL, version, and executable SHA256 are preserved in the
receipt. No driver installation, driver replacement, or reboot was performed.
Neither executable nor SDK/driver binaries are included in this evidence.

The result establishes Windows host Vulkan dispatch and readback on this GPU
and driver. It does not establish EFI Metal acceleration, a macOS guest driver,
Apple shader translation, or macOS boot.
