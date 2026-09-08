# SGPU wire to RX 6800 XT validation — 2026-09-08

Current result: an independently authored payload in the existing SGPU command
format passed through checked byte decoding, session resource/kernel mapping,
CommandExecutor, and real Windows Vulkan execution on an AMD Radeon RX 6800 XT.
The payload contains one host buffer copy followed by two GPU compute commands.
Its final 256 values, offset guards, rejection behavior, and resource lifetime
checks passed. Desired product work still includes an actual guest driver and
physical EFI GPU command submission.

[receipt.json](receipt.json) preserves the tested implementation hashes and
validation environment. [hardware-example.json](hardware-example.json) contains
the program stdout data, including the exact 87-byte payload and input/readback
arrays (line endings normalized to LF). An independent Python check recomputed
all 1,024 persisted values across the existing example and new wire path.
The example internally checked 1,536 values after eight physical GPU dispatches;
two successive dispatches are compared through their final result. Three native
pipeline compilations include the original cache lifecycle checks; the SGPU
adapter reuses the existing compiled pipeline.

The payload was decoded at offset one in a snapshot with guard bytes. It carries
full-width opaque resource identifiers `1234567800000001` and
`fedcba9800000002`, with no host pointers. Unknown kernels, unsupported Present,
and trailing bytes were rejected before any earlier copy could change data.
A registered kernel prevented revocation of its buffer; revocation succeeded
after unregistering the kernel and later access to the handle failed.

The Windows acceptance example was cross-built under WSL from the exact source
snapshot in the receipt, using the same tools and dependency lock as the
[earlier physical Vulkan validation](../windows-rx6800xt-20260908/README.md).
[cross-build.log](cross-build.log) records the successful build. Run the resulting
`vulkan_compute.exe 1002 73bf <trusted-spirv-val.exe>` on Windows under an external
process deadline (45 seconds here; the backend fence timeout is three seconds).
No driver replacement or reboot was performed. No binaries are published here.

The host suites passed 109 GPU default tests, 112 GPU Vulkan tests (one ignored),
and 48 APLS tests, including ten new GPU wire/session regressions and the APLS
shared-codec regression. The Linux-only ignored software lifecycle test was not
rerun for this change; physical Vulkan execution above exercises the changed
path.

This validates authored SGPU byte submissions against a physical host GPU. It
is not evidence that an ARM/x86 guest driver submitted these bytes, that Metal
shader translation exists, that the EFI product can submit GPU commands, or
that macOS boots. The APLS default session retains its software-copy behavior;
compute execution requires explicitly constructing and registering the new GPU
adapter. See [the SGPU contract](../../SGPU_COMPUTE.md).
