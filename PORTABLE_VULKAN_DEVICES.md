# BP28 portable host Vulkan device selection

Current: the existing explicit backend is not vendor-filtered, but its two PCI
IDs cannot distinguish two devices of the same model. It rejects host-visible
storage allocations without HOST_COHERENT, although Vulkan provides explicit
flush/invalidate operations. Device diagnostics omit identity and relevant
compute/memory limits. These are host backend portability gaps, not evidence
that an untested NVIDIA or Intel device currently fails.

Decision: retain the existing two-ID config and constructor API. Add explicit
Vulkan 1.1 deviceUUID selection to the backend and compute manager; reject no
match and duplicate matches, including duplicate reports of a device UUID. There is
no first-device choice, vendor allowlist or CPU/virtual fallback. Read-only
enumeration reports actual driver identity, queues, features, storage/compute
limits and memory flags/heaps, with concrete baseline unsupported reasons.
A candidate passing those baseline checks is not proof that an allocation,
shader, driver submission, or guest Metal will succeed. Per-buffer Vulkan
memoryTypeBits and limits remain authoritative at dispatch.

Select an eligible HOST_VISIBLE memory type using its actual memoryTypeBits,
heap index and allocation size; prefer coherent/cached memory without assuming
DEVICE_LOCAL implies host visibility or that unified memory is discrete VRAM.
The backend enables no protected/device-coherent memory features, so memory types
requiring them must not be allocated. In particular, actual AMD inventory exposes
DEVICE_COHERENT_AMD types; vkAllocateMemory VUID 02790 prohibits choosing them
without deviceCoherentMemory. Exclude those types while retaining ordinary AMD,
NVIDIA, Intel or other compatible memory without a vendor filter.
For noncoherent memory map the whole allocation, flush offset zero / WHOLE_SIZE
after upload, and invalidate the same mapped range after the GPU-to-host barrier
and completed fence before readback. Mapping the full object makes the end-of-
allocation exception explicit for nonCoherentAtomSize. Unmap even on maintenance
failure. Existing uncertain-submission poisoning and atomic host commit remain.

Ownership: only this GPU worktree. Implementation is restricted to Vulkan device
selection/memory, the manager constructor, existing example diagnostics and
independent tests/docs. No EFI OS-loader dependency, PCI adapters, guest protocol,
metadata, commits or pushes. Root owns integration and publication.

Validation: independent synthetic selection/bounds fixtures, normal crate tests,
explicit software ICD lifecycle test, and the unchanged arithmetic/readback/cache
acceptance path on the available RX 6800 XT, now also selected by observed UUID.
Synthetic capabilities are unit-test inputs only. NVIDIA/Intel hardware and a
noncoherent hardware allocation are unverified unless separately observed.
The product x86 EFI path and real macOS/Metal remain unverified.

Public contracts:
- https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceIDProperties.html
- https://docs.vulkan.org/refpages/latest/refpages/source/vkAllocateMemory.html
- https://docs.vulkan.org/refpages/latest/refpages/source/VkMappedMemoryRange.html
- https://docs.vulkan.org/refpages/latest/refpages/source/vkFlushMappedMemoryRanges.html
- https://docs.vulkan.org/refpages/latest/refpages/source/vkInvalidateMappedMemoryRanges.html

OPEN_QUESTION: none within this delegated host-backend scope.

Final verification: default tests 109, Vulkan tests 119 (one ignored software
lifecycle), explicit Linux software lifecycle 1, and native Windows library tests
10 all pass. Independent review found and resolved Linux-only cfg placement on
the new shared fixture; memory synchronization follows the quoted Vulkan rules.
Actual AMD inventory exposed optional device-coherent memory types, leading to
feature-gated memory eligibility and an independent rejection regression. Both
legacy-ID and observed-UUID hardware executions pass on the final source.
[The final receipt](validation/bp28-portable-rx6800xt-20260909/receipt.json) preserves
source/binary hashes, raw driver inventory and results. No physical noncoherent,
NVIDIA/Intel, EFI GPU or guest Metal success is claimed.
