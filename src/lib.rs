pub mod canonical_spec;
pub mod virtual_device;
pub mod adapters;
pub mod translator;
pub mod framebuffer;
pub mod texture;
pub mod sync;
pub mod render_pipeline;
pub mod command_executor;
pub mod compute;
#[cfg(feature = "vulkan")]
pub mod vulkan_compute;
pub mod rasterizer;
pub mod sgpu_command;
pub mod sgpu_compute;
/// Design D10 honesty contract (Metal Driver Track M0). Always compiled so
/// default tests cannot silently omit Metal acceptance locks. Feature
/// `metal_track` marks the authorized engineering track for future gated work.
pub mod metal_acceptance;
/// Metal Driver Track M1: ADP window `0x6` / AIC line 2 display-path freeze.
pub mod display_path_freeze;
/// Metal Driver Track M2: typed guest-facing ABI with explicit reject paths.
pub mod metal_public_abi;
/// Metal Driver Track M3: host↔guest transport sketch (buffer/copy shapes only).
pub mod metal_transport;
/// Metal Driver Track M4: non-Metal host accel probe + receipt (x86-oriented).
pub mod metal_accel_probe;
/// Metal Driver Track M5: guest Metal probe harness (executed + fail reasons).
pub mod metal_guest_probe;
/// Metal Driver Track M6: Design D10 acceptance gate evaluator (transcript).
pub mod metal_d10_gates;
