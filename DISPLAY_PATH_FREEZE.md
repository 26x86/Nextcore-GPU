# Display-path freeze (Metal Driver Track M1)

Graph-local ADP L1 scanout wiring is frozen while the Metal track advances.
Remapping these ids requires updating this file **and**
`docs/research/METAL_DRIVER_TRACK.md` (same machine marker).

Machine marker (do not edit casually):

```text
M1_DISPLAY_PATH_FREEZE:window=0x6,irq=2
```

| Constant | Value | C symbol |
| --- | --- | --- |
| ADP MMIO window | `0x6` | `VF_M1_MMIO_WINDOW_ADP` |
| ADP AIC IRQ line | `2` | `VF_M1_ADP_IRQ_LINE` / Rust `DISPLAY_SOURCE` |

Metal feature flags and `metal_acceptance` must not clear or remap this path.
`metal_verified` stays false until Design D10 guest evidence.
