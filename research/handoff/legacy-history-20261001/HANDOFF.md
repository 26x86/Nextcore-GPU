# Historical public history handoff — nextcore-gpu

## Current Status

The authoritative main at `b931a09c9f563fdd55deba29764e2b126c2a8353` is the active module API. The older local public main `7d23e4bdfe1041449c5eef97eeaeeda722f5ce81` and frozen source HEAD `eaab9fcfded3a8ee298453397790f13edace5694` contain 3 commit identities absent from that initial main ancestry. Ordinary local merge commits preserve those identities without replacing the current API or dependency pins. Publication must be verified separately.

## Target State

Keep the current module implementation active. Retain the distinct older conflict variants below as historical research source with their exact bytes, original paths, Git blob identifiers and SHA-256 hashes in [manifest.json](manifest.json). The preserved variants are unfinished historical work; their presence makes no compiler, runtime, operating-system boot, Metal or device acceptance claim.

## Conflict disposition

The two conflicts differ in English comments and trailing newlines. Current comments and formatting are retained; exact older variants and their license are retained here. Executable behavior is unchanged.

## License

Each recorded source revision includes its original `LICENSE.txt` in the manifest. The current module [LICENSE.txt](../../../LICENSE.txt) also remains unchanged.
