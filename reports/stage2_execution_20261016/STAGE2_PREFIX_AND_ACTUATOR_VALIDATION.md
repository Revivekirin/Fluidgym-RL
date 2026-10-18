# Stage 2 prefix and actuator validation — local preparation

**VERIFIED:** The Stage 1 adapter implementation is retained, with only the approved condition names/horizon/onset changed in gain_at. Before CFD, source/AST checks enforce that restriction. The exact verified Stage 1 boundary observer is reused. Local software tests verify schedule/reset/memory preservation, nominal-prefix rejection, warm/RNG comparison, fixed tolerances, accepted-counter offset handling, weight/config guards, 960-control budget, stop-before-shift behavior and the 45-minute external watchdog. See local_validation.json for the final test record.

**VERIFIED prior evidence:** The returned 784-file Stage 1 evidence set passes integrity checks. Nominal gain1 equivalence, independent boundary readback and native transition validity were verified there. Stage 2 requires the same evidence and current pinned provenance before execution.

**UNRESOLVED:** Actual Stage 2 S0/S2 numerical prefixes and frozen-model invariance under its runs. These are checked per transition, not assumed from matching seeds. Full-state replay and physical accuracy remain uncertified. No tolerances are widened.
