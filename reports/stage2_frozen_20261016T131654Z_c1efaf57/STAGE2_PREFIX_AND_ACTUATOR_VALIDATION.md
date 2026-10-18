# Stage 2 prefix and actuator validation

UNRESOLVED: complete acceptance; inspect worker_status.json for the exact blocker and saved partial prefix_differences.json. Per-run prefix_differences.json includes maximum absolute/relative errors, bitwise numeric equality and exact RNG identity; the relative denominator floor is 1e-12 for reporting only. Acceptance stays atol=rtol=1e-5, clocks atol=1e-6/rtol=0, counters exact; actuator readback atol=1e-7/rtol=1e-6.

Requested actions, clipped commands, native smoother memory, expected profile products and measured boundary readback are separate. C++ setter arguments are not directly intercepted. Planner RNG and warm state before/after inference are recorded; no candidate, latent or value diagnostics added. Frozen online/target weight hashes and config/std invariants are checked before each physical control. Native source/solver are unchanged. Full hidden-cache replay and constructor divergence-correction internals remain UNRESOLVED.

Prefix difference summaries (full field detail remains in JSON):

e0_p0_S2: exact identity checks 481/481.

| Pair | Field category | Max absolute difference | Max relative difference | Numeric checks pass |
|---|---|---:|---:|---|
| e0_p0_S2 | observations/clocks | 4.05311584e-06 | 0.000433015167 | True |
| e0_p0_S2 | native state | 3.16202641e-05 | 25.4222222 | False |
| e0_p0_S2 | planner action/warm | 3.57627869e-07 | 2.11820814e-05 | True |
| e0_p0_S2 | transition/actuator | 4.17232513e-07 | 7.21210419e-05 | True |
