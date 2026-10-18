# Stage 2B — prospective outcome-focused frozen-control pilot

Specification fixed before execution, 2026-10-18. **NEW experiment. Original Stage 2 strict native-pressure gate remains FAILED.** This neither resumes nor repairs that experiment. No result-driven changes or replacement trajectories.

## Fixed design and estimands

CylinderJet2D-easy-v0; VAL split; environment seeds {0,1} × planner seeds {0,1}; for each pair execute S0, S1, S2 in that order. Fresh environment and frozen validation-selected TD-MPC 50k agent per episode. Original reset randomize=True, horizon 80, observation (302 velocity sensor values), action clipping, reward, smoothing and planner settings remain unchanged. Physical initial-domain identity must match across conditions within a seed pair and is recorded, not inferred from the seed.

S0: gain 1 throughout. S1: gain .5 at t0–79 after nominal gain1 reset. S2: gain1 at t0–39, gain .5 at t40–79. First shifted transition is o40→o41. No policy, smoother or environment reset at t40. Instance-scoped post-smoothing adapter is byte-identical to the reviewed Stage 2 adapter; its implementation outside gain_at is checked against verified Stage 1. No command-scaling substitute.

Maximum 12 episodes, 960 attempted controls; no retries. External watchdog 2700 seconds (45 minutes), including provenance, resets, planning and logging. Reset/warmup solver calls are counted separately; the 960 limit is control transitions, not native solver iterations. No training/optimization. No automatic subsequent experiment.

Primary exploratory quantity: for each environment/planner pair, sum of rewards t40–79 in S2 minus S0 (Delta_post). Also report full return, pre40/post40 returns, Cd and absolute control-interval Cl, requested/clipped action magnitudes, clipping/saturation and their segment changes; descriptive S1−S0 comparisons. Show every seed pair and descriptive variation; no significance test or powered robustness claim. Saturation means |clipped action|≥1. No threshold is selected after observing returns. Pre40 differences are reported without post-hoc subtraction from Delta_post.

## Preregistered provenance before CFD

The launcher creates a unique reports/stage2b_frozen_<UTC>_<UUID> directory and saves this contract and package_hashes.json before starting the worker. expected.json, source_gate.json, frozen_reference.json, Stage 1's fixed manifest and unchanged inference loaders define expected source/backend/data/checkpoint/config hashes. The worker verifies them and saves provenance.json and frozen_provenance.json, then preregistration_ready.json binds their hashes and the contract/package manifest **before environment construction/reset**. A checkpoint is restored and frozen invariants checked before each episode's CFD. Failure before readiness means no experiment execution was accepted. Exact server paths and loaded source/backend are recorded; clean pinned Physics checkout is not modified.

Pinned FluidGym: 50071140b73a2a917fb96d33522a769f903f8c27. Expected compiled backend SHA256: 294edfc263cd57a7d4da0aeea72e76abcba3c586901511bac703659717ea2bea. Checkpoint SHA/config are bound by frozen_reference.json and expected.json, not overridden. Planner: horizon20, 512 samples, 64 elites, 6 iterations, discount .99, saved std .05000900000000001; existing eval_mode=True, step=50000 interfaces.

## Hard stops versus numerical diagnostics

STOP on failed source/backend/config/data/checkpoint provenance; unexpected initial-domain/config mismatch; nonfinite state, observation, action or planner output; native solver failure; broken smoothing/profile/independent measured boundary/flux contract; invalid gain/onset/reset behavior; unexpected horizon/counter behavior; frozen weight/config/optimizer invariant violation; or resource/control/watchdog budget violation. Existing actuator readback tolerance stays atol1e-7/rtol1e-6. No retries or missing-trajectory substitution.

Cross-run differences in finite native arrays, sensor observations, actions, warm state, RNG hashes, rewards, actuation or adaptive clocks are **diagnostic**, not stops. This accommodates otherwise valid closed-loop numerical divergence; each run must still satisfy its own hard contracts. Retain original prefix diagnostic envelope atol=rtol=1e-5, clock atol1e-6/rtol0 and exact identity comparisons; never label diagnostic disagreement as strict-native PASS. S1/S2 reset diagnostics and S2 nominal-prefix diagnostics t0–39 are retained. S0/S2 source/config/physical-initialization mismatches remain hard stops. No aggregate reward value is a stop criterion.

## Logs and integrity

Each transition records reward, drag, signed and absolute lift, requested/clipped action, gain, raw/flattened policy observation, planner RNG hashes/full states and warm arrays, physical time/counters; every outer call records native status, finite fields, smoothing memory, independent effective boundary velocity/flux before/after solve, and source-derived expected values separately. Retain available native snapshots/geometry/workarrays, immutable profiles, initialization identities and runtime provenance. Constructor divergence correction precedes the single-step observer; its internal iterations are not independently logged. Native success is not physical-accuracy proof. Persist partial logs and explicit failure status. Output hashes and preservation checks are finalized without CFD.

## Interpretation and progression

VERIFIED will mean measured/logged contract and results, not simulator physical accuracy. INFERRED effects concern this frozen closed-loop controller under assigned schedules, with numerical variability. UNRESOLVED: isolated learned-model error, optimal achievable control, history necessity and generalization. Two planner seeds sharing an environment seed are not independent physical cases. No retuning, checkpoint reselection or new gain schedule.

Stage 3 recommendation is evidence-based and conditional: controller failure is not required for history identifiability. Any collection needs separate authorization, causal label alignment, excitation, trajectory-level splits and leakage/noise controls. No Stage 3, TD-MPC2 or training is launched by this package.
