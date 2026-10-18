import pathlib,json,hashlib,csv
O=pathlib.Path(__file__).resolve().parent
F=O.parents[2]/'Fluidgym-Physics/third_party/fluidgym/src/fluidgym'
# Workspace sibling, not the RL source tree.
F=pathlib.Path('/Users/jihyekim/Desktop/Fluidgym-Physics/third_party/fluidgym/src/fluidgym')
h=[
('H1','envs/fluid_env.py',507,'Reset/domain initialization mismatch','Different loaded geometry/native fields at o0; different private RNG or domain identity.','VERIFIED: saved o0/o1 native arrays and reset clocks agree exactly; domain 8 and seed agree. This mechanism is not supported for saved initialization. Unsaved internal state remains UNRESOLVED.'),
('H2','simulation/extensions/resampling.cu',300,'Floating-point atomic scatter during observation resampling','Same native fields can produce last-bit resampled sensor differences without consuming planner RNG.','INFERRED: strongest supported candidate for the earliest observation discrepancy. atomicAdd at 333/335 accumulates data/weights; normalization at 499–527. o0/o1 native equality with unequal sensors fits. No isolated extraction repeat was performed; root cause UNRESOLVED.'),
('H3','envs/cylinder/cylinder_env_base.py',741,'Small policy differences propagate through smoothing and CFD','Sensor differences, then planner/warm/action differences, then measured boundary differences, then native field differences.','VERIFIED chronology fits; INFERRED propagation is plausible. No identical-action replay exists here, so contribution versus independent solver numerical variability is UNRESOLVED.'),
('H4','simulation/pict/PISOtorch_simulation.py',1840,'Pressure linear-solve sensitivity and reduction/convergence effects','Small RHS differences can produce nonuniform pressure residuals; iteration counts or residual histories could differ.','INFERRED compatible with o2/o3 RHS/pressure differences. UNRESOLVED: linear iterations/residual traces/fallback branch are not saved. Solver success does not imply equal numerical trajectories.'),
('H5','simulation/pict/PISOtorch_simulation.py',1922,'Uniform pressure gauge offset alone','Demeaning would remove most pressure discrepancy.','VERIFIED contradicted as a sufficient explanation: b4 demeaned maximum remains 3.10744e-5 versus 3.16203e-5. Source conditionally subtracts torch.mean; this is not evidence that reduction nondeterminism caused the failure.'),
('H6','simulation/pict/PISOtorch_simulation.py',2004,'Adaptive timestep branch divergence','Different accepted increments or clocks near first divergence.','VERIFIED no such signature in 75 paired outer calls: clocks match and each accepts one increment. Unsaved transient solver details cannot be reconstructed.'),
('H7','envs/cylinder/cylinder_env_base.py',273,'Advective outlet update / pressure coupling sensitivity','Residual concentration near downstream outlet and subsequent outlet velocity differences.','INFERRED spatially compatible: b4 residual lies at x≈18.52–19.98; b4_boundary1_velocity differs at o3. PRE callback uses update_advective_boundaries(tol=5e-6). UNRESOLVED causality: no per-inner-solve outlet trace.'),
('H8','simulation/extensions/cg_solver_kernel.cu',93,'GPU norm/reduction and convergence threshold sensitivity','Different convergence behavior or residual histories despite comparable RHS.','UNRESOLVED: cuBLAS norm/sum/max operations are concrete numerical operations, not proof of nondeterministic execution. No residual/iteration evidence isolates them.')]
records=[dict(id=a,source=str(F/b),line=c,hypothesis=d,expected_signature=e,evidence=f) for a,b,c,d,e,f in h]
(O/'source_hypotheses.json').write_text(json.dumps(records,indent=2))
with (O/'source_hypotheses.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
extra=['envs/util/obs_extraction.py','simulation/pict/util/output.py','simulation/pict/data/resample.py','envs/cylinder/jet_cylinder_env_2d.py','simulation/simulation.py']
paths=sorted(set([r['source'] for r in records]+[str(F/p) for p in extra]))
(O/'source_hashes.json').write_text(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths},indent=2))
report='''# Stage 2 nominal-prefix discrepancy investigation

## Disposition

**BOUNDED VERIFICATION REQUIRED.** **VERIFIED:** the original Stage 2 native-pressure gate failed and remains failed. No directly supported implementation defect has been isolated. **INFERRED:** numerical variability is present in the observation/control path. **UNRESOLVED:** saved evidence cannot separate its downstream contribution from independent solver sensitivity/nondeterminism. For the requested A/B/C classification: **C is the defensible root-cause classification**; B describes observed numerical variability but is not an isolated explanation of native-pressure failure. A is not established.

This investigation used existing files only: **0 CFD controls, 0 checkpoint inference calls, 0 training updates, 0 retries**. No acceptance criterion, checkpoint, simulator source, or original artifact was modified. Proposed future verification below is not authorized for automatic execution.

## Evidence and integrity

Source evidence: `reports/stage2_frozen_20261016T131654Z_c1efaf57`, comparing `e0_p0_S0` and `e0_p0_S2`. Both use VAL environment seed 0, planner seed 0, actual initial domain 8; gain is 1 throughout the available paired prefix. S2 never reaches its t=40 gain change.

**VERIFIED:** all 1,349 manifest-listed output file hashes were recomputed before and after analysis. All 408 individual native array hashes (51 arrays × 4 boundaries × 2 runs) match their saved state metadata. This verifies correspondence with the returned manifest, not independent authentication of the remote execution. Derived comparison covers 976 field records. `analyze_saved.py` uses NumPy and standard-library ZIP inspection, without importing Torch, FluidGym or policy code.

**VERIFIED inventory:** at o0 (reset), o1 (after t0), o2 (after t1), o3 (after t2), each run has:

- `state_00..03.npz/json`: five blocks' velocity, pressure, vertex coordinates, transforms and face transforms; saved boundary velocity/transform arrays; domain velocity RHS/result and pressure RHS/RHSdiv/result; smoothing memory. Exact 51-key shapes/dtypes are in `inventory.json`. Saved boundaries: b0 indices 0/1; b1 2/3; b2 0; b3 2/3; b4 1/2/3. This is a snapshot subset, not all possible solver internals.
- `obs_00..03.npz`: extracted velocity (151×2), pressure (151), flattened policy input (302 velocity values). These are resampled/extracted sensor observations, **not native mesh fields**. Pressure sensors are diagnostic, not part of this flattened policy input. A complete resampled uniform grid is not saved.
- State JSON clocks/counters and RNG hashes; `reset_boundary.json` contains measured jet velocity/flux at reset. Per-transition `planner_00..02.json/npz` contains requested/clipped actions, warm state before/after and RNG hashes; `planner_rng_00..02.pt` contains serialized RNG records. Python, NumPy, Torch CPU and CUDA hashes agree before/after each compared plan. ZIP tensor-storage payloads were compared without unpickling; this is not a replay of RNG consumption.
- `transition_00..02_begin.json`, `actuator_00..02.json` and `steps.json`: gain, action, smoothing memory, measured top/bottom boundary velocity/flux before and after each solver call, solver status, clocks, reward, drag, lift, absolute lift and completion flags. There are 25 outer solver calls per transition, 75 per run in this prefix.

**UNRESOLVED / unavailable:** intermediate native fields within those outer calls; linear-solver residual histories and iterations; every fallback/inner callback state; a full CUDA scheduling trace; a separately saved resampling-coordinate tensor; full-episode S0/S2 comparisons. Experiment provenance/runtime timestamps are contextual; numerical alignment uses transition index and solver clocks, not matched wall-clock timestamps. Unsaved states were not reconstructed.

## Numerical method

Let x=S0, y=S2 and d=y−x, evaluated in float64 from saved arrays. Original criterion is retained: |d| ≤ 1e-5 + 1e-5|y|. RMS is sqrt(mean(d²)). CSV/JSON report max absolute error, RMS, exceedance count, means, signed mean difference, and max/RMS after subtracting the spatial mean difference. For multi-component velocity, separate component rows prevent a combined-component mean from obscuring directional differences. Means are unweighted sample/cell means, not volume integrals.

Relative diagnostics include |d|/max(|x|,1e-12), and a safer symmetric |d|/max(|x|,|y|,1e-5), plus near-zero counts and absolute errors. Neither relative diagnostic replaces the original gate. Native localization uses the arithmetic average of four saved vertices, not an exact curved-cell centroid. Sensor locations are not asserted from unavailable saved coordinate tensors.

## Earliest discrepancy versus first failed gate

| Boundary | Solver time; solver counter | Max native b4 pressure error | Max policy-input error | Max pressure-sensor error |
|---|---|---:|---:|---:|
| o0 reset | 1.2099997997; 22 | 0 | 2.38419e-7 | 1.19209e-7 |
| o1 after t0 | 1.4599995613; 47 | 0 | 5.96046e-8 | 2.38419e-7 |
| o2 after t1 | 1.7099993229; 72 | 1.60933e-6 | 4.76837e-7 | 2.50340e-6 |
| o3 after t2 | 1.9599990845; 97 | 3.16203e-5 | 8.34465e-7 | 4.05312e-6 |

**VERIFIED:** earliest recorded difference is **o0 sensor extraction**, while all 51 saved native arrays are identical at both o0 and o1. Earliest saved native differences are o2, where all native fields pass. Earliest recorded failed gate is **o3 / b4_pressure**, after t2. Observation differences at reset therefore precede any differing requested action or measured jet boundary.

**VERIFIED:** o3 b4 pressure has 722/10,680 exceedances; max 3.1620264053344727e-5; RMS 5.189338757925223e-6; maximum error/tolerance ratio 2.87924495. Signed mean difference is −5.45861808e-7; mean-subtracted max remains 3.107440224517318e-5 (about 98.3% of the original maximum). A uniform gauge shift is insufficient.

Maximum residual is at native grid [y=12,x=444], approximate physical center (19.9797754,0.1134214). All exceeding cells lie in grid columns 408–444; their center bounding box is x=[18.5235949,19.9797754], y=[−1.8884997,1.9884996]. This is downstream-edge localization, not a demonstrated outlet defect. Full localization and per-axis counts are in `spatial_localization.json`.

**VERIFIED:** offline comparison finds two additional o3 failures beyond the runner's first-stop field: domain_velocityRHS max 1.068115234375e-4 with 23 exceedances; domain_pressureResult max 3.1620264053344727e-5 with 722 exceedances. They are not earlier gates, and packed pressure data must not be counted as independent physical evidence. Native velocities and sensor fields pass at all available boundaries. Full per-block and per-component metrics are supplied in `comparison.csv/json`.

## Action/planner chronology

| Transition | Requested/clipped action max difference | Warm-after max difference | Measured delivery |
|---|---:|---:|---|
| t0 | 0 (both −1) | 1.49012e-7 | identical |
| t1 | 2.98023e-8 | 1.19209e-7 | first difference at outer call 24 (zero-based) |
| t2 | 4.47035e-8 | 3.57628e-7 | small differences persist |

**VERIFIED:** no warm state is present before t0; the warm-before states at t1/t2 inherit the preceding small differences. RNG hashes match before and after all three plans; identical RNG does not imply identical floating-point inference on different inputs. Saturation at t0 masks any differences in the final requested action.

At t1's 25th outer call, smoothing memory and measured top/bottom jet velocities first differ by 5.960464477539063e-8; measured flux differs by 1.862645149230957e-9, both before and after solver execution. These are runtime readbacks, not profile-derived expected values. Gain stays 1. All 75 paired before/after solver clocks match exactly; each call increments the accepted solver counter by one. Solver status passes. Source smoothing uses factor 0.1, and applied jet profiles are assigned after smoothing; command scaling is not substituted.

**VERIFIED:** small action and measured boundary differences precede the o3 pressure failure and coincide with the transition leading to first saved native differences at o2. **UNRESOLVED:** this chronology cannot establish causation. Across t0–t2, max reward/drag/lift differences are respectively 4.17233e-7, 2.38419e-7, 1.82539e-7; all pass the original scalar tolerance. No full-episode outcome reproducibility follows from three controls.

## Bounded source audit

Local pinned source is under `/Users/jihyekim/Desktop/Fluidgym-Physics/third_party/fluidgym`, reference pin `50071140b73a2a917fb96d33522a769f903f8c27`. Returned provenance records the clean pinned source and backend SHA256 `294edfc263cd57a7d4da0aeea72e76abcba3c586901511bac703659717ea2bea`. This audit reads source and saved provenance; it does not load the server binary. `source_hashes.json` identifies the local files inspected. **VERIFIED:** all 10 inspected source files match their exact pinned Git blobs (`source_pin_verification.json`), including CUDA source; this does not independently prove how the server binary was compiled.

Observation path: `envs/util/obs_extraction.py:28–59` → `simulation/pict/util/output.py:381–405` → `simulation/pict/data/resample.py:254–358` → compiled multi-grid resampling. Native data, resampling and sensor indexing are separate operations. The nondifferentiable CUDA resampler uses floating-point atomic accumulation. This supplies a concrete candidate for unequal observations from equal saved native states; it does not prove that this candidate caused downstream pressure failure.

Reset also uses private seeded generators (`envs/fluid_env.py:852–917`), domain loading (`:507–545`), cylinder boundary refresh/noise/warmup (`envs/cylinder/cylinder_env_base.py:334–404`). Boundary profile assignment is `jet_cylinder_env_2d.py:185–188`; native step success/failure handling is in `simulation/simulation.py:210–280`. Pressure audit follows the active nonorthogonal branch, not an unrelated orthogonal branch. GPU reduction presence alone is not evidence of nondeterminism.

'''
report+='| ID | Source | Candidate and expected signature | Evidence assessment |\n|---|---|---|---|\n'
for r in records:
 report+=f"| {r['id']} | [{pathlib.Path(r['source']).name}:{r['line']}]({r['source']}:{r['line']}) | {r['hypothesis']}. {r['expected_signature']} | {r['evidence']} |\n"
report+='''

## Three reproducibility targets

1. **VERIFIED failure — strict native state:** original o3 gate failed. It remains failed regardless of any future policy-level result.
2. **VERIFIED only for recorded prefix — policy interface:** o0–o3 observations and t0–t2 actions/warm states/actuation are within original numerical tolerance; RNG hashes match. **UNRESOLVED** for 80 controls.
3. **UNRESOLVED — closed-loop outcomes:** only three matched controls exist. S0 and S1 completed 80 controls each, but S1 has a different gain and is not a nominal repeat. Historical aggregate reward is not a paired reproducibility test.

## Proposed smallest next verification — not executed

**INFERRED recommendation:** preregister a separately named nominal policy/outcome repeatability test, with no shifted condition. Question: for the same physical initialization and planner seed, are the complete policy-interface trajectory and outcomes reproducible within the existing 1e-5 absolute/relative numerical envelope, even if strict native-state identity fails?

- **Exactly two fresh closed-loop nominal runs × 80 controls = maximum 160 CFD controls**, one VAL pair (environment seed 0, planner seed 0, physical initial domain 8), gain=1 throughout. Use frozen validation-selected TD-MPC 50k, unchanged planner/config/reward/initialization. No TD-MPC2, optimization or retry. Count reset/warmup solver calls separately from the 160 control budget. External watchdog: 15 minutes for the complete pair; stop on timeout, no retry.
- Before execution, preregister exact source/backend/checkpoint/config/data hashes, reset parameters, seed procedures, and this acceptance/stop rule in a new output directory. This is a proposal requiring separate authorization, not a restart or acceptance of original Stage 2.
- Log every real raw sensor observation and flattened input; requested/clipped action; warm state before/after; Python/NumPy/Torch CPU/CUDA RNG hashes; gain; physical initial-domain identity; per-outer smoothing memory and measured pre/post boundary velocity/flux; solver status, finite native fields, clocks and counters; reward/Cd/Cl/absolute Cl; runtime/provenance/cost and output hashes. Keep native endpoint snapshots and strict native discrepancies as diagnostics. No broader research dataset or training is required.
- **Prospective acceptance:** both runs complete 80 controls; initial identities/provenance and RNG hashes match exactly; all compared policy observations, actions, warm-state arrays, smoothing memory and measured actuation pass |S2−S0|≤1e-5+1e-5|S2| at every logged boundary (use repeat B as reference); solver clocks/counters match exactly; all per-control reward, Cd, Cl and absolute Cl pass that same criterion. Report returns and mean drag/absolute lift; aggregate discrepancy must not exceed the sum (or mean) of the corresponding per-control tolerance budgets. Native solver status must be successful and all native fields finite.
- **Stop immediately:** provenance/checkpoint/initialization mismatch; native solver failure or nonfinite field; actuation/profile contract failure; time/control budget exceeded; any prospective interface/outcome criterion fails when the second run reaches that point. No retries, margin tuning or condition changes. A native-only tolerance failure is retained and reported as a native failure, but is not the endpoint of this separately preregistered policy/outcome test. The original Stage 2 gate is neither waived nor reclassified.
- **If passed:** establishes one-seed, one-initialization, nominal 80-control policy/outcome repeatability within this envelope. It can support a new, explicitly policy/outcome-focused evaluation protocol. It cannot establish native-state identity, physical accuracy, all-seed robustness, shifted-control performance, history identifiability, or a solver root cause. If failed, preserve the first divergence and stop; do not automatically broaden tolerances or declare the research globally impossible.

This is a conservative numerical repeatability screen, not a scientifically calibrated smallest effect size. Any later performance margin must be prospectively justified against the research estimand and variability; the present evidence does not supply that calibration. An identical-input resampling micro-test could isolate H2 without CFD, but is not required to answer the immediate full-episode policy/outcome question and is not executed here.

## Research progression decisions

| Decision | Assessment | Conditions and scope |
|---|---|---|
| A. Stage 2 frozen-control performance under a newly preregistered protocol | **CONDITIONAL GO** | First establish nominal full-episode policy/outcome repeatability with the bounded pair above, then preregister outcome-focused comparison, paired seeds/initialization, uncertainty reporting, and native-failure reporting. Original Stage 2 remains stopped/failed; current data do not justify resuming its original protocol. |
| B. Stage 3 history-identifiability data collection | **CONDITIONAL GO** | May proceed independently of controller failure, including prescribed-action trajectories, once separately authorized with verified actuation/time/reset/solver contracts and a prospectively defined noise/repeat baseline. Use matched or randomized initialization, excitation, causal history windows, trajectory-level splits, and controls against schedule/time/initialization leakage. Strict native trajectory identity need not be a dataset assumption. Existing evidence does not yet demonstrate identifiable dynamics. No Stage 3 collection or estimator training is launched. |
| C. Change primary simulator to ManiSkill3 | **NO-GO on present evidence** | One strict pressure-prefix failure with working actuator/solver contracts supplies no comparative evidence that simulator replacement is necessary or improves this research question. No ManiSkill3 benchmark was performed. This is not a claim that migration can never be justified. |

**UNRESOLVED** root cause does not itself imply RESEARCH BLOCKED. The minimum defensible next empirical result is a bounded nominal policy/outcome pair, or separately preregistered prescribed-action identifiability collection with explicit noise controls. Neither requires proving bitwise CFD identity or demonstrating that the existing controller fails under a shift.

## Files and reproducibility

`comparison.csv/json`, `inventory.json`, `spatial_localization.json`, `action_planner_chronology.json`, `outer_actuation_comparison.json`, `verification.json`, `source_hypotheses.csv/json`, `source_hashes.json`, and the analysis script accompany this report. All are derived files in this separate directory. `analysis_output_hashes.json` hashes the final derived package (excluding itself). Run `analyze_saved.py` only to reproduce saved-array analysis; it contains no simulation or inference entry point.
'''
(O/'STAGE2_PREFIX_DISCREPANCY_ANALYSIS.md').write_text(report)
(O/'analysis_output_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(O.iterdir()) if p.is_file() and p.name!='analysis_output_hashes.json'},indent=2))
print('Report and source hypothesis table written.')
