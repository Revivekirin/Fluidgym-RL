# Stage 1 reviewed paired transition results

**VERIFIED:** 8/8 runs, 96/96 controls, zero solver failures and zero retries. Worker time 114.81 s; launcher time 122.08 s, within 600 s. All runs used VAL domain index 8, seed=0, randomize=True and the original horizon 80. Each stopped after its prescribed 12 controls.

There were 2,400 control outer solver calls plus 168 reset warmup calls = **2,568 observed successful single_step calls**. The returned cost field `accepted_solver_steps_including_resets=2576` sums final counters. Each run starts at counter 1, so 8 initial offsets must be subtracted: **observed counter increments=2,568**. Do not interpret 2,576 as measured executed substeps. The pre-observer initialization solve is not independently counted/certified.

| Condition (both repeats identical scalar results) | 12-control reward sum | Mean Cd | Mean signed Cl |
|---|---:|---:|---:|
| N-bare | -9.582380574 | 3.332958599 | -0.340151744 |
| N-hook | -9.582380574 | 3.332958599 | -0.340151744 |
| Fixed .5 | -9.249357924 | 3.326323092 | -0.283872683 |
| Jump at t=6 | -9.869973347 | 3.333771328 | -0.316190557 |

**VERIFIED:** The jump first applies in o6→o7. At its first outer call, native smoother memory is 0.1742885410785675 and the argument supplied to native apply is 0.08714427053928375; actual boundary readback agrees with profile×that effective input. Memory is not halved.

**VERIFIED numerical observation:** For Fixed, flat-velocity RMS and lift first exceed the predeclared threshold in consecutive transitions t=1,2; drag at t=2,3. For Jump, all three first satisfy it at t=6,7 and remain above threshold through t=11. Offline recomputation agrees with every recorded difference and threshold. Nominal noise and the fixed numerical floor, not a post-hoc threshold, define this criterion.

**INFERRED:** This short matched-reset experiment supports delivered actuator-effectiveness intervention and a detectable downstream sensor response. Its trajectory reward differences are not frozen-policy performance or irreducible authority-loss estimates.

**UNRESOLVED:** Generalization across initial domains/seeds, statistical gain identifiability, history necessity and policy robustness. Two repeats of one seeded initialization are numerical repeats, not independent physical cases.
