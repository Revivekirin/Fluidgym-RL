# TD-MPC vs TD-MPC2 nominal implementation and performance audit

Audit date:2026-10-13. **No retraining, GPU diagnostic, algorithm fix, physics change, remote login, W&B write or Git push was performed.** New audit outputs and opt-in diagnostic code only. The prior report aggregation fix predates this audit and is not an algorithm change.

## Verdict

- **Verified:** both recorded runs reached50,000 real environment steps;49,940 vs49,999 learner calls. Core pinned objectives/model/planners are native with documented compatibility patches. TD-MPC's runner is adapted, not a literal upstream train.py reproduction.
- **Verified:**21 shared evaluations (16 validation candidates,4 tests,1 train reference), checkpoint hashes, validation-only selection and per-episode recomputed aggregates pass. One common physics/data/seed-budget cohort. These are valid conditional comparisons of the saved runs.
- **Inference:** same physical task and correct common wrapper are strongly supported. **Unavailable:** fresh real-CFD cross-wrapper transition equivalence and bitwise GPU determinism.
- **Verified:** seed1 TD-MPC outperforms seed1 TD-MPC2 on the recorded common test episodes. **Not established:** general superiority, sample efficiency, learning instability mechanism, or a causal explanation based on horizon/architecture/losses.
- **Research decision:** ready for bounded validation-only diagnostics; defer TD-MPC2 as sole primary dynamics-shift mechanism-analysis system until nominal planner/variability checks. Do not tune with the already reported test results.

## Deliverables and provenance

Read upstream_fidelity.md for exact source/function/line evidence, environment_contract.md for physical and reset semantics, config_comparison.csv for all resolved fields and effective quantities, checkpoint_audit.csv for16 precise checkpoint identities, metrics_dictionary.csv for mathematical definitions/reductions, diagnostic_plan.md for opt-in budgets/commands, research_decision.md for ranked hypotheses. artifact_inventory.csv inventories296 supplied run/evaluation files plus source/patch/test paths. input_hashes.json and preservation.json establish original preservation. source_snapshots contains exact pinned source text and licenses; no third_party checkout was modified.

Raw W&B .wandb histories and summary/config files exist in the supplied run directories and are hashed in the inventory. Their binary event streams were not decoded or synced. Analysis uses the actual CSV/JSONL records emitted by the same logger path and manifest config; no additional hidden W&B metrics are assumed. Historical TD-MPC2 dirty harness cannot be reconstructed byte-for-byte from a git SHA alone.

## Common evaluation facts

CylinderJet2D-easy-v0,80 control steps,10 episodes per split,environment seeds0–9,seeded randomize=True. Shared wrapper checks302 float32 observations,1 bounded action,nominal physical parameters and reward identity. One cohort based on full installed source/data hashes and contract. Native planners remain seeded sampling planners, not deterministic actor means. Original logs never substitute for common validation.

Recomputation in build_evidence.py reads every episodes.csv, recomputes reward/step,return,horizon,drag,signed/absolute lift,total wall seconds and measured-reference drag reduction; absolute discrepancies are <1e-10. Every episode return/length agrees with its mean reward. It verifies reported checkpoint hashes against actual local checkpoint bytes and selection candidate manifest hashes after translating the server path prefix only in memory. All candidates selected from validation; frozen winner hashes are linked into test_best manifests. No original path is rewritten.

| Independent test | TD-MPC val-best50k | TD-MPC final50k | TD-MPC2 val-best44k | TD-MPC2 final50k |
|---|---:|---:|---:|---:|
| Reward/step |0.03762838|0.03812451|−0.07737686|−0.08246609|
| Episode return |3.01027071|3.04996078|−6.19014869|−6.59728717|
| Mean Cd |3.11004720|3.11010831|3.22068048|3.22553160|
| Mean abs Cl |0.18047993|0.17992270|0.18485190|0.18509001|
| Drag reduction % |6.54430|6.54246|3.21981|3.07404|

Sources: results/nominal_comparison/20261013T061228Z/{tdmpc,tdmpc2}/{test_best,test_final}/episodes.csv and summary.json; recomputed in evaluation_recomputed.csv. Reference Cd=3.3278303676843644 from reference/summary.json, measured on TRAIN. That reference's reward is not an independent same-split test baseline. Native reward Cd_ref is separately preserved in physics manifests.

**Verified discovery:** TD-MPC best and final have different serialized archive hashes but all76 saved online+target tensors are bitwise identical and both planner_std values are0.05000900000000001. See tdmpc_best_final_weights.json. Their small evaluation score differences therefore are NOT different learned weights. They may arise from numerical/planning/solver execution variation; root cause is unavailable. Different hashes alone were insufficient to infer different models. Keep both original evidence records; do not average them into independent seeds or select using test.

TD-MPC shared validation best0.02886147 vs final0.02863720;latest48k0.01586679. Only these3 archives survive (2 steps). Thus neither a complete learning curve nor a cross-algorithm sample-efficiency claim is identifiable. TD-MPC2 has13 archives (4k through48k plus50k); it has more opportunities for validation selection. Final-vs-final is an important complementary50k-budget comparison.

## 40k–50k TD-MPC2 investigation

| Checkpoint | Shared val reward/step | Cd | Mean abs Cl | Mean commanded action | Historical TRAIN eval reward/step |
|---|---:|---:|---:|---:|---:|
|40k|−0.10844786|3.25224508|0.18435829|−0.01724421|−0.11353599|
|44k|−0.08228433|3.22406237|0.18637748|−0.01757093|−0.07753210|
|48k|−0.32461120|3.27787859|0.37488812|−0.07537061|−0.31586504|
|50k|−0.08769601|3.23028768|0.18556385|−0.01644449|−0.08929228|

Sources: evaluation_recomputed.csv and tdmpc2_training_eval.csv, with original source paths. Historical train evaluations are shown as a separate corroborating protocol, NEVER pooled with val. All10 paired common validation seeds are worse at48k than44k. The0.24232686 reward drop decomposes into +0.05381622 Cd and +0.18851064 absolute lift;77.79% is the lift penalty. This is an arithmetic decomposition of the observed reward, not a learned-mechanism attribution. The command bias and larger absolute lift support a changed control behavior at48k; causal origin remains untested.

Training-window medians over logged final-update records:

| TD-MPC2 diagnostic |40k–44k|44k–48k|48k–50k|
|---|---:|---:|---:|
|Consistency loss|0.00003465|0.00005049|0.00005732|
|Reward CE|0.293680|0.302591|0.297136|
|Value CE|0.309719|0.311232|0.308158|
|Model grad norm|0.253011|0.335495|0.226614|
|Policy objective|0.326399|0.301449|0.298314|

training_windows.csv contains counts,min,max,median/mean and reduction. These are sparse last-update logs, not every update or fixed replay-batch probes. Logged gradient norms remain finite; max in44k–48k window1.72097, below clip20. This does NOT rule out unlogged spikes or model/value errors. Policy entropy/scaled entropy/scale/pi_grad_norm exist; raw Q calibration and replay priorities do not. No cross-algorithm loss-magnitude comparison is valid. Separate historical loss panels already exist in the comparison dashboard. No new simulator or diagnostic outcomes have been invented.

## Differences and confounders

1. **Expected native algorithms:** latent50 vs512;20 vs3 rollout/planning horizon;scalar MSE vs101-bin symlog soft CE;PER vs sequence slices;twin-Q vs ensemble5;discount.99 vs.95;different optimizers/policy objectives. None alone is a bug or causal diagnosis.
2. **FluidGym-specific adaptations:**302 velocity inputs,80 steps,action repeat1,wrappers/time-limit semantics,seed100 and budget gates,original custom episode loop,TD-MPC2 inclusive-loop correction/logging/meta-template compatibility. These require explicit provenance when claiming reproduction.
3. **Confirmed protocol confounders:**100 vs1001 random actions;updates in bursts vs online;different training IC/RNG consumption;3 vs13 saved candidate archives;one training seed;different precision setup and compile behavior. Same50k interactions do not mean matched sample reuse, compute or optimizer work.
4. **Potential implementation/reproducibility errors:** new TD-MPC2 Box action space unseeded in inspected adapter (tested); historical reward_formula metadata incorrect; adapter drops explicit randomize kwargs; not complete historical dirty-tree provenance. No silent fixes applied. Actual shared reset bypasses the latter interface, and actual reward is checked.
5. **Unidentifiable:** why48k drift occurs; actor-only competence;horizon sensitivity;reward/value calibration;latent transition error on candidate rollouts;gradient spikes between logged samples;multi-seed expected performance.

## Executed vs unexecuted checks

Executed:14 tests in lightweight_tests.log; existing_tests.log records additional48 passed,5 skipped,18 subtests with FGRL_TDMPC_SOURCE set (read log for authoritative counts). Exact source pin/patch application checks passed.16 checkpoint archives inspected on CPU; finite tensors.21 existing real-evaluation records and selection provenance checked; scientific originals unchanged (one Finder .DS_Store changed during the audit; see preservation.json). CLI dry-run budget preview completed without CUDA.

Not executed: new real-CFD equivalence, new GPU inference diagnostics, CPU-vs-GPU action equivalence, multi-seed retraining. Existing real common evaluation success is evidence of fresh-process GPU loading from the previous user-run suite, not a newly executed local test. TensorDict gradient/meta regression is skipped locally because TensorDict is absent; Hydra composition regression skipped without Hydra; pinned-checkout location tests can skip because snapshots are outside third_party. Do not convert these skips to passes.

## Next decision

Priority A320-step frozen44k plan-vs-actor validation contrast, with latency, is the smallest high-information next step. Priority C repeats44/48/50k with fixed environment seeds and changed planner seeds. Priority B frozen horizon sensitivity is not a retrained-algorithm comparison. All commands, costs and caveats are in diagnostic_plan.md. Research decision is provisional: retain both methods; TD-MPC2 should not yet be the sole basis for causal dynamics-shift conclusions. One-seed nominal benchmark comparison is valid within the recorded protocol; paper reproduction and mechanism identification remain separate claims.

## Restoration cross-check

Verified config_restoration_checks.json: all20 controlled evaluations reference the same uncontrolled manifest hash and identical full physics objects. TD-MPC restored parameters match saved sidecars; TD-MPC2 differs only in temporary work_dir and seed_steps (1000 during training versus unresolved convenience field during inference). Its act/plan methods do not read seed_steps, so this is not an action-selection override. Saved model size, horizon, samples, iterations, discount inputs and architecture fields are preserved.
