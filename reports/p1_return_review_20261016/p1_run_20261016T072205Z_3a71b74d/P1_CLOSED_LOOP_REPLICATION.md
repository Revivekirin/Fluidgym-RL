# P1_CLOSED_LOOP_REPLICATION

Scope: current-pinned-environment verification. Historical binary equivalence is UNRESOLVED.

## C_tdmpc: VERIFIED — passed

```json
{
  "status": "passed",
  "algorithm": "tdmpc",
  "control_steps": 800,
  "attempted_control_steps": 800,
  "solver_calls_including_resets": 20199,
  "solver_failures": 0,
  "mean_reward_per_step": 0.037778737819753584,
  "interpretation": "contract/restoration/completion passed; reward difference is not a failure or robustness result",
  "wall_seconds": 662.2734932536259,
  "peak_torch_cuda_allocated_bytes": 30677504
}
```

## C_tdmpc2: VERIFIED — passed

```json
{
  "status": "passed",
  "algorithm": "tdmpc2",
  "control_steps": 800,
  "attempted_control_steps": 800,
  "solver_calls_including_resets": 20199,
  "solver_failures": 0,
  "mean_reward_per_step": -0.07674576454097407,
  "interpretation": "contract/restoration/completion passed; reward difference is not a failure or robustness result",
  "wall_seconds": 661.0049646273255,
  "peak_torch_cuda_allocated_bytes": 66411520
}
```


Numerical comparison table: [C_tdmpc/historical_comparison.csv](C_tdmpc/historical_comparison.csv).
VERIFIED: reward/step=0.037778738; Cd=3.110115987; abs Cl=0.180260793.
Per-episode results: [C_tdmpc/episodes.csv](C_tdmpc/episodes.csv). Initial-domain identities: [C_tdmpc/initializations.json](C_tdmpc/initializations.json).

Numerical comparison table: [C_tdmpc2/historical_comparison.csv](C_tdmpc2/historical_comparison.csv).
VERIFIED: reward/step=-0.076745765; Cd=3.220350184; abs Cl=0.184551099.
Per-episode results: [C_tdmpc2/episodes.csv](C_tdmpc2/episodes.csv). Initial-domain identities: [C_tdmpc2/initializations.json](C_tdmpc2/initializations.json).
INFERRED: differences describe this fixed runtime and the tested initializations only.
UNRESOLVED: physical accuracy, representation robustness and dynamics-shift sensitivity.
Numerical gate thresholds were declared before execution; failed/unrun stages are not passes.
