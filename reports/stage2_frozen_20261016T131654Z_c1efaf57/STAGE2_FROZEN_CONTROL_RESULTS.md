# Stage 2 frozen TD-MPC control results

Status: failed

VERIFIED: recorded execution cost (partial on failure):
```json
{
  "attempted_controls": 163,
  "completed_controls": 163,
  "runs_started": 3,
  "runs_completed": 2,
  "control_outer_calls": 4075,
  "reset_outer_calls": 63,
  "control_accepted_increments": 4075,
  "reset_accepted_increments": 63,
  "initial_counter_offsets": 3,
  "observed_solver_entries": 4138,
  "solver_failures": 0,
  "optimizer_updates": 0,
  "worker_wall_seconds": 195.12821842357516
}
```

| Run | VAL domain | Episode return | Pre40 return | Post40 return | Mean Cd | Mean signed Cl | Mean absolute interval Cl | Clipping | Saturation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| e0_p0_S0 | 8 | 2.99027345 | -4.68046274 | 7.67073619 | 3.11356015 | 0.0139433416 | 0.177216953 | 0 | 0.0125 |
| e0_p0_S1 | 8 | -3.35728357 | -8.09095067 | 4.7336671 | 3.14094472 | -0.0241557721 | 0.229176843 | 0 | 0.0125 |

Paired post-shift outcomes (not isolated learned-model errors):
```json
[]
```

Full whole/pre40/post40 action and force metrics: episodes.json. Per-step actions, memory, reward and time: each run steps.json; independent measured boundary tensors: actuator_*.json and solver.jsonl. Successful outer calls and actual accepted counter increments are separate; constructor offsets are excluded from increments. Partial/failing runs may only have steps.json.

INFERRED: gain delivery concerns physical actuator amplitude; changed actions and trajectories describe this frozen controller/planner response. UNRESOLVED: learned-model mismatch, irreducible control-authority loss, optimal achievable reward, history necessity and physical accuracy. Reward improvement or decrease is not itself a gate. Two environment seeds with two planner seeds form a small exploratory cohort; planner repeats are not independent physical initial conditions. No statistical significance or generalization claim.
