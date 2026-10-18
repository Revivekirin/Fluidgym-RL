# P1_FROZEN_POLICY_RNG

Scope: current-pinned-environment verification. Historical binary equivalence is UNRESOLVED.

## A_tdmpc: VERIFIED — passed

```json
{
  "status": "passed",
  "algorithm": "tdmpc",
  "control_steps": 0,
  "planner_calls": 22,
  "observations": 2,
  "frozen_model_hash": "1906aa73b83aeb86c3ec6eb626640a0ec49b099da734043c43113297c2677fa4",
  "contrasts": [
    {
      "observation": 0,
      "contrast": "same_rng_reset_state",
      "required_to_pass": true,
      "shape_equal": true,
      "max_abs": 0.0,
      "allclose": true,
      "bitwise_equal": true
    },
    {
      "observation": 0,
      "contrast": "changed_rng_reset_state",
      "required_to_pass": false,
      "shape_equal": true,
      "max_abs": 0.06956702470779419,
      "allclose": false,
      "bitwise_equal": false
    },
    {
      "observation": 0,
      "contrast": "same_rng_same_warm",
      "required_to_pass": true,
      "shape_equal": true,
      "max_abs": 0.0,
      "allclose": true,
      "bitwise_equal": true
    },
    {
      "observation": 0,
      "contrast": "changed_rng_same_warm",
      "required_to_pass": false,
      "shape_equal": true,
      "max_abs": 0.028591692447662354,
      "allclose": false,
      "bitwise_equal": false
    },
    {
      "observation": 0,
      "contrast": "same_rng_changed_valid_warm",
      "required_to_pass": false,
      "shape_equal": true,
      "max_abs": 0.0003319978713989258,
      "allclose": false,
      "bitwise_equal": false
    },
    {
      "observation": 1,
      "contrast": "same_rng_reset_state",
      "required_to_pass": true,
      "shape_equal": true,
      "max_abs": 0.0,
      "allclose": true,
      "bitwise_equal": true
    },
    {
      "observation": 1,
      "contrast": "changed_rng_reset_state",
      "required_to_pass": false,
      "shape_equal": true,
      "max_abs": 0.0526452362537384,
      "allclose": false,
      "bitwise_equal": false
    },
    {
      "observation": 1,
      "contrast": "same_rng_same_warm",
      "required_to_pass": true,
      "shape_equal": true,
      "max_abs": 0.0,
      "allclose": true,
      "bitwise_equal": true
    },
    {
      "observation": 1,
      "contrast": "changed_rng_same_warm",
      "required_to_pass": false,
      "shape_equal": true,
      "max_abs": 0.04451659321784973,
      "allclose": false,
      "bitwise_equal": false
    },
    {
      "observation": 1,
      "contrast": "same_rng_changed_valid_warm",
      "required_to_pass": false,
      "shape_equal": true,
      "max_abs": 0.051204293966293335,
      "allclose": false,
      "bitwise_equal": false
    }
  ],
  "rng_sources": [
    "Python global",
    "NumPy legacy global",
    "Torch CPU",
    "Torch CUDA all devices"
  ],
  "limitations": [
    "Gaussian candidates/prior/Q subsampling/elite selection share RNG streams; not independently varied",
    "Compiled planner intermediate candidates not exposed; no hooks or graph changes used",
    "Different RNG or memory need not change the selected action",
    "Finite outputs and repeatability do not establish prediction accuracy"
  ],
  "wall_seconds": 10.14728597458452,
  "peak_torch_cuda_allocated_bytes": 25067520
}
```

## A_tdmpc2: VERIFIED — passed

```json
{
  "status": "passed",
  "algorithm": "tdmpc2",
  "control_steps": 0,
  "planner_calls": 22,
  "observations": 2,
  "frozen_model_hash": "48ef11645964ec94df7620cf7686fccf41abf159bd451a21a483ba31420fd59d",
  "contrasts": [
    {
      "observation": 0,
      "contrast": "same_rng_reset_state",
      "required_to_pass": true,
      "shape_equal": true,
      "max_abs": 0.0,
      "allclose": true,
      "bitwise_equal": true
    },
    {
      "observation": 0,
      "contrast": "changed_rng_reset_state",
      "required_to_pass": false,
      "shape_equal": true,
      "max_abs": 0.07983773946762085,
      "allclose": false,
      "bitwise_equal": false
    },
    {
      "observation": 0,
      "contrast": "same_rng_same_warm",
      "required_to_pass": true,
      "shape_equal": true,
      "max_abs": 0.0,
      "allclose": true,
      "bitwise_equal": true
    },
    {
      "observation": 0,
      "contrast": "changed_rng_same_warm",
      "required_to_pass": false,
      "shape_equal": true,
      "max_abs": 0.07039332389831543,
      "allclose": false,
      "bitwise_equal": false
    },
    {
      "observation": 0,
      "contrast": "same_rng_changed_valid_warm",
      "required_to_pass": false,
      "shape_equal": true,
      "max_abs": 0.05613356828689575,
      "allclose": false,
      "bitwise_equal": false
    },
    {
      "observation": 1,
      "contrast": "same_rng_reset_state",
      "required_to_pass": true,
      "shape_equal": true,
      "max_abs": 0.0,
      "allclose": true,
      "bitwise_equal": true
    },
    {
      "observation": 1,
      "contrast": "changed_rng_reset_state",
      "required_to_pass": false,
      "shape_equal": true,
      "max_abs": 0.0341930091381073,
      "allclose": false,
      "bitwise_equal": false
    },
    {
      "observation": 1,
      "contrast": "same_rng_same_warm",
      "required_to_pass": true,
      "shape_equal": true,
      "max_abs": 0.0,
      "allclose": true,
      "bitwise_equal": true
    },
    {
      "observation": 1,
      "contrast": "changed_rng_same_warm",
      "required_to_pass": false,
      "shape_equal": true,
      "max_abs": 0.003296464681625366,
      "allclose": false,
      "bitwise_equal": false
    },
    {
      "observation": 1,
      "contrast": "same_rng_changed_valid_warm",
      "required_to_pass": false,
      "shape_equal": true,
      "max_abs": 0.019628018140792847,
      "allclose": false,
      "bitwise_equal": false
    }
  ],
  "rng_sources": [
    "Python global",
    "NumPy legacy global",
    "Torch CPU",
    "Torch CUDA all devices"
  ],
  "limitations": [
    "Gaussian candidates/prior/Q subsampling/elite selection share RNG streams; not independently varied",
    "Compiled planner intermediate candidates not exposed; no hooks or graph changes used",
    "Different RNG or memory need not change the selected action",
    "Finite outputs and repeatability do not establish prediction accuracy"
  ],
  "wall_seconds": 25.859337966889143,
  "peak_torch_cuda_allocated_bytes": 66411520
}
```


Numerical comparison table: [A_tdmpc/contrasts.csv](A_tdmpc/contrasts.csv).

Numerical comparison table: [A_tdmpc2/contrasts.csv](A_tdmpc2/contrasts.csv).
INFERRED: differences describe this fixed runtime and the tested initializations only.
UNRESOLVED: physical accuracy, representation robustness and dynamics-shift sensitivity.
Numerical gate thresholds were declared before execution; failed/unrun stages are not passes.
