# Physical dynamics shift preparation

Status: **nominal architectural preparation only; physical sweeps are NOT READY**.
Original TD-MPC is the immediate research target. TD-MPC2 and SB3 SAC remain
independent baselines. No new objective, adaptation, diagnostic or shift is implemented.

## Boundaries after this change

| Responsibility | Owner |
|---|---|
| Architecture, objectives, planner, optimizer, replay | Pinned original TD-MPC `src/algorithm/{tdmpc,helper}.py`, loaded by `tdmpc/runtime.py`; unchanged device patch |
| Algorithm parameters and their provenance | Existing `configs/tdmpc/cylinderjet2d_easy.yaml`; unchanged bytes and resolved parameters |
| Environment identity, interface expectations, nominal physics validation | `environment_contract.py`; independently serializable `configs/environments/cylinderjet2d_nominal.json` |
| Native construction, flattening, split selection | Existing `envs.make_fluid_env`; unchanged |
| NumPy/Torch conversion, action clipping, boundary checks | Existing `tdmpc.environment.CheckedEnv`; only physical checks delegated |
| Training budget, seed/reset sequence, cadence, validation selection | `tdmpc/runner.py` and entry point; unchanged |
| Episode protocol, aggregates, independent comparison and selection | `evalcontract.py`, `comparison/*`; unchanged |
| Clone integrity, checkpoint loading, logging, safety | `upstream.py`, algorithm runtimes/evaluation, tracking modules, `safety.py`; unchanged except TD-MPC config composition |

The new physics mapping describes **expected values**. It is not a dictionary of
kwargs to forward to the simulator. The only accepted specification is the
historical nominal one. Unknown fields, changed physics, malformed types and
misplaced configuration keys fail before simulator creation. The original TD-MPC
factory now rejects arbitrary constructor overrides, which have no retained callers.
The generic factory and TD-MPC2/SAC interfaces are untouched.

The legacy algorithm namespace still includes task, horizon and I/O dimensions
because upstream code and checkpoint sidecars require them. These are checked
against the separate environment specification; no physics/experiment fields are
added to the agent namespace. Existing saved resolved configs with `tracking` remain
accepted. Splits and reset seeds remain protocol-owned, not physics settings.

## Configuration and CLI compatibility

Existing usage remains valid:

```bash
python scripts/tdmpc_train.py --print-config
```

Explicit composition, with exactly the same printed legacy configuration:

```bash
python scripts/tdmpc_train.py \
  --config configs/tdmpc/cylinderjet2d_easy.yaml \
  --environment-config configs/environments/cylinderjet2d_nominal.json \
  --print-config
```

A future contract version may support another environment without adding physical
knobs to TD-MPC. This version deliberately rejects a changed Reynolds number,
viscosity, smoothing, reward penalty, flattening or episode horizon. Missing
`--environment-config` resolves to nominal in code, so historical invocation and
checkpoint restoration do not acquire a new file dependency. JSON scalar types are
strict: use the provided float values for the physics fields.

New TD-MPC run manifests add `environment_spec` and the contract-module source hash.
The historical `environment_contract` dictionary is unchanged and re-exported at
its previous import path. Saved model weights, parameter/provenance dictionaries,
checkpoint sidecars' existing fields and inference probes are unchanged. There is
no exact-resume capability: optimizer, replay and RNG states remain absent.

## Verified upstream mechanism (source inspection, not a shift execution)

Inspected clean FluidGym revision `50071140b73a2a917fb96d33522a769f903f8c27`
from the local checkout `/tmp/fgrl-original-fluidgym` and archived audit sources.
The source root below is `src/fluidgym/` in that repository:

- `registry.py:EnvRegistry.make`: merges registered defaults with supplied kwargs,
  then calls the registered class. `reynolds_number` really is accepted here.
- `__init__.py`: easy CylinderJet2D registers Re=100 and resolution=24; medium
  registers Re=250 and resolution=32. Changing environment difficulty would also
  change resolution, so it is not an isolated Reynolds sweep.
- `envs/cylinder/jet_cylinder_env_2d.py:__init__`: explicit `reynolds_number`,
  `resolution`, `adaptive_cfl`, `dt`, `step_length`, `episode_length`, `lift_penalty`,
  device and initial-domain flags. There is no `viscosity` or smoothing-alpha
  constructor argument.
- `envs/cylinder/cylinder_env_base.py`: constructor computes `_viscosity` as
  `_U_mean / _reynolds_number`. `initial_domain_id` includes integer Reynolds number
  and angular resolution; noninteger Reynolds values can therefore alias an ID.
- `envs/fluid_env.py`: initialization prepares datasets using this ID, reset loads
  the selected train/val/test domain, `_load_initial_domain` deserializes a solver
  domain and calls `PrepareSolve()`. Changing a Python attribute does not establish
  that the loaded solver domain has the desired physical viscosity.
- `envs/cylinder/cylinder_env_base.py:_step_impl`: reward is
  `Cd_ref - mean_substep(Cd) - lift_penalty * abs(mean_substep(Cl))` in 2D.
  `_cd_ref` comes from loaded domain statistics. Changing Reynolds/domain data may
  change that reference; reward comparability needs an explicit decision.

No upstream file, pin, patch, PDE solver, reward or initial-domain file was changed.
The extracted validator still checks the same four observed attributes with the
same `numpy.isclose` defaults. It does **not** newly certify dt, CFL, resolution,
solver-domain viscosity or dataset suitability. Synthetic test doubles without
`cuda_device` retain the previous interface-only path. This is documented rather
than silently widening historical physical claims.

## Remaining blockers for a controlled sweep

1. Choose a single physical intervention and a justified supported creation/reset
   mechanism. Verify the effective loaded solver viscosity, grid and boundary
   conditions, rather than trusting a Python Reynolds label.
2. Identify/generate matched initial domains and statistics with explicit hashes,
   availability, split definitions and non-aliasing IDs. Do not silently download or
   regenerate them during an experiment.
3. Decide whether the research holds initial conditions, resolution, control
   interval and reward reference fixed. Distinguish physics effects from dataset,
   discretization or reward-reference effects.
4. Define train dynamics vs evaluation dynamics in a versioned experiment protocol;
   select checkpoints on validation only, freeze test conditions independently.
5. Extend the comparison cohort and manifests explicitly for shifted physics. The
   existing `NominalEnv`, 80-step aggregate checks and nominal loader guards are
   intentionally not relaxed here.
6. Validate actual CUDA same-seed/same-action nominal equivalence before execution;
   repeat for any authorized shifted creation path, with numerical tolerances and
   solver/data provenance. Local CPU mocks are insufficient evidence.
7. Historical original TD-MPC keeps rolling `latest.pt`, `best.pt`, `final.pt`;
   overwritten earlier checkpoints cannot be recreated without retraining. Do not
   invent historical curve points or claim full temporal latent-model evidence.

No remote command was run. For the existing remote runtime, the environment-only
validation command is below (one nominal episode; no learner updates):

```bash
python scripts/tdmpc_train.py --mode validate --device cuda:0 \
  --environment-config configs/environments/cylinderjet2d_nominal.json
```

Existing bounded learning smoke remains available as a separately authorized next
step (`--mode smoke --smoke-steps 240 --device cuda:0`); it was **NOT RUN** here.
Do not run full training or physical sweeps as part of this preparation.
