# Stage 1 implementation validation — local, no CFD

**VERIFIED:** An opt-in, instance-scoped post-smoothing adapter and bounded pilot launcher have been implemented. No production training/evaluation module, pinned FluidGym source, checkpoint or prior result was edited for this work. Local source/artifact preservation and test details are in `local_validation.json`.

**VERIFIED (CPU software fixtures only):** gain=1 bare/hook equality; unchanged requested command; unchanged smoother memory across gain jump; reset gain=1/zero memory/fresh boundary references; isolation and detach restoration; invalid schedule rejection; native failure latch; actual-readback mismatch rejection; profile mutation/stale-reference rejection; nonfinite/flux rejection; fixed tolerance failure records; 8-run/96-control cap; nominal gate blocks F/J; external watchdog kills at 600 seconds without retry. These are not real CFD observations.

**INFERRED:** The reviewed source seam and readback/flux APIs support the intended native integration. Runtime import/signature/hash and physics gates fail closed. The adapter forwards `g*u` only after native smoothing; it does not scale the requested command or alter profiles. `N-bare` does not patch `_apply_action`, while sharing read-only numerical instrumentation with other arms.

**UNRESOLVED:** Native CUDA execution, actual gain=1 numerical equivalence, runtime boundary setters/readback, full-state reset/prefix agreement and downstream physical response. Those gates execute on the server before any shifted arm. Actual C++ setter arguments are not intercepted; measured boundary tensors and expected profile products are labeled separately. Constructor divergence correction is not independently instrumented; exceptions and subsequent finite fields are checked. Unexposed caches and full-state replay remain unresolved.

See `README.md` for the exact bounded user-executed command. No remote execution was performed by the authoring agent.
