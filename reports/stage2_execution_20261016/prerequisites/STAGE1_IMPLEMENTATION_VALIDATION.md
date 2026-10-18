# Stage 1 returned implementation validation

**VERIFIED:** Returned output hashes: 784/784 match. Runtime package manifest matches the prepared Stage 1 package. FluidGym 72-source fingerprint and PISOtorch backend match the pinned reference; returned post-preservation checks are consistent. Offline review used 36,308 assertions; no additional CFD or model inference was run.

**VERIFIED:** All 2,400 control outer calls have successful native returns, finite-field flags, correct gain timing and measured boundary delivery. Recomputed smoother and boundary-profile residual maxima are both 0. Boundary readback persists through the solver. Maximum observed domain flux imbalance is 4.76837158203125e-7, below the unchanged native tolerance 1e-5. Measured boundary tensors remain distinct from source-derived expected values; C++ setter arguments were not directly intercepted.

**VERIFIED:** Native memory remains identical across gain treatments under the common prescribed command. N-bare/N-hook and within-condition repeats have maximum raw observation difference 2.384185791015625e-7. Compared exposed native states, reward, drag/lift, clocks and counters agree exactly. J/N exposed state and observations agree through o6 within the declared tolerances.

**UNRESOLVED:** Unexposed C++ caches/full-state replay, constructor divergence-correction internal success, and physical accuracy. These numerical checks do not constitute a physical validation.

Evidence: verification.json; original per-run actuator JSON, state/observation NPZ and solver journals in the returned result directory. Original files were preserved unchanged.
