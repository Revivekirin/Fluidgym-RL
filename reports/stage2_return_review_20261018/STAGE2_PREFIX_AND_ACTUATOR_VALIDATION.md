# Stage 2 prefix and actuator review

**VERIFIED blocker:** S0/S2 nominal-prefix comparison failed after transition t=2 at observation o3, on native block4 pressure (`native_state3/b4_pressure`). Both conditions still used gain1.

Raw arrays independently confirm maximum absolute difference 3.1620264053344727e-5. At that element, S0=-0.09818213433027267 and S2=-0.09821375459432602; allowed difference under atol=rtol=1e-5 is 1.0982137609971687e-5. Maximum error/tolerance ratio is 2.87924; 722 of 10,680 pressure elements exceed tolerance. RMS difference=5.18934e-6. The spatially varying difference remains 3.10744e-5 after mean subtraction, so a uniform pressure offset alone cannot explain it. No pressure correction was applied.

**VERIFIED chronological observations:** reset flattened-observation max difference=2.38419e-7. Planner RNG hashes before and after each of t0,t1,t2 match exactly. Requested-action differences are 0, 2.98023e-8 and 4.47035e-8, respectively. Warm-state maximum differences remain at most 3.57628e-7. At o3, flattened observation difference=8.34465e-7 and sensor-pressure difference=4.05312e-6, both within the declared tolerance. Full native pressure is the first recorded failed field; the runner stopped there, so later field checks were not completed.

**VERIFIED:** all completed transitions have successful native solver returns and finite-field flags. Recomputed measured boundary readback matches profile×gain×native smoother memory within the fixed actuator tolerance. Online and target weight hashes are unchanged for all three runs. This is a nominal-prefix reproducibility gate failure, not a recorded solver, checkpoint, or gain-delivery failure.

**INFERRED:** small differences can propagate through observation extraction, stochastic planning with matched RNG, changed commands and CFD. The observed sequence is compatible with such numerical propagation, but it does not isolate its initiating mechanism or quantify the causal contribution of each component.

**UNRESOLVED:** initial numerical difference source, pressure sensitivity under these closed-loop commands, and a valid 40-transition nominal prefix under the unchanged contract. A small absolute error does not retroactively satisfy a failed predeclared gate. No tolerance, source, algorithm or artifact was changed.
