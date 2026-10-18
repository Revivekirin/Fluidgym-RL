# Stage 2 returned results — incomplete

**VERIFIED:** 1,349 returned output hashes match; execution package identity matches the prepared package. All 221 returned protected-artifact/source checks match. Offline review added zero CFD and zero model inference.

Completed 163 of 960 authorized controls: e0/p0 S0=80, S1=80, S2=3. Only 3 of 12 runs started and 2 completed. Launcher time 198.33 s (3 min 18 s); worker 195.13 s. No watchdog expiry, automatic retry, optimizer update or native solver failure. Successful control outer calls/accepted increments=4,075; reset calls/increments=63; total=4,138. Three initial counter offsets are recorded separately.

| Run | Controls | Return (partial if incomplete) |
|---|---:|---:|
| e0_p0_S0 | 80 | 2.990273453 |
| e0_p0_S1 | 80 | -3.357283570 |
| e0_p0_S2 | 3 | -1.944624580 |

All use VAL initial domain 8, environment seed0/planner seed0. The S1–S0 return difference is -6.347557023 for this single pair; it is descriptive only. Original whole/pre40/post40 force/action metrics for the two completed runs remain in the returned episodes.json.

**UNRESOLVED:** S2 post-shift response and Delta_post. S2 never reached transition40; no reduced-gain S2 transition ran. The full seed matrix is incomplete. Do not report these results as a completed Stage 2 pilot or attribute the early stop to shifted physics, model error, or controller failure.
