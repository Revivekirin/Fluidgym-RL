# Paired transition feasibility pilot — review proposal

작성일 2026-10-16. **Stage 0 설계만 완료. 구현·CFD·추정기 학습 0.** 모든 gain/step/budget는 검토 제안이며 승인된 실행 상수가 아니다. VERIFIED=소스/기존 P1 근거, INFERRED=조건부 설계 판단, UNRESOLVED=미검증. 후속 구현/CFD는 명시적 사용자 승인 후에만 가능하다.


## R3. 최소 검토 matrix

**INFERRED 제안, 실행 승인 아님:** VAL split, seed=0, randomize=True, 같은 saved-domain bytes를 사용해 모든 run을 새 environment/reset으로 시작한다. 실제 index와 perturbation RNG를 기록한다. P1 TEST 결과를 새 학습 데이터로 재사용하지 않는다(S6/S7/S18).

| 조건 | 적용 | 반복 | Control steps/run |
|---|---|---:|---:|
| N-bare | 원본 actuator, gain hook 없음 | 2 | 12 |
| N-hook | 제안 adapter, g=1 항상 | 2 | 12 |
| F | g=.5, t=0부터 고정 | 2 | 12 |
| J | g=1 for t=0..5; g=.5 for t=6..11 | 2 | 12 |

합계 **8 runs ×12=96 controls**, 최대 2,400 outer solver calls + reset calls 별도. Horizon은 원본 80으로 유지하고 검증자는 12번째 전이 뒤 멈춘다. terminated=False/truncated=False가 정상이며 이것을 12-step episode truncation으로 재정의하지 않는다. 모든 조건의 command sequence는 `[0,.2,.2,-.2,-.2,.2,.2,-.2,-.2,.2,.2,0]`. g=.5와 amplitude .2는 리뷰 가능한 최초 후보이지 검증된 안전/식별 가능 상수가 아니다.

Shift는 **o6를 얻은 뒤, a6=+.2로 진행하는 일곱 번째 전이의 첫 boundary assignment**에서 시작한다. a5=a6=+.2라 그 순간 command jump와 gain jump를 겹치지 않는다. Nominal/J의 o0..o6 및 내부 prefix 상태가 비교 대상이다. Fixed F는 o0만 공통이며 이미 이후 상태가 달라졌으므로 F/N의 후기 관측 차이를 같은-state one-step effect라고 부르면 안 된다. 근거 S3/S4, 인과 목적: memory 이후 amplitude 개입을 분리.

## 실행 gate와 로그: 아직 구현하지 않음

1. **시뮬레이션 전:** source/backend/data/config hashes, boundary class/readback API, profile normalization과 지원 bounds 확인. Native source를 수정하거나 initial domains를 재생성하지 않음.
2. **각 reset 후:** 실제 domain ID/index/split 및 JSON/NPZ hash, env/private NumPy/Torch RNG states, CPU/CUDA global RNG, native blocks u/p, vertex geometry/transforms, viscosities, boundary tensors/types, solver clock/counter, native u_memory=0 확인. Bare/hook nominal은 reset 직후 값도 비교한다.
3. **각 outer call:** requested raw a, clipped a, source-derived expected u, runtime u, g, expected P_top/bottom*(g*u), actual setter arguments, `.velocity` 또는 `getVelocity(False)` readback을 별개 필드로 기록. Boundary-normal flux 및 domain net-flux readback은 좌표/부호 정의를 함께 기록한다. Profile hashes가 변하지 않는지 확인. Gain이 다음 call에 upstream에 의해 덮여 없어지는지 검사한다.
4. **각 control transition:** raw velocity/pressure, flat velocity input, reward/Cd/Cl, termination/truncation, requested dt, outer-call count, accepted-step counter, reset/전이 전후 raw solver clocks, 모든 success flags 및 native fields finite 여부 기록.
5. **전이 6 시작 직전:** J/N의 u_memory, available domain/solver/RNG state를 대조한 뒤 gain만 전환. 최초 boundary readback 차이가 .5 비율인지 검증한다. u가 거의0인 위치/시간의 ratio는 사용하지 않고 absolute tensor residual을 사용한다.

**Full-state identity 요구와 한계:** domain u/p만 같거나 obs가 같다고 full state 같다고 하지 않는다. Native accessible domain tensors(geometry, boundary, flux/history buffers 포함 여부를 inventory로 확인), solver control/counters/clock, action-memory, RNG, adapter counter까지 identity manifest를 요구한다. Exact hash와 tolerance 차이를 둘 다 기록한다. C++ hidden buffers/solver caches를 열거·검증하지 못하면 **full-state identity=UNRESOLVED**로 남긴다. 이 경우 허용되는 해석은 deterministic matched-reset/matched-prefix comparison이며 true branching/counterfactual replay가 아니다. Full-state replay가 꼭 필요한 연구 주장으로 진행하려면 중단하고 별도 복원 검증을 요청한다(S9/S10/S14).

## 수치 tolerance와 식별 gate

**VERIFIED 출발 근거:** P1의 관측 반복 최대차는 2.3842e-7, scalar/clock/counter 차이0(S18). **INFERRED 제안:** 동일 조건 반복과 gain1/bare 비교는 float arrays/reward/Cd/Cl atol=rtol=1e-5, counter/shape/split/ID/config exact, 반복 clock atol=1e-6/rtol=0. Actuator/boundary expected-vs-measured는 atol=1e-7/rtol=1e-6. Tolerances는 float32 repeatability용이며 physical accuracy 기준이 아니다. Native zero entries, signed flux cancellation은 absolute residual도 함께 보고한다. 결과를 보고 tolerance를 넓히지 않는다.

**Intervention relevance:** boundary gain relation이 맞으면 물리 입력 intervention delivery를 확인할 수 있다. O/reward 차이가 없다고 implementation 실패는 아니다. 관측가능성의 최소 운영 기준 후보는 미리 고정한 flat-velocity RMS 및 Cd/Cl 절대차 중 하나가, 같은 시점 nominal repeat의 해당 error와 float32 acceptance floor 중 큰 값의 **5배**를 연속 **2 transitions** 초과하는 것. Raw 값/전 시점을 모두 보고하며 이 기준은 통계 검정·역학 파라미터 식별 증명이 아니다. 통과하지 않으면 history study용 identifiability는 NO-GO/UNRESOLVED로 남기고 amplitude/gain/horizon을 몰래 늘리지 않는다. Ground-truth boundary scaling만 확인된 결과와 downstream sensor response가 분리된 결과를 구분한다.

## 예산·중단

Stage 0 실제 비용0. 후속 승인 시 위96 controls만; P1 약 .8s/control을 기준으로 rollout 부분 약77s이나 reset/readback/full-state audit·startup 포함 총시간은 미확정. 성공한 native outer calls는 최대2,400+reset calls; adaptive accepted steps는 cap로 오인하지 않는다. 별도 watchdog 제안10분, 초과 시 실패/미완료 보존, 자동 재시도 없음. N-bare/N-hook equivalence 실패 시 F/J를 시작하지 않는다.

False solver return, exception, native field nonfinite, flux imbalance, profile mutation, reset memory/gain leakage, timing misalignment, measured boundary mismatch, 반복 tolerance 초과 시 중단. Full readback 불가면 boundary-effectiveness 검증을 PASS로 표기하지 않는다. 소스 S1–S16 및 P1 S18이 이 gate의 근거다. 실패/불충분 시 frozen control 단계는 자동 실행하지 않는다.


## 소스 증거 인덱스

- **S1** [CheckedEnv.step: validation, clipping, device](/Users/jihyekim/Desktop/Fluidgym-RL/src/fluidgym_rl/tdmpc/environment.py:48)
- **S2** [FluidEnv.step: shape/boundary and step counter](/Users/jihyekim/Desktop/Fluidgym-Physics/third_party/fluidgym/src/fluidgym/envs/fluid_env.py:772)
- **S3** [Cylinder step: smoothing, actuation, solver, observations/reward](/Users/jihyekim/Desktop/Fluidgym-Physics/third_party/fluidgym/src/fluidgym/envs/cylinder/cylinder_env_base.py:741)
- **S4** [Jet profile initialization and boundary assignment](/Users/jihyekim/Desktop/Fluidgym-Physics/third_party/fluidgym/src/fluidgym/envs/cylinder/jet_cylinder_env_2d.py:130)
- **S5** [Jet peak normalization](/Users/jihyekim/Desktop/Fluidgym-Physics/third_party/fluidgym/src/fluidgym/envs/util/profiles.py:6)
- **S6** [Initial-domain selection and initialization](/Users/jihyekim/Desktop/Fluidgym-Physics/third_party/fluidgym/src/fluidgym/envs/fluid_env.py:507)
- **S7** [Reset and zero-action assignment](/Users/jihyekim/Desktop/Fluidgym-Physics/third_party/fluidgym/src/fluidgym/envs/fluid_env.py:862)
- **S8** [Boundary references, memory reset, randomization](/Users/jihyekim/Desktop/Fluidgym-Physics/third_party/fluidgym/src/fluidgym/envs/cylinder/cylinder_env_base.py:334)
- **S9** [Cylinder state API and actuator memory](/Users/jihyekim/Desktop/Fluidgym-Physics/third_party/fluidgym/src/fluidgym/envs/cylinder/cylinder_env_base.py:847)
- **S10** [Domain clone and simulator reconstruction](/Users/jihyekim/Desktop/Fluidgym-Physics/third_party/fluidgym/src/fluidgym/envs/fluid_env.py:1320)
- **S11** [Boundary velocity/readback/flux Python bindings](/Users/jihyekim/Desktop/Fluidgym-Physics/third_party/fluidgym/src/fluidgym/simulation/extensions/PISOtorch.cpp:105)
- **S12** [C++ setter validation and coordinate transforms](/Users/jihyekim/Desktop/Fluidgym-Physics/third_party/fluidgym/src/fluidgym/simulation/extensions/domain_structs.cpp:437)
- **S13** [Solver flux check and failure return](/Users/jihyekim/Desktop/Fluidgym-Physics/third_party/fluidgym/src/fluidgym/simulation/simulation.py:210)
- **S14** [Outlet PRE callback and adaptive simulation](/Users/jihyekim/Desktop/Fluidgym-Physics/third_party/fluidgym/src/fluidgym/envs/cylinder/cylinder_env_base.py:273)
- **S15** [Raw sensor extraction](/Users/jihyekim/Desktop/Fluidgym-Physics/third_party/fluidgym/src/fluidgym/envs/util/obs_extraction.py:28)
- **S16** [Official flatten operator](/Users/jihyekim/Desktop/Fluidgym-Physics/third_party/fluidgym/src/fluidgym/wrappers/flatten_obs.py:9)
- **S17** [Frozen policy reset/act](/Users/jihyekim/Desktop/Fluidgym-RL/src/fluidgym_rl/tdmpc/evaluation.py:10)
- **S18** [Completed P1 verification](/Users/jihyekim/Desktop/Fluidgym-RL/reports/p1_return_review_20261016/P0_P1_FINAL_ASSESSMENT.md:1)
