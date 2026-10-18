# Frozen TD-MPC control shift design — conditional proposal

작성일 2026-10-16. **Stage 0 설계만 완료. 구현·CFD·추정기 학습 0.** 모든 gain/step/budget는 검토 제안이며 승인된 실행 상수가 아니다. VERIFIED=소스/기존 P1 근거, INFERRED=조건부 설계 판단, UNRESOLVED=미검증. 후속 구현/CFD는 명시적 사용자 승인 후에만 가능하다.


## R4. 조건부 비교

**전제:** R3 pilot에서 adapter nominal equivalence, boundary delivery, numerical validity 및 최소 downstream response를 검토한 후 별도 실행 승인. 지금 실행하지 않는다. TD-MPC2는 보조 비교 후보로만 유지하며 이번 최소 budget에 포함하지 않는다.

| 조건 | Proposed g_t | Horizon |
|---|---|---:|
| S0 | 1 always | 80 |
| S1 | .5 for t=0..79, native reset 끝난 뒤부터 | 80 |
| S2 | 1 for t=0..39, .5 for t=40..79 | 80 |

S2는 o40→o41, 즉 **41번째 action 전이**에서 처음 변경한다. Step40 직전까지 controller는 schedule/gain을 받지 않는다. Env가 action을 받은 후 해당 transition gain을 결정하며 policy에는 clock/schedule side channel을 전달하지 않는다. S0/S2 prefix가 수치적으로 일치하는지 확인한다. 이후 서로 다른 actions는 정상적인 closed-loop feedback 결과다(S3/S17).

**예산 제안:** VAL environment seeds {0,1} × planner seeds {0,1} ×3조건 ×80 = **12 runs, 960 controls**. 각 tuple마다 fresh env/agent reset. 실제 domain IDs와 perturbation states를 기록한다. 동일 env seed의 두 planner seeds는 독립 physical initialization 표본이 아니라 같은 초기조건의 sampling contrast다. Shared TEST benchmark 결과로 보고하지 않는다. 검증 이후 TEST 확장, TD-MPC2, gradual/reversible schedules는 별도 승인 전0 controls. 기본 pilot96+control960을 모두 승인해도 전체 최대 **1,056 controls**이며 이번 Stage0 실행량은0.

## 고정 요소와 기록

TD-MPC frozen `results/tdmpc_full/20261012T060719Z_full/checkpoints/best.pt` (50k), 기존 selection hash/sidecar/config, horizon20 planner와 saved std/discount/samples/elites/iterations, raw velocity302/no normalization, original reward, 80-step episode를 그대로 유지한다. Reward ref를 shifted rollout으로 다시 계산하지 않는다. Neural weights/optimizer updates0. TD-MPC2 44k는 이 예산에 미포함. source S1/S3/S15/S16/S17, checkpoint/P1 S18.

Environment seed와 planner seed는 별도 저장한다. Env reset 완료 후 기존 Policy.reset과 Python/NumPy/Torch CPU/CUDA state 설정으로 planner seed를 고정한다. Episode t0=True 및 native warm-start 경로를 유지하고 shift 때 warm-start를 지우지 않는다. Shift-trigger reset은 별도 algorithm intervention이므로 금지한다. 조건별 같은 planner seed라도 trajectory에 따라 소비/계산 경로와 후보 내용이 달라질 수 있어 모든 후보가 같다고 가정하지 않는다. Per-episode RNG hash와 initial memory, runtime flags를 남긴다.

기존 metric인 per-step reward, episode return, Cd, signed/abs Cl, requested action, runtime applied state, clocks/solver status를 저장한다. Pre40/post40 구간 summaries는 원래 reward를 재정의하지 않은 진단 구간이다. Max/min/전체 paired episode 차이를 제시하고 반복 seeds를 training seeds 또는 독립학습 모델로 계산하지 않는다. 결과로 checkpoint를 reselect하거나 gains를 튜닝하지 않는다.

## 인과 구분과 제한

| 질문 | 관측/대조 | 가능한 해석 | 아직 불가능한 해석 |
|---|---|---|---|
| Authority가 줄었는가 | R3 동일 prescribed action/memory, boundary readback P*g*u | source-defined velocity amplitude effectiveness 감소 | optimal control 성능 한계의 정확한 감소량 |
| Frozen controller가 변화에 맞지 않는가 | S0/S1/S2 paired closed-loop action/reward, prefix 비교 | 현재 nominal-trained controller의 intervention 반응 | reward drop을 authority 감소와 model/planner mismatch로 유일 분해 |
| Planner randomness인가 | 같은 IC·조건의 두 planner seed, P1 고정입력 결과 | 제한된 conditional sampling sensitivity | 두 seeds만으로 전체 분산/유의성 추정 |
| IC dependence인가 | 두 env seeds, actual domain+perturbation 분리 | 해당 초기조건에서의 차이 | basin population/generalization 또는 training-seed superiority |

**INFERRED:** R3는 intervention delivery와 action→observation response를, R4는 frozen closed-loop response를 평가한다. **UNRESOLVED:** authority-vs-controller-mismatch의 분리에는 추가 matched-state/model/oracle 대조가 필요하며 이번에 확정하지 않는다. Reward drop alone은 history conditioning의 필요성/충분성 증거가 아니다. 개선이 없어도 history 신호가 없다는 결론은 아니다. Gain 변화를 알려주는 oracle, inverse-gain command compensation, adaptation은 이 단계에 추가하지 않는다.

## 중단과 비용

960 controls는 nominal P1 throughput 기준 약13분 rollout + reset/compile/logging; 보장 아님. 약24,000 outer calls+reset 별도, adaptive accepted steps 별도. 승인 시 watchdog 제안45분. 소스/hash/config drift, boundary/solver/observation contract 실패, prefix가 declared repeat envelope를 벗어남, frozen weights 변화, budget 초과 시 중단하고 다음 run을 시작하지 않는다. Reward가 나빠지거나 |a|가 합법적 범위 안에서 saturation되는 것 자체는 실패 사유가 아니다. Effect가 작아도 조건 수를 늘리지 않는다.

**R5 conditional extensions:** gradual/reversibility는 latency/recovery 질문에만 의미가 있고 abrupt feasibility 결과를 본 뒤 새 hypothesis와 confound를 검토해야 한다(S3/S8/S17). 현재 budget0, 승인 없으면 stop. 특히 S2에서 gain 복구 시 memory/flow가 이미 변했으므로 recovery를 initial-state 복원이라고 부르면 안 된다.


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
