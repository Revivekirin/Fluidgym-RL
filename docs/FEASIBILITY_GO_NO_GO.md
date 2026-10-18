# Stage 0 feasibility — conditional go / no-go

작성일 2026-10-16. **Stage 0 설계만 완료. 구현·CFD·추정기 학습 0.** 모든 gain/step/budget는 검토 제안이며 승인된 실행 상수가 아니다. VERIFIED=소스/기존 P1 근거, INFERRED=조건부 설계 판단, UNRESOLVED=미검증. 후속 구현/CFD는 명시적 사용자 승인 후에만 가능하다.


## 결론

**INFERRED: 구현 검토에 conditional GO. 실행은 아직 NO-GO(명시적 승인 대기).** Post-smoothing boundary effectiveness는 원본 source를 바꾸지 않고 private actuation seam과 original boundary setter를 통해 구현할 수 있는 구체적 근거가 있다. 단, 검증되지 않은 intervention을 지금 “물리적으로 실현/식별 가능”으로 선언하지 않는다. History가 유용하다는 가설도 미결정이다.

## VERIFIED / INFERRED / UNRESOLVED

| 항목 | 판정 |
|---|---|
| Nominal baseline | VERIFIED: P0/P1 완료. TD-MPC50k/TD-MPC2 44k frozen loads, nominal control 경로 검증 |
| Post-smoothing 삽입점 | VERIFIED source: memory update 후 _apply_action, profile×scalar를 양 boundary에 매 outer call 설정 |
| Boundary readback | VERIFIED source binding: .velocity/getVelocity(False)/GetFluxes. 이번 gain runtime 결과는 UNRESOLVED |
| Sign/normalization | VERIFIED: peak-normalized profile, 두 boundary 같은 scalar. INFERRED: opposite physical blowing/suction; discrete balance는 runtime gate |
| Minimal adapter | INFERRED: instance-scoped forwarding hook, official smoother/solver/reward 그대로. Private API version dependency 있음 |
| Full-state replay | UNRESOLVED: clone API만으로 clocks/RNG/caches 복원 안 됨. 계획에서 사용하지 않음 |
| Same-prefix transition effect | INFERRED: deterministic fresh-reset/prefix를 검증하면 제한된 paired comparison 가능. Full counterfactual clone 주장 금지 |
| Gain identifiability | UNRESOLVED: excitation·sensor sensitivity·noise/phase confound 검증 필요 |
| History conditioning 필요성 | UNRESOLVED: reward drop이나 actuator memory 존재만으로 필요성 입증 안 됨 |

## 추천별 evidence / 인과 목적 / 예산 / 중단

| 제안 | Source evidence | 인과 목적 | 후속 최대 예산(제안) | Stop |
|---|---|---|---|---|
| R1 control-index schedule | S2/S3/S13 | shift가 adaptive dt와 섞이지 않음 | 별도0, R3 내 포함 | outer-call/step boundary 불일치 |
| R2 post-smoothing hook | S3/S4/S11/S12 | command/memory와 boundary authority 분리 | 구현0 CFD, R3에 포함 | gain1 mismatch/readback 불가/reset leakage |
| R3 paired pilot | S6–S10/S13/S18 | delivery와 downstream response 및 repeat noise 구분 | 8×12=96 controls; 10min watchdog 제안 | solver/contract/repeat failure; signal 미분리면 확장 보류 |
| R4 frozen TD-MPC | S1/S3/S15–S18 | 확인된 intervention에서 fixed controller 반응 | 12×80=960 controls; 45min watchdog 제안 | pilot 미통과/승인 없음/config·source·prefix failure |
| R5 gradual/reversible | S3/S8/S17 | 변화속도/회복 질문의 조건부 확장 | 현재0 | 별도 가설/승인 전 진행 금지 |
| R6 scoped schema | S3/S11/S15/S16/S18 | ground truth와 actor input·측정/추론 분리 | 새 CFD0; R3/R4 내 logging만 | hidden label/clock 유출 또는 missing data 위장 |
| R7 estimation baselines | S3/S15–S18 | observation vs action/history 추가 정보 평가 | Stage0 fit0/CFD0; 향후 별도 검토 | group 부족/leakage/future information |

Total Stage0: **0 controls, 0 solver runs, 0 estimator fits, intervention 구현0**. Stage1 후보 R3만96 controls. R4까지 모두 별도 승인될 때만1,056 controls. Reset/warmup solver calls는 추가로 따로 계수하며 control budget에 숨기지 않는다. 기존 P1 1,616 steps를 새 예산이나 새 독립 관측으로 합산하지 않는다. 새 pilot의 96 controls는 원래80 horizon을 유지한 조기 종료 trials이므로 공식 nominal benchmark score가 아니다.

## 승인 가능한 다음 단위

**권고:** 먼저 R2의 isolated adapter 구현 및 R3의96-control pilot까지만 승인 대상으로 삼는다. Gain=.5, bounded command=.2, t=6 change는 변경 가능한 review proposal이다. 구현 승인과 CFD 실행 승인 범위를 구분해서 답할 수 있다. Nominal no-op/측정/수치 gate가 통과하기 전 R4를 자동 실행하지 않는다. R4의960 controls, gradual/reversible, TD-MPC2 control extension, estimator 학습은 별도 승인 전 보류한다.

별도 remote 실행은 기존 방침대로 사용자가 진행한다. 승인 후에만 구체적인 구현·검증·실행 명령을 준비한다. 지금 source 수정, intervention hook, CFD launch script를 생성하지 않았다.

## 문서

- [Physical intervention contract](TIME_VARYING_INTERVENTION_CONTRACT.md)
- [Paired transition pilot](PAIRED_TRANSITION_PILOT_DESIGN.md)
- [Frozen control protocol](FROZEN_CONTROL_SHIFT_DESIGN.md)
- [History identifiability data](HISTORY_IDENTIFIABILITY_DATA_SPEC.md)

P1에서 성공한 reset/action observation은 nominal only이며, 이번 source 기반 제안의 승인이나 shifted 결과가 아니다. 검토 후 physical delivery가 실패하면 알고리즘 변경으로 우회하지 않고 intervention/measurement contract부터 수정 검토한다.


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
