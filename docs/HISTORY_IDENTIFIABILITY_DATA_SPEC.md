# History identifiability — minimum data and leakage contract

작성일 2026-10-16. **Stage 0 설계만 완료. 구현·CFD·추정기 학습 0.** 모든 gain/step/budget는 검토 제안이며 승인된 실행 상수가 아니다. VERIFIED=소스/기존 P1 근거, INFERRED=조건부 설계 판단, UNRESOLVED=미검증. 후속 구현/CFD는 명시적 사용자 승인 후에만 가능하다.


## R6. 최소 transition schema — 구현/학습 안 함

**INFERRED:** P1은 finite/restoration/repeatability만 검증했다. Hidden gain의 observation-history identifiability는 별도 문제다(S3/S15/S18). 아래 schema는 향후 approved pilot/control logging의 최소 요건이다. Stage0 추가 CFD/estimator fit=0. Proposal R3/R4를 둘 다 실행하더라도 최대1,056 controls는 feasibility 자료이며 estimator benchmark에 충분한 표본 수라고 주장하지 않는다.

| 필드 | 정의/shape와 provenance | estimator 사용 |
|---|---|---|
| o_t, o_next | official flattened velocity float32[302], t 전/후. raw velocity[151,2], pressure[151]도 감사용 보존 | actor-visible o만 기본 입력. pressure는 oracle/additional-sensor 설정으로 분리 |
| a_requested_raw, a_t | 원래 output 및 [-1,1] clip 후 command float32[1] | a_t만 history input 기본; clipping indicator 별도 |
| r_t, Cd_t, Cl_t | 해당 control transition의 original reward/forces, abs Cl 정의 보존 | 기본 gain estimator 입력에서 제외해 sensor set을 통제; label/진단용 |
| gain_t | 25 outer calls에 적용한 ground-truth scalar | target/감사용만. feature 누출 금지 |
| smoothing_state_before/after | native u memory readback. 가능하면 25-call u vector | privileged audit만; hidden memory 실측을 history 모델에 주면 질문이 달라짐 |
| applied_actuator_t | expected g*u/P*g*u, runtime smoothing, setter arguments, boundary readback을 별도 subfields; outer-call timestamps 포함 | 기본 estimator에 제외. g와 u의 비로 label이 직접 노출될 수 있음 |
| physical_time_t | 시작/끝 raw solver clock, nominal control interval, accepted-step counts, outer-call count, reset clock | 기본 features 제외. 완전한 accepted-dt history가 없으면 정확한 integration duration이라고 부르지 않음 |
| episode_id | immutable unique run ID | grouping only |
| initial_domain_id | family+dataset revision+split+index+file hashes+reset seed/randomize+perturbation RNG/identity | grouping only; 파일명/index feature 금지 |
| shift_schedule_id | immutable full schedule definition/hash; τ와 gain law | grouping/audit only |
| t, phase, reset/boundary | control index, genuine term/trunc, pilot early-stop status | alignment only. episode clock/shift phase feature 금지 |
| numerical/provenance | all success flags, source/backend/config/checkpoint hashes, device/dtype/deps, actual boundary type/profile hashes, finite/flux checks, planner RNG + warm-state ID, measurement availability flags | audit only |

Reward r_t는 a_t로 진행한 o_t→o_{t+1}의 결과다. No-reset gap/failed transition은 유효 표본으로 채우지 않고 invalid flag를 둔다. Missing readback을 expected value로 대체하지 않는다. Raw boundary tensor를 매 call 모두 저장하는 비용이 크면 고정 profile tensor+scalar+readback residual/hash를 활용할 수 있지만, pilot에는 실제 readback 배열을 남겨 압축 표현을 먼저 검증한다(S4/S11/S12). 모든 time/condition files은 actor 경로에서 접근 불가능한 logger side channel에 둔다.

## Estimation timing: 예측 불가능한 jump와 식별을 구별

**INFERRED causal limit:** g_t가 action t 직후 적용되며 미래 schedule이 숨겨져 있고 pre-prefix가 같다면 o_t 또는 causal history에서 새 g_t를 즉시 예측할 정보가 없다. 이 사례를 history estimator 실패로 세면 부정확하다. 비교는 우선 전이 완료 후 g_t를 추정하는 filtering/detection 문제로 명세한다. Action 선택 시점에는 마지막 관측된 g_(t−1)에 대한 belief만 사용할 수 있다. Hidden parameter ground truth를 neural input에 직접 넣지 않는다.

## R7. 같은 관측 시점의 baseline 제안

모든 baseline은 o_(t+1)을 받은 뒤 동일 g_t label을 추정한다. 학습/튜닝은 Stage0에서 하지 않는다. 근거 S3(미관측 memory), S15/S16(허용 관측), S17(정책의 action generation), 인과 목적: 시간적 정보와 입력 정보량의 기여를 구분.

1. **No-observation constant prior:** train-only gain mean/majority; class imbalance 기준.
2. **Observation-only:** o_(t+1)→g_t. 먼저 regularized linear/ridge 또는 고정된 작은 estimator family 후보; raw time/id 없음.
3. **Single-transition:** (o_t,a_t,o_(t+1))→g_t. Action-conditioning 효과를 multi-step history와 분리하기 위한 최소 비교.
4. **History:** 최근 H=4 completed transitions의 causal (o,a) window→g_t; H=4는 사전 후보이며 현재 fit0. 같은 estimator family/regularization search 예산을 맞추고 더 큰 입력 차원 자체가 confound임을 보고한다. 필요하면 observation-history-only 대조로 action history 기여를 분리하되 추가 조건은 승인 전0.

**Known-filter baseline/공변량:** command history와 reset u=0, α=.1, 25 outer calls로 u를 결정적으로 재구성하는 non-neural filter를 명시한다. 이 재구성은 hidden gain 추정기가 아니며 g를 직접 주지 않는다. Single-transition estimator에 causal reconstructed u를 추가한 대조를 후보로 두면 “history가 알려진 smoothing memory를 복원한 효과”와 gain/flow 추정의 추가 효과를 구분하는 데 도움이 된다. 실측 privileged u를 입력한 결과와 구분한다. Fit/CFD budget은 Stage0에서0, 향후 동일 offline 비교 범위 검토; 근거 S3/S8. 재구성 residual gate 실패 또는 runtime hidden state를 몰래 입력하면 해당 대조를 중단한다.

기본 metric 후보: held-out group gain MAE, condition-balanced error, change 직후 latency/false alarm, matched-time windows의 error. 한 rollout의 overlapping windows를 IID로 CI 계산하지 않는다. Gain 값 두 개만으로 continuous gain regression/generalization을 주장하지 않는다. Identifiability의 근거는 held-out groups에서 prior/observation-only 대비 history의 추가 정보이며, estimator가 더 잘 맞는 것만으로 control benefit은 입증되지 않는다. 현재 budget은 fitting0/CFD0; 나중에 기존 approved data만 읽는 offline 비교를 별도 검토한다.

## Leakage 방지와 현재 작은 설계의 한계

- 동일 initial-domain 파일 및 reset root에서 나온 N/F/J, S0/S1/S2, deterministic repeats, planner-seed variants, benchmark/extended prefixes를 하나의 **connected origin group**으로 묶은 뒤 split한다. 시간 row 또는 겹친 window를 무작위 분할하지 않는다. 동일 base domain의 다른 perturbation을 split 간 허용하면 주장이 달라지므로 기본은 base-domain 그룹까지 묶는다.
- 동일 episode의 prefix와 post-shift를 train/test로 나누지 않는다. Schedule-only group도 별도로 관리해 held-out τ/gain family를 검사해야 한다. 원본 TRAIN/VAL/TEST 의미를 보존하고 기존 TEST(P1 포함)를 estimator 학습에 사용하지 않는다.
- clock, t, schedule_id/filename, reset mask의 padding 길이, hidden gain, boundary readback, applied control, seed/domain ID, future observations를 feature에서 제외한다. Full-length causal windows만 사용하면 padding-length leakage를 줄일 수 있다. 하지만 flow phase/state가 시간을 간접 encode할 수 있어 metadata 삭제만으로 충분하지 않다.
- 고정 τ와 condition별 고정 action을 학습하면 쉽게 schedule을 외울 수 있다. 향후 estimator data에는 same action protocol across labels, hidden randomized τ, matched-time no-shift negatives, held-out schedule families와 초기조건이 필요하다. **현재 pilot τ=6/control τ=40은 구현 검증용이며 identifiability benchmark로 승인하지 않는다.** 새 random schedule matrix나 CFD 예산은 Stage0에서 확정하지 않는다.
- Frozen controller의 a가 condition label을 드러내는 policy-response confound가 있다. 첫 식별 분석은 동일 prescribed actions를 우선 고려하고 closed-loop 분석은 별도 cohort로 둔다. Two env seeds/one pilot IC로 group-held-out 추정기 평가를 충분히 할 수 없다.
- Train-only feature normalization/선택/regularization, validation-only tuning, immutable final group split. Deterministic repeats는 measurement noise 대조이지 독립 샘플 증대가 아니다.

**Stop:** feature schema에 privileged/time leakage 발견, connected origin groups가 양 split에 존재, independent groups 부족, failed solver rows 사용, future 정보 사용이면 estimator 효과 주장을 중단한다. Representation training, history-conditioned TD-MPC 또는 inference-time neural update는 현재 scope 밖이다.


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
