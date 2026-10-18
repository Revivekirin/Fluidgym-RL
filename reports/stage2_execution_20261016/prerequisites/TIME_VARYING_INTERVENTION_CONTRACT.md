# Time-varying hidden actuator intervention contract

작성일 2026-10-16. **Stage 0 설계만 완료. 구현·CFD·추정기 학습 0.** 모든 gain/step/budget는 검토 제안이며 승인된 실행 상수가 아니다. VERIFIED=소스/기존 P1 근거, INFERRED=조건부 설계 판단, UNRESOLVED=미검증. 후속 구현/CFD는 명시적 사용자 승인 후에만 가능하다.


## 판정과 범위

**INFERRED: post-smoothing boundary-effectiveness gain은 조건부 구현 가능성이 높다.** 단순 Python patch 가능성 때문이 아니라, 원본 `_step_impl`의 actuation 경계, 원본 두 boundary setter, C++ setter/readback 및 solver 소비 경로가 확인된다(S3/S4/S11/S12). **UNRESOLVED:** 실제 gain intervention의 readback·solver 동작. P1은 gain=1의 nominal 경로만 확인했다(S18).

## 0A. 정확한 실행 경로

| 순서 | VERIFIED 동작 | 근거 |
|---|---|---|
| Requested command | Policy가 scalar a_raw를 반환. CheckedEnv에서 float32 변환, shape=(1,), finite 검사, [-1,1] clip 후 CUDA 전달 | S1, S17 |
| Native validation | raw FluidEnv는 reset 여부, action shape, horizon을 검사. Box range를 자체 clip하지 않음 | S2 |
| Smoothing | 각 outer call j 전에 `u[t,j]=0.9*u[t,j-1]+0.1*a[t]`; private `__last_control`에 저장 | S3 |
| Boundary | `_apply_action(u)`가 top/bottom 고정 profile×u를 두 `setVelocity`에 매번 전달 | S4:185 |
| Integration | assignment 직후 `single_step()`. 25 outer calls/control, 요청 dt=.01. Adaptive accepted substeps는 별도 카운터 | S2:839, S3, S13 |
| Measurements | 각 outer call 뒤 Cd/Cl; 25개를 산술 평균. 마지막 global obs 획득. `r=Cd_ref-Cd-abs(Cl)` | S3 |
| Policy observation | raw velocity (151,2), pressure (151,); official flatten은 velocity만 sensor-major (302,) | S15/S16 |

**VERIFIED profile:** jet angular support는 10° parameter와 mesh index 선택으로 결정된다. Parabolic scalar profile은 `max=1`로 정규화하며 평균 유량=1 정규화가 아니다(S4:136–183, S5). Global components는 `[p*sin(theta), p*cos(theta)]`. Top/bottom 모두 같은 scalar를 곱하며 bottom에 별도 `-u`를 추가하지 않는다. Jet 근처 두 profile의 global y 성분은 양수 방향이다. **INFERRED:** positive u는 위쪽에서 cylinder→fluid, 아래쪽에서 fluid→cylinder인 paired blowing/suction; negative u는 반대. Boundary-normal 부호와 discrete flux는 geometry/readback으로 확인해야 한다. “자동으로 정확한 zero net mass flux”라는 수치 주장은 보류한다.

Boundary handle은 top block `-y`, bottom block `+y`(S8). C++ setter는 CUDA/shape/dtype를 검증하고 tensor를 `m_velocity`에 할당한다. 자체 saturation이나 amplitude normalization은 없다(S12). `getVelocity(computational=False)` 또는 `.velocity`는 설정된 physical-coordinate tensor를 읽는 후보 API이고 `True`는 transform을 적용하므로 비교 좌표계를 혼동하면 안 된다. `GetFluxes`와 domain `GetBoundaryFluxBalance`도 노출되어 있다(S11/S13). Readback이 solver 전체 해의 정확성 측정은 아니다.

## Memory, reset, timing

**VERIFIED:** `last_control`은 control steps 사이에 남고 reset의 `_additional_initialization`에서 0으로 초기화된다. Profile와 boundary references도 새 domain 기준으로 다시 생성된다. Randomized reset은 velocity/pressure noise와 여러 solver calls를 수행하며 마지막 reset에서 zero boundary를 적용한다(S6/S7/S8). 관측에는 이 memory가 포함되지 않는다. `get_state().additional_info['last_control']`는 public state API로 접근 가능하지만 매번 domain clone을 유발한다(S9/S10). Scoped runtime logging은 확인된 private field read만 사용하는 편이 작으며 버전 gate가 필요하다.

**제안 R1 — 시점 정의:** `o_0`는 reset 결과, `a_t`는 `o_t→o_{t+1}` 전이의 command. Step 진입 때 `g_t`를 한 번 결정해 25 outer calls 동안 유지한다. 첫 변경 t=τ는 action τ의 첫 smoothing 후 boundary assignment부터 적용한다. 모든 adaptive 내부 substeps는 해당 outer call에 설정한 boundary를 받는다. Gain을 inner adaptive count나 반올림 clock으로 전환하지 않는다. 근거 S2/S3/S13, 인과 목적: numerical adaptive subdivision이 treatment timing을 바꾸지 않음. 예산: Stage 0=0, 검증은 pilot 96 controls에 포함. 중단: 전이당 gain이 바뀌거나 25회 call 매핑이 깨질 때.

## Command scaling vs post-smoothing effectiveness

Nominal clipped command를 a, smoother를 F라고 하자.

- Command scaling: `u'=F(g_t*a_t)` → `B=P*u'`. Gain 변화가 internal smoothing memory에 흡수된다.
- 제안 effectiveness: `u=F(a_t)`, `v_eff=g_t*u`, `B=P*v_eff`. Memory u를 유지하며 gain 변화는 다음 boundary assignment에 즉시 나타난다.

**INFERRED:** 고정 g, zero initial memory, 추가 saturation 없음이면 선형성으로 두 방식이 수학적으로 같을 수 있다. 따라서 fixed-gain 비교만으로 두 구현을 구별할 수 없다. Episode-internal jump와 nonzero memory가 있어야 구별된다. 같은 command를 유지하는 jump boundary를 pilot에 넣는 이유다(S3/S4). g<=1, |a|<=1, u0=0이면 |g*u|<=1이므로 별도 saturation이 필요하지 않다. g>1/negative gain은 이번 제안 범위 밖이며 조용히 clip하지 않고 거부한다.

**INFERRED — 알려진 smoothing과 미지 gain을 분리:** α=.1와 25 calls가 알려져 있으므로 control-step 종료 memory는 수학적으로 `u_end=β*u_before+(1−β)*a`, `β=.9^25≈.0717898`이다(실제 float32 recurrence residual은 별도 측정). Reset u=0 및 command history가 있으면 nominal smoother state는 neural model 없이 재구성할 수 있다. History의 이점이 단지 알려진 actuator memory 복원에서 오는지, 미지 gain/flow state 정보에서 오는지 분리해야 한다. u가 거의0이면 boundary의 g*u 곱으로부터 g를 식별할 excitation이 부족하다. 새 CFD/fit budget0, 근거 S3/S8; memory reconstruction이 실측 recurrence와 불일치하면 known-filter 가정을 중단한다.

## 0B. 최소 adapter/hook 제안 R2

**선택 후보:** RL 연구 전용 opt-in wrapper가 정확한 단일 `CylinderJetEnv2D` instance의 `_apply_action` 호출을 감싸고, 원래 bound method에 `g*u`를 전달한다. 기존 `_step_impl`, smoother update, profile 생성·정규화, reward, solver를 복사/재작성하지 않는다. Class-global monkeypatch 대신 instance 수명에 국한하고 detach 시 원래 method를 복원한다. Generic observation/action wrapper에서 command에 gain을 곱하는 구현은 이 contract가 아니다.

삽입 순서: original action validation/clip → original smoothing 및 memory update → **gain multiplication** → original normalized profile multiplication → original boundary assignment → original solver. Gain은 profile normalization **후**의 amplitude에 작용하고 normalized profile tensor 자체를 in-place 수정하지 않는다. Setter 후 매 outer call마다 boundary tensor readback을 수행할 수 있도록 별도 read-only observer를 둔다. Nominal g=1도 adapter를 통과한 경로와 bare upstream을 비교해야 한다.

Schedule은 외부 audit config에만 두고 policy input, reward, action_space, info에 노출하지 않는다. Logger 전용 side channel로 gain/clock/schedule_id를 저장한다. Reset 시작 시 adapter phase=initializing, gain=1, transition counter=0으로 설정한다. Native reset/warmup 전체는 gain=1로 유지한다. Reset 완료 후 u=0 및 zero boundary를 검증하고 schedule을 arm한다. Fixed gain도 첫 policy transition t=0부터 적용한다. Schedule마다 다른 reset flow를 만들지 않는다. 중간 episode gain change에서 u를 reset하지 않는다. Reset 실패 시 arm하지 않는다.

**Public API 한계:** gain setter나 post-smoothing callback은 없다. 제안은 private actuation seam과 pinned class semantics에 의존한다. Class/method signature/hash, single-agent, differentiable=False, enable_actions=True를 검사하고 예상과 다르면 중단한다. Reset이 새 domain을 만들기 때문에 캐시한 boundary handle을 다음 episode에 재사용하면 안 된다. Python attribute를 바꿀 수 있다는 것만으로 채택하지 않는다: setter-readback-profile 관계와 solver-side flux checks가 runtime acceptance gate다.

근거 S3/S4/S6–S13. 인과 목적: command 및 internal memory와 physical boundary authority를 분리. 예산: 구현 자체 0 CFD; pilot에서만 최대96 controls. 중단: gain1 bare equivalence 실패, readback 미지원/불일치, cumulative profile mutation, reset leakage, source/hash 변경. Readback 불가 시 setter 인자를 “measured boundary”로 이름만 바꿔 진행하지 않는다.

## 실패와 복원 한계

**VERIFIED:** `single_step`는 boundary flux imbalance에서 예외, linear-solve failure에서 False를 반환할 수 있고 family loop는 반환값을 무시한다(S13/S3). 제안 observer는 원본을 그대로 호출한 뒤 False/exception을 실패로 기록하고 중단한다. Flux tolerance를 완화하거나 success로 바꾸지 않는다. P1 observer 패턴을 재사용할 수 있으나 physical intervention 검증은 아직 없다.

**VERIFIED:** base state API는 domain Clone/n_steps/mode를 보관하고 restore 시 simulation을 재생성한다. `_get_simulation`은 `make_divergence_free()`도 호출한다(S10/S14). Cylinder는 last_control을 복원하지만 solver clock, adaptive state/caches, 모든 RNG 및 adapter schedule을 완전히 저장/복원한다는 보장이 없다. **UNRESOLVED:** true full-state restart equivalence. 이 설계는 state restore를 사용하지 않는다.


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
