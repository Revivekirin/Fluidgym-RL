# Nominal representation / planning readiness 조사

조사일: 2026-10-16. 기존 코드·체크포인트·기록에 대한 조사이며 새 CFD, 학습, physical shift, loss 또는 adaptation 구현은 수행하지 않았다. 기존 working-tree 변경과 원본 결과는 보존했다.

**결론: CylinderJet2D-easy의 저장된 TD-MPC/TD-MPC2 nominal control 비교와 모델 구조 분석은 가능하다. 실제 관측에 대한 nominal representation 및 planning prediction의 정량 검증은 아직 자료가 부족하다. 현재 자산으로 6환경 dynamics-shift 연구 준비가 완료되었다고 판단할 수 없다.**

VERIFIED = 현재 읽은 코드 또는 파일/계산으로 확인한 사실. 기록의 VERIFIED는 해당 기록을 검증했다는 뜻이며 새 CFD 재현을 뜻하지 않는다. INFERRED = 확인된 사실의 제한적 해석. UNRESOLVED = 필요한 데이터·실행 검증이 없는 주장. Physics의 상세 CFD 분석 결과는 해당 프로젝트의 감사 보고서를 확인했으며 이번에 58,200-step 전체 재분석을 반복하지 않았다.

## 1. 사용 가능한 checkpoint와 조건

| 구분 | VERIFIED: 실제 로컬 자산 | VERIFIED: 학습 조건 | 평가와 한계 |
|---|---|---|---|
| TD-MPC | `results/tdmpc_full/20261012T060719Z_full/checkpoints/`: latest.pt=48k, best.pt=50k, final.pt=50k | Cylinder easy, seed 1, 50,000 transitions, 625×80 episodes, 49,940 learner calls | historical val 및 별도 shared val/test 존재. best/final 온라인+target 76개 tensor가 동일하다. |
| TD-MPC2 | `results/tdmpc2_full/20261008T124249Z_4be46ae7/upstream/models/`: step_4000부터 step_48000까지 4k 간격 12개, final.pt=50k | Cylinder easy, seed 1, model_size=5, 50,000 transitions, 49,999 learner calls, compile=true | historical eval은 train split. shared validation winner는 44k. |
| SAC | 현 workspace 결과에서 `.zip` checkpoint 및 완료된 SAC CFD run 없음 | config만 존재: seed 0, 50k 예정, MLP 256×256, lr=3e-4, gamma=.99, tau=.005, replay=1e6, batch=256, learning_starts=100, train_freq=1, gradient_steps=-1, auto entropy | 구현/CPU fixture 검증과 학습 완료를 구분해야 한다. 공식 public checkpoint 다운로드·호환성 실패도 입증되지 않았다. |

**VERIFIED:** 16개 `.pt`를 현재 bytes로 재해시해 기존 inventory와 일치 확인. CPU `weights_only=True` 역직렬화와 모든 tensor의 finite 검사를 통과했다. 이것은 TD-MPC2 fresh-agent 실행이나 CUDA 재현 검증이 아니다. TD-MPC는 weights+sidecar의 config/step/planner_std/probe가 필요하고, TD-MPC2는 weights+`run_config.json`+pinned runtime/patch가 필요하다. 두 모델 모두 replay/optimizer/RNG/simulator를 포함하는 exact resume 자산이 아니다.

**VERIFIED:** TD-MPC는 warmup 100 actions 후 episode 단위 update burst; TD-MPC2는 inclusive loop를 `steps=49999`로 맞추며 1,001 random actions, 초기 1,000 updates 후 online update. 같은 50k는 같은 데이터 소비·optimizer work·compute를 뜻하지 않는다. TD-MPC2 마지막 episode transitions가 종료 전에 replay에 추가되지 않는 native loop 특성도 있다.

**INFERRED:** 두 TD 계열은 inference 분석의 출발점으로 사용할 수 있다. SAC는 현재 학습된 대조군으로 사용할 수 없다.

**UNRESOLVED:** 외부 서버에만 존재하는 추가 checkpoint, 다른 training seeds, Cylinder medium/hard 또는 RBC trained policy는 확인되지 않았다. TD-MPC2의 당시 dirty harness 전체를 SHA 하나로 복원할 수 없다.

근거: `tdmpc/runner.py`, `tdmpc/evaluation.py`, `tdmpc2_runtime.py`, `sac/{training,runtime}.py`, 각 run summary/status/config, `configs/sac_nominal.json`, 이 조사 `verification.json`, `tensors.json`.

## 2. Nominal control 성능 및 selection

**VERIFIED:** shared evaluation 21건 = val 후보 16건 + test 4건 + uncontrolled TRAIN reference 1건. 모두 Cylinder easy, seeded reset, seeds 0–9, randomize=True, episode 80 steps, split 외 공통 cohort 1개. 결과 파일/체크포인트 해시 전부 일치. episode mean으로 재계산한 reward aggregate의 최대 차이는 5.56e-17 미만이다.

| 독립 test | Reward/step | Return | Mean Cd | Mean abs Cl | TRAIN-reference drag reduction |
|---|---:|---:|---:|---:|---:|
| TD-MPC val-best 50k | 0.03762838 | 3.01027071 | 3.11004720 | 0.18047993 | 6.54430% |
| TD-MPC final 50k | 0.03812451 | 3.04996078 | 3.11010831 | 0.17992270 | 6.54246% |
| TD-MPC2 val-best 44k | -0.07737686 | -6.19014869 | 3.22068048 | 0.18485190 | 3.21981% |
| TD-MPC2 final 50k | -0.08246609 | -6.59728717 | 3.22553160 | 0.18509001 | 3.07404% |

**VERIFIED:** selection은 val mean reward 최대, tie는 작은 step 및 hash 순. 후보 manifest hash와 winner를 다시 계산해 TD-MPC 3개 / TD-MPC2 13개 selection을 확인했다. TD-MPC shared val-best=0.02886147, TD-MPC2 44k=-0.08228433. test_best가 frozen winner hash를 사용한다. TD-MPC historical best score=0.03206033은 다른 reset protocol의 점수라 shared score와 혼합하면 안 된다.

**VERIFIED:** uncontrolled reference Cd=3.3278303676843644는 TRAIN에서 측정했다. reward 내 native Cd_ref=3.3281555422147115와 다르다. 위 감소율은 `100*(1-controlled mean Cd / measured TRAIN mean Cd)`이며 paired test baseline 대비 개선율이 아니다. 음수 reward가 drag 개선 부재를 뜻하지 않는다.

**VERIFIED:** TD-MPC best/final은 동일 weights 및 planner_std인데 평가 값은 약간 다르다. 3개 archive가 3개 독립 학습 모델은 아니다. TD-MPC2는 더 많은 validation 후보를 보유한다. final-vs-final도 함께 유지해야 하며 test가 더 좋다는 이유로 final을 새 best로 선택해서는 안 된다.

**VERIFIED — 과거 문서 이후 결과:** `results/nominal_diagnostics/20261013T111711Z`의 44k plan/actor 검증은 이미 완료되어 있다. 각각 val seeds 0,1 ×80 steps, 같은 checkpoint. manifest 결과 hash를 재확인했다.

| 44k validation diagnostic | Plan | Actor mean |
|---|---:|---:|
| Reward/step | -0.07566534 | -0.08683327 |
| Mean Cd | 3.21980322 | 3.21272386 |
| Mean abs Cl | 0.18401764 | 0.20226492 |
| Action latency median | 5.996 ms | 0.975 ms |

두 episode 모두 plan reward가 높다. 평균 차이 +0.01116793은 actor보다 drag가 0.00707936 높지만 abs lift가 0.01824728 낮아서 발생한다. 첫 action latency는 plan 9.167 s, actor 0.0595 s로 compile/copy를 포함한다. median만으로 총 비용을 판단할 수 없다. 원래 standalone CLI는 cleanup으로 삭제되었지만 결과와 일반 actor inference는 남아 있다.

**INFERRED:** 이 두 validation IC에서는 planning이 reward에 유리했다. “actor가 더 좋아 planner가 문제”라는 단순 설명은 이 기록으로 지지되지 않는다. TD-MPC가 이 seed의 공통 test에서 우수하다는 조건부 비교는 타당하다.

**UNRESOLVED:** training-seed 일반화, convergence, sample efficiency, 모델 정확도가 성능 차이의 원인인지, identical-weight 점수 차이의 원인, 48k TD-MPC2 악화의 원인. 44k→48k reward 하락의 약 77.8%가 abs-lift 항이라는 것은 산술 분해이지 원인 규명은 아니다. 이 test는 이미 열람되었으므로 앞으로의 가설 선택에 대해 untouched holdout이라고 할 수 없다.

## 3. Encoder, dynamics, reward/value, planner 실행

**VERIFIED:** 현재 shared 경로는 `scripts/compare_nominal.py` → `comparison.evaluate.evaluate_to_directory` → `load_policy` → `evalcontract.run_evaluation` → 각 policy `act` → `NominalEnv.step`이다. 환경과 learner는 별도로 생성된다. 학습 코드를 실행하지 않고 저장된 config로 복원한다.

| 항목 | TD-MPC | TD-MPC2 |
|---|---|---|
| Encoder | raw 302 → Linear 256 → ELU → Linear 50 | raw state 302 → state encoder, latent 512, SimNorm(group 8); task_dim=0, single task |
| Transition | `model.next(z,a)`의 MLP, concat(50+1) → 512×2 → 50 | `model.next(z,a,None)`, concat(512+1) → 512×2 → 512, SimNorm |
| Reward | 같은 z,a의 scalar prediction, MSE target | 101-bin logits, symlog two-hot CE; inference에서 `two_hot_inv`로 scalar 환원 |
| Value | twin scalar Q; planner terminal에서 min Q | Q ensemble 5개; planner terminal에서는 무작위 2개를 선택해 decoded Q 평균 |
| 학습 target | target encoder latent consistency, scalar reward/Q; rho=.5, full target EMA | detached next encoding consistency, reward/Q soft CE, target Q EMA; episodic=false |
| Horizon / discount | 학습 unroll 20; planner schedule 1→20, gamma=.99 | 학습/plan H=3, effective gamma=.95 |
| Candidate budget | Gaussian 512 + policy 25 = **537**, iterations 6, elites 64 | **512 total** 중 policy 24, iterations 6, elites 64 |
| 최종 선택 | elite 가중 분포에서 NumPy sampling; 첫 action | elite 가중 분포에서 Gumbel sampling; 첫 action |

TD-MPC `Policy.act` → `agent.plan(eval_mode=True,step=saved_step,t0=...)` → encode → policy prior trajectories → warm-start shifted mean → Gaussian candidates → `estimate_value` → elite weighted mean/std refinement → elite sample. TD-MPC2 `TDMPC2Policy.act` → `agent.act` → `plan/_plan` → 같은 유형의 latent rollout/terminal value/ranked refinement. TD-MPC 소스는 루프를 CEM이라고 명명한다. 두 경로를 deterministic actor 또는 순수한 analytic MPPI 최적해로 설명하면 부정확하다.

Planner score는 `sum(gamma^t * predicted_reward_t) + gamma^H * terminal_Q`이다. 매 실제 step마다 새 observation을 encode하고 첫 action만 적용한다. `_prev_mean`은 action-sequence warm start이며 recurrent physical latent memory가 아니다. 평가 모드는 마지막 추가 exploration noise를 끄지만 candidate/prior/elite/Q sampling까지 끄지 않는다. TD-MPC2 actor mode는 `encode` 후 `pi(...).info['mean']`를 반환하고 planning을 우회한다.

**VERIFIED:** decoder 또는 full-field reconstruction objective가 없다. 물리 state의 완전 복원이 아니라 task-oriented representation이다. TD-MPC loss 계수 consistency/reward/value=2/.5/.1, TD-MPC2=20/.1/.1. latent dimension·scale와 MSE/CE가 달라 raw training loss 크기의 알고리즘 간 비교는 타당하지 않다.

**INFERRED:** latent phase/물리량 보존과 reward/Q calibration은 개념상 다른 질문이며 따로 판단해야 한다. 같은 표면적 `num_samples=512`도 실제 후보 수와 계산량은 다르다.

**UNRESOLVED:** 실제 nominal observations에 대한 latent rank/collapse, transition residual, reward bias, Q calibration, 후보 ranking 및 선택 regret. 유한 weights나 training loss는 이 항목들의 검증이 아니다.

근거: pinned `source_snapshots/tdmpc/src/algorithm/{tdmpc,helper}.py`, `source_snapshots/tdmpc2/tdmpc2/{tdmpc2,common/world_model,common/layers,common/math}.py`, 현재 `tdmpc/evaluation.py`, `tdmpc2_runtime.py`, restored config. Device 및 TensorDict 호환 patch는 `patches/`에 따로 있으며 original source와 혼동하지 않았다.

## 4. 실제 CFD contract 및 프로젝트 간 동등성

**VERIFIED:** checked shared nominal contract는 CylinderJet2D-easy-v0, Re100, action_repeat=1, 80 steps에 한정된다. `environment_contract.resolve_environment`는 다른 physical/config 값으로의 변경을 거부한다. generic FluidGym factory가 환경 ID를 받을 수 있다는 사실은 RBC shared evaluation이 구현되었다는 뜻이 아니다.

| Contract | VERIFIED: 코드/기록 | UNRESOLVED / 해석 제한 |
|---|---|---|
| Observation | native Dict에는 velocity와 pressure. FlattenObservation은 temperature/velocity만 선택하므로 Cylinder는 velocity (151,2)만 sensor-major flatten하여 float32 (302,). normalization 없음, frame_stack=1 | pressure, Cd/Cl, applied-action history는 encoder 입력이 아님. 다른 설치본의 센서 추출 동등성은 아래 source drift 검증 필요 |
| Action | bounded scalar [-1,1], clipping 후 Torch/device 전달. native top/bottom jet velocity profiles에 smoothed scalar 적용 | normalized command는 실제 질량유량 또는 applied actuator history 자체가 아님 |
| Smoothing | 각 native `single_step` 호출 전 `u_s ← u_s + .1*(a-u_s)`; 한 control step에 25회 | adaptive 내부 subdivision마다 새 smoothing을 하는 것은 아님. 과거 문서의 “each solver step”은 이 호출 경계로 해석해야 함 |
| Reward | `Cd_ref - mean_25calls(Cd) - abs(mean_25calls(Cl))`, scale=1 | `mean(abs(Cl))`와 다름. adaptive 모든 accepted substeps의 시간 가중 평균도 아님 |
| Time | 요청 dt=.01, control interval=.25, 25 native calls, CFL=.8, adaptive-capable solver; episode nominal duration=20 | RL 파일은 actual solver clock/count 시계열을 기록하지 않음. 모든 평가에서 fixed physical interval을 새로 검증한 것은 아님 |
| Reset | train/val/test saved-domain 선택, seeded randomize=True에서 private RNG로 domain 선택; actuator memory는 domain 초기화 시 reset | seed 번호는 domain index가 아님. 중복 선택 가능. cross-wrapper exact state/trajectory equality는 미검증 |
| Boundary | step 80 truncation, physical terminated=False; checked wrapper는 예기치 않은 termination을 거부 | time limit을 terminal TD target으로 사용하지 않음. TDMPC2 done은 term OR trunc이지만 별도 info['terminated'] 사용 |

**VERIFIED — 중요한 신규 대조:** 두 프로젝트 `UPSTREAM.lock`의 FluidGym commit은 `50071140b73a2a917fb96d33522a769f903f8c27`로 같다. 하지만 RL shared evaluation의 **installed Python source hashes**와 Physics pinned checkout의 실제 bytes를 비교하면 다음 5개가 다르다.

- `envs/fluid_env.py`
- `envs/util/__init__.py`
- `envs/util/obs_extraction.py`
- `simulation/pict/data/resample.py`
- `simulation/pict/util/output.py`

해시 목록은 `verification.json`에 저장했다. RL 실행 당시 설치본 bytes 전체가 이 조사에 제공된 것은 아니므로 내용 diff나 영향은 확정하지 않는다. 같은 version=0.1.2/pin 선언만으로 installed wheel=clean pinned source라고 할 수 없다. 핵심 Cylinder implementation 및 flatten wrapper 해시는 대조에서 일치하지만 관측 extraction 경로까지 완전히 동일하다고 인증할 수 없다. Physics lock의 과거 “RL pin UNKNOWN” 문구 역시 현재 이 대조를 반영한 상태가 아니다.

**VERIFIED:** native Cylinder code는 `single_step()` 반환 성공 flag를 확인하지 않는다. 현재 RL checked evaluator는 obs/reward의 finite 여부와 boundary/reward 식을 검사하지만 Physics runner의 single_step failure guard와 동일하지 않다. **UNRESOLVED:** 과거 RL에서 실패가 발생했는지; 유한 output만으로 native solver 성공을 보증할 수 없다.

**INFERRED:** actuator memory가 observation에 없고 sparse velocity만 관측하므로 `z,a→z'` 오차에는 partial observability도 섞일 수 있다. 이것을 곧바로 dynamics shift로 분류할 수 없다.

## 5. Physics nominal 결과와 연결 가능한 진단량

**VERIFIED:** Physics report는 60 runs, 58,200 records, 2,970 resampled field snapshots, 6환경의 zero-action nominal characterization을 기록한다. 해당 runner는 native observation의 finite/shape를 검사하지만 policy observation time series를 저장하지 않는다. `snapshot()`은 public `get_velocity/get_pressure/...` field를 저장한다. RL encoder의 302개 native sensor observations와 같은 배열이 아니다.

| Physics evidence | 연결할 수 있는 질문 / INFERRED | 현재 상태 |
|---|---|---|
| Cylinder easy St 후보 ≈.274; 시간축 검증 통과 | nominal latent가 shedding phase나 lift phase 정보를 보존하는가 | UNRESOLVED: 원래 관측과 동기화된 latent 없음. approximate field interpolation을 원래 policy obs로 취급 불가 |
| Easy IC 간 평균 drag는 가깝고 phase alignment 후 lift 상관 증가 | pointwise prediction error에서 phase offset과 평균 물리량 차이를 구분할 필요 | INFERRED: 좋은 diagnostic 구분이나 현재 모델 결과는 아님 |
| Medium/hard adaptive subdivision, FFT 보류 | 시간 인덱스와 실제 적분시간을 분리해야 함 | VERIFIED: 해당 Physics 한계. RL easy의 nominal dt만으로 다른 환경 spectrum 복원 불가 |
| RBC medium/hard IC별 Nu/KE 및 roll structure 차이 | representation이 spatial organization과 heat transport를 보존하는지 | UNRESOLVED: 현재 RBC checkpoint/contract 없음. Cylinder encoder에 RBC fields를 reshape/pad해 넣을 근거 없음 |
| benchmark와 extended prefix 일치 | 같은 origin의 시간 구간을 train/test로 나누면 leakage 가능 | VERIFIED: Physics 감사 기록. root trajectory 및 initial-domain identity를 묶어 split 검토해야 함 |

**VERIFIED:** RL 기존 metric은 reward, Cd, signed Cl, abs(control-interval Cl), command aggregate. Physics는 더 긴 Cd/Cl 시계열과 RBC Nu/volume KE, resampled fields를 보유한다. 통계 정의, window, split, IC, source/dataset provenance를 맞추기 전에는 Physics benchmark mean Cd를 RL의 matched baseline으로 대체할 수 없다.

**INFERRED:** easy에서 .274와 nominal .25를 사용하면 주기는 약 14.6 control steps이다. 따라서 TD-MPC H20과 TD-MPC2 H3는 서로 다른 시간 범위를 본다. 이는 zero-action 후보 주파수에 기반한 해석일 뿐 controlled shedding period나 horizon 성능 우열의 증거가 아니다.

**UNRESOLVED:** 두 프로젝트의 실제 acquisition dataset 전체 bytes 동등성, original state/observation 재구성 가능성, nonzero action-response, 실제 parameter shift. RBC IC 차이는 동일 nominal dynamics 아래의 초기조건 차이이며 그 자체로 physical dynamics shift가 아니다.

## 6. 추가 CFD 없이 가능한 범위와 새 데이터가 필요한 범위

| 분석 | 현재 가능 여부 | 필요 조건 / 한계 |
|---|---|---|
| checkpoint hash/finite/동일 weight, config 및 selection 감사 | VERIFIED: 이번 실행 완료 | 16 checkpoints, 21 evaluations, 2 diagnostic manifests |
| 기존 reward/Cd/Cl/command aggregate, training curve, 44k vs48k 비교 | VERIFIED: 기존 records로 가능 | 같은 IC의 전체 state pairing, all-update history는 없음 |
| 고정 입력에서 encoder/latent/reward/Q 수치·Jacobian·candidate repeatability | INFERRED: simulator 없이 weights로 가능 | 맞는 runtime/patch 필요. synthetic 입력 결과는 실제 nominal 분포 정확도 증거가 아님; 이번에 실행하지 않음 |
| 실제 관측에서 representation rank, phase/물리량 probe | UNRESOLVED: 현재 그대로는 불가 | native sensor obs와 동기화된 physical labels 필요. resampled field 변환은 독립적으로 검증되어야 함 |
| one/multi-step latent consistency와 reward prediction calibration | UNRESOLVED | aligned `(o_t,a_t,r_t,o_{t+1})`, episode boundaries, preprocessing 필요. 현재 replay/transition dataset 없음 |
| offline candidate score 분해, Q ensemble 분산, action ranking | INFERRED: 입력이 있으면 CFD 없이 계산 가능 | “모델이 무엇을 선호하는가”만 알 수 있음. 실제 candidate 성능/uncertainty calibration과 다름 |
| Q 대 realized return 비교 | UNRESOLVED | discount/continuation semantics 일치 필요. 80-step undiscounted return을 continuing discounted Q의 정답으로 쓰면 안 됨 |
| closed-loop actor/planner/horizon/repeatability | 일부 VERIFIED: 기존 44k 2-episode contrast | 새로운 조건의 실제 제어효과에는 새 CFD가 필요. 이번에 조건/예산을 확정하지 않음 |
| predicted-vs-realized candidate ranking/regret | UNRESOLVED | 완전한 state/actuator/RNG restore와 same-action replay 검증 후 branching CFD가 필요 |
| action-response / actual dynamics shift / adaptation | UNRESOLVED: 자산에 없음 | zero-action 자료나 nominal loss로 대체할 수 없음; 이번 범위 밖 |

**VERIFIED:** `evalcontract.run_evaluation`은 observation을 다음 action 계산에 사용하지만 저장 반환값은 EpisodeRecord다. candidate sequences, z, predicted r/Q, selected predicted return, actual per-step a/r/o, actual solver time을 보존하지 않는다. latency JSON은 이 누락을 메우지 못한다. 모델에는 observation decoder도 없어 latent만으로 CFD state를 복구할 수 없다.

## 7. 가설 확정 전에 해결할 blocker

| 우선 검토 항목 | 상태와 영향 | 해결됐다고 판단할 증거 |
|---|---|---|
| RL installed source vs Physics pin | VERIFIED: 5-file hash 불일치, 원인/영향 UNRESOLVED | 당시 설치본 확보 후 내용 diff, sensor extraction·reset·resampling 의미 대조. 기존 checkpoint 입력 contract 보존 |
| Native observation / trajectory 부재 | VERIFIED: 현재 export는 aggregates/fields | 원래 observation/action/reward/time alignment를 검증한 자료 또는 기존 원본 자료의 발견. 조용한 interpolation/reshape로 대체 불가 |
| Physical time 및 solver validity | VERIFIED: RL은 clock/count/failure guard evidence 부족 | actual accepted steps/clock 및 solver-success evidence. Physics easy 검증을 RL 실행으로 자동 전이하지 않음 |
| Dataset / initial-domain / prefix leakage | VERIFIED: Physics prefix 중복; cross-project 전체 동등성 UNRESOLVED | source trajectory/domain ID와 split provenance, 중복 grouping 및 실제 파일 identity 대조 |
| Planner stochasticity 및 causal attribution | VERIFIED: identical weights 다른 점수, 기존 contrast 2 episodes뿐 | sampling/IC/numerical variability를 분리한 증거. 현재 수치만으로 원인 선택 금지 |
| TD-MPC2 reset / warmup reproducibility | VERIFIED: 새 Box action_space를 seed하지 않음; reset randomize kwargs 버림 | 정확한 versioned reset/RNG contract 확인. common evaluator는 우회하므로 그 결과를 무효로 만들 이유는 아님 |
| Historical provenance / runtime | VERIFIED: TD-MPC2 dirty harness, 로컬 RL third_party 미존재 | pinned clean source와 patch/dependency 조합의 fresh inference 확인. 현 CPU deserialization을 CUDA 복원으로 표시하지 않음 |
| Scope coverage | VERIFIED: easy TD 계열만 있음, SAC/RBC trained baseline 없음 | 필요한 범위에 맞는 자산이 별도로 확인되어야 함. 연구 대상 확장은 이번에 결정하지 않음 |
| Protocol 혼용 / 낡은 문서 | VERIFIED: official_callback vs seeded; 삭제된 diagnostic CLI 안내 잔존 | 현재 executable/manifest를 기준으로 판단. 과거 “미실행” 문서보다 이후 diagnostic artifacts 우선 |

이 항목들은 관측된 성능 차이의 확정 원인이 아니다. 실행 경로·자산 보존·기존 nominal 비교는 확인되었으나, 물리적 표현 보존과 planning prediction fidelity를 결론내릴 증거는 별도로 필요하다. 새로운 physical parameter, robust loss, adaptation algorithm, 추가 실험 matrix는 선정하지 않았다.

## 이번 검증과 산출물

- `verification.json`: checkpoint archive hash, 공통 평가 결과 hash/reward 재계산, validation selection, 기존 diagnostics hash, 프로젝트 간 source hash 차이.
- `tensors.json`: CPU torch 역직렬화, finite tensor 확인, TD-MPC best/final identity. 16/16 통과.
- 기존 contract 관련 테스트: `test_environment_contract.py`, `test_evalcontract.py`, `test_nominal_audit.py`, `test_adapter_contract.py`: **28 passed, 19 subtests passed**. CUDA/FluidGym 수치 재현 테스트가 아니다.
- 기존 audit 및 Physics 보고서는 보존했다. 이 문서는 과거 진단 계획의 실행 지시를 승인하거나 재활성화하지 않는다.

주요 원본: `results/nominal_comparison/20261013T061228Z`, `results/nominal_diagnostics/20261013T111711Z`, `reports/nominal_implementation_audit_20261013`, 현재 `src/fluidgym_rl`, `docs/DIAGNOSTIC_CLEANUP.md`; 별도 `/Users/jihyekim/Desktop/Fluidgym-Physics/docs/{FINAL_NOMINAL_CHARACTERIZATION,TIME_AXIS_AND_EXECUTION_VALIDATION}.md`, 해당 runner와 pinned source. 모든 경로는 별도 표시 외 Fluidgym-RL root 기준이다.
