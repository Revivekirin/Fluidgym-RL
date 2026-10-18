# P1-A frozen-policy RNG attribution

**VERIFIED — 두 알고리즘 PASS.** P1-B의 실제 reset 관측과 +0.2 action 이후 관측을 사용했다. 각 모델 22 planning calls, CFD 0 steps. Checkpoint/config는 고정되었고 모델 state hash가 inference 전후 및 A/C 간 일치한다. Encoder와 reward/transition/Q/prior 출력은 유한하며 반복 encoder equality가 서버에서 검사되었다. 저장된 model-output NPZ의 유한성을 로컬에서도 재검사했다.

| Algorithm | 동일 RNG+reset 최대 action 차이 | 동일 RNG+동일 warm 최대 차이 | RNG 변경 시 action 차이 범위 | Warm 변경 시 차이 범위 |
|---|---:|---:|---:|---:|
| tdmpc | 0 | 0 | 0.02859169–0.06956702 | 0.00033200–0.05120429 |
| tdmpc2 | 0 | 0 | 0.00329646–0.07983774 | 0.01962802–0.05613357 |

**VERIFIED:** calls.json의 pre-call RNG hash 및 warm-state 배열을 직접 대조해 비교 쌍에서 한 요인만 바뀌었음을 확인했다. Warm states는 실제 planner 호출로 생성했으며 같은 t0=False 조건에서 비교했다. 사전 compile warmup은 두 t0 경로를 포함한다. 동일 RNG+state의 action은 이 검사에서 bitwise 일치했다.

**INFERRED:** 평가 모드의 planner도 stochastic하며, warm-start memory도 action에 영향을 준다. 동일 observation만 고정해서는 충분하지 않다.

**UNRESOLVED:** 각각의 Gaussian candidate/prior, Q subsampling, elite selection이 차이에 기여한 비율. 공유 RNG stream이라 각 요인을 독립 분리하지 않았다. Compiled planner intermediate candidates는 기록하지 않았다. 관측 두 개와 seed 대조의 결과를 모든 관측·GPU·실행으로 일반화할 수 없다. Finite 및 repeatable model output은 prediction calibration/nominal accuracy의 증거가 아니다.

원자료: [p1_run_20261016T072205Z_3a71b74d/A_tdmpc](p1_run_20261016T072205Z_3a71b74d/A_tdmpc), [p1_run_20261016T072205Z_3a71b74d/A_tdmpc2](p1_run_20261016T072205Z_3a71b74d/A_tdmpc2).

이번 문서는 반환된 서버 실행과 로컬 재계산을 구분한다. VERIFIED=기록/배열/계산으로 확인, INFERRED=제한적 해석, UNRESOLVED=증거 없음. 원본 tar 및 반환 보고서는 보존했다. 로컬 검토에서 새 CFD는 실행하지 않았다.
