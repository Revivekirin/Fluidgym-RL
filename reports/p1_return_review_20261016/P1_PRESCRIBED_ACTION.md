# P1-B prescribed-action verification

**VERIFIED — PASS.** 현재 pinned CylinderJet2D-easy, TEST seed=0, randomize=True, 실제 initial-domain index=8에서 zero와 [0,+.2,−.2,0] 시퀀스를 각각 두 번 실행했다. 4 trials ×4=16 control steps. Outer solver calls는 reset 포함 484 = control 400 + reset 84. 실패 flag/exception 0.

| 반복 간 최대 절대차 | Zero | Bounded nonzero |
|---|---:|---:|
| Raw velocity / flat observation | 2.3841858e-7 | 1.1920929e-7 |
| Raw pressure | 1.1920929e-7 | 1.1920929e-7 |
| Reward, drag, lift | 0 | 0 |
| Solver clock, internal step counter | 0 | 0 |

**VERIFIED:** 사전 선언 atol=rtol=1e-5를 통과했다. 관측은 bitwise 동일하지 않다. Raw velocity (151,2), pressure (151,), float32; policy input은 velocity sensor-major flatten (302,)와 정확히 일치한다. Reward는 Cd_ref−Cd−abs(Cl)와 일치한다. 코드 경로는 25 outer calls/control이며 각 call의 native success를 확인했다.

**VERIFIED:** 요청 action, source recurrence로 산출한 expected control, 실제 `_apply_action` 호출 인자 및 private smoothing memory를 구분해 저장했다. Bounded sequence의 control-step 종료 actuator memory는 0, +0.18564205, −0.17231484, −0.01237045다. 마지막 요청이 0이어도 memory는 즉시 0이 되지 않는다. Runtime 적용 인자와 source recurrence는 지정 tolerance 이내, 마지막 인자와 memory는 정확히 일치한다. 독립적인 jet mass-flux 측정은 아니다.

Reset solver clock=1.2099997997, 종료=2.2099988461. Nominal 4×.25=1과 실제 float32 누적 clock은 구분한다. 반복 간 clocks/counters 일치는 모든 accepted dt 또는 시간적 수렴성 증명이 아니다.

**INFERRED:** 이 IC와 짧은 action 범위에서 인터페이스와 수치적 반복성이 유지된다.

**UNRESOLVED:** 다른 IC·장시간·큰 action, physical accuracy, grid/time convergence, 과거 wheel과 현재 pin의 직접 transition equality. 이 검증은 action-response의 광범위한 식별 실험이 아니다.

원자료: [p1_run_20261016T072205Z_3a71b74d/B](p1_run_20261016T072205Z_3a71b74d/B). 재계산: [verified_results.json](verified_results.json).


이번 문서는 반환된 서버 실행과 로컬 재계산을 구분한다. VERIFIED=기록/배열/계산으로 확인, INFERRED=제한적 해석, UNRESOLVED=증거 없음. 원본 tar 및 반환 보고서는 보존했다. 로컬 검토에서 새 CFD는 실행하지 않았다.
