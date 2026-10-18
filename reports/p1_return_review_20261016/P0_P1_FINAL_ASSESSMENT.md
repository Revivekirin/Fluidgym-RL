# P0/P1 final assessment

**결론: 요청한 bounded current-pinned-environment P1 검증은 모두 통과했다. 현재 환경에서 frozen checkpoint를 로드하고 nominal planning/evaluation을 실행하는 데 확인된 blocker는 없다.** 정확한 historical binary equivalence 또는 model prediction accuracy가 검증됐다는 뜻은 아니다.

## VERIFIED

- Source/backend/dependencies/checkpoint hashes가 승인된 기대값과 일치한다. 실행 script/expected.json hash도 전달된 패키지와 일치한다.
- B: matched IC의 16 controls. 관측 차이 최대 2.3842e-7, reward/force/clock/counter 반복 차이 0. 요청·예상·실측 actuator 상태를 구분했다.
- A: 알고리즘별 실제 관측 2개/22 planner calls. 동일 RNG 및 memory에서는 선택 action 정확히 동일; RNG 또는 memory를 바꾸면 관측된 action이 달라졌다. Frozen model state hash 보존.
- C: 알고리즘별 10×80, 합계 1,600 controls. 과거 대비 mean reward/step 차이는 TD-MPC +0.000150354, TD-MPC2 +0.000631094. 기존 selection 보존.
- 원본 보존 체크 통과. 반환 파일에 대한 독립 로컬 검증 222개 통과. 새 training updates=0.

## 실제 비용

전체 wall time **1,393.058 s (23분 13초)**. B 24.974 s, A TD-MPC 10.147 s, A TD-MPC2 25.859 s, C TD-MPC 662.273 s, C TD-MPC2 661.005 s. 나머지는 프로세스 시작/정리 등이다.

Control steps **B=16, C=1,600, 총 1,616**. Native outer solver calls **B=484, C=20,199+20,199, 총 40,882**. 그중 control에 40,400 calls, reset에 482 calls가 사용되었다. Outer call과 adaptive internal accepted-step count는 별개다. Solver 실패 0. 최대 보고된 PyTorch CUDA allocation 66,411,520 bytes (약 63.34 MiB); 전체 GPU 사용량 측정은 아니다.

## INFERRED

P0의 nominal false-branch 정적 동등성과 P1의 현재 환경 관측/제어 검증을 합치면 소스 차이를 이유로 nominal 평가를 중단할 필요는 없다. 현재 환경을 명시적으로 고정해 후속 nominal model/planning 평가의 실행 기반으로 사용할 수 있다. 기존 nominal 성능 비교의 큰 방향도 유지되었다.

## UNRESOLVED 및 해석 한계

과거/current 소스를 같은 state/action으로 직접 비교하지 않았다. P1-B는 한 randomized TEST IC와 짧은 bounded actions만, P1-A는 실제 관측 두 개만, P1-C는 알고리즘별 한 번의 common protocol 실행만 포함한다. 수치 반복성은 physical accuracy/convergence 증명이 아니다. Planner prediction error, Q/reward calibration, candidate regret, representation 보존 및 dynamics-shift sensitivity는 미평가다. 기존 TEST를 살펴본 이 검증을 새로운 untouched holdout으로 취급할 수 없다.

## 다음 최소 조치 — 구현하지 않음

1. 이번 pinned source/backend/deps/checkpoint/config/hash와 RNG/warm-state 계약을 후속 nominal 평가의 기준으로 유지한다. Installation rollback 또는 model 수정은 불필요하다.
2. nominal model/planning 분석의 질문과 필요한 관측/target을 먼저 정한다. 이번 B의 짧은 scoped verification arrays를 광범위한 학습·진단 dataset으로 확대 해석하지 않는다.
3. prediction calibration이나 counterfactual planning을 하려면 별도 승인 범위에서 aligned transitions 및 state replay 요건을 검토한다. 이번에 broader transition logging, loss, adaptation, shift를 추가하지 않았다.
4. 정확한 historical equivalence나 반복 분산이 연구 주장에 꼭 필요한 경우에만 해당 추가 검증을 고려한다. 현재 결과만으로 무조건 재실행할 이유는 없다.

상세: [P1-B](P1_PRESCRIBED_ACTION.md), [P1-A](P1_FROZEN_POLICY_RNG.md), [P1-C](P1_CLOSED_LOOP_REPLICATION.md), [기계 판독 검증](verified_results.json), [episode별 비교](per_episode_comparison.csv).


이번 문서는 반환된 서버 실행과 로컬 재계산을 구분한다. VERIFIED=기록/배열/계산으로 확인, INFERRED=제한적 해석, UNRESOLVED=증거 없음. 원본 tar 및 반환 보고서는 보존했다. 로컬 검토에서 새 CFD는 실행하지 않았다.
