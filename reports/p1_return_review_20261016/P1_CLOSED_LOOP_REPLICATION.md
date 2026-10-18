# P1-C current-pinned closed-loop replication

**VERIFIED — 두 알고리즘 PASS.** B와 두 A gate 통과 후 각각 TEST seeds 0–9, randomize=True, 80 steps, frozen val-selected TD-MPC 50k / TD-MPC2 44k를 평가했다. 각 800, 합계 1,600 controls. 재선정 없음. 20개 episode 전부 horizon truncation으로 완료되었고 solver 실패는 0이다.

| Algorithm | 과거 reward/step | 현재 reward/step | 평균 차이 | 최대 episode별 절대 차이 |
|---|---:|---:|---:|---:|
| tdmpc | 0.037628384 | 0.037778738 | +0.000150354 | 0.004963581 |
| tdmpc2 | -0.077376859 | -0.076745765 | +0.000631094 | 0.003015045 |

| Algorithm | 현재 Cd | 과거 대비 Cd 차이 | 현재 abs Cl | 과거 대비 abs Cl 차이 |
|---|---:|---:|---:|---:|
| tdmpc | 3.110115987 | +0.000068784 | 0.180260793 | -0.000219138 |
| tdmpc2 | 3.220350184 | -0.000330293 | 0.184551099 | -0.000300802 |

**VERIFIED:** 실제 선택된 base-domain indices는 두 알고리즘 모두 `[8,4,8,8,7,6,4,9,7,4]`. 10 seeds는 10개의 서로 다른 saved domains를 뜻하지 않으며 5개 base domains가 선택되었다. 같은 domain에서도 다른 seed의 reset perturbation은 별개다. 과거 실행에는 동일 수준의 actual-domain trace가 없어 과거와 실제 상태가 정확히 같았다고 인증하지 않는다.

**VERIFIED:** episode CSV에서 mean/return/boundary/seed를 재계산했고 summary와 일치한다. [per_episode_comparison.csv](per_episode_comparison.csv)는 모든 20 episode의 5개 metric 대조를 포함한다. 평균 차이가 작아도 모든 episode가 동일한 것은 아니다.

**INFERRED:** 현재 pinned 환경에서 과거 nominal 성능 수준과 TD-MPC/TD-MPC2의 조건부 순위가 유지되었다. 이 평가의 완료 gate는 계약·복원·유한성·예산이며, reward가 과거와 같아야 한다는 gate가 아니다.

**UNRESOLVED:** 역사적 차이 중 소스·CUDA 수치·sampling trajectory가 차지하는 비율. Current-pinned closed loop를 각각 한 번 실행했으므로 current-versus-current 전체 episode repeat 분포나 통계적 equivalence margin을 측정한 것은 아니다. 학습 seed 일반화, representation robustness, shift sensitivity도 평가하지 않았다.

원자료: [p1_run_20261016T072205Z_3a71b74d/C_tdmpc](p1_run_20261016T072205Z_3a71b74d/C_tdmpc), [p1_run_20261016T072205Z_3a71b74d/C_tdmpc2](p1_run_20261016T072205Z_3a71b74d/C_tdmpc2).

이번 문서는 반환된 서버 실행과 로컬 재계산을 구분한다. VERIFIED=기록/배열/계산으로 확인, INFERRED=제한적 해석, UNRESOLVED=증거 없음. 원본 tar 및 반환 보고서는 보존했다. 로컬 검토에서 새 CFD는 실행하지 않았다.
