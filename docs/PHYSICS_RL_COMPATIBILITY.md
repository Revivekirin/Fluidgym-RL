# Physics–RL compatibility — 2026-10-16

> 최신 상태(2026-10-16): 사용자 실행 P1-B/A/C가 모두 완료되었으며 반환 자료의 로컬 재검증 222개가 통과했다. 아래 과거 NOT RUN/BLOCKED 문구는 실행 전 상태다. 현재 판정은 [P0/P1 최종 평가](../reports/p1_return_review_20261016/P0_P1_FINAL_ASSESSMENT.md)를 참조한다. Current-pinned replication이며 historical binary equivalence는 미검증이다.


PHYSICS_TO_RL_HANDOFF.md를 읽고 기록된 RL contract와 대조했다. full physical equivalence는 UNRESOLVED다.

| Handoff 조건 | 상태 | 근거 / 제한 |
|---|---|---|
| Pin/tree 및 실제 source identity | VERIFIED: Physics clean pin/tree; RL 역사적 71/71 source는 v0.1.2 release와 일치 | pinned editable source와 직접 일치하는 것은 아님. nominal false resampler body 동일은 확인 |
| Runtime import/backend | UNRESOLVED | historical path/versions 있음. 현재 서버 actual import 및 compiled binary hash 필요 |
| Dataset | 부분 VERIFIED | RL manifests는 train/val/test 파일 hash를 기록. Physics는 TEST campaign. 전체 cross-project 원본 bytes 및 현재 경로 검증 미완료 |
| Cylinder easy Re100 / 80-step / 302 velocity | VERIFIED code + historical manifest | raw velocity (151,2), pressure (151,), wrapper가 pressure 제외. runtime raw 관측 재비교 미실행 |
| Action/smoothing | VERIFIED source | scalar [-1,1], 매 outer call alpha=.1; 동일 command가 같은 transition을 주는지는 미실행 |
| Reward | VERIFIED source + historical live check | Cd_ref−mean(Cd)−abs(mean(Cl)); native ref와 측정 TRAIN baseline 구분 |
| Reset / actual IC | 부분 VERIFIED | common seeded randomize=True, split 지정. seed는 index가 아니며 실제 선택 ID/perturbation의 fresh equality 미확인 |
| Time/failure | UNRESOLVED runtime | 요청 .01×25와 actual accepted steps 구분. RL aggregate는 solver clock/failure trace 미포함 |
| Split/protocol | VERIFIED difference | Physics는 TEST zero-action, RL은 train/val/test policy evaluation. Physics extended는 benchmark 대체 불가 |
| Stored fields as policy input | VERIFIED incompatible as direct substitution | resampled public fields와 native sensor obs 다름; 모든 step obs 미저장 |
| Other 5 environments | UNRESOLVED / out of current RL scope | 현재 nominal checked evaluator와 checkpoints는 Cylinder easy만 지원 |

판정(INFERRED): 다섯 source diff는 nominal differentiable=False 경로에서 동일 compiled resampler로 연결되므로 이전의 광범위한 observation-drift 우려를 줄인다. 실제 reset/observation/transition equality는 아직 증명되지 않았다. full physical equivalence 또는 reproducibility passed를 선언하지 않는다.

## 최소 다음 조치, 우선순위

1. 사용자 실행 remote_preflight 결과로 current installed bytes/import/RECORD/direct_url/backend/deps/data hashes를 확정한다. 교체나 업그레이드는 하지 않는다.
2. 실제 native observation과 matching runtime 확보 후 fixed-input RNG/warm-start 분리 검사를 수행한다. synthetic vector는 별도 numerical smoke로만 표시한다.
3. source isolation 가능 여부를 확인한 뒤 짧은 prescribed-action reset/transition 검증의 정확한 범위를 정한다. 실패/시간/IC 자료가 없으면 equivalence 판정을 보류한다.
4. B가 해석 가능해진 다음 frozen common closed-loop 반복을 실행하고 episode별로 비교한다. selection 변경 없음.
5. 해당 증거가 갖춰진 후 nominal representation/planning 평가의 자료 요건을 재검토한다. 지금 transition logging, 새 metric, algorithm 또는 shift 구현을 추가하지 않는다.

이 조치들은 자동 실행하지 않는다. 사용자 요청에 따라 서버 실행은 사용자에게 맡긴다.


## 서버 반환 증거 업데이트 — 2026-10-16 06:47 UTC

VERIFIED: 사용자 실행 preflight archive를 확인했다. 현재 import는 `/nfs4/jhkim/repos/Fluidgym-Physics/third_party/fluidgym/src/fluidgym/__init__.py`이며 direct_url은 같은 checkout의 editable 설치다. archive 내 Python 72개는 보고된 hash 및 로컬 clean pinned source와 모두 일치한다. compiled PISOtorch hash `294edfc263cd57a7d4da0aeea72e76abcba3c586901511bac703659717ea2bea`도 Physics handoff와 일치한다. 현재 서버 checkout의 git clean 여부 자체는 preflight가 수집하지 않았으므로 소스 bytes 일치와 구분한다.

VERIFIED: 과거 RL installed source 71개와 비교하면 기존 5개 변경 및 `envs/util/diff_tools.py` 추가가 있다. 현재 환경은 과거 wheel release 소스와 동일하지 않다. 전환 시점/설치 명령은 UNRESOLVED. 현재 editable 설치는 확인했으나 언제 바뀌었는지는 이 기록으로 알 수 없다.

VERIFIED: 현재 서버의 checkpoint 16개와 기존 evaluation 파일 132개의 hash가 로컬 원본과 모두 일치한다. Cylinder easy 초기조건 파일 91개도 과거 shared evaluation manifest와 모두 일치한다. import된 initial_domains_path는 `/home/jhkim/.local/share/FluidGym/initial_domains`이다. 비교 가능한 과거 dependency version에 차이가 없으며 Torch 2.9.1+cu126/CUDA 12.6/RTX3090, deterministic_algorithms=False이다. 설정 flag의 현재 값이 모든 과거 실행에서 동일했다는 뜻은 아니다.

UNRESOLVED: source-level nominal false branch 동일성은 실제 numerical observation/transition equivalence를 대신하지 못한다. P1 A/B는 NOT RUN. C는 역사적 source identity gate 미충족으로 보류한다. 지금 common evaluator를 실행하면 changed-source reevaluation이므로 historical repeat로 표기할 수 없다. 원본 설치를 되돌리거나 Physics editable checkout을 수정하지 않는다.

다음 최소 순서: (1) historical hash-matched Python source와 현재 binary/dependencies를 process-local하게 분리한 import 검증, (2) 격리 가능할 때만 작은 prescribed-action 비교로 observation/reset/transition을 확인, (3) 확보한 실제 native observation으로 frozen-policy RNG/warm-start 분리, (4) 두 알고리즘의 기존 800-step protocol을 각각 반복. 과거 compiled backend hash가 RL 기록에 없어 동일 binary라는 역사적 주장도 보류한다. 필요한 격리 검증 없이 CFD 명령을 제공하지 않는다. 이 순서는 새 실험 설계 확정이 아니라 실행 전 gate 순서다.

원격 preflight 실제 wall time: 13.1635 seconds. CFD steps=0, training updates=0. 현재까지 P1 결과를 새로 생성하지 않았다.

기계 판독 증거: `../reports/p0_p1_audit_20261016T063722Z/server_review_20261016T064745Z/verification.json`.
