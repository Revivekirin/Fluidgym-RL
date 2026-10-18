# P1 Nominal reproducibility — 2026-10-16

> 최신 상태(2026-10-16): 사용자 실행 P1-B/A/C가 모두 완료되었으며 반환 자료의 로컬 재검증 222개가 통과했다. 아래 과거 NOT RUN/BLOCKED 문구는 실행 전 상태다. 현재 판정은 [P0/P1 최종 평가](../reports/p1_return_review_20261016/P0_P1_FINAL_ASSESSMENT.md)를 참조한다. Current-pinned replication이며 historical binary equivalence는 미검증이다.


상태: 원격 실행은 사용자가 진행한다. 이번에는 소스 감사와 실행 전 자료 준비만 수행했다. CFD 0 steps, learner updates 0. P1을 통과했다고 표시하지 않는다.

## A. Fixed-input policy

UNRESOLVED / NOT RUN: 기존 export에는 raw native 302-vector 시계열이 없고 Physics fields는 같은 observation이 아니다. 로컬 Torch 2.14 CPU는 기존 CUDA 2.9.1 runtime과도 다르다. zero vector를 물리적으로 검증된 valid observation으로 꾸미지 않았다. weights finite 및 best/final identity는 기존 조사에서 확인했으나 fixed-input planner 재현 실험은 아니다.

VERIFIED source: TD-MPC encoder는 deterministic feed-forward이나 pi(z,min_std), Gaussian candidates, NumPy elite sampling이 stochastic이다. TD-MPC2 pi Gaussian, Q 2-of-5 subsampling, Gaussian candidates 및 Gumbel elite selection도 stochastic이다. eval_mode는 마지막 exploration noise만 제거한다. warm start는 _prev_mean과 t0로 제어된다.

필요한 분리: 동일 observation에서 (1) RNG+warm state 모두 복원, (2) RNG만 복원, (3) warm state만 복원, (4) 둘 다 진행. 동일 pre-call RNG를 encode/prior probe가 소비하여 planner baseline을 바꾸지 않도록 각 probe 전 상태 복원 필요. encoder, prior mean/std, accessible candidates, action, RNG hash, _prev_mean 비교는 inference 검사이며 representation diagnostics를 추가하는 작업이 아니다. 실제 실행 코드는 아직 작성/실행하지 않았다.

## B. Prescribed-action environment

UNRESOLVED / NOT RUN: 실제 설치본, dataset bytes, backend가 확인되기 전 CFD 비교를 시작하지 않는다. selected domain index/split/input hashes 및 reset clock을 확인해야 한다. raw velocity/pressure와 flattened velocity, command와 smoothed applied control, reward/Cd/Cl, actual clock/count, false single_step status를 구분해야 한다. 기존 aggregate CSV는 이 검사를 충족하지 않는다.

P0 runtime gate 통과 후 검토 가능한 최소 범위는 동일 seed/reset 설정으로 두 번의 짧은 prescribed sequence이다. 0→양수→음수→0의 4 control steps×2=8 steps는 reset 및 smoothing 초기 분기에 대한 제한된 검사일 뿐 80-step equivalence 인증이 아니다. full-horizon 경계를 비교하려면 별도 80-step 검사가 필요하다. source 비교는 각 source를 별도 process에서 사용하고 installed 파일을 덮어쓰지 않아야 한다. 현재 두 runtime의 안전한 import isolation이 미확정이므로 이 예시는 실행 승인/최종 matrix가 아니다.

## C. Frozen closed loop

VERIFIED: 기존 shared evaluation은 frozen val-best TD-MPC 50k / TD-MPC2 44k, test split, seeded randomize=True, seeds 0..9, 80 steps, saved planner config다. 알고리즘별 800 steps. 두 알고리즘을 한 차례씩 같은 조건으로 반복하면 1,600 control steps이며 새 selection은 없다. 역사적 step throughput 약 0.8–0.9 s/step을 사용하면 약 21–24분 + startup/compile 추정이다; runtime 보장은 아니다. 1–2 episode로 줄이면 subset probe이고 전체 common-protocol repeat라고 부르면 안 된다.

UNRESOLVED / NOT RUN: fresh closed-loop repeat. 초기 예산은 source/runtime와 prescribed-action 검사 결과 이후 확정해야 한다. 원격 runtime deps를 업데이트하거나 checkpoint/config를 변경하지 않는다.

`historical_episode_comparison.csv`는 기존 test_best vs test_final 20개 episode pair를 제시한다. TD-MPC는 weights가 같지만 serialization/evaluation은 별개이며 원인이 분리되지 않았다. TD-MPC2는 weights가 달라 reproducibility test가 아니다. 이 테이블은 새 repeat 결과가 아니다. 기존 44k actor/plan 2-episode 결과도 inference mode가 달라 repeat로 간주하지 않는다.

## 명령 및 자원

첫 단계는 보고서 폴더의 remote_preflight.py만 실행한다. environment 생성/reset/step, checkpoint deserialization, pip install, W&B, solver 변경을 하지 않는다. 현재 import, distribution metadata, 71개 Python source, binary hash, dependency versions, GPU 상태, checkpoint/evaluation/data hash를 새 고유 폴더에 수집한다. 서버 관측값과 historical provenance를 먼저 대조해야 한다.

실행: `/nfs4/jhkim/conda_envs/fluidgym-test/bin/python -B remote_preflight.py --repo /nfs4/jhkim/repos/Fluidgym-RL --data-root /home/jhkim/.local/share/FluidGym/initial_domains`.

정확한 Python/data 경로는 기존 기록 및 Physics handoff 기준이다. script는 존재하지 않는 data를 다운로드/생성하지 않으며 import 실패도 JSON에 남긴다. 경로 오류는 출력으로 보고하고 다른 환경으로 자동 대체하지 않는다. source/NPZ/checkpoint hash 계산의 CPU/I/O 비용만 있으며 actual wall seconds는 출력 JSON에 기록한다. P1 GPU 자원 사용은 아직 0이다.


## 서버 반환 증거 업데이트 — 2026-10-16 06:47 UTC

VERIFIED: 사용자 실행 preflight archive를 확인했다. 현재 import는 `/nfs4/jhkim/repos/Fluidgym-Physics/third_party/fluidgym/src/fluidgym/__init__.py`이며 direct_url은 같은 checkout의 editable 설치다. archive 내 Python 72개는 보고된 hash 및 로컬 clean pinned source와 모두 일치한다. compiled PISOtorch hash `294edfc263cd57a7d4da0aeea72e76abcba3c586901511bac703659717ea2bea`도 Physics handoff와 일치한다. 현재 서버 checkout의 git clean 여부 자체는 preflight가 수집하지 않았으므로 소스 bytes 일치와 구분한다.

VERIFIED: 과거 RL installed source 71개와 비교하면 기존 5개 변경 및 `envs/util/diff_tools.py` 추가가 있다. 현재 환경은 과거 wheel release 소스와 동일하지 않다. 전환 시점/설치 명령은 UNRESOLVED. 현재 editable 설치는 확인했으나 언제 바뀌었는지는 이 기록으로 알 수 없다.

VERIFIED: 현재 서버의 checkpoint 16개와 기존 evaluation 파일 132개의 hash가 로컬 원본과 모두 일치한다. Cylinder easy 초기조건 파일 91개도 과거 shared evaluation manifest와 모두 일치한다. import된 initial_domains_path는 `/home/jhkim/.local/share/FluidGym/initial_domains`이다. 비교 가능한 과거 dependency version에 차이가 없으며 Torch 2.9.1+cu126/CUDA 12.6/RTX3090, deterministic_algorithms=False이다. 설정 flag의 현재 값이 모든 과거 실행에서 동일했다는 뜻은 아니다.

UNRESOLVED: source-level nominal false branch 동일성은 실제 numerical observation/transition equivalence를 대신하지 못한다. P1 A/B는 NOT RUN. C는 역사적 source identity gate 미충족으로 보류한다. 지금 common evaluator를 실행하면 changed-source reevaluation이므로 historical repeat로 표기할 수 없다. 원본 설치를 되돌리거나 Physics editable checkout을 수정하지 않는다.

다음 최소 순서: (1) historical hash-matched Python source와 현재 binary/dependencies를 process-local하게 분리한 import 검증, (2) 격리 가능할 때만 작은 prescribed-action 비교로 observation/reset/transition을 확인, (3) 확보한 실제 native observation으로 frozen-policy RNG/warm-start 분리, (4) 두 알고리즘의 기존 800-step protocol을 각각 반복. 과거 compiled backend hash가 RL 기록에 없어 동일 binary라는 역사적 주장도 보류한다. 필요한 격리 검증 없이 CFD 명령을 제공하지 않는다. 이 순서는 새 실험 설계 확정이 아니라 실행 전 gate 순서다.

원격 preflight 실제 wall time: 13.1635 seconds. CFD steps=0, training updates=0. 현재까지 P1 결과를 새로 생성하지 않았다.

기계 판독 증거: `../reports/p0_p1_audit_20261016T063722Z/server_review_20261016T064745Z/verification.json`.
