

## 서버 반환 증거 업데이트 — 2026-10-16 06:47 UTC

VERIFIED: 사용자 실행 preflight archive를 확인했다. 현재 import는 `/nfs4/jhkim/repos/Fluidgym-Physics/third_party/fluidgym/src/fluidgym/__init__.py`이며 direct_url은 같은 checkout의 editable 설치다. archive 내 Python 72개는 보고된 hash 및 로컬 clean pinned source와 모두 일치한다. compiled PISOtorch hash `294edfc263cd57a7d4da0aeea72e76abcba3c586901511bac703659717ea2bea`도 Physics handoff와 일치한다. 현재 서버 checkout의 git clean 여부 자체는 preflight가 수집하지 않았으므로 소스 bytes 일치와 구분한다.

VERIFIED: 과거 RL installed source 71개와 비교하면 기존 5개 변경 및 `envs/util/diff_tools.py` 추가가 있다. 현재 환경은 과거 wheel release 소스와 동일하지 않다. 전환 시점/설치 명령은 UNRESOLVED. 현재 editable 설치는 확인했으나 언제 바뀌었는지는 이 기록으로 알 수 없다.

VERIFIED: 현재 서버의 checkpoint 16개와 기존 evaluation 파일 132개의 hash가 로컬 원본과 모두 일치한다. Cylinder easy 초기조건 파일 91개도 과거 shared evaluation manifest와 모두 일치한다. import된 initial_domains_path는 `/home/jhkim/.local/share/FluidGym/initial_domains`이다. 비교 가능한 과거 dependency version에 차이가 없으며 Torch 2.9.1+cu126/CUDA 12.6/RTX3090, deterministic_algorithms=False이다. 설정 flag의 현재 값이 모든 과거 실행에서 동일했다는 뜻은 아니다.

UNRESOLVED: source-level nominal false branch 동일성은 실제 numerical observation/transition equivalence를 대신하지 못한다. P1 A/B는 NOT RUN. C는 역사적 source identity gate 미충족으로 보류한다. 지금 common evaluator를 실행하면 changed-source reevaluation이므로 historical repeat로 표기할 수 없다. 원본 설치를 되돌리거나 Physics editable checkout을 수정하지 않는다.

다음 최소 순서: (1) historical hash-matched Python source와 현재 binary/dependencies를 process-local하게 분리한 import 검증, (2) 격리 가능할 때만 작은 prescribed-action 비교로 observation/reset/transition을 확인, (3) 확보한 실제 native observation으로 frozen-policy RNG/warm-start 분리, (4) 두 알고리즘의 기존 800-step protocol을 각각 반복. 과거 compiled backend hash가 RL 기록에 없어 동일 binary라는 역사적 주장도 보류한다. 필요한 격리 검증 없이 CFD 명령을 제공하지 않는다. 이 순서는 새 실험 설계 확정이 아니라 실행 전 gate 순서다.

원격 preflight 실제 wall time: 13.1635 seconds. CFD steps=0, training updates=0. 현재까지 P1 결과를 새로 생성하지 않았다.

기계 판독 증거: `../reports/p0_p1_audit_20261016T063722Z/server_review_20261016T064745Z/verification.json`.
