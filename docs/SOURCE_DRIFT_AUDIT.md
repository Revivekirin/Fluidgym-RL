# P0 Source drift audit — 2026-10-16

> 최신 상태(2026-10-16): 사용자 실행 P1-B/A/C가 모두 완료되었으며 반환 자료의 로컬 재검증 222개가 통과했다. 아래 과거 NOT RUN/BLOCKED 문구는 실행 전 상태다. 현재 판정은 [P0/P1 최종 평가](../reports/p1_return_review_20261016/P0_P1_FINAL_ASSESSMENT.md)를 참조한다. Current-pinned replication이며 historical binary equivalence는 미검증이다.


VERIFIED는 로컬 소스/기록 확인, INFERRED는 조건부 해석, UNRESOLVED는 실행 증거 부재를 뜻한다. 사용자 지시에 따라 원격 실행은 수행하지 않았다. 새 CFD/학습/파일 교체 없음.

## 정확한 소스 확보

VERIFIED: 공통 RL 평가 manifest가 기록한 installed Python 소스 71개 전부가 로컬 Git 이력의 `60bf1f4` (v0.1.2) bytes와 SHA-256이 일치한다. 다섯 파일의 이전 bytes를 이 이력에서 복원했다. Physics checkout은 clean `50071140b73a2a917fb96d33522a769f903f8c27`, src tree `8a8476b966e496341676ea36384589028f4dc450`이다. 따라서 단순 hash 차이에서 한 단계 나아가 기록된 설치본에 대한 exact diff를 확보했다.

기록된 import root는 `/nfs4/jhkim/conda_envs/fluidgym-test/lib/python3.12/site-packages/fluidgym`. 당시 Python 3.12, Torch 2.9.1+cu126이다. 현재 로컬 Python 3.11.16/Torch 2.14.0 CPU이며 FluidGym은 미설치다. 현재 서버 import 경로는 UNRESOLVED.

INFERRED: setup script의 `fluidgym==0.1.2` wheel 설치와 71/71 소스 일치는 wheel release와 7 commits 이후 pinned revision 간 차이로 설명된다. 현재 서버의 editable 여부, RECORD와 실제 파일, direct_url, 이후 local modifications는 아직 검증하지 않았으므로 설치 메커니즘을 확정하지 않는다.

## 파일별 변화와 caller

| 파일 | 변화/분류 | nominal evaluation 호출 여부 |
|---|---|---|
| envs/fluid_env.py | get_velocity/get_pressure에서 differentiable 인자 전달. observation/field-affecting conditional API 변경. reset/step/state construction 변경 없음 | Cylinder sensor 경로는 아래 extraction이 직접 resample하므로 이 두 getter를 직접 통과하지 않음. rendering/public-field 경로는 호출 가능 |
| envs/util/__init__.py | get_flat_state, mark_state_differentiable import/export 추가. import/API-only | package import 시 실행되지만 이 함수 자체를 RL evaluator는 호출하지 않음. import dependency 부작용은 실제 runtime 점검 필요 |
| envs/util/obs_extraction.py | extract_global_2d_obs, extract_global_3d_obs에서 differentiable 전달. observation-affecting conditional dispatch | Cylinder._get_global_obs → extract_global_2d_obs → _resample_block_data. reset 및 매 step 실제 코드 경로에 있음; 3D는 미사용 |
| simulation/pict/data/resample.py | 기존 함수가 dispatcher가 되고 old body는 *_nondiff로 이동. *_diff 및 _fill_empty_cells 추가. observation-affecting conditional implementation | differentiable=False이면 compiled nondiff 경로. nominal True override 증거 없음 |
| simulation/pict/util/output.py | _resample_block_data의 인자/전달 확장. 이름이 output이어도 diagnostics-only 아님 | sensor extraction이 직접 호출함. field/output writer도 transitive caller |

VERIFIED: 기존 resampler body와 새 nondiff body는 docstring 제외 AST가 동일하다. sensor 좌표, indexing, velocity permute/vector-last 순서, flatten wrapper, dtype 처리의 변경은 이 diff에 없다. 새 torch differentiable 분기의 dtype/연산 변경을 nominal false 경로에 잘못 귀속하면 안 된다. 새 분기는 분석 대상 코드지만 nominal에서 실행된다는 증거는 없다.

INFERRED: differentiable=False, 동일 native inputs/backend/kwargs라는 조건에서는 같은 compiled resampling 연산으로 도달한다. 이 다섯 변경이 nominal 관측 의미를 바꾸었다는 근거는 발견하지 못했다.

UNRESOLVED: 실제 observation/transition 수치 동등성. AST 동일성은 CUDA kernel, binary, dataset 및 RNG 동등성의 증명이 아니다. 새 source로 자동 교체하지 않았다. 실제 설치본과 pinned source 비교 실행도 하지 않았다.

## 증거

`../reports/p0_p1_audit_20261016T063722Z/provenance.json`: 전체 hash, 역사적 runtime, Git SHA.
`../reports/p0_p1_audit_20261016T063722Z/diffs/`: 5개 exact unified diff.
`../reports/p0_p1_audit_20261016T063722Z/recorded_runtime_sources/`: 역사적 hash-matched source bytes.
`../reports/p0_p1_audit_20261016T063722Z/static_semantics.json`: AST 확인과 미실행 상태.


## 서버 반환 증거 업데이트 — 2026-10-16 06:47 UTC

VERIFIED: 사용자 실행 preflight archive를 확인했다. 현재 import는 `/nfs4/jhkim/repos/Fluidgym-Physics/third_party/fluidgym/src/fluidgym/__init__.py`이며 direct_url은 같은 checkout의 editable 설치다. archive 내 Python 72개는 보고된 hash 및 로컬 clean pinned source와 모두 일치한다. compiled PISOtorch hash `294edfc263cd57a7d4da0aeea72e76abcba3c586901511bac703659717ea2bea`도 Physics handoff와 일치한다. 현재 서버 checkout의 git clean 여부 자체는 preflight가 수집하지 않았으므로 소스 bytes 일치와 구분한다.

VERIFIED: 과거 RL installed source 71개와 비교하면 기존 5개 변경 및 `envs/util/diff_tools.py` 추가가 있다. 현재 환경은 과거 wheel release 소스와 동일하지 않다. 전환 시점/설치 명령은 UNRESOLVED. 현재 editable 설치는 확인했으나 언제 바뀌었는지는 이 기록으로 알 수 없다.

VERIFIED: 현재 서버의 checkpoint 16개와 기존 evaluation 파일 132개의 hash가 로컬 원본과 모두 일치한다. Cylinder easy 초기조건 파일 91개도 과거 shared evaluation manifest와 모두 일치한다. import된 initial_domains_path는 `/home/jhkim/.local/share/FluidGym/initial_domains`이다. 비교 가능한 과거 dependency version에 차이가 없으며 Torch 2.9.1+cu126/CUDA 12.6/RTX3090, deterministic_algorithms=False이다. 설정 flag의 현재 값이 모든 과거 실행에서 동일했다는 뜻은 아니다.

UNRESOLVED: source-level nominal false branch 동일성은 실제 numerical observation/transition equivalence를 대신하지 못한다. P1 A/B는 NOT RUN. C는 역사적 source identity gate 미충족으로 보류한다. 지금 common evaluator를 실행하면 changed-source reevaluation이므로 historical repeat로 표기할 수 없다. 원본 설치를 되돌리거나 Physics editable checkout을 수정하지 않는다.

다음 최소 순서: (1) historical hash-matched Python source와 현재 binary/dependencies를 process-local하게 분리한 import 검증, (2) 격리 가능할 때만 작은 prescribed-action 비교로 observation/reset/transition을 확인, (3) 확보한 실제 native observation으로 frozen-policy RNG/warm-start 분리, (4) 두 알고리즘의 기존 800-step protocol을 각각 반복. 과거 compiled backend hash가 RL 기록에 없어 동일 binary라는 역사적 주장도 보류한다. 필요한 격리 검증 없이 CFD 명령을 제공하지 않는다. 이 순서는 새 실험 설계 확정이 아니라 실행 전 gate 순서다.

원격 preflight 실제 wall time: 13.1635 seconds. CFD steps=0, training updates=0. 현재까지 P1 결과를 새로 생성하지 않았다.

기계 판독 증거: `../reports/p0_p1_audit_20261016T063722Z/server_review_20261016T064745Z/verification.json`.
