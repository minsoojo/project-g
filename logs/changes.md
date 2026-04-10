# Change Log

## 2026-04-09
- 변경 내용: `src/ai_video_detector/`에 데이터 로딩, 전처리, VideoMAE 래퍼, MLP 분류기, 학습/평가/추론 코드 추가
- 이유: TASK.md 기준 베이스라인 파이프라인이 비어 있었음
- 영향: 학습/검증/단일 영상 추론까지 가능한 최소 실행 구조 확보

## 2026-04-09
- 변경 내용: `tests/test_pipeline.py`, `tests/__init__.py`, `tests/conftest.py` 추가 및 `unittest` 기반 검증 구성
- 이유: 테스트 필수 규칙과 sandbox 환경의 `pytest` 부재를 동시에 만족하기 위함
- 영향: 외부 다운로드 없이 핵심 흐름 검증 가능

## 2026-04-09
- 변경 내용: `docs/overview.md`, `logs/plan.md`, `logs/test_report.md` 업데이트
- 이유: Harness 문서가 요구하는 계획/결과 기록 반영
- 영향: 작업 이력과 검증 상태 추적 가능
