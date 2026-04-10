# Test Report

## 테스트 날짜
2026-04-09

## 결과
- Accuracy: 단위 테스트 기준 정상 계산 확인
- F1-score: 단위 테스트 기준 정상 계산 확인
- 추가 검증: 학습 1 epoch, validation loop, 단일 영상 추론, JSON 출력 형식 통과

## 문제점
- `pytest` 미설치
- 바이트코드 캐시(`__pycache__`) 쓰기 권한 제한 존재

## 개선 사항
- CI 또는 로컬 환경에 `pytest` 추가 시 더 풍부한 테스트 리포트 가능
- 실제 비디오 파일(mp4/avi) 기반 integration test fixture 추가 권장
