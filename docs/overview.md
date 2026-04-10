# AI Video Detection Baseline

이 프로젝트는 AI 생성 영상 탐지를 위한 베이스라인 파이프라인을 제공합니다.

## 구성
- `src/ai_video_detector/data.py`: 영상 로딩, 프레임 샘플링, 전처리
- `src/ai_video_detector/model.py`: VideoMAE 기반 인코더 래퍼와 MLP 분류기
- `src/ai_video_detector/train.py`: 학습 및 검증 루프
- `src/ai_video_detector/infer.py`: 단일 영상 추론
- `tests/test_pipeline.py`: 단위/통합 성격의 기본 검증

## 특징
- `transformers` 사용 가능 시 VideoMAE 로드
- 외부 모델이 없을 때는 경량 fallback encoder 사용
- 학습/추론 출력은 JSON 스펙 준수
- seed 고정 유틸리티 포함
