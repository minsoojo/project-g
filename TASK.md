# Task Definition

## 목표
AI 생성 영상 탐지를 위한 베이스라인 모델 구현

## 문제 정의
입력 영상이 실제 영상인지 AI 생성 영상인지 분류하는 이진 분류 문제

## 입력
- Video (mp4, avi 등)
- 프레임 시퀀스

## 출력
- Class: [Real, AI Generated]
- Probability score

## 세부 Task

### Task 1: 데이터 로딩
- 영상 → 프레임 시퀀스 변환
- 일정 길이 클립 생성

### Task 2: 전처리
- Resize
- Normalize
- Temporal Sampling

### Task 3: 모델 구현
- Video Encoder (VideoMAE)
- Classifier (MLP)

### Task 4: 학습
- Loss: BCEWithLogitsLoss
- Optimizer: AdamW

### Task 5: 평가
- Accuracy
- F1-score
- ROC-AUC

### Task 6: 추론
- 단일 영상 입력 → 결과 출력

## 완료 조건
- 학습 loop 정상 동작
- validation score 출력
- inference 가능