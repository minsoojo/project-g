# Codebase Context

## 프로젝트 개요
본 프로젝트는 AI 생성 영상 탐지를 위한 베이스라인 모델 구현을 목표로 한다.

## 핵심 구조
- Video Encoder: 시공간 특징 추출
- Classifier: 이진 분류 수행

## 사용 기술
- PyTorch
- HuggingFace Transformers
- VideoMAE

## 데이터 흐름
Video → Frames → Tensor → Model → Prediction

## 모델 구조
Input Video
    ↓
Frame Sampling
    ↓
Video Encoder (VideoMAE)
    ↓
Feature Vector
    ↓
MLP Classifier
    ↓
Prediction

## 향후 확장
- Transformer head 추가
- XAI (Grad-CAM, Attention)
- Temporal modeling 강화

## 제약 조건
- GPU 자원 제한
- 데이터셋 크기 제한

## 중요 포인트
- Temporal 정보 유지
- 과도한 augmentation 금지