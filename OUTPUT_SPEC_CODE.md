# Output Specification

## 학습 결과 출력
{
    "epoch": int,
    "train_loss": float,
    "val_loss": float,
    "accuracy": float,
    "f1_score": float
}

## 추론 결과 출력
{
    "prediction": "real" | "ai_generated",
    "confidence": float
}

## 로그 출력 형식
[INFO] message
[ERROR] message
[WARNING] message

## 파일 저장
- 모델: .pt
- 로그: .md
- 결과: .json

## 규칙
- 모든 결과는 JSON 형태로 출력
- key 이름 변경 금지