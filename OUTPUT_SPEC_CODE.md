# Output Specification

## 학습 결과 파일

`train_metrics.json`은 epoch summary 객체를 epoch 순서대로 누적한 JSON 배열이다.

```json
[
  {
    "epoch": 1,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "accuracy": 0.0,
    "f1_score": 0.0,
    "roc_auc": 0.0
  }
]
```

## 단일 영상 추론 결과

`infer` 명령의 기본 출력이다. `--output-path`가 있으면 같은 payload를 JSON 파일로 저장한다.

```json
{
  "prediction": "real",
  "confidence": 0.0
}
```

- `prediction`: `"real"` 또는 `"ai_generated"`
- `confidence`: 모델 sigmoid score. `0.5` 이상이면 `"ai_generated"`로 판정한다.

## 단일 영상 XAI 추론 결과

`infer --with-xai` 사용 시 기본 추론 결과에 XAI 필드가 추가된다.

```json
{
  "prediction": "real",
  "confidence": 0.0,
  "frame_importance": [0.0],
  "segments": [
    {
      "start_frame": 0,
      "end_frame": 0,
      "type": "movement anomaly",
      "confidence": 0.0
    }
  ],
  "explanations": [
    "Frames 0 to 0 show movement anomaly (confidence 0.00)"
  ],
  "xai_method": "attention_rollup"
}
```

- `frame_importance`: sampled frame별 중요도 점수 배열
- `segments`: `xai_threshold` 이상인 연속 frame 구간 목록
- `segments[].type`: `"movement anomaly"`, `"texture jitter"`, `"lighting anomaly"`, `"object inconsistency"`, `"unknown"` 중 하나
- `segments[].confidence`: 해당 segment의 평균 중요도
- `explanations`: segment별 사람이 읽을 수 있는 설명 문자열
- `xai_method`: `"attention_rollup"` 또는 `"activation_energy"`

## Manifest 추론 결과 파일

`infer-manifest` 명령은 전체 payload를 `--output-path`에 저장한다.

```json
{
  "manifest": "manifest.csv",
  "split": "test",
  "checkpoint": "model.pt",
  "num_samples": 0,
  "num_predictions": 0,
  "num_failures": 0,
  "metrics": {
    "accuracy": 0.0,
    "f1_score": 0.0,
    "roc_auc": 0.0
  },
  "predictions": [
    {
      "path": "video.mp4",
      "label": 0,
      "prediction": "real",
      "confidence": 0.0
    }
  ],
  "failures": [
    {
      "path": "broken.mp4",
      "label": 1,
      "error_type": "OSError",
      "error": "error message"
    }
  ]
}
```

`infer-manifest --with-xai` 사용 시 `predictions[]`의 각 항목에도 단일 영상 XAI 추론 결과와 동일한 XAI 필드가 추가된다.

## 로그 출력 형식

```text
[INFO] message
[ERROR] message
[WARNING] message
```

진행 로그:

```text
[INFO] train progress step=100 running_loss=0.1234
[INFO] infer progress processed=100 predicted=99 failed=1
```

## 파일 저장

- 모델: `.pt`
- 결과: `.json`
- 로그: `.md`

## 규칙

- 모든 결과는 JSON 형태로 출력한다.
- key 이름은 코드와 문서에서 동일하게 유지한다.
