from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from ai_video_detector.server import AnalyzeResponse, Evidence, ModelAnalyzer, XAIVisualization, app, get_t2v_analyzer


class StubAnalyzer:
    name = "t2v"

    def analyze_url(self, url: str, *, request_id: str | None = None) -> AnalyzeResponse:
        return AnalyzeResponse(
            decision="REAL",
            t2v_prob=0.25,
            model_used="VideoMAE",
            evidence=Evidence(
                frame_importance=[0.25],
                segments=[],
                explanations=[],
            ),
            xai_visualization=XAIVisualization(method="attention_rollout", heatmaps=[]),
        )


class ServerTests(unittest.TestCase):
    def test_t2v_analyze_accepts_s3_url_payload(self) -> None:
        app.dependency_overrides[get_t2v_analyzer] = lambda: StubAnalyzer()
        try:
            client = TestClient(app)
            response = client.post(
                "/t2v/analyze",
                json={
                    "request_id": "req-1",
                    "s3_url": "https://bucket.s3.ap-northeast-2.amazonaws.com/input.mp4?signature=test",
                },
            )
        finally:
            app.dependency_overrides.clear()

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["decision"], "REAL")
        self.assertEqual(payload["t2v_prob"], 0.25)
        self.assertEqual(payload["model_used"], "VideoMAE")
        self.assertEqual(payload["evidence"]["frame_importance"], [0.25])
        self.assertEqual(payload["xai_visualization"]["method"], "attention_rollout")

    def test_t2v_analyze_rejects_non_http_url(self) -> None:
        app.dependency_overrides[get_t2v_analyzer] = lambda: StubAnalyzer()
        try:
            client = TestClient(app)
            response = client.post("/t2v/analyze", json={"s3_url": "s3://bucket/input.mp4"})
        finally:
            app.dependency_overrides.clear()

        self.assertEqual(response.status_code, 422)

    def test_model_analyzer_cleans_up_downloaded_file(self) -> None:
        analyzer = ModelAnalyzer.__new__(ModelAnalyzer)
        analyzer.name = "t2v"

        class Config:
            max_download_bytes = 1024
            num_frames = 4
            image_size = 8
            with_xai = False
            xai_threshold = 0.6
            max_heatmaps = 5
            xai_output_dir = None
            model_used = "VideoMAE"

        analyzer.config = Config()
        analyzer.device = "cpu"
        analyzer.model = object()
        created_paths = []

        def fake_download(url: str, destination, *, max_bytes: int) -> None:
            destination.write_bytes(b"video")
            created_paths.append(destination)

        def fake_predict_video(model, video_path, **kwargs):
            self.assertTrue(video_path.exists())
            return {"prediction": "ai_generated", "confidence": 0.8}

        with patch("ai_video_detector.server.download_url", fake_download), patch(
            "ai_video_detector.server.predict_video",
            fake_predict_video,
        ), patch.object(analyzer, "_load_model", lambda: analyzer.model):
            result = analyzer.analyze_url("https://example.com/video.gif")

        self.assertEqual(result.decision, "FAKE")
        self.assertEqual(result.t2v_prob, 0.8)
        self.assertTrue(created_paths)
        self.assertFalse(created_paths[0].exists())

    def test_model_analyzer_generates_heatmap_files(self) -> None:
        tmp_path = Path(".tmp_test_outputs") / self._testMethodName
        tmp_path.mkdir(parents=True, exist_ok=True)
        video_path = tmp_path / "video.npy"
        np.save(video_path, np.random.randint(0, 255, size=(3, 8, 8, 3), dtype=np.uint8))

        analyzer = ModelAnalyzer.__new__(ModelAnalyzer)
        analyzer.name = "t2v"

        class Config:
            max_heatmaps = 2
            xai_threshold = 0.6
            xai_output_dir = tmp_path / "xai"
            model_used = "VideoMAE"

        analyzer.config = Config()
        result = analyzer._format_response(
            {
                "prediction": "ai_generated",
                "confidence": 0.91,
                "xai": {
                    "frame_importance": [0.12, 0.88, 0.76],
                    "sampled_frames": [
                        {"sampled_frame_index": 0, "original_frame_index": 0, "timestamp_sec": 0.0},
                        {"sampled_frame_index": 1, "original_frame_index": 2, "timestamp_sec": 0.2},
                        {"sampled_frame_index": 2, "original_frame_index": 1, "timestamp_sec": 0.1},
                    ],
                    "segments": [{"start_frame": 1, "end_frame": 2, "type": "movement anomaly", "confidence": 0.82}],
                    "explanations": ["Frames 1 to 2 show movement anomaly."],
                    "method": "attention_rollup",
                },
            },
            video_path,
            request_id="video123",
        )

        self.assertEqual(result.decision, "FAKE")
        self.assertEqual(result.evidence.sampled_frames[1]["original_frame_index"], 2)
        self.assertEqual(result.xai_visualization.method, "attention_rollout")
        self.assertEqual(len(result.xai_visualization.heatmaps), 2)
        self.assertEqual(result.xai_visualization.heatmaps[0].original_frame_index, 2)
        for heatmap in result.xai_visualization.heatmaps:
            self.assertTrue((tmp_path / heatmap.heatmap_url.lstrip("/")).exists())
            self.assertTrue((tmp_path / heatmap.overlay_url.lstrip("/")).exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
