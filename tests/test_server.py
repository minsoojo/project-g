from __future__ import annotations

import unittest
from typing import Any, Dict
from unittest.mock import patch

from fastapi.testclient import TestClient

from ai_video_detector.server import ModelAnalyzer, app, get_t2v_analyzer


class StubAnalyzer:
    name = "t2v"

    def analyze_url(self, url: str) -> Dict[str, Any]:
        return {"prediction": "real", "confidence": 0.25, "received_url": url}


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
        self.assertEqual(payload["request_id"], "req-1")
        self.assertEqual(payload["model"], "t2v")
        self.assertEqual(payload["result"]["prediction"], "real")
        self.assertTrue(payload["result"]["received_url"].startswith("https://bucket.s3"))

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

        self.assertEqual(result["prediction"], "ai_generated")
        self.assertTrue(created_paths)
        self.assertFalse(created_paths[0].exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
