from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ai_video_detector.data import VideoDataset, VideoSample, load_video, load_video_samples_from_manifest
from ai_video_detector.infer import predict_video
from ai_video_detector.metrics import compute_classification_metrics
from ai_video_detector.model import VideoClassifier, VideoClassifierConfig
from ai_video_detector.train import build_optimizer, evaluate_model, make_epoch_summary, train_one_epoch
from ai_video_detector.utils import log_message, save_json, set_seed


class PipelineTests(unittest.TestCase):
    def _make_tmp_path(self) -> Path:
        path = Path(".tmp_test_outputs") / self._testMethodName
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_video_dataset_sampling_and_shape(self) -> None:
        tmp_path = self._make_tmp_path()
        sample_path = tmp_path / "video.npy"
        frames = np.random.randint(0, 255, size=(6, 12, 12, 3), dtype=np.uint8)
        np.save(sample_path, frames)

        dataset = VideoDataset([VideoSample(path=str(sample_path), label=1)], num_frames=4, image_size=(8, 8))
        pixel_values, label = dataset[0]

        self.assertEqual(pixel_values.shape, (4, 3, 8, 8))
        self.assertEqual(label.item(), 1.0)

    def test_load_video_supports_gif_via_imageio(self) -> None:
        tmp_path = self._make_tmp_path()
        gif_path = tmp_path / "sample.gif"
        gif_path.write_bytes(b"GIF89a")
        frames = np.random.randint(0, 255, size=(3, 8, 8, 3), dtype=np.uint8)

        fake_imageio = type("FakeImageIO", (), {"imread": staticmethod(lambda _: frames)})
        with patch.dict(sys.modules, {"imageio": type("FakeImageIOPackage", (), {"v3": fake_imageio}), "imageio.v3": fake_imageio}):
            loaded = load_video(gif_path)

        np.testing.assert_array_equal(loaded, frames)

    def test_load_video_samples_from_manifest_csv(self) -> None:
        tmp_path = self._make_tmp_path()
        manifest_path = tmp_path / "manifest_train.csv"
        manifest_path.write_text(
            "\n".join(
                [
                    "id,split,source,label,label_name,relative_path,ext,index,status,is_zero_byte,note",
                    "1,train,source_a,1,fake,clips/a.gif,.gif,1,ok,0,",
                    "2,val,source_b,0,real,clips/b.mp4,.mp4,2,ok,0,",
                    "3,train,source_c,1,fake,clips/c.gif,.gif,3,missing,0,",
                    "4,train,source_d,0,real,clips/d.gif,.gif,4,ok,1,",
                ]
            ),
            encoding="utf-8",
        )

        samples = load_video_samples_from_manifest(manifest_path, base_dir=tmp_path / "dataset", split="train")

        self.assertEqual(samples, [VideoSample(path=str(tmp_path / "dataset" / "clips" / "a.gif"), label=1)])

    def test_metrics_output_keys(self) -> None:
        logits = torch.tensor([-3.0, 2.0, 1.0, -2.0])
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])
        metrics = compute_classification_metrics(logits, labels)

        self.assertEqual(set(metrics), {"accuracy", "f1_score", "roc_auc"})
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1_score"], 1.0)
        self.assertEqual(metrics["roc_auc"], 1.0)

    def test_training_evaluation_and_inference_pipeline(self) -> None:
        set_seed(7)
        device = torch.device("cpu")
        model = VideoClassifier(VideoClassifierConfig(hidden_dim=64, use_pretrained=False))
        optimizer = build_optimizer(model, lr=1e-3)

        inputs = torch.randn(4, 4, 3, 8, 8)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=2, shuffle=False)

        train_loss = train_one_epoch(model, loader, optimizer, device)
        metrics = evaluate_model(model, loader, device)
        summary = make_epoch_summary(1, train_loss, metrics)

        self.assertEqual(set(summary), {"epoch", "train_loss", "val_loss", "accuracy", "f1_score"})
        self.assertEqual(summary["epoch"], 1)
        self.assertGreaterEqual(summary["train_loss"], 0.0)
        self.assertGreaterEqual(summary["val_loss"], 0.0)

        tmp_path = self._make_tmp_path()
        sample_path = tmp_path / "infer.npy"
        np.save(sample_path, np.random.randint(0, 255, size=(4, 8, 8, 3), dtype=np.uint8))
        prediction = predict_video(model, sample_path, device=device, num_frames=4, image_size=(8, 8))

        self.assertEqual(set(prediction), {"prediction", "confidence"})
        self.assertIn(prediction["prediction"], {"real", "ai_generated"})
        self.assertGreaterEqual(prediction["confidence"], 0.0)
        self.assertLessEqual(prediction["confidence"], 1.0)

    def test_freeze_encoder_option(self) -> None:
        model = VideoClassifier(VideoClassifierConfig(hidden_dim=64, use_pretrained=False, freeze_encoder=True))

        self.assertTrue(all(not parameter.requires_grad for parameter in model.encoder.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.classifier.parameters()))

    def test_json_and_log_output(self) -> None:
        tmp_path = self._make_tmp_path()
        payload = {"prediction": "real", "confidence": 0.25}
        output_path = save_json(tmp_path / "result.json", payload)
        loaded = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(loaded, payload)
        self.assertEqual(log_message("info", "message"), "[INFO] message")


if __name__ == "__main__":
    unittest.main(verbosity=2)
