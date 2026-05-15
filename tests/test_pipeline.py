from __future__ import annotations

import json
import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from ai_video_detector.cli import load_manifest, parse_args, run_train
from ai_video_detector.data import VideoDataset, VideoSample, load_video, load_video_samples_from_manifest
from ai_video_detector.infer import predict_video
from ai_video_detector.metrics import compute_classification_metrics
from ai_video_detector.model import VideoClassifier, VideoClassifierConfig, VideoMAEEncoder, load_video_classifier_state_dict
from ai_video_detector.preprocessing import temporal_sample_clips_with_indices
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
        frames = [
            Image.fromarray(np.full((8, 8, 3), fill_value=value, dtype=np.uint8), mode="RGB")
            for value in (0, 64, 128)
        ]
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

        loaded = load_video(gif_path)

        self.assertEqual(loaded.shape, (3, 8, 8, 3))
        self.assertEqual(int(loaded[1, 0, 0, 0]), 64)

    def test_load_video_reports_corrupt_gif_path(self) -> None:
        tmp_path = self._make_tmp_path()
        gif_path = tmp_path / "broken.gif"
        gif_path.write_bytes(b"not-a-real-gif")

        with self.assertRaises(OSError) as context:
            load_video(gif_path)

        self.assertIn(str(gif_path), str(context.exception))
        self.assertIn("Failed to load GIF", str(context.exception))

    def test_load_video_samples_from_manifest_csv(self) -> None:
        tmp_path = self._make_tmp_path()
        dataset_root = tmp_path / "dataset"
        (dataset_root / "clips").mkdir(parents=True, exist_ok=True)
        (dataset_root / "clips" / "a.gif").write_bytes(b"gif-placeholder")
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

        samples = load_video_samples_from_manifest(manifest_path, base_dir=dataset_root, split="train")

        self.assertEqual(samples, [VideoSample(path=str(dataset_root / "clips" / "a.gif"), label=1)])

    def test_load_video_samples_from_manifest_skips_missing_files(self) -> None:
        tmp_path = self._make_tmp_path()
        dataset_root = tmp_path / "dataset"
        (dataset_root / "clips").mkdir(parents=True, exist_ok=True)
        valid_path = dataset_root / "clips" / "a.gif"
        valid_path.write_bytes(b"gif-placeholder")
        manifest_path = tmp_path / "manifest_train.csv"
        manifest_path.write_text(
            "\n".join(
                [
                    "id,split,source,label,label_name,relative_path,ext,index,status,is_zero_byte,note",
                    "1,train,source_a,1,fake,clips/a.gif,.gif,1,ok,0,",
                    "2,train,source_b,0,real,clips/missing.gif,.gif,2,ok,0,",
                ]
            ),
            encoding="utf-8",
        )

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            samples = load_video_samples_from_manifest(manifest_path, base_dir=dataset_root, split="train")

        self.assertEqual(samples, [VideoSample(path=str(valid_path), label=1)])
        self.assertIn("missing sample", stderr.getvalue())
        self.assertIn("missing.gif", stderr.getvalue())

    def test_cli_load_manifest_filters_csv_by_split(self) -> None:
        tmp_path = self._make_tmp_path()
        dataset_root = tmp_path / "dataset"
        (dataset_root / "clips").mkdir(parents=True, exist_ok=True)
        (dataset_root / "clips" / "b.mp4").write_bytes(b"mp4-placeholder")
        manifest_path = tmp_path / "manifest_train.csv"
        manifest_path.write_text(
            "\n".join(
                [
                    "id,split,source,label,label_name,relative_path,ext,index,status,is_zero_byte,note",
                    "1,train,source_a,1,fake,clips/a.gif,.gif,1,ok,0,",
                    "2,val,source_b,0,real,clips/b.mp4,.mp4,2,ok,0,",
                ]
            ),
            encoding="utf-8",
        )

        samples = load_manifest(manifest_path, data_root=dataset_root, split="val")

        self.assertEqual(samples, [VideoSample(path=str(dataset_root / "clips" / "b.mp4"), label=0)])

    def test_cli_parse_args_accepts_single_manifest_mode(self) -> None:
        argv = [
            "ai-video-detector",
            "train",
            "--manifest",
            "shared.csv",
            "--output-dir",
            "outputs/run",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.manifest, Path("shared.csv"))
        self.assertEqual(args.train_split, "train")
        self.assertEqual(args.val_split, "val")

    def test_cli_parse_args_accepts_manifest_inference(self) -> None:
        argv = [
            "ai-video-detector",
            "infer-manifest",
            "--manifest",
            "manifest.csv",
            "--split",
            "test",
            "--checkpoint",
            "runs/head_only_30k/model.pt",
            "--output-path",
            "runs/head_only_30k/test_predictions.json",
            "--with-xai",
            "--xai-threshold",
            "0.7",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.command, "infer-manifest")
        self.assertEqual(args.manifest, Path("manifest.csv"))
        self.assertEqual(args.split, "test")
        self.assertEqual(args.checkpoint, Path("runs/head_only_30k/model.pt"))
        self.assertEqual(args.output_path, Path("runs/head_only_30k/test_predictions.json"))
        self.assertTrue(args.with_xai)
        self.assertEqual(args.xai_threshold, 0.7)

    def test_video_dataset_skips_corrupt_sample_and_logs_warning(self) -> None:
        samples = [
            VideoSample(path="broken.gif", label=1),
            VideoSample(path="good.npy", label=0),
        ]
        good_frames = np.random.randint(0, 255, size=(4, 8, 8, 3), dtype=np.uint8)

        def loader(path: Path) -> np.ndarray:
            if str(path) == "broken.gif":
                raise OSError("Failed to load GIF 'broken.gif': decoder error")
            return good_frames

        dataset = VideoDataset(samples, num_frames=4, image_size=(8, 8), video_loader=loader)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            pixel_values, label = dataset[0]

        self.assertEqual(pixel_values.shape, (4, 3, 8, 8))
        self.assertEqual(label.item(), 0.0)
        self.assertIn("[WARN] failed to load sample", stderr.getvalue())
        self.assertIn("broken.gif", stderr.getvalue())

    def test_training_pipeline_continues_after_corrupt_sample(self) -> None:
        set_seed(3)
        device = torch.device("cpu")
        model = VideoClassifier(VideoClassifierConfig(hidden_dim=64, use_pretrained=False))
        optimizer = build_optimizer(model, lr=1e-3)
        good_frames = np.random.randint(0, 255, size=(4, 8, 8, 3), dtype=np.uint8)
        samples = [
            VideoSample(path="broken.gif", label=1),
            VideoSample(path="good_a.npy", label=0),
            VideoSample(path="good_b.npy", label=1),
        ]

        def loader(path: Path) -> np.ndarray:
            if str(path) == "broken.gif":
                raise OSError("Failed to load GIF 'broken.gif': decoder error")
            return good_frames

        dataset = VideoDataset(samples, num_frames=4, image_size=(8, 8), video_loader=loader)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            train_loss = train_one_epoch(model, dataloader, optimizer, device, log_interval=1)

        self.assertGreaterEqual(train_loss, 0.0)
        self.assertIn("broken.gif", stderr.getvalue())
        self.assertIn("train progress", stderr.getvalue())

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

        self.assertEqual(set(summary), {"epoch", "train_loss", "val_loss", "accuracy", "f1_score", "roc_auc"})
        self.assertEqual(summary["epoch"], 1)
        self.assertGreaterEqual(summary["train_loss"], 0.0)
        self.assertGreaterEqual(summary["val_loss"], 0.0)
        self.assertGreaterEqual(summary["roc_auc"], 0.0)

        tmp_path = self._make_tmp_path()
        sample_path = tmp_path / "infer.npy"
        np.save(sample_path, np.random.randint(0, 255, size=(4, 8, 8, 3), dtype=np.uint8))
        prediction = predict_video(model, sample_path, device=device, num_frames=4, image_size=(8, 8))

        self.assertEqual(set(prediction), {"prediction", "confidence", "inference", "clip_predictions"})
        self.assertIn(prediction["prediction"], {"real", "ai_generated"})
        self.assertGreaterEqual(prediction["confidence"], 0.0)
        self.assertLessEqual(prediction["confidence"], 1.0)
        self.assertEqual(prediction["inference"]["video_score_strategy"], "max_clip_confidence")

    def test_adaptive_sampling_preserves_original_frame_indices(self) -> None:
        frames = torch.arange(10 * 2 * 2 * 3, dtype=torch.uint8).reshape(10, 2, 2, 3)

        clips, indices = temporal_sample_clips_with_indices(frames, num_frames=4, num_clips=3)

        self.assertEqual(clips.shape, (3, 4, 2, 2, 3))
        self.assertEqual(len(indices), 3)
        self.assertEqual(indices[0].tolist()[0], 0)
        self.assertTrue(all(0 <= value < 10 for clip in indices for value in clip.tolist()))
        self.assertEqual(indices[-1].tolist()[-1], 9)

    def test_predict_video_uses_max_confidence_clip_as_video_score(self) -> None:
        class CountingModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.batch_sizes: list[int] = []

            def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                self.batch_sizes.append(pixel_values.shape[0])
                return torch.tensor([-2.0, -1.0, 0.0, 3.0, 1.0])

        device = torch.device("cpu")
        model = CountingModel()
        tmp_path = self._make_tmp_path()
        sample_path = tmp_path / "long_infer.npy"
        np.save(sample_path, np.random.randint(0, 255, size=(600, 8, 8, 3), dtype=np.uint8))

        prediction = predict_video(model, sample_path, device=device, num_frames=4, image_size=(8, 8))

        self.assertEqual(model.batch_sizes, [5])
        self.assertEqual(prediction["inference"]["num_clips"], 5)
        self.assertEqual(prediction["inference"]["representative_clip_index"], 3)
        self.assertEqual(prediction["confidence"], prediction["clip_predictions"][3]["confidence"])
        self.assertEqual(len(prediction["clip_predictions"][3]["sampled_frames"]), 4)

    def test_freeze_encoder_option(self) -> None:
        model = VideoClassifier(VideoClassifierConfig(hidden_dim=64, use_pretrained=False, freeze_encoder=True))

        self.assertTrue(all(not parameter.requires_grad for parameter in model.encoder.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.classifier.parameters()))

    def test_load_video_classifier_state_dict_accepts_legacy_mlp_head_keys(self) -> None:
        source = VideoClassifier(VideoClassifierConfig(hidden_dim=64, use_pretrained=False))
        state_dict = source.state_dict()
        legacy_state_dict = {}
        for key, value in state_dict.items():
            legacy_key = key.replace("classifier.layers.", "classifier.")
            legacy_state_dict[legacy_key] = value
        target = VideoClassifier(VideoClassifierConfig(hidden_dim=64, use_pretrained=False))

        load_video_classifier_state_dict(target, legacy_state_dict)

        self.assertTrue(torch.equal(target.state_dict()["classifier.layers.0.weight"], state_dict["classifier.layers.0.weight"]))

    def test_transformer_head_forward_with_fallback_encoder(self) -> None:
        model = VideoClassifier(
            VideoClassifierConfig(
                hidden_dim=64,
                use_pretrained=False,
                head_type="transformer",
                transformer_head_layers=1,
                transformer_head_heads=4,
                transformer_head_ff_dim=128,
            )
        )
        inputs = torch.randn(2, 4, 3, 8, 8)

        logits = model(inputs)

        self.assertEqual(logits.shape, (2,))

    def test_xai_output_structure(self) -> None:
        model = VideoClassifier(VideoClassifierConfig(hidden_dim=64, use_pretrained=False))
        inputs = torch.randn(1, 4, 3, 8, 8)

        outputs = model.predict_with_xai(inputs)

        self.assertIn("segments", outputs)
        self.assertIn("explanations", outputs)

    def test_transformer_xai_enables_eager_attention_and_marks_missing_attention(self) -> None:
        class FakeTransformer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = type("Config", (), {"hidden_size": 64})()
                self.attn_implementation = None

            def set_attn_implementation(self, value: str) -> None:
                self.attn_implementation = value

            def forward(self, pixel_values: torch.Tensor, output_attentions: bool = False):
                return type(
                    "Outputs",
                    (),
                    {
                        "last_hidden_state": torch.ones(pixel_values.shape[0], 2, 64),
                        "attentions": None,
                    },
                )()

        encoder = VideoMAEEncoder(VideoClassifierConfig(hidden_dim=64, use_pretrained=False))
        fake_model = FakeTransformer()
        encoder.model = fake_model
        encoder.uses_transformers = True
        encoder.hidden_dim = 64

        outputs = encoder.encode(torch.randn(1, 4, 3, 8, 8), return_attention=True)

        self.assertEqual(fake_model.attn_implementation, "eager")
        self.assertIsNone(outputs.frame_importance)
        self.assertEqual(outputs.xai_method, "unavailable")

    def test_predict_video_with_xai_returns_json_ready_payload(self) -> None:
        device = torch.device("cpu")
        model = VideoClassifier(VideoClassifierConfig(hidden_dim=64, use_pretrained=False))
        tmp_path = self._make_tmp_path()
        sample_path = tmp_path / "infer_xai.npy"
        np.save(sample_path, np.random.randint(0, 255, size=(4, 8, 8, 3), dtype=np.uint8))

        prediction = predict_video(
            model,
            sample_path,
            device=device,
            num_frames=4,
            image_size=(8, 8),
            return_xai=True,
            xai_threshold=0.6,
        )

        self.assertIn("xai", prediction)
        self.assertEqual(
            set(prediction["xai"]),
            {
                "method",
                "threshold",
                "scope",
                "clip_index",
                "frame_importance_scope",
                "frame_importance",
                "sampled_frames",
                "segments",
                "explanations",
                "visualizations",
                "summary",
            },
        )
        self.assertEqual(prediction["xai"]["scope"], "representative_clip")
        self.assertEqual(prediction["xai"]["frame_importance_scope"], "clip_sampled_frame_index")
        self.assertEqual(prediction["xai"]["sampled_frames"][0]["original_frame_index"], 0)
        self.assertEqual(
            set(prediction["xai"]["summary"]),
            {"num_frames", "num_segments", "max_frame_importance", "num_visualizations"},
        )
        self.assertNotIn("frame_importance", prediction)
        self.assertNotIn("segments", prediction)
        self.assertNotIn("explanations", prediction)
        json.dumps(prediction)

    def test_cli_parse_args_accepts_transformer_head_options(self) -> None:
        argv = [
            "ai-video-detector",
            "train",
            "--manifest",
            "shared.csv",
            "--output-dir",
            "outputs/run",
            "--head-type",
            "transformer",
            "--transformer-head-layers",
            "1",
            "--transformer-head-heads",
            "4",
            "--transformer-head-ff-dim",
            "128",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.head_type, "transformer")
        self.assertEqual(args.transformer_head_layers, 1)
        self.assertEqual(args.transformer_head_heads, 4)
        self.assertEqual(args.transformer_head_ff_dim, 128)

    def test_train_cli_saves_all_epoch_summaries(self) -> None:
        tmp_path = self._make_tmp_path()
        train_manifest = tmp_path / "train_manifest.json"
        val_manifest = tmp_path / "val_manifest.json"
        output_dir = tmp_path / "outputs"
        sample_a = tmp_path / "sample_a.npy"
        sample_b = tmp_path / "sample_b.npy"
        np.save(sample_a, np.random.randint(0, 255, size=(4, 8, 8, 3), dtype=np.uint8))
        np.save(sample_b, np.random.randint(0, 255, size=(4, 8, 8, 3), dtype=np.uint8))
        manifest_payload = [
            {"path": str(sample_a), "label": 0},
            {"path": str(sample_b), "label": 1},
        ]
        train_manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")
        val_manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")

        args = type(
            "Args",
            (),
            {
                "seed": 11,
                "encoder_name": "MCG-NJU/videomae-base",
                "no_pretrained": True,
                "freeze_encoder": False,
                "head_type": "mlp",
                "transformer_head_layers": 2,
                "transformer_head_heads": 8,
                "transformer_head_ff_dim": 2048,
                "manifest": None,
                "train_manifest": train_manifest,
                "val_manifest": val_manifest,
                "train_data_root": None,
                "val_data_root": None,
                "data_root": None,
                "train_split": "train",
                "val_split": "val",
                "num_frames": 4,
                "image_size": 8,
                "batch_size": 2,
                "epochs": 2,
                "output_dir": output_dir,
            },
        )()

        run_train(args)

        summaries = json.loads((output_dir / "train_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(len(summaries), 2)
        self.assertEqual([summary["epoch"] for summary in summaries], [1, 2])
        self.assertTrue(all("roc_auc" in summary for summary in summaries))

    def test_json_and_log_output(self) -> None:
        tmp_path = self._make_tmp_path()
        payload = {"prediction": "real", "confidence": 0.25}
        output_path = save_json(tmp_path / "result.json", payload)
        loaded = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(loaded, payload)
        self.assertEqual(log_message("info", "message"), "[INFO] message")


if __name__ == "__main__":
    unittest.main(verbosity=2)
