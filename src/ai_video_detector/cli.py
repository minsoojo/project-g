"""Simple entry points for training summaries and inference output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .data import VideoDataset, VideoSample, load_video_samples_from_manifest
from .infer import predict_video
from .model import VideoClassifier, VideoClassifierConfig
from .train import build_optimizer, evaluate_model, make_epoch_summary, save_checkpoint, save_epoch_summary, train_one_epoch
from .utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI-generated video detection baseline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--manifest", type=Path)
    train_parser.add_argument("--train-manifest", type=Path)
    train_parser.add_argument("--val-manifest", type=Path)
    train_parser.add_argument("--data-root", type=Path)
    train_parser.add_argument("--train-data-root", type=Path)
    train_parser.add_argument("--val-data-root", type=Path)
    train_parser.add_argument("--train-split", type=str, default="train")
    train_parser.add_argument("--val-split", type=str, default="val")
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--num-frames", type=int, default=16)
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--no-pretrained", action="store_true")
    train_parser.add_argument("--encoder-name", type=str, default="MCG-NJU/videomae-base")
    train_parser.add_argument("--freeze-encoder", action="store_true")

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--video-path", type=Path, required=True)
    infer_parser.add_argument("--checkpoint", type=Path, required=True)
    infer_parser.add_argument("--output-path", type=Path)
    infer_parser.add_argument("--num-frames", type=int, default=16)
    infer_parser.add_argument("--image-size", type=int, default=224)
    infer_parser.add_argument("--with-xai", action="store_true")
    infer_parser.add_argument("--no-pretrained", action="store_true")
    infer_parser.add_argument("--encoder-name", type=str, default="MCG-NJU/videomae-base")
    infer_parser.add_argument("--freeze-encoder", action="store_true")

    args = parser.parse_args()
    if args.command == "train":
        has_shared_manifest = args.manifest is not None
        has_split_manifests = args.train_manifest is not None or args.val_manifest is not None
        if has_shared_manifest and has_split_manifests:
            parser.error("Use either --manifest or both --train-manifest/--val-manifest, not both.")
        if not has_shared_manifest and not (args.train_manifest and args.val_manifest):
            parser.error("Training requires either --manifest or both --train-manifest and --val-manifest.")
    return args


def load_manifest(path: Path, *, data_root: Optional[Path] = None, split: Optional[str] = None) -> list[VideoSample]:
    if path.suffix.lower() == ".csv":
        return load_video_samples_from_manifest(path, base_dir=data_root, split=split)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [VideoSample(path=item["path"], label=int(item["label"])) for item in payload]


def run_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = VideoClassifierConfig(
        encoder_name=args.encoder_name,
        use_pretrained=not args.no_pretrained,
        freeze_encoder=args.freeze_encoder,
    )
    model = VideoClassifier(config).to(device)
    optimizer = build_optimizer(model)

    shared_manifest = args.manifest
    train_manifest = args.train_manifest or shared_manifest
    val_manifest = args.val_manifest or shared_manifest
    train_data_root = args.train_data_root or args.data_root
    val_data_root = args.val_data_root or args.data_root
    train_split = args.train_split if shared_manifest is not None else None
    val_split = args.val_split if shared_manifest is not None else None

    if train_manifest is None or val_manifest is None:
        raise ValueError("Training manifests must be resolved before loading datasets.")

    train_samples = load_manifest(train_manifest, data_root=train_data_root, split=train_split)
    val_samples = load_manifest(val_manifest, data_root=val_data_root, split=val_split)
    train_dataset = VideoDataset(train_samples, num_frames=args.num_frames, image_size=(args.image_size, args.image_size))
    val_dataset = VideoDataset(val_samples, num_frames=args.num_frames, image_size=(args.image_size, args.image_size))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    summary = {}
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, device)
        summary = make_epoch_summary(epoch, train_loss, val_metrics)
        print(json.dumps(summary))

    checkpoint_path = args.output_dir / "model.pt"
    summary_path = args.output_dir / "train_metrics.json"
    save_checkpoint(model, checkpoint_path)
    save_epoch_summary(summary_path, summary)


def run_infer(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = VideoClassifierConfig(
        encoder_name=args.encoder_name,
        use_pretrained=not args.no_pretrained,
        freeze_encoder=args.freeze_encoder,
    )
    model = VideoClassifier(config).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    payload = predict_video(
        model,
        args.video_path,
        device=device,
        num_frames=args.num_frames,
        image_size=(args.image_size, args.image_size),
        return_xai=args.with_xai,
    )
    print(json.dumps(payload))
    if args.output_path:
        args.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.command == "train":
        run_train(args)
        return
    if args.command == "infer":
        run_infer(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
