import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 512
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORES_TO_USE = 4

class FLIRPersonDataset(Dataset):
    def __init__(self, data_root: Path, split: str):
        self.split_root = data_root / f"images_thermal_{split}"
        annotation_path = self.split_root / "coco.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing COCO annotations: {annotation_path}")

        import json

        with annotation_path.open(encoding="utf-8") as annotation_file:
            coco = json.load(annotation_file)

        person_ids = {
            category["id"]
            for category in coco["categories"]
            if category["name"].casefold() == "person"
        }
        if not person_ids:
            raise ValueError(f"No 'person' category found in {annotation_path}")

        positive_image_ids = {
            annotation["image_id"]
            for annotation in coco["annotations"]
            if annotation["category_id"] in person_ids
        }
        self.samples = [
            (
                self.split_root / image["file_name"],
                float(image["id"] in positive_image_ids),
            )
            for image in coco["images"]
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as source_image:
            image = source_image.convert("L")
            if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                raise ValueError(
                    f"Expected {IMAGE_WIDTH}x{IMAGE_HEIGHT} image, got "
                    f"{image.width}x{image.height}: {image_path}"
                )
            pixels = np.array(image, dtype=np.uint8, copy=True)

        image_tensor = torch.from_numpy(pixels).unsqueeze(0).to(torch.float32).div_(255.0)
        return image_tensor, torch.tensor(label, dtype=torch.float32)


class DepthwiseBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=input_channels,
                bias=False,
            ),
            nn.BatchNorm2d(input_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class FullResolutionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            DepthwiseBlock(16, 32, stride=2),
            DepthwiseBlock(32, 48, stride=2),
            DepthwiseBlock(48, 64, stride=2),
            DepthwiseBlock(64, 96, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(96, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).flatten(1)).squeeze(1)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return torch.device(requested)


def evaluate(
    model: nn.Module,
    batches: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
    model.eval()
    loss_total = 0.0
    correct = 0
    true_positives = 0
    predicted_positives = 0
    actual_positives = 0

    with torch.no_grad():
        for images, labels in batches:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss_total += loss_function(logits, labels).item() * labels.numel()
            predictions = logits >= 0.0
            targets = labels >= 0.5
            correct += int((predictions == targets).sum().item())
            true_positives += int((predictions & targets).sum().item())
            predicted_positives += int(predictions.sum().item())
            actual_positives += int(targets.sum().item())

    sample_count = len(batches.dataset)
    precision = true_positives / max(predicted_positives, 1)
    recall = true_positives / max(actual_positives, 1)
    return loss_total / sample_count, correct / sample_count, precision, recall


def train(arguments: argparse.Namespace) -> None:
    torch.manual_seed(arguments.seed)
   # torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.set_num_threads(CORES_TO_USE)
    device = select_device(arguments.device)

    data_root = arguments.data_root.resolve()
    training_data = FLIRPersonDataset(data_root, "train")
    validation_data = FLIRPersonDataset(data_root, "val")
    positive_count = sum(label for _, label in training_data.samples)
    negative_count = len(training_data) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Training split must contain both positive and negative samples.")

    loader_options = {
        "batch_size": arguments.batch_size,
        "num_workers": arguments.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": arguments.workers > 0,
    }
    training_batches = DataLoader(training_data, shuffle=True, **loader_options)
    validation_batches = DataLoader(validation_data, shuffle=False, **loader_options)

    model = FullResolutionCNN().to(device)
    positive_weight = torch.tensor(negative_count / positive_count, device=device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate)
    output_path = arguments.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")

    print(
        f"Training on {len(training_data)} native {IMAGE_WIDTH}x{IMAGE_HEIGHT} images; "
        f"validation={len(validation_data)}, device={device}, batch={arguments.batch_size}"
    )
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        training_loss_total = 0.0
        for images, labels in training_batches:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
            training_loss_total += loss.item() * labels.numel()

        validation_loss, accuracy, precision, recall = evaluate(
            model, validation_batches, loss_function, device
        )
        print(
            f"Epoch {epoch:02d}/{arguments.epochs} "
            f"train_loss={training_loss_total / len(training_data):.4f} "
            f"val_loss={validation_loss:.4f} accuracy={accuracy:.3f} "
            f"precision={precision:.3f} recall={recall:.3f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
            torch.save(
                {
                    "architecture": "FullResolutionCNN",
                    "input_shape": [1, IMAGE_HEIGHT, IMAGE_WIDTH],
                    "class_names": ["background", "person"],
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "model_state_dict": model.state_dict(),
                },
                temporary_path,
            )
            temporary_path.replace(output_path)

    print(f"Best full-resolution checkpoint saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a separate 640x512 FLIR thermal person classifier."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "resources" / "model" / "FLIR",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "resources"
        / "model"
        / "FLIR"
        / "human_detector_fullres.pt",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    if arguments.epochs < 1 or arguments.batch_size < 1 or arguments.workers < 0:
        parser.error("epochs and batch-size must be positive; workers cannot be negative")
    train(arguments)


if __name__ == "__main__":
    main()
