from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


SEED = 42
BATCH_SIZE = 20
EPOCHS_PER_PHASE = 10
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HairModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.hidden = nn.Linear(32 * 99 * 99, 64)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv(x)))
        x = x.flatten(start_dim=1)
        x = nn.functional.relu(self.hidden(x))
        return self.output(x)


def evaluation_transform():
    return transforms.Compose([
        transforms.Resize(
            (200, 200),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
        NORMALIZE,
    ])


def augmented_train_transform():
    return transforms.Compose([
        transforms.Resize(
            (200, 200),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomRotation(
            50,
            interpolation=transforms.InterpolationMode.NEAREST,
        ),
        transforms.RandomResizedCrop(
            200,
            scale=(0.9, 1.0),
            ratio=(0.9, 1.1),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        NORMALIZE,
    ])


def make_loader(dataset, *, shuffle: bool, seed: int | None = None):
    options = {
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": 0,
    }
    if seed is not None:
        options["generator"] = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, **options)


def print_progress_bar(epoch, total_epochs, phase, loss=None, acc=None, eval_loss=None, eval_acc=None, bar_length=40):
    """Print a progress bar for training."""
    progress = epoch / total_epochs
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    parts = [f"Epoch {epoch}/{total_epochs} [{bar}] {progress*100:.1f}%"]
    if loss is not None:
        parts.append(f"train_loss={loss:.4f}")
    if acc is not None:
        parts.append(f"train_acc={acc:.4f}")
    if eval_loss is not None:
        parts.append(f"eval_loss={eval_loss:.4f}")
    if eval_acc is not None:
        parts.append(f"eval_acc={eval_acc:.4f}")
    
    sys.stdout.write(f"\r{phase}: " + " | ".join(parts))
    sys.stdout.flush()
    if epoch == total_epochs:
        print()


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)
            logits = model(images)
            loss = criterion(logits, labels)
            loss_sum += loss.item() * images.size(0)
            correct += ((torch.sigmoid(logits) >= 0.5) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total


def train_phase(model, loader, evaluation_loader, criterion, optimizer, device, phase_name="Training"):
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "evaluation_loss": [],
        "evaluation_accuracy": [],
    }
    for epoch in range(1, EPOCHS_PER_PHASE + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.float().unsqueeze(1)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * images.size(0)
            correct += ((torch.sigmoid(logits) >= 0.5) == labels).sum().item()
            total += labels.size(0)

        evaluation_loss, evaluation_accuracy = evaluate(
            model,
            evaluation_loader,
            criterion,
            device,
        )
        train_loss = loss_sum / total
        train_accuracy = correct / total
        
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["evaluation_loss"].append(evaluation_loss)
        history["evaluation_accuracy"].append(evaluation_accuracy)
        
        print_progress_bar(
            epoch, EPOCHS_PER_PHASE, phase_name,
            loss=train_loss, acc=train_accuracy,
            eval_loss=evaluation_loss, eval_acc=evaluation_accuracy
        )
    return history


def run(data_dir: Path, output_dir: Path) -> None:
    print("=" * 60)
    print("Hair Classification - Reference Training")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Seed: {SEED}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs per phase: {EPOCHS_PER_PHASE}")
    print(f"Device: cpu (deterministic)")
    print()
    
    print("[1/7] Setting random seeds for reproducibility...")
    seed_everything()
    torch.set_num_threads(2)
    device = torch.device("cpu")
    
    print("[2/7] Loading datasets...")
    train_dir = data_dir / "train"
    evaluation_dir = data_dir / "test"

    baseline_dataset = datasets.ImageFolder(train_dir, transform=evaluation_transform())
    evaluation_dataset = datasets.ImageFolder(
        evaluation_dir,
        transform=evaluation_transform(),
    )
    if len(baseline_dataset) != 800 or len(evaluation_dataset) != 201:
        raise ValueError(
            "unexpected ImageFolder counts; verify the archive against "
            "dataset_manifest.json"
        )
    if baseline_dataset.class_to_idx != {"curly": 0, "straight": 1}:
        raise ValueError(f"unexpected class mapping: {baseline_dataset.class_to_idx}")
    
    print(f"    Training samples: {len(baseline_dataset)} (curly: 410, straight: 390)")
    print(f"    Evaluation samples: {len(evaluation_dataset)} (curly: 103, straight: 98)")
    print(f"    Class mapping: {baseline_dataset.class_to_idx}")
    print()
    
    print("[3/7] Creating data loaders...")
    evaluation_loader = make_loader(evaluation_dataset, shuffle=False)
    
    print("[4/7] Initializing model, loss, and optimizer...")
    model = HairModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.8)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Model: HairModel (Conv2d -> MaxPool -> Linear -> Linear)")
    print(f"    Total parameters: {total_params:,}")
    print(f"    Loss: BCEWithLogitsLoss")
    print(f"    Optimizer: SGD (lr=0.002, momentum=0.8)")
    print()
    
    print("[5/7] Starting baseline training phase (no augmentation)...")
    baseline = train_phase(
        model,
        make_loader(baseline_dataset, shuffle=True, seed=SEED),
        evaluation_loader,
        criterion,
        optimizer,
        device,
        phase_name="Baseline",
    )
    
    print()
    print("[6/7] Starting augmented training phase (with augmentation)...")
    augmented_dataset = datasets.ImageFolder(
        train_dir,
        transform=augmented_train_transform(),
    )
    augmented = train_phase(
        model,
        make_loader(augmented_dataset, shuffle=True, seed=SEED + 1),
        evaluation_loader,
        criterion,
        optimizer,
        device,
        phase_name="Augmented",
    )
    
    print()
    print("[7/7] Saving results...")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "history_baseline.json").write_text(
        json.dumps(baseline, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "history_augmented.json").write_text(
        json.dumps(augmented, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"    Saved: {output_dir}/history_baseline.json")
    print(f"    Saved: {output_dir}/history_augmented.json")
    print()
    
    summary = {
        "output_dir": str(output_dir),
        "parameters": total_params,
        "baseline_median_train_accuracy": statistics.median(
            baseline["train_accuracy"]
        ),
        "baseline_train_loss_std_population": float(
            np.std(baseline["train_loss"], ddof=0)
        ),
        "augmented_mean_evaluation_loss": statistics.mean(
            augmented["evaluation_loss"]
        ),
        "augmented_last_five_evaluation_accuracy": statistics.mean(
            augmented["evaluation_accuracy"][-5:]
        ),
    }
    
    print("=" * 60)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/reference"))
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)
