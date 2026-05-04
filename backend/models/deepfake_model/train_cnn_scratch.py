import os
from pathlib import Path
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms


CELEB_ROOT = r"D:\celeb\frames_celeb"
FF_ROOT    = r"D:\ff\frames"

SAVE_DIR    = str(Path(__file__).resolve().parent.parent / "cnn_scratch")
EPOCHS      = 20
LR         = 1e-4 
BATCH_SIZE = 64
PATIENCE   = 7 

IMG_SIZE    = 128
MAX_FRAMES  = 30

NUM_WORKERS = 2
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
    transforms.RandomRotation(15),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class DeepfakeCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 512),
            nn.Sigmoid(),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        att = self.attention(x).unsqueeze(-1).unsqueeze(-1)
        x = x * att
        x = self.gap(x)
        x = self.classifier(x)
        return x

class FaceFrameDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, source = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long), source


def collate_fn(batch):
    pixels = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    groups = [b[2] for b in batch]
    return pixels, labels, groups

def load_split(split, max_per_video=MAX_FRAMES):

    samples = []

    sources = [
        (CELEB_ROOT, "celeb"),
        (FF_ROOT,    "ff"),
    ]

    for base_dir, source_name in sources:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            print(f"  [WARN] No foulder: {split_dir}")
            continue

        for label_name in ["real", "fake"]:
            label_idx = 0 if label_name == "real" else 1
            label_dir = os.path.join(split_dir, label_name)
            if not os.path.exists(label_dir):
                print(f"  [WARN] No foulder: {label_dir}")
                continue

            video_dirs = []
            for root, dirs, files in os.walk(label_dir):
                if any(f.lower().endswith(('.jpg', '.png', '.jpeg')) for f in files):
                    video_dirs.append(root)

            total_frames = 0
            for vdir in video_dirs:
                frames = [
                    os.path.join(vdir, f)
                    for f in os.listdir(vdir)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))
                ]
                if not frames:
                    continue
                if len(frames) > max_per_video:
                    frames = random.sample(frames, max_per_video)
                for fp in frames:
                    samples.append((fp, label_idx, source_name))
                total_frames += len(frames)

            print(f"  {split}/{source_name}/{label_name}: "
                  f"{len(video_dirs)} video, {total_frames} frames")

    return samples

def evaluate_by_group(model, test_samples, transform, save_dir):
    model.eval()
    all_groups = sorted(set(g for _, _, g in test_samples))
    group_data = {g: {"preds": [], "labels": []} for g in all_groups}

    with torch.no_grad():
        for fp, label, group in test_samples:
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                continue
            img = transform(img).unsqueeze(0).to(device)
            pred = model(img).argmax(dim=1).item()
            group_data[group]["preds"].append(pred)
            group_data[group]["labels"].append(label)

    lines = ["\n=== Metrics by source ==="]
    print(lines[0])

    for group in all_groups:
        preds  = np.array(group_data[group]["preds"])
        labels = np.array(group_data[group]["labels"])
        if len(labels) == 0:
            continue

        accuracy  = (preds == labels).mean()
        fake_mask = labels == 1
        real_mask = labels == 0
        FAR  = (preds[real_mask] == 1).sum() / max(real_mask.sum(), 1)
        FRR  = (preds[fake_mask] == 0).sum() / max(fake_mask.sum(), 1)
        HTER = (FAR + FRR) / 2

        line = (f"{group.upper():8s}: Acc={accuracy:.4f} | "
                f"FAR={FAR:.4f} | FRR={FRR:.4f} | HTER={HTER:.4f} "
                f"({len(labels)} frames)")
        print(line)
        lines.append(line)

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "metrics_by_source.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved: {save_dir}/metrics_by_source.txt")

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    model = DeepfakeCNN(num_classes=2).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    print("\nLoading splits...")
    train_samples = load_split("train", MAX_FRAMES)
    val_samples   = load_split("val",   MAX_FRAMES)
    test_samples  = load_split("test",  MAX_FRAMES)
    print(f"\nTrain: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}\n")

    train_ds = FaceFrameDataset(train_samples, train_transforms)
    val_ds   = FaceFrameDataset(val_samples,   val_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn,
                              pin_memory=(device == "cuda"), persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn,
                              pin_memory=(device == "cuda"), persistent_workers=True)

    real_count = sum(1 for _, l, _ in train_samples if l == 0)
    fake_count = sum(1 for _, l, _ in train_samples if l == 1)
    weight = torch.tensor([1.0, real_count / fake_count]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    print(f"Class weight: real=1.0, fake={real_count/fake_count:.3f}")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_val_acc = 0.0
    no_improve   = 0
    best_model_path = os.path.join(SAVE_DIR, "best_model.pth")

    for epoch in range(EPOCHS):
        model.train()
        train_correct = 0
        train_total   = 0
        running_loss  = 0.0

        for batch_idx, (pixels, labels, _) in enumerate(train_loader):
            pixels = pixels.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                outputs = model(pixels)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)
            running_loss  += loss.item()

            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] "
                      f"loss={loss.item():.4f}", end='\r')

        train_acc = train_correct / train_total

        model.eval()
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for pixels, labels, _ in val_loader:
                pixels = pixels.to(device)
                labels = labels.to(device)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    outputs = model(pixels)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train={train_acc:.4f} | Val={val_acc:.4f} | "
              f"LR={scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Saved best (val={val_acc:.4f})")
        else:
            no_improve += 1
            print(f"  No improve ({no_improve}/{PATIENCE})")
            if no_improve >= PATIENCE:
                print(f"Early stop at epoch {epoch+1}")
                break

    print("\nLoading best model for test...")
    best_model = DeepfakeCNN(num_classes=2).to(device)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    best_model.eval()

    test_ds     = FaceFrameDataset(test_samples, val_transforms)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, collate_fn=collate_fn)

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for pixels, labels, _ in test_loader:
            pixels = pixels.to(device)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                outputs = best_model(pixels)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy  = (all_preds == all_labels).mean()
    fake_mask = all_labels == 1
    real_mask = all_labels == 0
    FAR       = (all_preds[real_mask] == 1).sum() / real_mask.sum()
    FRR       = (all_preds[fake_mask] == 0).sum() / fake_mask.sum()
    HTER      = (FAR + FRR) / 2

    result = (
        f"Model:    CascadeVerify-CNN (scratch)\n"
        f"IMG_SIZE: {IMG_SIZE}\n"
        f"Best val: {best_val_acc:.4f}\n"
        f"Accuracy: {accuracy:.4f}\n"
        f"FAR:      {FAR:.4f}\n"
        f"FRR:      {FRR:.4f}\n"
        f"HTER:     {HTER:.4f}\n"
    )
    print("\n" + result)

    with open(os.path.join(SAVE_DIR, "metrics.txt"), "w") as f:
        f.write(result)

    evaluate_by_group(best_model, test_samples, val_transforms, SAVE_DIR)
    print(f"\nDone. Best model: {best_model_path}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    train()