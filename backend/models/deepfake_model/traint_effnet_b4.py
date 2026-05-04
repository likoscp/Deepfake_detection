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
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

CELEB_ROOT = r"D:\celeb\frames_celeb"
FF_ROOT    = r"D:\ff\frames"
SAVE_DIR   = str(Path(__file__).resolve().parent.parent / "efficientnet_b4_v2")

IMG_SIZE    = 160
BATCH_SIZE  = 32
MAX_FRAMES  = 30
NUM_WORKERS = 2

EPOCHS_PHASE1   = 5
LR_HEAD_PHASE1  = 3e-3

EPOCHS_PHASE2   = 15
LR_HEAD_PHASE2  = 3e-4
LR_BACKBONE     = 1e-5
UNFREEZE_BLOCKS = 2

PATIENCE = 7

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
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.10)),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def build_model():
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 2),
    )

    return model


def unfreeze_blocks(model, n_blocks):
    blocks = list(model.features.children())
    for block in blocks[-n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")


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
        return self.transform(img), torch.tensor(label, dtype=torch.long), source

def collate_fn(batch):
    pixels = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    groups = [b[2] for b in batch]
    return pixels, labels, groups

def load_split(split, max_per_video=MAX_FRAMES):
    samples = []
    sources = [(CELEB_ROOT, "celeb"), (FF_ROOT, "ff")]

    for base_dir, source_name in sources:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            print(f"  [WARN] No foulder: {split_dir}")
            continue

        for label_name in ["real", "fake"]:
            label_idx = 0 if label_name == "real" else 1
            label_dir = os.path.join(split_dir, label_name)
            if not os.path.exists(label_dir):
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

            print(f"  {split}/{source_name}/{label_name}: {len(video_dirs)} video, {total_frames} frames")

    return samples

def run_training_phase(model, train_loader, val_loader, criterion, optimizer,
                       scheduler, scaler, epochs, patience, best_val_acc,
                       best_model_path, phase_name):
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_correct = 0
        train_total   = 0

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

            if batch_idx % 50 == 0:
                print(f"  [{phase_name}] Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] "
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

        print(f"[{phase_name}] Epoch {epoch+1:02d}/{epochs} | "
              f"Train={train_acc:.4f} | Val={val_acc:.4f} | "
              f"LR={optimizer.param_groups[-1]['lr']:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Saved best (val={val_acc:.4f})")
        else:
            no_improve += 1
            print(f"  No improve ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break
    return best_val_acc

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
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
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
        FAR  = (preds[labels == 0] == 1).sum() / max((labels == 0).sum(), 1)
        FRR  = (preds[labels == 1] == 0).sum() / max((labels == 1).sum(), 1)
        HTER = (FAR + FRR) / 2
        line = (f"{group.upper():8s}: Acc={accuracy:.4f} | "
                f"FAR={FAR:.4f} | FRR={FRR:.4f} | HTER={HTER:.4f} ({len(labels)} frames)")
        print(line)
        lines.append(line)

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "metrics_by_source.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("\nLoading splits...")
    train_samples = load_split("train", MAX_FRAMES)
    val_samples   = load_split("val",   MAX_FRAMES)
    test_samples  = load_split("test",  MAX_FRAMES)
    print(f"\nTrain: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}\n")

    train_ds = FaceFrameDataset(train_samples, train_transforms)
    val_ds   = FaceFrameDataset(val_samples,   val_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn,
                              pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn,
                              pin_memory=True, persistent_workers=True)

    real_count = sum(1 for _, l, _ in train_samples if l == 0)
    fake_count = sum(1 for _, l, _ in train_samples if l == 1)
    weight     = torch.tensor([1.0, real_count / fake_count]).to(device)
    criterion  = nn.CrossEntropyLoss(weight=weight)
    print(f"Class weight: real=1.0, fake={real_count/fake_count:.3f}")

    scaler          = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    best_val_acc    = 0.0
    best_model_path = os.path.join(SAVE_DIR, "best_model.pth")

    print(f"\n{'='*50}")
    print(f"PHASE 1: only head, backbone freeze")
    print(f"{'='*50}")

    model = build_model().to(device)
    unfreeze_blocks(model, 0) 

    optimizer1 = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR_HEAD_PHASE1, weight_decay=0.01
    )
    scheduler1 = CosineAnnealingLR(optimizer1, T_max=EPOCHS_PHASE1, eta_min=1e-5)

    best_val_acc = run_training_phase(
        model, train_loader, val_loader, criterion,
        optimizer1, scheduler1, scaler,
        EPOCHS_PHASE1, PATIENCE, best_val_acc,
        best_model_path, "P1"
    )

    print(f"\n{'='*50}")
    print(f"PHASE 2: unfreeze {UNFREEZE_BLOCKS} bloks backbone")
    print(f"{'='*50}")

    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    unfreeze_blocks(model, UNFREEZE_BLOCKS)

    optimizer2 = AdamW([
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and "classifier" not in n], "lr": LR_BACKBONE},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and "classifier" in n],     "lr": LR_HEAD_PHASE2},
    ], weight_decay=0.01)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=EPOCHS_PHASE2, eta_min=1e-6)

    best_val_acc = run_training_phase(
        model, train_loader, val_loader, criterion,
        optimizer2, scheduler2, scaler,
        EPOCHS_PHASE2, PATIENCE, best_val_acc,
        best_model_path, "P2"
    )

    print("\nLoading best model for test...")
    best_model = build_model().to(device)
    unfreeze_blocks(best_model, UNFREEZE_BLOCKS)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    best_model.eval()

    test_ds     = FaceFrameDataset(test_samples, val_transforms)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, collate_fn=collate_fn)

    all_preds, all_labels = [], []

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
    FAR       = (all_preds[all_labels == 0] == 1).sum() / (all_labels == 0).sum()
    FRR       = (all_preds[all_labels == 1] == 0).sum() / (all_labels == 1).sum()
    HTER      = (FAR + FRR) / 2

    result = (
        f"Model:         EfficientNet-B4 fine-tune v2 (2-phase)\n"
        f"IMG_SIZE:      {IMG_SIZE}\n"
        f"UNFREEZE:      {UNFREEZE_BLOCKS} blocks (phase 2)\n"
        f"Best val acc:  {best_val_acc:.4f}\n"
        f"Accuracy:      {accuracy:.4f}\n"
        f"FAR:           {FAR:.4f}\n"
        f"FRR:           {FRR:.4f}\n"
        f"HTER:          {HTER:.4f}\n"
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