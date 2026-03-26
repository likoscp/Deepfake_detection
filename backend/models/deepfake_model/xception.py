import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
import random
import timm

# xception OR' efficientnet_b4
MODEL_NAME  = "efficientnet_b4"

FRAMES_DIR  = r"G:\dataset\dataset_ready"
SAVE_DIR    = r"E:\Projects\diploma\Deepfake_detection\backend\models"
BATCH_SIZE  = 16
EPOCHS      = 10
LR          = 1e-4
MAX_FRAMES  = 30
PATIENCE    = 3
IMG_SIZE    = 299 if "xception" in MODEL_NAME else 380


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device} | Model: {MODEL_NAME}")

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
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class FaceFrameDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, group = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long), group


def collate_fn(batch):
    pixels = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    groups = [b[2] for b in batch]
    return pixels, labels, groups


def load_split(frames_dir, split, max_per_video=30):
    samples = []
    split_dir = os.path.join(frames_dir, split)

    for group in ["asian", "others"]:
        for label_name, label_idx in [("real", 0), ("fake", 1)]:
            label_dir = os.path.join(split_dir, group, label_name)
            if not os.path.exists(label_dir):
                print(f"  folder not found: {label_dir}")
                continue

            video_dirs = []
            for root, dirs, files in os.walk(label_dir):
                if any(f.endswith('.jpg') for f in files):
                    video_dirs.append(root)

            for video_path in video_dirs:
                frames = [
                    os.path.join(video_path, f)
                    for f in os.listdir(video_path)
                    if f.endswith('.jpg')
                ]
                if len(frames) > max_per_video:
                    frames = random.sample(frames, max_per_video)
                for fp in frames:
                    samples.append((fp, label_idx, group))

            print(f"  {split}/{group}/{label_name}: {len(video_dirs)} video")

    return samples


def prepare_model(model_name):
    model = timm.create_model(model_name, pretrained=True, num_classes=2)

    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    return model

def evaluate_by_group(model, test_samples, transform, save_dir, suffix=""):
    model.eval()
    group_data = {"asian":  {"preds": [], "labels": []},
                  "others": {"preds": [], "labels": []}}

    with torch.no_grad():
        for fp, label, group in test_samples:
            img = Image.open(fp).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)
            outputs = model(img)
            pred = outputs.argmax(dim=1).item()
            group_data[group]["preds"].append(pred)
            group_data[group]["labels"].append(label)

    lines = [f"\n Metrics by groups({MODEL_NAME}) \n"]
    print(lines[0])

    for group in ["asian", "others"]:
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

        line = (f"{group.upper()}: Acc={accuracy:.4f} | "
                f"FAR={FAR:.4f} | FRR={FRR:.4f} | HTER={HTER:.4f} "
                f"({len(labels)} frames)")
        print(line)
        lines.append(line)

    fname = f"metrics_by_group_{MODEL_NAME}{suffix}.txt"
    with open(os.path.join(save_dir, fname), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved: {fname}")


def train():
    model_save_dir = os.path.join(SAVE_DIR, MODEL_NAME)
    os.makedirs(model_save_dir, exist_ok=True)

    model = prepare_model(MODEL_NAME)
    model.to(device)
    train_samples = load_split(FRAMES_DIR, "train", MAX_FRAMES)
    val_samples   = load_split(FRAMES_DIR, "val",   MAX_FRAMES)
    test_samples  = load_split(FRAMES_DIR, "test",  MAX_FRAMES)

    print(f"\nTrain: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}\n")

    train_dataset = FaceFrameDataset(train_samples, train_transforms)
    val_dataset   = FaceFrameDataset(val_samples,   val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0,
                              collate_fn=collate_fn, pin_memory=True)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=0.05
    )
    scheduler  = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion  = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    no_improve   = 0

    for epoch in range(EPOCHS):
        model.train()
        train_correct = 0
        train_total   = 0

        for batch_idx, (pixels, labels, _) in enumerate(train_loader):
            pixels = pixels.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(pixels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] "
                      f"loss: {loss.item():.4f}", end='\r')

        train_acc = train_correct / train_total

        model.eval()
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for pixels, labels, _ in val_loader:
                pixels = pixels.to(device)
                labels = labels.to(device)
                outputs = model(pixels)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train acc: {train_acc:.3f} | "
              f"Val acc: {val_acc:.3f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save(model.state_dict(),
                       os.path.join(model_save_dir, "best_model.pth"))
            print(f"Saved (val_acc: {val_acc:.3f})")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{PATIENCE})")
            if no_improve >= PATIENCE:
                print(f"Early stopping {epoch+1}")
                break

    best_model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=2)
    best_model.load_state_dict(
        torch.load(os.path.join(model_save_dir, "best_model.pth"))
    )
    best_model.to(device)
    best_model.eval()

    all_preds  = []
    all_labels = []

    test_dataset = FaceFrameDataset(test_samples, val_transforms)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0,
                              collate_fn=collate_fn)

    with torch.no_grad():
        for pixels, labels, _ in test_loader:
            pixels = pixels.to(device)
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

    print(f"\n metrics ({MODEL_NAME}) ")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"FAR:      {FAR:.4f}")
    print(f"FRR:      {FRR:.4f}")
    print(f"HTER:     {HTER:.4f}")

    with open(os.path.join(model_save_dir, "metrics.txt"), "w") as f:
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Best val acc: {best_val_acc:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"FAR:      {FAR:.4f}\n")
        f.write(f"FRR:      {FRR:.4f}\n")
        f.write(f"HTER:     {HTER:.4f}\n")

    evaluate_by_group(best_model, test_samples, val_transforms, model_save_dir)
    print(f"\nDone. Model: {model_save_dir}/best_model.pth")

def test_only():
    model_save_dir = os.path.join(SAVE_DIR, MODEL_NAME)
    
    test_samples = load_split(FRAMES_DIR, "test", MAX_FRAMES)
    print(f"Test: {len(test_samples)} frames\n")

    best_model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=2)
    best_model.load_state_dict(
        torch.load(os.path.join(model_save_dir, "best_model.pth"), weights_only=True)
    )
    best_model.to(device)
    best_model.eval()

    evaluate_by_group(best_model, test_samples, val_transforms, model_save_dir)

if __name__ == "__main__":
    test_only()
# if __name__ == "__main__":
#     train()