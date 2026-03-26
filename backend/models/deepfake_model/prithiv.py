import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
import random
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.utils.data import WeightedRandomSampler

MODEL_NAME           = "prithivMLmods/Deep-Fake-Detector-v2-Model"
FRAMES_DIR           = r"G:\dataset\dataset_ready"
SAVE_DIR             = r"E:\Projects\diploma\Deepfake_detection\backend\models"
BATCH_SIZE           = 32
EPOCHS               = 10
LR                   = 2e-5
MAX_FRAMES_PER_VIDEO = 30
PATIENCE             = 3

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


class FaceFrameDataset(Dataset):
    def __init__(self, samples, feature_extractor, augment=False):
        self.samples = samples
        self.feature_extractor = feature_extractor
        self.augment = augment

        self.aug_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.RandomGrayscale(p=0.05),
            
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, group = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.augment:
            img = self.aug_transforms(img)

        inputs = self.feature_extractor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        return pixel_values, torch.tensor(label, dtype=torch.long), group


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


def collate_fn(batch):

    pixels = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    groups = [b[2] for b in batch]
    return pixels, labels, groups


def prepare_model(model_name):
    feature_extractor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)

    for param in model.parameters():
        param.requires_grad = False

    try:

        for layer in model.vit.encoder.layer[-1:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in model.vit.layernorm.parameters():
            param.requires_grad = True
    except:
        pass

    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    return feature_extractor, model


def evaluate_by_group(model, test_samples, feature_extractor, save_dir):
    model.eval()

    group_data = {"asian": {"preds": [], "labels": []},
                  "others": {"preds": [], "labels": []}}

    with torch.no_grad():
        for fp, label, group in test_samples:
            img = Image.open(fp).convert("RGB")
            inputs = feature_extractor(images=img, return_tensors="pt")
            pixels = inputs["pixel_values"].to(device)
            outputs = model(pixel_values=pixels)
            pred = outputs.logits.argmax(dim=1).item()

            group_data[group]["preds"].append(pred)
            group_data[group]["labels"].append(label)

    lines = ["\n Metrics\n"]

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

        line = (f"{group.upper()}: "
                f"Acc={accuracy:.4f} | FAR={FAR:.4f} | FRR={FRR:.4f} | HTER={HTER:.4f} "
                f"({len(labels)} frames)")
        print(line)
        lines.append(line)

    metrics_path = os.path.join(save_dir, "metrics_by_group.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nMetrics saved : {metrics_path}")


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    feature_extractor, model = prepare_model(MODEL_NAME)
    model.to(device)
    print(model.config.id2label)

    train_samples = load_split(FRAMES_DIR, "train", MAX_FRAMES_PER_VIDEO)
    val_samples   = load_split(FRAMES_DIR, "val",   MAX_FRAMES_PER_VIDEO)
    test_samples  = load_split(FRAMES_DIR, "test",  MAX_FRAMES_PER_VIDEO)
    
    evaluate_by_group(model, test_samples, feature_extractor, SAVE_DIR)
    return
    # print(f"\nTrain: {len(train_samples)} frames")
    # print(f"Val:   {len(val_samples)} frames")
    # print(f"Test:  {len(test_samples)} frames\n")
    
    # asian_train  = sum(1 for _, l, g in train_samples if g == "asian")
    # others_train = sum(1 for _, l, g in train_samples if g == "others")
    # print(f"Train asian: {asian_train} | others: {others_train}")
    
    # train_dataset = FaceFrameDataset(train_samples, feature_extractor, augment=True)
    # val_dataset   = FaceFrameDataset(val_samples,   feature_extractor, augment=False)

    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
    #                           shuffle=True,  num_workers=0,  
    #                           collate_fn=collate_fn, pin_memory=True)
    
    # val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
    #                           shuffle=False, num_workers=0,
    #                           collate_fn=collate_fn, pin_memory=True)

    # optimizer = AdamW(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=LR, weight_decay=0.05  
    # )
    # scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    # criterion = torch.nn.CrossEntropyLoss()

    # best_val_acc = 0.0
    # no_improve   = 0

    # for epoch in range(EPOCHS):
    #     model.train()
    #     train_correct = 0
    #     train_total   = 0

    #     for batch_idx, (pixels, labels, _) in enumerate(train_loader):
    #         pixels = pixels.to(device)
    #         labels = labels.to(device)

    #         optimizer.zero_grad()
    #         outputs = model(pixel_values=pixels)
    #         loss = criterion(outputs.logits, labels)
    #         loss.backward()
    #         optimizer.step()

    #         preds = outputs.logits.argmax(dim=1)
    #         train_correct += (preds == labels).sum().item()
    #         train_total   += labels.size(0)

    #         if batch_idx % 20 == 0:
    #             print(f"  Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] "
    #                   f"loss: {loss.item():.4f}", end='\r')

    #     train_acc = train_correct / train_total

    #     model.eval()
    #     val_correct = 0
    #     val_total   = 0

    #     with torch.no_grad():
    #         for pixels, labels, _ in val_loader:
    #             pixels = pixels.to(device)
    #             labels = labels.to(device)
    #             outputs = model(pixel_values=pixels)
    #             preds = outputs.logits.argmax(dim=1)
    #             val_correct += (preds == labels).sum().item()
    #             val_total   += labels.size(0)

    #     val_acc = val_correct / val_total
    #     scheduler.step()

    #     print(f"Epoch {epoch+1}/{EPOCHS} | "
    #           f"Train acc: {train_acc:.3f} | "
    #           f"Val acc: {val_acc:.3f} | "
    #           f"LR: {scheduler.get_last_lr()[0]:.2e}")

    #     if val_acc > best_val_acc:
    #         best_val_acc = val_acc
    #         no_improve   = 0
    #         model.save_pretrained(os.path.join(SAVE_DIR, "best_model"))
    #         feature_extractor.save_pretrained(os.path.join(SAVE_DIR, "best_model"))
    #         print(f"Saved (val_acc: {val_acc:.3f})")
    #     else:
    #         no_improve += 1
    #         print(f"  No improvement ({no_improve}/{PATIENCE})")
    #         if no_improve >= PATIENCE:
    #             print(f"Early stopping {epoch+1}")
    #             break

    # best_model = AutoModelForImageClassification.from_pretrained(
    #     os.path.join(SAVE_DIR, "best_model")
    # ).to(device)
    # best_model.eval()

    # all_preds  = []
    # all_labels = []

    # test_dataset = FaceFrameDataset(test_samples, feature_extractor, augment=False)
    # test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
    #                           shuffle=False, num_workers=0,
    #                           collate_fn=collate_fn)

    # with torch.no_grad():
    #     for pixels, labels, _ in test_loader:
    #         pixels = pixels.to(device)
    #         outputs = best_model(pixel_values=pixels)
    #         preds = outputs.logits.argmax(dim=1)
    #         all_preds.extend(preds.cpu().numpy())
    #         all_labels.extend(labels.numpy())

    # all_preds  = np.array(all_preds)
    # all_labels = np.array(all_labels)

    # accuracy  = (all_preds == all_labels).mean()
    # fake_mask = all_labels == 1
    # real_mask = all_labels == 0
    # FAR       = (all_preds[real_mask] == 1).sum() / real_mask.sum()
    # FRR       = (all_preds[fake_mask] == 0).sum() / fake_mask.sum()
    # HTER      = (FAR + FRR) / 2

    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"FAR:      {FAR:.4f}")
    # print(f"FRR:      {FRR:.4f}")
    # print(f"HTER:     {HTER:.4f}")

    # with open(os.path.join(SAVE_DIR, "metrics.txt"), "w") as f:
    #     f.write(f"Model: {MODEL_NAME}\n")
    #     f.write(f"Best val acc: {best_val_acc:.4f}\n")
    #     f.write(f"Accuracy: {accuracy:.4f}\n")
    #     f.write(f"FAR:      {FAR:.4f}\n")
    #     f.write(f"FRR:      {FRR:.4f}\n")
    #     f.write(f"HTER:     {HTER:.4f}\n")

    # evaluate_by_group(best_model, test_samples, feature_extractor, SAVE_DIR)

if __name__ == "__main__":
    train()