
import json
import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))

JSON_FILES = [
    "Test_Phase1_Time_All_result.json"
]

LABEL_MAP = {
    "real_celeb":    0,
    "real_celeb2":   0,
    "real_ff":       0,
    "deepfake_ff":   1,
    "deepfake_celeb": 1,
    "DFDC":          1,
    "mask":          2,
    "print_attack":  2,
    "replay_attack": 2,
}

CLASS_NAMES = {0: "real", 1: "deepfake", 2: "physical"}

MODEL_PATH    = os.path.join(TRAIN_DIR, "lgbm_model.pkl")
FEATURES_PATH = os.path.join(TRAIN_DIR, "lgbm_features.json")

FEATURE_NAMES = [
    "no_blink",
    "static_head",
    "rppg_absence",
    "temporal_freq",
    "gan_fingerprint",
    "texture",
    "compression_artifacts",
    "temporal_inconsistency",
    "mask_edges",
    "skin_tone",
    "face_warping",
    "color_inconsistency",
    "face_flicker",
    "temporal_texture",
    "halftone_pattern",
    "lbp_entropy",
    "color_depth",
    "specular_consistency",
    "face_bg_sharpness",
    "eye_region_temporal",
    "blending_boundary",
    "prnu_inconsistency",
]


def extract_record(video, label):
    raw = video.get("raw", {})

    p2 = raw.get("phase1_details") or raw.get("phase2_details")
    if not p2:
        return None, None

    row = []
    for feat in FEATURE_NAMES:
        det = p2.get(feat, {})
        score = det.get("raw_score", np.nan)
        if score == -1.0:
            score = np.nan
        row.append(score)

    return row, label

def load_all():
    X, y = [], []
    label_counts = {0: 0, 1: 0, 2: 0}

    for filename in JSON_FILES:
        path = os.path.join(TRAIN_DIR, filename)
        if not os.path.exists(path):
            print(f"skip{filename} not found")
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        for section in ("real", "fake"):
            for cat_name, videos in data.get(section, {}).items():
                label = LABEL_MAP.get(cat_name)
                if label is None:
                    print(f"  warn unknown category: {cat_name}, skipping")
                    continue

                for video in videos:
                    row, lbl = extract_record(video, label)
                    if row is None:
                        continue
                    X.append(row)
                    y.append(lbl)
                    label_counts[lbl] += 1

        print(f"loaded {filename}")

    print(f"\nDataset: real={label_counts[0]}  deepfake={label_counts[1]}  physical={label_counts[2]}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def train(X, y):
    X_df = pd.DataFrame(X, columns=FEATURE_NAMES)
    X_train, X_val, y_train, y_val = train_test_split(
        X_df, y, test_size=0.15, random_state=42, stratify=y
    )

    CLASS_WEIGHTS = {0: 2.5, 1: 1.0, 2: 1.0}
    counts = np.bincount(y_train)
    base_weights = len(y_train) / (len(counts) * counts)
    sample_weights = np.array([base_weights[c] * CLASS_WEIGHTS[c] for c in y_train])

    model = lgb.LGBMClassifier(
        num_class=3,
        objective="multiclass",
        metric="multi_logloss",

        n_estimators=800,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,

        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)],
    )

    return model, X_val, y_val

def predict_with_threshold(model, X, real_t=0.40, physical_t=0.70):
    probs = model.predict_proba(X)

    preds = []
    for p in probs:
        real_p = p[0]
        deepfake_p = p[1]
        physical_p = p[2]

        if physical_p > physical_t:
            preds.append(2)

        elif real_p > real_t:
            preds.append(0)

        else:
            preds.append(1)

    return np.array(preds)

def evaluate(model, X_val, y_val):
    y_pred_default = model.predict(X_val)
    y_pred_thresh = predict_with_threshold(model, X_val)

    print("\n── DEFAULT LightGBM ──")
    print(classification_report(y_val, y_pred_default,
                                target_names=["real", "deepfake", "physical"]))

    print("\n── THRESHOLD SWEEP (real_t) ──")
    for t in [0.30, 0.35, 0.40, 0.50, 0.55, 0.60]:
        preds = predict_with_threshold(model, X_val, real_t=t)
        print(f"\n  real_t={t}")
        print(classification_report(y_val, preds, target_names=["real", "deepfake", "physical"]))

    print("\n── THRESHOLD MODEL (real_t=0.40) ──")
    print(classification_report(y_val, y_pred_thresh,
                                target_names=["real", "deepfake", "physical"]))

    cm = confusion_matrix(y_val, y_pred_thresh)
    print("\nConfusion Matrix (threshold)")
    header = f"{'':>12}" + "".join(f"  pred_{CLASS_NAMES[i]:<10}" for i in range(3))
    print(header)
    for i, row in enumerate(cm):
        print(f"true_{CLASS_NAMES[i]:<8}" + "".join(f"  {v:<14}" for v in row))

    print("\nFeature Importance")
    importances = model.feature_importances_
    pairs = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
    for name, imp in pairs:
        bar = "█" * int(imp / max(importances) * 30)
        print(f"  {name:<28} {imp:>4}  {bar}")

def main():
    print("Loading data...")
    X, y = load_all()

    if len(X) == 0:
        print("No data found. Check JSON files.")
        return

    print(f"Features: {len(FEATURE_NAMES)}, Samples: {len(X)}\n")
    print("Training LightGBM...")
    model, X_val, y_val = train(X, y)

    evaluate(model, X_val, y_val)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved: {MODEL_PATH}")

    with open(FEATURES_PATH, "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    print(f"Features saved: {FEATURES_PATH}")


if __name__ == "__main__":
    main()