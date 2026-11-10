"""
Graded modality-ablation sweep.

Instead of removing a modality completely, keep it for X% of the test patients
(100, 90, 80, ... 0) and measure how performance degrades.

Design
------
* "keep rate" = % of patients that still have the modality. 100 = nothing
  removed, 0 = modality fully removed (what the old ablations did).
* Sample-level: a dropped patient loses the whole modality (image or clinical).
* Masks are nested. Each test patient gets one fixed random number u in [0,1);
  the patient keeps the modality when u < keep. So everyone dropped at 90% is
  also dropped at 80%, and the curves are comparable instead of re-rolled noise.
* Train/val/test are three separate splits (60/20/20, stratified). The best
  checkpoint is picked on val and the sweep runs on test, so the test set is
  not also used for model selection.
* Every run is repeated over several seeds (new split, new training, new masks).
  The test set is small, so single-seed curves are mostly noise.
* Two variants: with BI-RADS in the clinical features and without it. In
  data2.xlsx BI-RADS maps 1:1 onto the class, so it leaks the label.
* The model is trained with modality dropout (--train-drop), otherwise zeros in
  place of a missing modality would be out-of-distribution at test time.

Run from the "Ablation Experiments - text, random ,gan" folder, or anywhere:
    python ablation_experiments/ablation_sweep.py --seeds 5
"""
import argparse
import copy
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(os.path.dirname(HERE))  # ROOT_DIR in ablation_train is relative: "AISSLab"

from ablation_train import DEVICE, get_image_path  # noqa: E402

CLASS_TO_IDX = {"Normal": 0, "Benign": 1, "Malignant": 2}
KEEP_RATES = [round(k / 10, 1) for k in range(10, -1, -1)]  # 1.0, 0.9, ... 0.0
VARIANTS = {"with_birads": True, "without_birads": False}
MODALITIES = ["image", "text"]


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def clinical_vector(row, use_birads):
    side = row["Side"].strip().lower()
    view = row["View"].strip().upper()
    feats = [1, 0] if side == "left" else [0, 1]
    feats += [1, 0] if view == "CC" else [0, 1]
    if use_birads:
        feats.append(float(row["BI-RADS"].split("-")[-1]) / 5.0)
    return feats


def load_data(image_size=64):
    df = pd.read_excel(os.path.join("AISSLab", "data2.xlsx"))
    df["ImagePath"] = df.apply(get_image_path, axis=1)
    found = df["ImagePath"].apply(os.path.exists)
    if not found.all():
        print(f"WARNING: {(~found).sum()} of {len(df)} images not found; using {found.sum()}.")
    df = df[found].reset_index(drop=True)

    tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    imgs = []
    for p in df["ImagePath"]:
        arr = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        imgs.append(tf(Image.fromarray(arr)))
    images = torch.stack(imgs)
    labels = torch.tensor([CLASS_TO_IDX[c] for c in df["Class"]], dtype=torch.long)
    feats = {
        name: torch.tensor(
            [clinical_vector(r, use_b) for _, r in df.iterrows()], dtype=torch.float32
        )
        for name, use_b in VARIANTS.items()
    }
    print("Class counts:", df["Class"].value_counts().to_dict())
    return df, images, feats, labels


def three_way_split(labels, seed):
    idx = np.arange(len(labels))
    y = labels.numpy()
    trainval, test = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    train, val = train_test_split(trainval, test_size=0.25, stratify=y[trainval], random_state=seed)
    return train, val, test


# ----------------------------------------------------------------------------
# Model: missing modality -> its feature vector is zeroed, so the content of the
# (dummy) input is irrelevant.
# ----------------------------------------------------------------------------
class SweepModel(nn.Module):
    def __init__(self, text_dim):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.text_encoder = nn.Sequential(nn.Linear(text_dim, 128), nn.ReLU(), nn.Dropout(0.3))
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 3)
        )

    def forward(self, image, text, has_image, has_text):
        img_f = self.image_encoder(image).flatten(1) * has_image[:, None].float()
        txt_f = self.text_encoder(text) * has_text[:, None].float()
        return self.classifier(torch.cat([img_f, txt_f], dim=1))


@torch.no_grad()
def predict(model, images, feats, idx, has_image, has_text):
    model.eval()
    out = model(
        images[idx].to(DEVICE), feats[idx].to(DEVICE),
        torch.as_tensor(has_image).to(DEVICE), torch.as_tensor(has_text).to(DEVICE),
    )
    return out.argmax(1).cpu().numpy()


def train_model(images, feats, labels, train_idx, val_idx, args, seed):
    torch.manual_seed(seed)
    model = SweepModel(feats.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.RandomState(seed)
    best_f1, best_state = -1.0, None
    ones_val = np.ones(len(val_idx), dtype=bool)

    for _ in range(args.epochs):
        model.train()
        order = rng.permutation(train_idx)
        for i in range(0, len(order), args.batch_size):
            b = order[i:i + args.batch_size]
            if len(b) < 2:  # skip trailing single-sample batch
                continue
            # modality dropout: with prob train_drop, lose image (half) or text (half)
            r = rng.rand(len(b))
            has_image = ~(r < args.train_drop / 2)
            has_text = ~((r >= args.train_drop / 2) & (r < args.train_drop))
            opt.zero_grad()
            out = model(
                images[b].to(DEVICE), feats[b].to(DEVICE),
                torch.as_tensor(has_image).to(DEVICE), torch.as_tensor(has_text).to(DEVICE),
            )
            loss_fn(out, labels[b].to(DEVICE)).backward()
            opt.step()
        preds = predict(model, images, feats, val_idx, ones_val, ones_val)
        f1 = f1_score(labels[val_idx].numpy(), preds, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1, best_state = f1, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_f1


# ----------------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------------
def sweep_one_model(model, images, feats, labels, test_idx, seed, mask_draws):
    """Evaluate one trained model at every keep rate for every modality."""
    rows = []
    y = labels[test_idx].numpy()
    n = len(test_idx)
    for modality in MODALITIES:
        for draw in range(mask_draws):
            # one fixed u per patient -> nested masks across keep rates
            u = np.random.RandomState(seed * 1000 + draw * 10 + MODALITIES.index(modality)).rand(n)
            for keep in KEEP_RATES:
                kept = u < keep  # keep=1.0 -> all kept, keep=0.0 -> none kept
                has_image = kept if modality == "image" else np.ones(n, dtype=bool)
                has_text = kept if modality == "text" else np.ones(n, dtype=bool)
                pred = predict(model, images, feats, test_idx, has_image, has_text)
                rows.append({
                    "modality": modality, "keep": keep, "draw": draw,
                    "accuracy": accuracy_score(y, pred),
                    "f1": f1_score(y, pred, average="weighted", zero_division=0),
                })
    return rows


def plot_curves(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    colors = {"with_birads": "#c0392b", "without_birads": "#2471a3"}
    titles = {"image": "Image removed for some patients", "text": "Clinical data removed for some patients"}
    for ax, modality in zip(axes, MODALITIES):
        for variant in VARIANTS:
            s = summary[(summary.variant == variant) & (summary.modality == modality)].sort_values("keep")
            x = s["keep"] * 100
            ax.plot(x, s["f1_mean"], marker="o", color=colors[variant], label=variant.replace("_", " "))
            ax.fill_between(x, s["f1_mean"] - s["f1_std"], s["f1_mean"] + s["f1_std"],
                            color=colors[variant], alpha=0.15)
        ax.set_title(titles[modality])
        ax.set_xlabel(f"% of patients that still have the {modality} data")
        ax.set_xlim(105, -5)  # 100 on the left, falling to 0
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Weighted F1 (mean ± std over seeds)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--train-drop", type=float, default=0.5,
                    help="prob. of dropping one modality per training sample")
    ap.add_argument("--mask-draws", type=int, default=5,
                    help="random masks per trained model (cheap, no retraining)")
    ap.add_argument("--out-dir", default=os.path.join("ablation_experiments", "results", "sweep"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df, images, feats_by_variant, labels = load_data()
    all_rows = []
    for seed in range(args.seeds):
        train_idx, val_idx, test_idx = three_way_split(labels, seed)
        print(f"\nseed {seed}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
        for variant in VARIANTS:
            feats = feats_by_variant[variant]
            model, val_f1 = train_model(images, feats, labels, train_idx, val_idx, args, seed)
            print(f"  {variant:15s} best val F1 = {val_f1:.3f}")
            for r in sweep_one_model(model, images, feats, labels, test_idx, seed, args.mask_draws):
                all_rows.append({"seed": seed, "variant": variant, **r})

    res = pd.DataFrame(all_rows)
    res.to_csv(os.path.join(args.out_dir, "sweep_raw.csv"), index=False)

    # average mask draws within a seed first, then mean/std across seeds
    per_seed = res.groupby(["variant", "modality", "keep", "seed"], as_index=False)[["accuracy", "f1"]].mean()
    summary = per_seed.groupby(["variant", "modality", "keep"], as_index=False).agg(
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
    ).fillna(0.0)
    summary.to_csv(os.path.join(args.out_dir, "sweep_summary.csv"), index=False)
    with open(os.path.join(args.out_dir, "sweep_summary.json"), "w") as f:
        json.dump({"args": vars(args), "summary": summary.to_dict(orient="records")}, f, indent=2)
    plot_curves(summary, os.path.join(args.out_dir, "sweep_curves.png"))

    pd.set_option("display.width", 140)
    for modality in MODALITIES:
        print(f"\n=== {modality} removed for some patients: weighted F1 (mean over seeds) ===")
        print(summary[summary.modality == modality]
              .pivot(index="keep", columns="variant", values="f1_mean")
              .sort_index(ascending=False).round(3))
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
