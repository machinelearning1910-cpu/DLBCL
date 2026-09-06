

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from methodology import (
    FEATURE_ROOT,
    RESULT_ROOT,
    STAINS,
    TASKS,
    PATCHES_PER_STAIN,
    build_datasets_and_model,
    collect_outputs,
    concordance_index,
    device,
)

OUT = RESULT_ROOT / "paper_evaluation"
OUT.mkdir(parents=True, exist_ok=True)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")


def safe_auprc(y: np.ndarray, p: np.ndarray) -> float:
    return average_precision_score(y, p) if len(np.unique(y)) == 2 else float("nan")


def load_thresholds() -> dict[str, float]:
    path = RESULT_ROOT / "validation_thresholds.json"
    if not path.exists():
        raise FileNotFoundError("Validation thresholds not found. Run methodology.py first.")
    return json.loads(path.read_text())


def task_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {
        "N": len(y),
        "Threshold": threshold,
        "AUROC": safe_auc(y, p),
        "AUPRC": safe_auprc(y, p),
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1": f1_score(y, pred, zero_division=0),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def evaluate_classification(
    predictions: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]],
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rows = []
    metric_cols = [
        "AUROC", "AUPRC", "Accuracy", "Precision",
        "Sensitivity", "Specificity", "F1",
    ]

    for split, (y, p) in predictions.items():
        split_rows = []

        for task in TASKS:
            row = {
                "Split": split,
                "Task": task.upper(),
                **task_metrics(y[task], p[task], thresholds[task]),
            }
            rows.append(row)
            split_rows.append(row)

        # Paper-style overall row: macro average across the five classification endpoints.
        overall = {
            "Split": split,
            "Task": "OVERALL",
            "N": int(sum(r["N"] for r in split_rows)),
            "Threshold": np.nan,
            "TN": np.nan,
            "FP": np.nan,
            "FN": np.nan,
            "TP": np.nan,
        }
        for col in metric_cols:
            overall[col] = float(np.nanmean([r[col] for r in split_rows]))
        rows.append(overall)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "classification_metrics.csv", index=False)
    return df


def plot_confusion_matrices(
    y: dict[str, np.ndarray],
    p: dict[str, np.ndarray],
    thresholds: dict[str, float],
    split: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8))
    axes = axes.ravel()
    for i, task in enumerate(TASKS):
        pred = (p[task] >= thresholds[task]).astype(int)
        cm = confusion_matrix(y[task], pred, labels=[0, 1])
        axes[i].imshow(cm)
        for r in range(2):
            for c in range(2):
                axes[i].text(c, r, int(cm[r, c]), ha="center", va="center", fontsize=13)
        axes[i].set_title(task.upper())
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Ground Truth")
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
    axes[-1].axis("off")
    fig.suptitle(f"Multi-Task {split} Confusion Matrices")
    plt.tight_layout()
    plt.savefig(OUT / f"{split.lower()}_confusion_matrices.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(
    y: dict[str, np.ndarray],
    p: dict[str, np.ndarray],
    split: str,
) -> None:
    plt.figure(figsize=(7.5, 5.8))
    for task in TASKS:
        if len(np.unique(y[task])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y[task], p[task])
        auc = roc_auc_score(y[task], p[task])
        plt.plot(fpr, tpr, linewidth=2, label=f"{task.upper()} (AUROC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Multi-Task {split} ROC Curves")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUT / f"{split.lower()}_roc_curves.png", dpi=600, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7.5, 5.8))
    for task in TASKS:
        if len(np.unique(y[task])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y[task], p[task])
        ap = average_precision_score(y[task], p[task])
        plt.plot(recall, precision, linewidth=2, label=f"{task.upper()} (AUPRC={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Multi-Task {split} Precision-Recall Curves")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUT / f"{split.lower()}_pr_curves.png", dpi=600, bbox_inches="tight")
    plt.close()


def collect_age(loader: DataLoader) -> np.ndarray:
    ages = []
    for batch in loader:
        ages.extend(batch["age"].numpy())
    return np.asarray(ages, dtype=float)


def subgroup_metrics(
    y: np.ndarray,
    p: np.ndarray,
    pred: np.ndarray,
    time: np.ndarray,
    risk: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    yy, pp, dd = y[mask], p[mask], pred[mask]
    tt, rr = time[mask], risk[mask]
    tn, fp, fn, tp = confusion_matrix(yy, dd, labels=[0, 1]).ravel()
    return {
        "N": int(mask.sum()),
        "Positive_Rate": float(dd.mean()) if len(dd) else np.nan,
        "TPR": tp / (tp + fn) if tp + fn else np.nan,
        "FPR": fp / (fp + tn) if fp + tn else np.nan,
        "AUROC": safe_auc(yy, pp),
        "C_index": concordance_index(tt, yy, rr),
    }


def evaluate_fairness(
    loader: DataLoader,
    event_y: np.ndarray,
    event_p: np.ndarray,
    risk: np.ndarray,
    time: np.ndarray,
    threshold: float,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ages = collect_age(loader)
    pred = (event_p >= threshold).astype(int)
    young, old = ages < 60, ages >= 60

    y_metrics = subgroup_metrics(event_y, event_p, pred, time, risk, young)
    o_metrics = subgroup_metrics(event_y, event_p, pred, time, risk, old)

    groups = pd.DataFrame(
        [
            {"Split": split, "Age_Group": "<60", **y_metrics},
            {"Split": split, "Age_Group": ">=60", **o_metrics},
        ]
    )
    gaps = pd.DataFrame(
        [
            {
                "Split": split,
                "DPD": abs(y_metrics["Positive_Rate"] - o_metrics["Positive_Rate"]),
                "EOD_TPR_Gap": abs(y_metrics["TPR"] - o_metrics["TPR"]),
                "FPR_Gap": abs(y_metrics["FPR"] - o_metrics["FPR"]),
                "AUROC_Gap": abs(y_metrics["AUROC"] - o_metrics["AUROC"]),
                "Cindex_Gap": abs(y_metrics["C_index"] - o_metrics["C_index"]),
            }
        ]
    )
    return groups, gaps


def parameter_efficiency(model) -> pd.DataFrame:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable
    df = pd.DataFrame(
        [
            {
                "Total_parameters": total,
                "Trainable_parameters": trainable,
                "Frozen_parameters": frozen,
                "Trainable_percent": 100 * trainable / total,
                "Frozen_percent": 100 * frozen / total,
            }
        ]
    )
    df.to_csv(OUT / "parameter_efficiency.csv", index=False)
    return df


def plot_training_history() -> None:
    path = RESULT_ROOT / "multitask_training_history.csv"
    if not path.exists():
        return
    hist = pd.read_csv(path)
    plt.figure(figsize=(7, 5))
    plt.plot(hist["epoch"], hist["train_loss"], marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Multi-Task Training Loss")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUT / "training_loss.png", dpi=600, bbox_inches="tight")
    plt.close()

    auc_cols = [f"{t}_auc" for t in TASKS if f"{t}_auc" in hist.columns]
    if auc_cols:
        plt.figure(figsize=(7.5, 5.5))
        for col in auc_cols:
            plt.plot(hist["epoch"], hist[col], label=col.replace("_auc", "").upper())
        plt.xlabel("Epoch")
        plt.ylabel("Validation AUROC")
        plt.title("Validation Multi-Task AUROC")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(OUT / "validation_multitask_auroc.png", dpi=600, bbox_inches="tight")
        plt.close()


def gradcam_examples(model, test_ds, threshold: float, n_examples: int = 4) -> None:
    """
    ViT token Grad-CAM for correctly classified EVENT test cases.
    The model's learned stain and patch attention identifies the pathology patch
    on which the spatial activation map is displayed.
    """
    model.eval()
    candidates = []

    for idx in range(len(test_ds)):
        sample = test_ds[idx]
        x = sample["images"].unsqueeze(0).to(device)
        sm = sample["stain_mask"].unsqueeze(0).to(device)
        tab = sample["tabular"].unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x, sm, tab)
            prob = float(torch.sigmoid(out["event"])[0].cpu())
        truth = int(sample["event"].item())
        pred = int(prob >= threshold)
        if pred == truth:
            candidates.append(
                {
                    "idx": idx,
                    "truth": truth,
                    "prob": prob,
                    "confidence": prob if truth == 1 else 1 - prob,
                }
            )

    selected = []
    for cls in (0, 1):
        part = sorted(
            [c for c in candidates if c["truth"] == cls],
            key=lambda z: z["confidence"],
            reverse=True,
        )
        selected.extend(part[: max(1, n_examples // 2)])
    selected = selected[:n_examples]

    cfg = __import__("timm").data.resolve_model_data_config(model.enc)
    mean = torch.tensor(cfg["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["std"]).view(3, 1, 1)

    cache = {}

    def hook(_, __, output):
        cache["activation"] = output
        if output.requires_grad:
            output.retain_grad()

    handle = model.enc.blocks[-1].norm1.register_forward_hook(hook)
    results = []

    for info in selected:
        sample = test_ds[info["idx"]]
        x = sample["images"].unsqueeze(0).to(device)
        sm = sample["stain_mask"].unsqueeze(0).to(device)
        tab = sample["tabular"].unsqueeze(0).to(device)
        x.requires_grad_(True)

        cache.clear()
        model.zero_grad(set_to_none=True)
        out = model(x, sm, tab)
        truth = int(sample["event"].item())
        target = out["event"][0] if truth == 1 else -out["event"][0]
        target.backward()

        act = cache["activation"][:, 1:, :]
        grad = cache["activation"].grad[:, 1:, :]
        side = int(np.sqrt(act.shape[1]))
        act = act.reshape(act.shape[0], side, side, act.shape[-1])
        grad = grad.reshape(grad.shape[0], side, side, grad.shape[-1])
        weights = grad.mean(dim=(1, 2), keepdim=True)
        cams = F.relu((act * weights).sum(dim=-1))

        stain_idx = int(out["stain_weights"][0].argmax().detach().cpu())
        patch_idx = int(out["patch_weights"][0, stain_idx].argmax().detach().cpu())
        flat_idx = stain_idx * PATCHES_PER_STAIN + patch_idx

        cam = F.interpolate(
            cams[flat_idx][None, None],
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        image = sample["images"][stain_idx, patch_idx].detach().cpu()
        image = (image * std + mean).permute(1, 2, 0).numpy().clip(0, 1)

        results.append(
            {
                "image": image,
                "cam": cam.detach().cpu().numpy(),
                "patient": test_ds.ids[info["idx"]],
                "truth": truth,
                "prob": info["prob"],
                "stain": "H&E" if STAINS[stain_idx] == "HE" else STAINS[stain_idx],
            }
        )

    handle.remove()
    if not results:
        return

    fig, axes = plt.subplots(len(results), 3, figsize=(10, 3.0 * len(results)))
    if len(results) == 1:
        axes = np.expand_dims(axes, 0)

    for r, result in enumerate(results):
        axes[r, 0].imshow(result["image"])
        axes[r, 0].set_title(f"Original — {result['stain']}")
        axes[r, 1].imshow(result["cam"])
        axes[r, 1].set_title("Grad-CAM")
        axes[r, 2].imshow(result["image"])
        axes[r, 2].imshow(result["cam"], alpha=0.45)
        axes[r, 2].set_title(f"GT={result['truth']} | P(Event=1)={result['prob']:.3f}")
        for c in range(3):
            axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(OUT / "test_gradcam_examples.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(
        [
            {
                "Patient_ID": r["patient"],
                "Ground_Truth_Event": r["truth"],
                "Predicted_Probability": r["prob"],
                "Selected_Stain": r["stain"],
            }
            for r in results
        ]
    ).to_csv(OUT / "gradcam_selected_test_samples.csv", index=False)


def main() -> None:
    model, train_ds, val_ds, test_ds = build_datasets_and_model()
    checkpoint = RESULT_ROOT / "best_fairlora_multitask.pt"
    if not checkpoint.exists():
        raise FileNotFoundError("Final model checkpoint not found. Run methodology.py first.")

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    loaders = {
        "Train": DataLoader(train_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True),
        "Validation": DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True),
        "Test": DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True),
    }
    thresholds = load_thresholds()

    raw = {}
    pred_for_metrics = {}
    survival_rows = []
    fairness_groups, fairness_gaps = [], []

    for split, loader in loaders.items():
        y, p, risk, time, event = collect_outputs(model, loader)
        raw[split] = (y, p, risk, time, event)
        pred_for_metrics[split] = (y, p)
        survival_rows.append(
            {
                "Split": split,
                "N": len(event),
                "C_index": concordance_index(time, event, risk),
            }
        )

    metrics = evaluate_classification(pred_for_metrics, thresholds)
    pd.DataFrame(survival_rows).to_csv(OUT / "survival_cindex.csv", index=False)

    # Age-based fairness is summarized across train, validation, and test;
    # classification figures are produced for validation and held-out test.
    for split in ("Train", "Validation", "Test"):
        y, p, risk, time, event = raw[split]
        groups, gaps = evaluate_fairness(
            loaders[split],
            y["event"],
            p["event"],
            risk,
            time,
            thresholds["event"],
            split,
        )
        fairness_groups.append(groups)
        fairness_gaps.append(gaps)

        if split in ("Validation", "Test"):
            plot_confusion_matrices(y, p, thresholds, split)
            plot_roc_pr(y, p, split)

    pd.concat(fairness_groups, ignore_index=True).to_csv(
        OUT / "age_subgroup_metrics.csv", index=False
    )
    pd.concat(fairness_gaps, ignore_index=True).to_csv(
        OUT / "age_fairness_gaps.csv", index=False
    )

    efficiency = parameter_efficiency(model)
    plot_training_history()
    gradcam_examples(model, test_ds, thresholds["event"])

    print("\nCLASSIFICATION METRICS")
    print(metrics.round(3).to_string(index=False))
    print("\nSURVIVAL")
    print(pd.DataFrame(survival_rows).round(3).to_string(index=False))
    print("\nPARAMETER EFFICIENCY")
    print(efficiency.round(3).to_string(index=False))
    print(f"\nSaved evaluation outputs to: {OUT}")


if __name__ == "__main__":
    main()
