

from __future__ import annotations

import json
import math
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import BatchSampler, DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path("/content/drive/MyDrive/DLBCL_Morph_Project")
ZIP_PATH = PROJECT_ROOT / "01_data/raw/figshare/DLBCL-Morph.zip"
AUDIT_ROOT = PROJECT_ROOT / "03_audit"
SPLIT_ROOT = PROJECT_ROOT / "01_data/paper_splits"
FEATURE_ROOT = PROJECT_ROOT / "01_data/paper_features"
RESULT_ROOT = PROJECT_ROOT / "04_results"
CACHE_ROOT = Path("/content/DLBCL_paper_cache")
MANIFEST_PATH = FEATURE_ROOT / "complete_image_manifest.csv"

STAINS = ["HE", "CD10", "BCL6", "MUM1", "BCL2", "MYC"]
TASKS = ["event", "hans", "myc", "bcl2", "bcl6"]
LABEL_COLUMNS = {
    "event": "Follow-up Status",
    "hans": "HANS",
    "myc": "MYC FISH",
    "bcl2": "BCL2 FISH",
    "bcl6": "BCL6 FISH",
}
CLINICAL_INPUTS = ["ECOG PS", "LDH", "EN", "Stage"]
SHAPE_FEATURES = [
    "shortAxis", "longAxis", "ellip_area", "hull_area",
    "minDiameter", "maxDiameter", "esf", "csf",
    "sf1", "sf2", "elogation", "convexity",
]
PATCHES_PER_STAIN = 4
SEED = 42

for p in (SPLIT_ROOT, FEATURE_ROOT, RESULT_ROOT, CACHE_ROOT):
    p.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_patient_features() -> pd.DataFrame:
    """Aggregate cell morphology to patient level and merge structured clinical variables."""
    clinical_path = AUDIT_ROOT / "clinical_table_from_zip.csv"
    if not clinical_path.exists():
        raise FileNotFoundError("Run dataset_audit.py first.")

    with zipfile.ZipFile(ZIP_PATH) as zf:
        cell_shape_files = [n for n in zf.namelist() if n.lower().endswith("cell_shapes.csv")]
        if not cell_shape_files:
            raise FileNotFoundError("cell_shapes.csv was not found in the DLBCL-Morph ZIP.")
        with zf.open(cell_shape_files[0]) as f:
            cells = pd.read_csv(f, usecols=["patient_id"] + SHAPE_FEATURES)

    cells["patient_id"] = cells["patient_id"].astype(str)
    agg = cells.groupby("patient_id")[SHAPE_FEATURES].agg(["mean", "std", "median"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["cell_count"] = cells.groupby("patient_id").size()
    agg = agg.reset_index()

    clinical = pd.read_csv(clinical_path)
    clinical["patient_id"] = clinical["patient_id"].astype(str)
    keep = [
        "patient_id", "Age", *CLINICAL_INPUTS, "OS", "PFS",
        "Follow-up Status", "HANS", "MYC FISH", "BCL2 FISH", "BCL6 FISH",
    ]
    missing = [c for c in keep if c not in clinical.columns]
    if missing:
        raise KeyError(f"Required clinical columns missing: {missing}")

    features = clinical[keep].merge(agg, on="patient_id", how="inner")
    features.to_csv(FEATURE_ROOT / "patient_features.csv", index=False)
    print(f"Patient-level feature table: {len(features)} patients, {features.shape[1]} columns")
    return features


def build_final_cohort(features: pd.DataFrame) -> pd.DataFrame:
    """Restrict the study to patients with both pathology and patient-level structured data."""
    image_audit = pd.read_csv(AUDIT_ROOT / "patient_stain_patch_counts.csv")
    image_audit["Patient_ID"] = image_audit["Patient_ID"].astype(str)
    features = features.copy()
    features["patient_id"] = features["patient_id"].astype(str)

    cohort = image_audit[["Patient_ID"]].merge(
        features, left_on="Patient_ID", right_on="patient_id", how="inner"
    )
    if len(cohort) != 170:
        raise RuntimeError(
            f"Paper cohort requires 170 aligned patients; found {len(cohort)}. "
            "Check the downloaded release and audit outputs."
        )
    cohort.to_csv(FEATURE_ROOT / "paper_cohort_170.csv", index=False)
    return cohort


def _multilabel_targets(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event": df["Follow-up Status"].astype(int),
            "age60": (df["Age"] >= 60).astype(int),
            "hans_pos": (df["HANS"] == 1).astype(int),
            "hans_obs": df["HANS"].notna().astype(int),
            "myc_pos": (df["MYC FISH"] == 1).astype(int),
            "myc_obs": df["MYC FISH"].notna().astype(int),
            "bcl2_pos": (df["BCL2 FISH"] == 1).astype(int),
            "bcl2_obs": df["BCL2 FISH"].notna().astype(int),
            "bcl6_pos": (df["BCL6 FISH"] == 1).astype(int),
            "bcl6_obs": df["BCL6 FISH"].notna().astype(int),
        }
    )


def create_patient_splits(cohort: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create 117/26/27 independent patient partitions with multi-label stratification."""
    y = _multilabel_targets(cohort)

    first = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=53, random_state=SEED
    )
    train_idx, temp_idx = next(first.split(cohort, y))

    temp = cohort.iloc[temp_idx].reset_index(drop=True)
    y_temp = y.iloc[temp_idx].reset_index(drop=True)

    second = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=27, random_state=SEED
    )
    val_rel, test_rel = next(second.split(temp, y_temp))

    splits = {
        "train": cohort.iloc[train_idx].reset_index(drop=True),
        "val": temp.iloc[val_rel].reset_index(drop=True),
        "test": temp.iloc[test_rel].reset_index(drop=True),
    }

    expected = {"train": 117, "val": 26, "test": 27}
    for name, df in splits.items():
        if len(df) != expected[name]:
            raise RuntimeError(f"{name} split has {len(df)} patients; expected {expected[name]}.")
        df.to_csv(SPLIT_ROOT / f"{name}.csv", index=False)

    ids = {k: set(v["Patient_ID"].astype(str)) for k, v in splits.items()}
    if ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"]:
        raise RuntimeError("Patient leakage detected across partitions.")

    print("Patient-independent split sizes:", {k: len(v) for k, v in splits.items()})
    return splits


def prepare_structured_features(splits: dict[str, pd.DataFrame]) -> list[str]:
    """
    Training-only median imputation and StandardScaler.
    Age is deliberately excluded from the predictive feature vector.
    """
    target_cols = [
        "Patient_ID", "patient_id", "Age", "OS", "PFS",
        "Follow-up Status", "HANS", "MYC FISH", "BCL2 FISH", "BCL6 FISH",
    ]
    morph = [c for c in splits["train"].columns if c not in target_cols + CLINICAL_INPUTS]
    input_cols = CLINICAL_INPUTS + morph

    train_medians = splits["train"][input_cols].median()
    scaler = StandardScaler()

    for name in splits:
        splits[name] = splits[name].copy()
        splits[name][input_cols] = splits[name][input_cols].fillna(train_medians)

    scaler.fit(splits["train"][input_cols])

    for name, df in splits.items():
        df[input_cols] = scaler.transform(df[input_cols])
        df.to_csv(FEATURE_ROOT / f"{name}_fusion.csv", index=False)

    state = {
        "feature_columns": input_cols,
        "train_medians": {k: float(v) for k, v in train_medians.items()},
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    (FEATURE_ROOT / "structured_preprocessing.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    print(f"Predictive structured inputs: {len(input_cols)}; Age excluded.")
    return input_cols


def _parse_patch_path(name: str) -> tuple[str, str] | None:
    parts = name.split("/")
    lower = [p.lower() for p in parts]
    if "patches" not in lower:
        return None
    i = lower.index("patches")
    if len(parts) <= i + 3:
        return None
    stain = parts[i + 1].strip().upper()
    stain = {"H&E": "HE", "H_E": "HE"}.get(stain, stain)
    return stain, parts[i + 2].strip()


def build_complete_image_cache() -> pd.DataFrame:
    """Extract all pathology patches belonging to the final 170-patient cohort."""
    final_ids = set(
        pd.read_csv(FEATURE_ROOT / "paper_cohort_170.csv")["Patient_ID"].astype(str)
    )
    rows = []

    with zipfile.ZipFile(ZIP_PATH) as zf:
        patch_names = [
            n for n in zf.namelist()
            if "/patches/" in n.lower()
            and n.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        for name in patch_names:
            parsed = _parse_patch_path(name)
            if parsed is None:
                continue
            stain, pid = parsed
            if pid not in final_ids or stain not in STAINS:
                continue

            dst = CACHE_ROOT / stain / pid / Path(name).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                with zf.open(name) as src, dst.open("wb") as out:
                    shutil.copyfileobj(src, out)

            rows.append(
                {
                    "patient_id": pid,
                    "stain": stain,
                    "local_path": str(dst),
                    "zip_path": name,
                }
            )

    manifest = pd.DataFrame(rows).drop_duplicates()
    manifest.to_csv(MANIFEST_PATH, index=False)

    counts = manifest.groupby("patient_id").size()
    missing = final_ids.difference(counts.index.astype(str))
    if missing:
        raise RuntimeError(f"Patients with no cached pathology patches: {sorted(missing)}")

    print(f"Complete pathology cache: {len(manifest):,} patches from {manifest.patient_id.nunique()} patients")
    return manifest


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Linear(base.in_features, r, bias=False)
        self.B = nn.Linear(r, base.out_features, bias=False)
        self.scale = alpha / r
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.B(self.A(x)) * self.scale


class PatchAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim // 2)
        self.score = nn.Linear(dim // 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B,S,K,D]
        score = self.score(torch.tanh(self.proj(x))).squeeze(-1)
        weight = torch.softmax(score.float(), dim=2).to(x.dtype)
        pooled = (x * weight.unsqueeze(-1)).sum(dim=2)
        return pooled, weight


class FairLoRAMultimodalDLBCL(nn.Module):
    """Paper-aligned ViT/FairLoRA + patch/stain attention + multimodal multi-task model."""

    def __init__(self, n_tabular: int):
        super().__init__()
        self.enc = timm.create_model(
            "vit_tiny_patch16_224", pretrained=True, num_classes=0
        )
        for p in self.enc.parameters():
            p.requires_grad = False

        for block in self.enc.blocks:
            block.attn.qkv = LoRALinear(block.attn.qkv, r=8, alpha=16)
            block.attn.proj = LoRALinear(block.attn.proj, r=8, alpha=16)

        d = self.enc.num_features
        self.patch_attention = PatchAttention(d)
        self.stain_attention = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.Tanh(),
            nn.Linear(d // 2, 1),
        )
        self.image_projection = nn.Sequential(
            nn.Linear(d, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.structured_projection = nn.Sequential(
            nn.Linear(n_tabular, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.30),
        )
        self.heads = nn.ModuleDict(
            {
                "event": nn.Linear(128, 1),
                "hans": nn.Linear(128, 1),
                "myc": nn.Linear(128, 1),
                "bcl2": nn.Linear(128, 1),
                "bcl6": nn.Linear(128, 1),
            }
        )
        self.survival_head = nn.Linear(128, 1)

    def forward(
        self,
        images: torch.Tensor,
        stain_mask: torch.Tensor,
        tabular: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, s, k, c, h, w = images.shape
        patch_features = self.enc(images.reshape(b * s * k, c, h, w))
        patch_features = patch_features.view(b, s, k, -1)

        stain_features, patch_weights = self.patch_attention(patch_features)

        stain_scores = self.stain_attention(stain_features).squeeze(-1).float()
        stain_scores = stain_scores.masked_fill(stain_mask <= 0, -1e4)
        stain_weights = torch.softmax(stain_scores, dim=1).to(stain_features.dtype)
        image_repr = (stain_features * stain_weights.unsqueeze(-1)).sum(dim=1)
        image_repr = self.image_projection(image_repr)

        structured_repr = self.structured_projection(tabular)
        fused = self.fusion(torch.cat([image_repr, structured_repr], dim=1))

        out = {name: head(fused).squeeze(1) for name, head in self.heads.items()}
        out["risk"] = self.survival_head(fused).squeeze(1)
        out["patch_weights"] = patch_weights
        out["stain_weights"] = stain_weights
        return out


def build_transforms(model: FairLoRAMultimodalDLBCL):
    cfg = timm.data.resolve_model_data_config(model.enc)
    size = cfg["input_size"][-1]
    mean, std = cfg["mean"], cfg["std"]

    train_tfm = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(10, fill=255),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_tfm = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tfm, eval_tfm


class FusionDataset(Dataset):
    def __init__(
        self,
        split: str,
        feature_columns: list[str],
        train_transform,
        eval_transform,
        manifest: pd.DataFrame | None = None,
    ):
        self.split = split
        self.feature_columns = feature_columns
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.clinical = pd.read_csv(FEATURE_ROOT / f"{split}_fusion.csv")
        self.clinical["Patient_ID"] = self.clinical["Patient_ID"].astype(str)
        self.ids = self.clinical["Patient_ID"].tolist()
        self.manifest = (
            pd.read_csv(MANIFEST_PATH) if manifest is None else manifest.copy()
        )
        self.manifest["patient_id"] = self.manifest["patient_id"].astype(str)

    def __len__(self) -> int:
        return len(self.ids)

    def _label(self, row: pd.Series, column: str) -> float:
        return float(row[column]) if pd.notna(row[column]) else -1.0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        pid = self.ids[index]
        row = self.clinical.iloc[index]
        images, stain_mask = [], []

        for stain in STAINS:
            part = self.manifest[
                (self.manifest["patient_id"] == pid)
                & (self.manifest["stain"] == stain)
            ].sort_values("local_path")

            if len(part):
                if self.split == "train":
                    chosen = part.sample(
                        PATCHES_PER_STAIN,
                        replace=len(part) < PATCHES_PER_STAIN,
                    )
                    tfm = self.train_transform
                else:
                    chosen = part.iloc[:PATCHES_PER_STAIN]
                    if len(chosen) < PATCHES_PER_STAIN:
                        repeat = int(math.ceil(PATCHES_PER_STAIN / len(chosen)))
                        chosen = pd.concat([chosen] * repeat).iloc[:PATCHES_PER_STAIN]
                    tfm = self.eval_transform

                stack = torch.stack(
                    [
                        tfm(Image.open(path).convert("RGB"))
                        for path in chosen["local_path"]
                    ]
                )
                images.append(stack)
                stain_mask.append(1.0)
            else:
                images.append(torch.zeros(PATCHES_PER_STAIN, 3, 224, 224))
                stain_mask.append(0.0)

        return {
            "images": torch.stack(images),
            "stain_mask": torch.tensor(stain_mask, dtype=torch.float32),
            "tabular": torch.tensor(
                row[self.feature_columns].values.astype("float32")
            ),
            "age": torch.tensor(float(row["Age"]), dtype=torch.float32),
            "os": torch.tensor(float(row["OS"]), dtype=torch.float32),
            "event": torch.tensor(float(row["Follow-up Status"]), dtype=torch.float32),
            "hans": torch.tensor(self._label(row, "HANS"), dtype=torch.float32),
            "myc": torch.tensor(self._label(row, "MYC FISH"), dtype=torch.float32),
            "bcl2": torch.tensor(self._label(row, "BCL2 FISH"), dtype=torch.float32),
            "bcl6": torch.tensor(self._label(row, "BCL6 FISH"), dtype=torch.float32),
        }


def positive_class_weight(dataset: FusionDataset, column: str) -> torch.Tensor:
    y = dataset.clinical[column].dropna().astype(int)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    return torch.tensor(neg / max(pos, 1), dtype=torch.float32, device=device)


def masked_weighted_bce(
    logit: torch.Tensor,
    target: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    valid = target >= 0
    if not valid.any():
        return logit.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        logit[valid], target[valid], pos_weight=pos_weight
    )


def survival_rank_loss(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    comparable = (event[:, None] == 1) & (time[:, None] < time[None, :])
    if not comparable.any():
        return risk.sum() * 0.0
    diff = risk[:, None] - risk[None, :]
    return F.softplus(-diff[comparable]).mean()


def concordance_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    concordant = comparable = 0.0
    for i in range(len(time)):
        for j in range(i + 1, len(time)):
            if event[i] == 1 and time[i] < time[j]:
                comparable += 1
                concordant += 1 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0
            elif event[j] == 1 and time[j] < time[i]:
                comparable += 1
                concordant += 1 if risk[j] > risk[i] else 0.5 if risk[j] == risk[i] else 0
    return concordant / comparable if comparable else float("nan")


@torch.no_grad()
def collect_outputs(
    model: FairLoRAMultimodalDLBCL,
    loader: DataLoader,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y = {task: [] for task in TASKS}
    p = {task: [] for task in TASKS}
    risks, times, events = [], [], []

    for batch in loader:
        out = model(
            batch["images"].to(device, non_blocking=True),
            batch["stain_mask"].to(device, non_blocking=True),
            batch["tabular"].to(device, non_blocking=True),
        )
        for task in TASKS:
            yy = batch[task].numpy()
            pp = torch.sigmoid(out[task]).float().cpu().numpy()
            valid = yy >= 0
            y[task].extend(yy[valid])
            p[task].extend(pp[valid])

        risks.extend(out["risk"].float().cpu().numpy())
        times.extend(batch["os"].numpy())
        events.extend(batch["event"].numpy())

    y = {k: np.asarray(v).astype(int) for k, v in y.items()}
    p = {k: np.asarray(v, dtype=float) for k, v in p.items()}
    return (
        y,
        p,
        np.asarray(risks, dtype=float),
        np.asarray(times, dtype=float),
        np.asarray(events, dtype=int),
    )


def validation_summary(
    model: FairLoRAMultimodalDLBCL,
    loader: DataLoader,
) -> dict[str, float]:
    y, p, risk, time, event = collect_outputs(model, loader)
    aucs = {}
    for task in TASKS:
        aucs[task] = (
            roc_auc_score(y[task], p[task])
            if len(np.unique(y[task])) == 2
            else float("nan")
        )
    macro_auc = float(np.nanmean(list(aucs.values())))
    cidx = concordance_index(time, event, risk)
    return {**{f"{k}_auc": v for k, v in aucs.items()}, "macro_auc": macro_auc, "cindex": cidx}


def train_multitask_model(
    model: FairLoRAMultimodalDLBCL,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_ds: FusionDataset,
) -> Path:
    pos_weights = {
        task: positive_class_weight(train_ds, LABEL_COLUMNS[task])
        for task in TASKS
    }

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=8e-5,
        weight_decay=2e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    epochs, patience = 30, 7
    aux_weight, survival_weight = 0.15, 0.40
    best, bad = -np.inf, 0
    history = []
    ckpt = RESULT_ROOT / "best_multitask_fairlora.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for batch in train_loader:
            images = batch["images"].to(device, non_blocking=True)
            stain_mask = batch["stain_mask"].to(device, non_blocking=True)
            tabular = batch["tabular"].to(device, non_blocking=True)
            os_time = batch["os"].to(device)
            event = batch["event"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(images, stain_mask, tabular)
                event_loss = masked_weighted_bce(out["event"], event, pos_weights["event"])

                aux_loss = sum(
                    masked_weighted_bce(
                        out[task],
                        batch[task].to(device),
                        pos_weights[task],
                    )
                    for task in ("hans", "myc", "bcl2", "bcl6")
                )
                surv_loss = survival_rank_loss(out["risk"], os_time, event)
                loss = event_loss + aux_weight * aux_loss + survival_weight * surv_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        val = validation_summary(model, val_loader)
        scheduler.step(val["macro_auc"])
        history.append({"epoch": epoch, "train_loss": np.mean(losses), **val})

        if val["macro_auc"] > best:
            best, bad = val["macro_auc"], 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1

        print(
            f"E{epoch:02d} | Loss {np.mean(losses):.3f} | "
            f"Macro-AUROC {val['macro_auc']:.3f} | C-index {val['cindex']:.3f}"
        )
        if bad >= patience:
            print("Early stopping.")
            break

    pd.DataFrame(history).to_csv(RESULT_ROOT / "multitask_training_history.csv", index=False)
    print(f"Best validation macro-AUROC: {best:.4f}")
    return ckpt


class AgeEventBatchSampler(BatchSampler):
    """Eight-patient batch: two from each Age (<60/>=60) x EVENT (0/1) subgroup."""

    def __init__(self, dataset: FusionDataset):
        df = dataset.clinical
        age = df["Age"].to_numpy()
        evt = df["Follow-up Status"].astype(int).to_numpy()
        self.groups = [
            np.where((age < 60) & (evt == 0))[0],
            np.where((age < 60) & (evt == 1))[0],
            np.where((age >= 60) & (evt == 0))[0],
            np.where((age >= 60) & (evt == 1))[0],
        ]
        if any(len(g) == 0 for g in self.groups):
            raise RuntimeError("At least one Age x Event subgroup is empty.")
        self.batches = int(math.ceil(len(dataset) / 8))

    def __iter__(self):
        for _ in range(self.batches):
            batch = []
            for group in self.groups:
                batch.extend(
                    np.random.choice(group, size=2, replace=len(group) < 2).tolist()
                )
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.batches


def equalized_odds_loss(logit: torch.Tensor, event: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logit)
    young, old = age < 60, age >= 60
    pos, neg = event == 1, event == 0
    tpr_gap = torch.abs(prob[young & pos].mean() - prob[old & pos].mean())
    fpr_gap = torch.abs(prob[young & neg].mean() - prob[old & neg].mean())
    return 0.5 * (tpr_gap + fpr_gap)


@torch.no_grad()
def fairness_validation(
    model: FairLoRAMultimodalDLBCL,
    loader: DataLoader,
) -> dict[str, float]:
    y, p, risk, time, event = collect_outputs(model, loader)
    yy, pp = y["event"], p["event"]

    ages = []
    for batch in loader:
        ages.extend(batch["age"].numpy())
    ages = np.asarray(ages)

    grid = np.linspace(0.05, 0.95, 181)
    threshold = float(grid[np.argmax([f1_score(yy, pp >= t, zero_division=0) for t in grid])])
    pred = (pp >= threshold).astype(int)

    def subgroup(mask):
        yg, pg = yy[mask], pred[mask]
        tp = int(((yg == 1) & (pg == 1)).sum())
        fn = int(((yg == 1) & (pg == 0)).sum())
        fp = int(((yg == 0) & (pg == 1)).sum())
        tn = int(((yg == 0) & (pg == 0)).sum())
        tpr = tp / (tp + fn) if tp + fn else np.nan
        fpr = fp / (fp + tn) if fp + tn else np.nan
        return tpr, fpr, float(pg.mean())

    young, old = ages < 60, ages >= 60
    tpr_y, fpr_y, rate_y = subgroup(young)
    tpr_o, fpr_o, rate_o = subgroup(old)
    eo_gap = 0.5 * (abs(tpr_y - tpr_o) + abs(fpr_y - fpr_o))

    return {
        "event_auc": roc_auc_score(yy, pp),
        "cindex": concordance_index(time, event, risk),
        "f1": f1_score(yy, pred, zero_division=0),
        "DPD": abs(rate_y - rate_o),
        "EOD": abs(tpr_y - tpr_o),
        "FPR_gap": abs(fpr_y - fpr_o),
        "EO_gap": eo_gap,
        "threshold": threshold,
    }


def fairness_finetune(
    model: FairLoRAMultimodalDLBCL,
    train_ds: FusionDataset,
    val_loader: DataLoader,
    source_checkpoint: Path,
) -> Path:
    model.load_state_dict(torch.load(source_checkpoint, map_location=device, weights_only=True))

    fair_loader = DataLoader(
        train_ds,
        batch_sampler=AgeEventBatchSampler(train_ds),
        num_workers=2,
        pin_memory=True,
    )

    lora_params = [
        p for name, p in model.named_parameters()
        if name.startswith("enc.") and p.requires_grad
    ]
    downstream = [
        p for name, p in model.named_parameters()
        if not name.startswith("enc.") and p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": 1e-5},
            {"params": downstream, "lr": 5e-5},
        ],
        weight_decay=2e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best, bad = -np.inf, 0
    epochs, patience = 12, 4
    history = []
    ckpt = RESULT_ROOT / "best_fairlora_multitask.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for batch in fair_loader:
            images = batch["images"].to(device)
            stain_mask = batch["stain_mask"].to(device)
            tabular = batch["tabular"].to(device)
            event = batch["event"].to(device)
            age = batch["age"].to(device)
            os_time = batch["os"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(images, stain_mask, tabular)
                cls = F.binary_cross_entropy_with_logits(out["event"], event)
                surv = survival_rank_loss(out["risk"], os_time, event)
                fair = equalized_odds_loss(out["event"], event, age)
                loss = cls + 0.20 * surv + 0.15 * fair

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        val = fairness_validation(model, val_loader)
        performance = 0.45 * val["event_auc"] + 0.35 * val["cindex"] + 0.20 * val["f1"]
        selection_score = performance - 0.15 * val["EO_gap"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": np.mean(losses),
                **val,
                "selection_score": selection_score,
            }
        )

        if selection_score > best:
            best, bad = selection_score, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1

        print(
            f"Fair E{epoch:02d} | Loss {np.mean(losses):.3f} | "
            f"AUC {val['event_auc']:.3f} | C {val['cindex']:.3f} | "
            f"EO-gap {val['EO_gap']:.3f}"
        )
        if bad >= patience:
            print("Fairness fine-tuning early stopping.")
            break

    pd.DataFrame(history).to_csv(RESULT_ROOT / "fairness_training_history.csv", index=False)
    return ckpt


def select_validation_thresholds(
    model: FairLoRAMultimodalDLBCL,
    val_loader: DataLoader,
) -> dict[str, float]:
    y, p, _, _, _ = collect_outputs(model, val_loader)
    grid = np.linspace(0.05, 0.95, 181)
    thresholds = {}
    for task in TASKS:
        scores = [f1_score(y[task], p[task] >= t, zero_division=0) for t in grid]
        thresholds[task] = float(grid[int(np.argmax(scores))])
    (RESULT_ROOT / "validation_thresholds.json").write_text(
        json.dumps(thresholds, indent=2), encoding="utf-8"
    )
    return thresholds


def load_feature_columns() -> list[str]:
    state = json.loads((FEATURE_ROOT / "structured_preprocessing.json").read_text())
    return list(state["feature_columns"])


def build_datasets_and_model():
    feature_columns = load_feature_columns()
    model = FairLoRAMultimodalDLBCL(len(feature_columns)).to(device)
    train_tfm, eval_tfm = build_transforms(model)
    manifest = pd.read_csv(MANIFEST_PATH)

    train_ds = FusionDataset("train", feature_columns, train_tfm, eval_tfm, manifest)
    val_ds = FusionDataset("val", feature_columns, train_tfm, eval_tfm, manifest)
    test_ds = FusionDataset("test", feature_columns, train_tfm, eval_tfm, manifest)
    return model, train_ds, val_ds, test_ds


def main() -> None:
    seed_everything()

    features = build_patient_features()
    cohort = build_final_cohort(features)
    splits = create_patient_splits(cohort)
    prepare_structured_features(splits)
    build_complete_image_cache()

    model, train_ds, val_ds, _ = build_datasets_and_model()

    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        num_workers=2, pin_memory=True
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Trainable parameters: {trainable:,}/{total:,} ({100*trainable/total:.2f}%)")

    base_ckpt = train_multitask_model(model, train_loader, val_loader, train_ds)
    final_ckpt = fairness_finetune(model, train_ds, val_loader, base_ckpt)

    model.load_state_dict(torch.load(final_ckpt, map_location=device, weights_only=True))
    thresholds = select_validation_thresholds(model, val_loader)
    print("Validation-selected thresholds:", thresholds)
    print(f"Final model: {final_ckpt}")


if __name__ == "__main__":
    main()
