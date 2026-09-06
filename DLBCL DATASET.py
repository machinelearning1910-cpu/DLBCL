

from __future__ import annotations

import hashlib
import json
import shutil
import time
import warnings
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/content/drive/MyDrive/DLBCL_Morph_Project")
DATA_ROOT = PROJECT_ROOT / "01_data"
RAW_ROOT = DATA_ROOT / "raw"
FIGSHARE_ROOT = RAW_ROOT / "figshare"
AUDIT_ROOT = PROJECT_ROOT / "03_audit"
FIGURE_ROOT = PROJECT_ROOT / "04_results" / "data_visualization"

ARTICLE_ID = 12964772
API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"

EXPECTED_STAINS = ["HE", "CD10", "BCL6", "MUM1", "BCL2", "MYC"]

for p in (DATA_ROOT, RAW_ROOT, FIGSHARE_ROOT, AUDIT_ROOT, FIGURE_ROOT):
    p.mkdir(parents=True, exist_ok=True)


def mount_drive_if_colab() -> None:
    """Mount Google Drive when this script is executed in Google Colab."""
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception:
        pass


def md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, output_path: Path, expected_size: int | None) -> None:
    """Resumable download with final size verification."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = output_path.stat().st_size if output_path.exists() else 0

    if expected_size is not None and existing == expected_size:
        print(f"✓ Already complete: {output_path.name}")
        return

    headers = {"Range": f"bytes={existing}-"} if existing else {}
    response = requests.get(url, headers=headers, stream=True, timeout=(30, 300))

    if existing and response.status_code == 206:
        mode, initial = "ab", existing
    elif response.status_code == 200:
        mode, initial = "wb", 0
    else:
        response.raise_for_status()
        mode, initial = "wb", 0

    with output_path.open(mode) as f, tqdm(
        total=expected_size,
        initial=initial,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=output_path.name,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    if expected_size is not None and output_path.stat().st_size != expected_size:
        raise RuntimeError(f"Size mismatch after downloading {output_path.name}")


def download_official_release() -> Path:
    """Download the official Figshare release and verify size/MD5."""
    usage = shutil.disk_usage("/content/drive")
    print(
        f"Google Drive: total={usage.total/1024**3:.2f} GB | "
        f"used={usage.used/1024**3:.2f} GB | free={usage.free/1024**3:.2f} GB"
    )

    article = requests.get(API_URL, timeout=60)
    article.raise_for_status()
    manifest = article.json()
    files = manifest.get("files", [])
    if not files:
        raise RuntimeError("Figshare API returned no downloadable files.")

    (AUDIT_ROOT / "figshare_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    for i, info in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {info['name']}")
        output = FIGSHARE_ROOT / info["name"]
        error = None
        for attempt in range(1, 6):
            try:
                download_file(info["download_url"], output, info.get("size"))
                error = None
                break
            except Exception as exc:
                error = exc
                print(f"Attempt {attempt}/5 failed: {exc}")
                if attempt < 5:
                    time.sleep(10)
        if error is not None:
            raise RuntimeError(f"Failed to download {info['name']}: {error}")

    verification = []
    for info in files:
        path = FIGSHARE_ROOT / info["name"]
        expected_size = info.get("size")
        expected_md5 = info.get("supplied_md5") or info.get("computed_md5")
        actual_size = path.stat().st_size if path.exists() else None
        size_ok = actual_size == expected_size if expected_size is not None else path.exists()
        actual_md5 = md5sum(path) if path.exists() and expected_md5 else None
        md5_ok = (
            actual_md5.lower() == expected_md5.lower()
            if actual_md5 is not None and expected_md5 is not None
            else None
        )
        verification.append(
            {
                "file": info["name"],
                "exists": path.exists(),
                "expected_size": expected_size,
                "actual_size": actual_size,
                "size_ok": size_ok,
                "expected_md5": expected_md5,
                "actual_md5": actual_md5,
                "md5_ok": md5_ok,
            }
        )
        print(f"{'✓' if size_ok else '✗'} {info['name']} | size_ok={size_ok} | md5_ok={md5_ok}")

    (AUDIT_ROOT / "download_verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )

    zip_candidates = sorted(FIGSHARE_ROOT.glob("*.zip"))
    preferred = FIGSHARE_ROOT / "DLBCL-Morph.zip"
    if preferred.exists():
        return preferred
    if len(zip_candidates) == 1:
        return zip_candidates[0]
    raise FileNotFoundError("Could not uniquely identify the downloaded DLBCL-Morph ZIP.")


def parse_patch_path(name: str) -> tuple[str, str] | None:
    """Return (stain, patient_id) from .../Patches/<stain>/<patient>/<image>."""
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


def find_clinical_table(zf: zipfile.ZipFile) -> tuple[pd.DataFrame, str]:
    """Locate the clinical CSV by its patient/outcome columns."""
    candidates = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    required = {"Age", "OS", "Follow-up Status"}
    best = None
    for name in candidates:
        try:
            with zf.open(name) as f:
                df = pd.read_csv(f)
        except Exception:
            continue
        score = len(required.intersection(df.columns))
        if score >= 2 and (best is None or score > best[0]):
            best = (score, df, name)
    if best is None:
        raise RuntimeError("Clinical table could not be identified inside the ZIP.")
    return best[1], best[2]


def audit_dataset(zip_path: Path) -> tuple[pd.DataFrame, dict[str, dict[str, list[str]]], pd.DataFrame]:
    """Audit pathology patches, stain coverage, clinical labels, and patient matching."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        patch_files = [
            n for n in names
            if "/patches/" in n.lower()
            and n.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not patch_files:
            raise RuntimeError("No pathology patch images were found.")

        stain_patient_files: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for name in patch_files:
            parsed = parse_patch_path(name)
            if parsed is None:
                continue
            stain, patient = parsed
            stain_patient_files[stain][patient].append(name)

        stain_rows = []
        for stain in sorted(stain_patient_files):
            counts = [len(v) for v in stain_patient_files[stain].values()]
            stain_rows.append(
                {
                    "Stain": stain,
                    "Patients": len(counts),
                    "Patches": int(sum(counts)),
                    "Mean_patches_per_patient": float(np.mean(counts)),
                    "Median_patches_per_patient": float(np.median(counts)),
                    "Min_patches": int(np.min(counts)),
                    "Max_patches": int(np.max(counts)),
                }
            )
        stain_df = pd.DataFrame(stain_rows)
        stain_df.to_csv(AUDIT_ROOT / "stain_level_summary.csv", index=False)

        all_patients = sorted(
            set().union(*(set(d.keys()) for d in stain_patient_files.values()))
        )
        rows = []
        for pid in all_patients:
            row = {"Patient_ID": str(pid)}
            total = available = 0
            for stain in EXPECTED_STAINS:
                n = len(stain_patient_files.get(stain, {}).get(pid, []))
                row[f"{stain}_patches"] = n
                total += n
                available += int(n > 0)
            row["Available_stains"] = available
            row["Total_patches"] = total
            rows.append(row)
        patient_df = pd.DataFrame(rows)
        patient_df.to_csv(AUDIT_ROOT / "patient_stain_patch_counts.csv", index=False)

        clinical_df, clinical_source = find_clinical_table(zf)
        clinical_df.to_csv(AUDIT_ROOT / "clinical_table_from_zip.csv", index=False)

        labels = [
            "Age", "ECOG PS", "LDH", "EN", "Stage", "OS", "PFS",
            "Follow-up Status", "HANS", "MYC FISH", "BCL2 FISH", "BCL6 FISH",
        ]
        completeness = [
            {
                "Label": c,
                "Available": int(clinical_df[c].notna().sum()),
                "Missing": int(clinical_df[c].isna().sum()),
                "Availability_percent": float(100 * clinical_df[c].notna().mean()),
            }
            for c in labels if c in clinical_df.columns
        ]
        pd.DataFrame(completeness).to_csv(
            AUDIT_ROOT / "clinical_label_completeness.csv", index=False
        )

        summary = {
            "zip_size_gb": zip_path.stat().st_size / 1024**3,
            "zip_entries": len(names),
            "total_patch_images": len(patch_files),
            "patients_with_any_patch": len(patient_df),
            "patients_with_all_6_stains": int(
                (patient_df[[f"{s}_patches" for s in EXPECTED_STAINS]] > 0).all(axis=1).sum()
            ),
            "stains_detected": sorted(stain_patient_files.keys()),
            "clinical_table": clinical_source,
            "clinical_rows": int(len(clinical_df)),
        }
        (AUDIT_ROOT / "stage1_dataset_audit_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    print("\nSTAIN SUMMARY")
    print(stain_df.to_string(index=False))
    print("\nPATIENT COVERAGE")
    print(patient_df[["Available_stains", "Total_patches"]].describe().round(2))
    print(f"\nClinical patients: {len(clinical_df)}")
    return patient_df, stain_patient_files, clinical_df


def plot_representative_multistain_sample(
    zip_path: Path,
    patient_df: pd.DataFrame,
    stain_patient_files: dict[str, dict[str, list[str]]],
) -> Path:
    """Create the six-stain representative pathology sample used for dataset inspection."""
    complete = patient_df[
        (patient_df[[f"{s}_patches" for s in EXPECTED_STAINS]] > 0).all(axis=1)
    ]
    pool = complete if len(complete) else patient_df.sort_values(
        ["Available_stains", "Total_patches"], ascending=False
    )
    patient = str(pool.sort_values("Total_patches", ascending=False).iloc[0]["Patient_ID"])

    fig, axes = plt.subplots(1, 6, figsize=(15, 3.3))
    with zipfile.ZipFile(zip_path, "r") as zf:
        for ax, stain in zip(axes, EXPECTED_STAINS):
            files = sorted(stain_patient_files.get(stain, {}).get(patient, []))
            if files:
                img = Image.open(BytesIO(zf.read(files[len(files) // 2]))).convert("RGB")
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Unavailable", ha="center", va="center")
            ax.set_title("H&E" if stain == "HE" else stain)
            ax.axis("off")

    fig.suptitle(f"Representative DLBCL-Morph Multi-Stain Patient — {patient}", fontsize=12)
    plt.tight_layout()
    out = FIGURE_ROOT / "representative_six_stain_patient.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Representative sample saved: {out}")
    return out


def main() -> None:
    mount_drive_if_colab()
    zip_path = download_official_release()
    patient_df, stain_patient_files, _ = audit_dataset(zip_path)
    plot_representative_multistain_sample(zip_path, patient_df, stain_patient_files)
    print("\n✓ Dataset download, integrity audit, clinical audit, and sample visualization complete.")


if __name__ == "__main__":
    main()
