import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


TASK_SPECS = {
    "lfmc": {
        "source_legible": ["nfmd"],
        "pred_key": "lfmc_preds",
        "true_key": "lfmc_true",
    },
    "vv": {
        "source_legible": ["vv"],
        "pred_key": "vv_preds",
        "true_key": "vv_true",
    },
    "vh": {
        "source_legible": ["vh", "vh_backscatter"],
        "pred_key": "vh_preds",
        "true_key": "vh_true",
    },
}


def _expand_member_dirs(member_dirs: List[str], member_globs: List[str]) -> List[str]:
    out = []
    for p in member_dirs:
        if os.path.isdir(p):
            out.append(os.path.abspath(p))
    for pat in member_globs:
        for p in sorted(glob.glob(pat)):
            if os.path.isdir(p):
                out.append(os.path.abspath(p))
    out = sorted(set(out))
    if not out:
        raise ValueError("No ensemble member directories found")
    return out


def _member_is_complete(member_dir: str) -> bool:
    return os.path.isdir(os.path.join(member_dir, "fold_9998"))


def _filter_completed_members(member_dirs: List[str], skip_incomplete: bool) -> Tuple[List[str], List[str]]:
    if not skip_incomplete:
        return member_dirs, []
    completed = []
    skipped = []
    for member_dir in member_dirs:
        if _member_is_complete(member_dir):
            completed.append(member_dir)
        else:
            skipped.append(member_dir)
    return completed, skipped


def _list_folds(member_dir: str, include_final_fold: bool) -> List[int]:
    folds = []
    for p in sorted(glob.glob(os.path.join(member_dir, "fold_*"))):
        name = os.path.basename(p)
        try:
            fold = int(name.split("_")[1])
        except Exception:
            continue
        if (not include_final_fold) and fold == 9998:
            continue
        folds.append(fold)
    return sorted(folds)


def _safe_array(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 0:
        if np.isnan(arr).all():
            return np.array([], dtype=float)
        return arr.reshape(1)
    return arr


def _key_cols_for_info(info_df: pd.DataFrame) -> List[str]:
    preferred = [
        "date",
        "latitude",
        "longitude",
        "source_legible",
        "site_name",
        "fuel_type",
        "fuel",
        "source",
    ]
    return [c for c in preferred if c in info_df.columns]


def _load_fold_task_frame(member_dir: str, fold: int, split: str, task: str) -> pd.DataFrame:
    spec = TASK_SPECS[task]
    out_path = os.path.join(member_dir, f"fold_{fold}", f"{split}_outputs.pth")
    info_path = os.path.join(member_dir, f"fold_{fold}", f"{split}_info.csv")
    out = torch.load(out_path, weights_only=False)
    info = pd.read_csv(info_path)
    source_labels = spec["source_legible"]
    info_task = info[info["source_legible"].isin(source_labels)].copy().reset_index(drop=True)

    preds = _safe_array(out[spec["pred_key"]])
    truth = _safe_array(out[spec["true_key"]])
    if preds.size == 0 and truth.size == 0:
        return pd.DataFrame()
    if preds.size != truth.size:
        raise ValueError(
            f"{member_dir} fold_{fold} {split} {task}: pred/true length mismatch "
            f"({preds.size} vs {truth.size})"
        )
    if len(info_task) != preds.size:
        raise ValueError(
            f"{member_dir} fold_{fold} {split} {task}: info length mismatch "
            f"({len(info_task)} vs {preds.size})"
        )

    key_cols = _key_cols_for_info(info_task)
    frame = info_task[key_cols].copy() if key_cols else pd.DataFrame(index=np.arange(preds.size))
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["fold"] = int(fold)
    frame["split"] = split
    frame["task"] = task
    frame["row_in_fold_task"] = np.arange(preds.size, dtype=np.int64)
    frame["true"] = truth.astype(float)
    frame["pred"] = preds.astype(float)
    return frame


def _r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - (ss_res / ss_tot)


def _aggregate_task(
    member_dirs: List[str],
    split: str,
    task: str,
    folds: List[int],
) -> pd.DataFrame:
    templates: Dict[int, pd.DataFrame] = {}
    pred_stacks: Dict[int, List[np.ndarray]] = {}

    for member_idx, member_dir in enumerate(member_dirs):
        for fold in folds:
            frame = _load_fold_task_frame(member_dir, fold, split, task)
            if frame.empty:
                continue
            key_cols = [c for c in frame.columns if c not in ["pred"]]
            if fold not in templates:
                templates[fold] = frame.drop(columns=["pred"]).copy()
                pred_stacks[fold] = [frame["pred"].to_numpy(dtype=float)]
                continue

            tmpl = templates[fold]
            if len(tmpl) != len(frame):
                raise ValueError(
                    f"Mismatch for task={task}, split={split}, fold={fold}: "
                    f"{member_dir} has {len(frame)} rows, expected {len(tmpl)}"
                )
            for col in key_cols:
                if col not in tmpl.columns:
                    raise ValueError(f"Template missing column {col} for fold {fold}")
                left = tmpl[col]
                right = frame[col]
                if np.issubdtype(np.asarray(left).dtype, np.number):
                    if not np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), equal_nan=True):
                        raise ValueError(
                            f"Row alignment mismatch in numeric column '{col}' "
                            f"for task={task}, split={split}, fold={fold}, member={member_dir}"
                        )
                else:
                    if not left.fillna("__nan__").astype(str).equals(right.fillna("__nan__").astype(str)):
                        raise ValueError(
                            f"Row alignment mismatch in column '{col}' "
                            f"for task={task}, split={split}, fold={fold}, member={member_dir}"
                        )
            pred_stacks[fold].append(frame["pred"].to_numpy(dtype=float))

    outputs = []
    for fold in sorted(templates):
        pred_stack = np.stack(pred_stacks[fold], axis=1)  # (N, M)
        out = templates[fold].copy()
        out["ensemble_n"] = pred_stack.shape[1]
        out["pred_mean"] = pred_stack.mean(axis=1)
        out["pred_std"] = pred_stack.std(axis=1, ddof=0)
        outputs.append(out)

    if not outputs:
        return pd.DataFrame()
    return pd.concat(outputs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate cross-validation predictions from multiple seed runs into "
            "ensemble mean/std uncertainty (spread across ensemble members)."
        )
    )
    parser.add_argument("--member-dir", action="append", default=[], help="Ensemble member model directory (repeatable)")
    parser.add_argument("--member-glob", action="append", default=[], help="Glob for ensemble member model directories (repeatable)")
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--tasks", nargs="+", default=["lfmc", "vh"], choices=list(TASK_SPECS.keys()))
    parser.add_argument("--include-final-fold", action="store_true", help="Include fold_9998 (all-data fit) if present")
    parser.add_argument(
        "--skip-incomplete-members",
        action="store_true",
        help="Skip ensemble member directories that do not yet contain fold_9998",
    )
    args = parser.parse_args()

    candidate_member_dirs = _expand_member_dirs(args.member_dir, args.member_glob)
    member_dirs, skipped_member_dirs = _filter_completed_members(
        candidate_member_dirs,
        skip_incomplete=args.skip_incomplete_members,
    )
    print(
        f"Found {len(candidate_member_dirs)} candidate member directories, "
        f"using {len(member_dirs)} complete members"
    )
    if skipped_member_dirs:
        print(f"Skipping {len(skipped_member_dirs)} incomplete members:")
        for member_dir in skipped_member_dirs:
            print(f"  {member_dir}")
    if not member_dirs:
        raise ValueError("No completed ensemble member directories found")

    folds = _list_folds(member_dirs[0], include_final_fold=args.include_final_fold)
    if not folds:
        raise ValueError(f"No fold_* directories found in {member_dirs[0]}")
    for member_dir in member_dirs[1:]:
        these_folds = _list_folds(member_dir, include_final_fold=args.include_final_fold)
        if these_folds != folds:
            raise ValueError(f"Fold mismatch for {member_dir}: {these_folds} vs {folds}")

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {
        "split": args.split,
        "candidate_member_dirs": candidate_member_dirs,
        "member_dirs": member_dirs,
        "skipped_member_dirs": skipped_member_dirs,
        "n_members": len(member_dirs),
        "folds": folds,
        "tasks": {},
    }

    for task in args.tasks:
        try:
            agg = _aggregate_task(member_dirs, split=args.split, task=task, folds=folds)
        except ValueError as exc:
            if task == "lfmc":
                raise
            print(f"Skipping task={task}: {exc}")
            continue
        if agg.empty:
            print(f"Skipping task={task}: no rows found")
            continue
        parquet_path = os.path.join(args.out_dir, f"ensemble_{args.split}_{task}.parquet")
        csv_path = os.path.join(args.out_dir, f"ensemble_{args.split}_{task}.csv")
        agg.to_csv(csv_path, index=False)
        try:
            agg.to_parquet(parquet_path, index=False)
        except Exception as exc:
            print(f"Warning: failed to write {parquet_path}: {exc}")

        y_true = agg["true"].to_numpy(dtype=float)
        y_pred = agg["pred_mean"].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        r2 = float(_r2_score_np(y_true, y_pred))
        mean_spread = float(np.mean(agg["pred_std"].to_numpy(dtype=float)))

        summary["tasks"][task] = {
            "n_rows": int(len(agg)),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mean_pred_std": mean_spread,
            "csv": csv_path,
            "parquet": parquet_path,
        }
        print(
            f"{task} ({args.split}): n={len(agg):,}, "
            f"MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}, mean ensemble std={mean_spread:.3f}"
        )

    summary_path = os.path.join(args.out_dir, f"ensemble_{args.split}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
