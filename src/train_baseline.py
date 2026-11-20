"""Leave-one-round-out evaluation of people-count (0/1/2) or presence (0/1+) classifiers.

Every recording round is held out once while the model trains on all other rounds.
Windows from one session never appear on both sides, so near-duplicate windows cannot
leak. An optional --final-block (for example a round from another day) stays out of
cross-validation entirely and checks drift once, after the model is fixed.
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error

from .data_collection.csi import DYNAMIC_FEATURES, FEATURES, MODEL_FEATURES, SCHEMA


def models(used):
    forest = dict(n_estimators=200, random_state=42, class_weight="balanced", min_samples_leaf=3)
    return (("majority", lambda: DummyClassifier(strategy="most_frequent"), used),
            ("rssi_only", lambda: RandomForestClassifier(**forest), ("rssi_mean", "rssi_sd")),
            ("random_forest", lambda: RandomForestClassifier(**forest), used))


def scores(frame, column, labels):
    truth, prediction = frame["target"].to_numpy(), frame[column].to_numpy()
    numeric = all(isinstance(label, (int, np.integer)) for label in labels)
    # One vote per session: windows of one recording are correlated, so this is the honest unit.
    votes = frame.groupby("session").agg(target=("target", "first"), vote=(column, lambda v: v.mode().iloc[0]))
    result = dict(
        windows=int(len(frame)), sessions=int(len(votes)),
        exact_accuracy=accuracy_score(truth, prediction),
        macro_f1=f1_score(truth, prediction, labels=labels, average="macro", zero_division=0),
        session_vote_accuracy=accuracy_score(votes.target, votes.vote),
        confusion_actual_by_predicted=confusion_matrix(truth, prediction, labels=labels).tolist(),
        per_round_accuracy={str(k): accuracy_score(g.target, g[column]) for k, g in frame.groupby("block")})
    if numeric:
        truth, prediction = truth.astype(int), prediction.astype(int)
        present = truth > 0
        result["empty_false_positive_rate"] = float(np.mean(prediction[~present] > 0)) if (~present).any() else None
        result["occupied_recall"] = float(np.mean(prediction[present] > 0)) if present.any() else None
    if numeric and len(labels) == 3:
        result["mean_absolute_count_error"] = mean_absolute_error(truth, prediction)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/processed/windows.csv"))
    parser.add_argument("--task", choices=("count", "presence", "activity"), default="count",
                        help="count predicts 0/1/2; presence predicts empty (0) vs occupied (1); "
                             "activity predicts the webcam's scene label (sitting, walking, lying...)")
    parser.add_argument("--min-agree", type=float, default=0.7,
                        help="activity: keep windows where at least this share of camera frames agree")
    parser.add_argument("--features", choices=("dynamic", "nondrifting", "all"), default="dynamic",
                        help="dynamic: movement + breathing (default). nondrifting: also static shape "
                             "features. all: also rssi_mean and amp_scale, which drift with the empty room")
    parser.add_argument("--final-block", help="Round kept out of cross-validation as a one-time drift check")
    parser.add_argument("--out", type=Path, default=Path("models/pilot-v2"))
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a new output directory for this model run")
    frame = pd.read_csv(args.csv, dtype={"block": str, "session": str}, keep_default_na=False)
    used = dict(dynamic=DYNAMIC_FEATURES, nondrifting=MODEL_FEATURES, all=FEATURES)[args.features]
    missing = [f for f in FEATURES if f not in frame]
    if missing:
        parser.error(f"{args.csv} lacks {missing}. Rebuild it with python -m src.build_dataset")
    if args.task == "activity":
        if "cam_activity" not in frame:
            parser.error("no camera labels in the CSV; record with serial_capture --camera")
        agree = pd.to_numeric(frame["cam_agree"], errors="coerce")
        frame = frame[(agree >= args.min_agree) & ~frame.cam_activity.isin(["", "unknown"])].copy()
        frame["target"] = frame["cam_activity"]
        labels = sorted(frame.target.unique())
        print(f"Activity labels: {dict(frame.target.value_counts())}")
    else:
        frame["count"] = pd.to_numeric(frame["count"], errors="coerce")
        frame = frame[frame["count"].isin([0, 1, 2])].copy()
        frame["count"] = frame["count"].astype(int)
        frame["target"] = frame["count"].clip(upper=1) if args.task == "presence" else frame["count"]
        labels = [0, 1] if args.task == "presence" else [0, 1, 2]
    final = frame[frame.block == args.final_block].copy() if args.final_block else frame.iloc[:0].copy()
    cv = frame[frame.block != args.final_block].copy() if args.final_block else frame.copy()
    rounds = sorted(cv.block.unique())
    if len(rounds) < 2:
        parser.error("record at least two rounds; each is held out once")
    if set(cv.target) != set(labels):
        parser.error(f"cross-validation data must contain every class {labels}")
    for name, group in cv.groupby("block"):
        lacking = sorted(set(labels) - set(group.target))
        if lacking:
            print(f"Note: round {name} has no class {lacking}; its per-round accuracy covers fewer classes.")

    report = dict(task=args.task, schema=SCHEMA, features=used, rounds=rounds,
                  sessions_per_round={str(k): int(v) for k, v in cv.groupby("block").session.nunique().items()})
    for name, make, columns in models(used):
        cv[name] = pd.Series(index=cv.index, dtype=object)
        for held in rounds:
            train, test = cv.block != held, cv.block == held
            estimator = make().fit(cv.loc[train, list(columns)].to_numpy(), cv.loc[train, "target"])
            cv.loc[test, name] = estimator.predict(cv.loc[test, list(columns)].to_numpy())
        cv[name] = cv[name].astype(cv["target"].dtype)
        report[name] = scores(cv, name, labels)

    forest = models(used)[2][1]().fit(cv[list(used)].to_numpy(), cv["target"])
    importances = sorted(zip(used, forest.feature_importances_), key=lambda x: -x[1])
    report["feature_importance"] = {k: round(float(v), 4) for k, v in importances}
    if len(final):
        final["random_forest"] = forest.predict(final[list(used)].to_numpy())
        report["final_block"] = dict(block=args.final_block, **scores(final, "random_forest", labels))

    result = json.dumps(report, indent=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as stage:
        run = Path(stage) / "run"
        run.mkdir()
        joblib.dump(dict(schema=SCHEMA, task=args.task, features=used, model=forest), run / "forest.joblib")
        cv.to_csv(run / "cv_predictions.csv", index=False)
        if len(final):
            final.to_csv(run / "final_block_predictions.csv", index=False)
        (run / "metrics.json").write_text(result, encoding="utf8")
        run.rename(args.out)
    rf = report["random_forest"]
    print(result)
    print(f"\nRandom forest, leave-one-round-out over {len(rounds)} rounds: "
          f"window accuracy {rf['exact_accuracy']:.2f}, session vote {rf['session_vote_accuracy']:.2f}, "
          f"majority baseline {report['majority']['exact_accuracy']:.2f}, RSSI-only {report['rssi_only']['exact_accuracy']:.2f}")


if __name__ == "__main__":
    main()
