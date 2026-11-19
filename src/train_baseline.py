"""Evaluate a 0/1/2 random forest on a preselected, independent recording block."""

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

from .data_collection.csi import FEATURES


def scores(truth, prediction):
    present = truth > 0
    return dict(exact_accuracy=accuracy_score(truth, prediction),
                macro_f1=f1_score(truth, prediction, labels=[0, 1, 2], average="macro", zero_division=0),
                mean_absolute_count_error=mean_absolute_error(truth, prediction),
                confusion_actual_by_predicted_012=confusion_matrix(truth, prediction, labels=[0, 1, 2]).tolist(),
                empty_false_positive_rate=float(np.mean(prediction[~present] > 0)),
                occupied_recall=float(np.mean(prediction[present] > 0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/processed/windows.csv"))
    parser.add_argument("--test-block", required=True)
    parser.add_argument("--out", type=Path, default=Path("models/pilot-v1"))
    args = parser.parse_args()
    frame = pd.read_csv(args.csv, dtype={"block": str, "session": str}, keep_default_na=False)
    train = frame[frame.block != args.test_block]
    test = frame[frame.block == args.test_block].copy()
    if set(train.session) & set(test.session):
        parser.error("a session occurs on both sides of the block split")
    if any(set(part["count"]) != {0, 1, 2} for part in (train, test)):
        parser.error("record 0, 1 and 2 in both training and held-out blocks")
    if args.out.exists():
        parser.error("choose a new output directory for this model run")
    report = dict(test_block=args.test_block, train_blocks=sorted(train.block.unique()),
                  training_sessions=sorted(train.session.unique()), test_sessions=sorted(test.session.unique()))
    for name, estimator, columns in (
            ("majority", DummyClassifier(strategy="most_frequent"), FEATURES),
            ("rssi_only", RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced"), FEATURES[:2]),
            ("random_forest", RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced"), FEATURES)):
        estimator.fit(train[list(columns)].to_numpy(), train["count"])
        prediction = estimator.predict(test[list(columns)].to_numpy())
        report[name] = scores(test["count"].to_numpy(), prediction)
        test[name] = prediction
        if name == "random_forest":
            artifact = dict(schema=1, features=FEATURES, model=estimator)
    result = json.dumps(report, indent=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as stage:
        run = Path(stage) / "run"
        run.mkdir()
        joblib.dump(artifact, run / "forest.joblib")
        test.to_csv(run / "held_out_predictions.csv", index=False)
        (run / "metrics.json").write_text(result, encoding="utf8")
        run.rename(args.out)
    print(result)


if __name__ == "__main__":
    main()
