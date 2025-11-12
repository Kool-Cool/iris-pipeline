# tests/test_evaluation.py
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score

EVENT_CSV = "data/iris_events.csv"
MODEL_PATH = "models/iris_clf.joblib"
BASELINE_JSON = "metrics/baseline.json"  # {"accuracy": 0.96, "f1": 0.96}

def latest_per_id(df):
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    return (df.sort_values(["iris_id","event_timestamp"])
              .groupby("iris_id", as_index=False)
              .tail(1))

def test_model_accuracy_threshold(tmp_path):
    df = pd.read_csv(EVENT_CSV)
    latest = latest_per_id(df)
    X = latest[["sepal_length","sepal_width","petal_length","petal_width"]]
    y = latest["species"]
    clf = joblib.load(MODEL_PATH)
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro")
    assert acc >= 0.9
    assert f1 >= 0.9

def test_regression_vs_baseline():
    import json
    df = pd.read_csv(EVENT_CSV)
    latest = latest_per_id(df)
    X = latest[["sepal_length","sepal_width","petal_length","petal_width"]]
    y = latest["species"]
    clf = joblib.load(MODEL_PATH)
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    with open(BASELINE_JSON) as f:
        base = json.load(f)
    # allow small tolerance 0.02
    assert acc + 1e-9 >= base["accuracy"] - 0.02
