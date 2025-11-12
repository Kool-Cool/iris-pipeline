# tests/test_data_validation.py
import pandas as pd
import numpy as np

CSV_PATH = "data/iris_events.csv"

def test_schema_columns():
    df = pd.read_csv(CSV_PATH)
    expected = [
        "event_timestamp","iris_id","sepal_length","sepal_width",
        "petal_length","petal_width","species","created_timestamp"
    ]
    assert list(df.columns) == expected

def test_types_and_non_nulls():
    df = pd.read_csv(CSV_PATH)
    # parse timestamps
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="raise")
    df["created_timestamp"] = pd.to_datetime(df["created_timestamp"], errors="raise")
    # numeric
    num_cols = ["sepal_length","sepal_width","petal_length","petal_width"]
    for c in num_cols:
        assert pd.api.types.is_numeric_dtype(df[c])
        assert df[c].isnull().sum() == 0
    # categorical species
    assert df["species"].isnull().sum() == 0
    assert df["iris_id"].isnull().sum() == 0

def test_timestamp_consistency():
    df = pd.read_csv(CSV_PATH, parse_dates=["event_timestamp","created_timestamp"])
    # event <= created
    assert ((df["event_timestamp"] <= df["created_timestamp"]).all())

def test_unique_event_per_id():
    df = pd.read_csv(CSV_PATH, parse_dates=["event_timestamp"])
    dup = df.duplicated(subset=["iris_id","event_timestamp"]).sum()
    assert dup == 0

def test_monotonic_events_per_id():
    df = pd.read_csv(CSV_PATH)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    ok = True
    for _, g in df.sort_values(["iris_id","event_timestamp"]).groupby("iris_id"):
        if not g["event_timestamp"].is_monotonic_increasing:
            ok = False
            break
    assert ok

def test_feature_ranges():
    df = pd.read_csv(CSV_PATH)
    assert df["sepal_length"].between(4.0, 8.5).all()
    assert df["sepal_width"].between(2.0, 4.5).all()
    assert df["petal_length"].between(1.0, 7.0).all()
    assert df["petal_width"].between(0.1, 2.5).all()

def test_species_domain():
    df = pd.read_csv(CSV_PATH)
    valid = {"setosa","versicolor","virginica"}
    assert set(df["species"].unique()).issubset(valid)
