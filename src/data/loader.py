from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# -------------------------
# Common Utilities
# -------------------------

def _ensure_dir(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p.resolve()}")
    return p


def partition_non_iid(
    x: np.ndarray,
    y: np.ndarray,
    num_clients: int,
) -> List[Dict[str, np.ndarray]]:
    """Partition data into num_clients while preserving label distribution as much as possible."""
    rng = np.random.default_rng(42)
    labels = np.unique(y)

    idx_per_label = {
        label: rng.permutation(np.where(y == label)[0])
        for label in labels
    }

    xs = [[] for _ in range(num_clients)]
    ys = [[] for _ in range(num_clients)]

    for label, idxs in idx_per_label.items():
        splits = np.array_split(idxs, num_clients)
        for cid, split in enumerate(splits):
            if split.size == 0:
                continue
            xs[cid].append(x[split])
            ys[cid].append(y[split])

    parts = []
    for cid in range(num_clients):
        if xs[cid]:
            x_c = np.concatenate(xs[cid], axis=0)
            y_c = np.concatenate(ys[cid], axis=0)
        else:
            # If empty, add some random samples
            ridx = rng.choice(len(x), size=len(x) // num_clients, replace=False)
            x_c = x[ridx]
            y_c = y[ridx]

        parts.append({"x": x_c, "y": y_c})

    return parts


# -------------------------
# MNIST (for testing)
# -------------------------

def load_mnist(max_samples: int = None, **_) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension for CNN
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    if max_samples is not None:
        x_train = x_train[:max_samples]
        y_train = y_train[:max_samples]

    return x_train, y_train, x_test, y_test


# -------------------------
# Bot-IoT (CSV-based, assumes binary classification)
# -------------------------

def load_bot_iot(
    data_path: str = "data/raw/Bot-IoT",
    max_samples: int = None,
    test_size: float = 0.2,
    random_state: int = 42,
    label_col: str = "label",
    **_,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Bot-IoT CSV files and return (x_train, y_train, x_test, y_test).

    Assumptions:
    - .csv files exist under data_path
    - label_col column contains 0/1 or integer labels
      (0=normal, 1=attack format)
    """

    root = _ensure_dir(data_path)
    csv_files = sorted([p for p in root.glob("*.csv")])
    if not csv_files:
        raise FileNotFoundError(f"Bot-IoT CSV files not found: {root.resolve()}")

    dfs = [pd.read_csv(p) for p in csv_files]
    df = pd.concat(dfs, axis=0, ignore_index=True)

    label_candidates = []
    if label_col is not None:
        label_candidates.append(label_col)
    label_candidates.extend([
        "attack",
        "label",
        "Label",
        "y",
        "target",
        "class",
    ])

    resolved_label_col = None
    for candidate in label_candidates:
        if candidate in df.columns:
            resolved_label_col = candidate
            break

    if resolved_label_col is None:
        raise KeyError(
            f"Label column not found. candidates={label_candidates}. Please check the label column name in CSV."
        )

    if resolved_label_col != label_col:
        print(f"[load_bot_iot] Using '{resolved_label_col}' instead of '{label_col}' as label column.")

    label_col = resolved_label_col

    y = df[label_col].values
    feature_df = df.drop(columns=[label_col])

    numeric_df = feature_df.apply(pd.to_numeric, errors="coerce")

    # Exclude columns that are all NaN (unusable)
    all_nan_cols = numeric_df.columns[numeric_df.isna().all()]
    if len(all_nan_cols) > 0:
        print(
            f"[load_bot_iot] Removed {len(all_nan_cols)} columns that could not be converted to numeric: "
            f"{list(all_nan_cols)}"
        )
        numeric_df = numeric_df.drop(columns=all_nan_cols)

    # Fill remaining NaN with column median, replace with 0 if median is NaN
    if numeric_df.isna().any().any():
        nan_rows = int(numeric_df.isna().any(axis=1).sum())
        print(
            f"[load_bot_iot] Filling {nan_rows} rows with NaN (after numeric conversion) using median values."
        )
        medians = numeric_df.median(skipna=True)
        medians = medians.fillna(0.0)
        numeric_df = numeric_df.fillna(medians)

    if numeric_df.empty:
        raise ValueError(
            "[load_bot_iot] All feature columns are unusable. "
            "Please check if CSV contains pure numeric features."
        )

    X = numeric_df.values

    # Integer encoding correction
    # (Map unique values to 0..C-1 even if not 0/1)
    uniq = np.unique(y)
    mapping = {v: i for i, v in enumerate(uniq)}
    y = np.vectorize(mapping.get)(y)

    if max_samples is not None and len(X) > max_samples:
        X, _, y, _ = train_test_split(
            X, y,
            train_size=max_samples,
            stratify=y,
            random_state=random_state,
        )

    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    return x_train, y_train, x_test, y_test


# -------------------------
# Public API
# -------------------------

def load_dataset(name: str, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Call actual loader with name from config and arguments."""

    # Map path -> data_path (works even if yaml uses path)
    if "path" in kwargs and "data_path" not in kwargs:
        kwargs["data_path"] = kwargs.pop("path")

    if name.lower() in ["mnist"]:
        return load_mnist(**kwargs)
    elif name.lower() in ["bot_iot", "bot-iot", "botiot"]:
        return load_bot_iot(**kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

