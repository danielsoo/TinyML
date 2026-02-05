import os
from pathlib import Path
from typing import Tuple, Dict, List
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Suppress pandas DtypeWarning
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
pd.options.mode.chained_assignment = None


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
    balance_ratio: float = None,
    use_smote: bool = False,
    **_,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Bot-IoT CSV files and return (x_train, y_train, x_test, y_test).

    Now includes IP addresses and categorical features (protocol, flags, state)
    by encoding them for machine learning compatibility.

    Assumptions:
    - .csv files exist under data_path
    - label_col column contains 0/1 or integer labels
      (0=normal, 1=attack format)
    """

    root = _ensure_dir(data_path)
    csv_files = sorted([p for p in root.glob("*.csv")])
    if not csv_files:
        raise FileNotFoundError(f"Bot-IoT CSV files not found: {root.resolve()}")

    # Suppress DtypeWarning during CSV reading
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
        dfs = [pd.read_csv(p, low_memory=False) for p in csv_files]
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

    # Convert IP addresses to integers
    def ip_to_int(ip_str):
        """Convert IP address string to integer using ipaddress module."""
        try:
            import ipaddress
            if pd.isna(ip_str) or ip_str == '':
                return 0
            return int(ipaddress.IPv4Address(str(ip_str)))
        except (ValueError, AttributeError, TypeError):
            # Invalid IP or NaN, return 0
            return 0

    # Process IP address columns (source/destination IPs)
    ip_columns = ['saddr', 'daddr', 'srcip', 'dstip', 'src_ip', 'dst_ip']
    for col in ip_columns:
        if col in feature_df.columns:
            print(f"[load_bot_iot] Converting IP addresses in column '{col}' to integers.")
            feature_df[col] = feature_df[col].apply(ip_to_int)

    # Encode categorical columns using Label Encoding
    from sklearn.preprocessing import LabelEncoder
    
    categorical_columns = ['proto', 'flgs', 'state', 'service']
    encoders = {}
    
    for col in categorical_columns:
        if col in feature_df.columns:
            print(f"[load_bot_iot] Encoding categorical column '{col}' using Label Encoding.")
            le = LabelEncoder()
            # Handle NaN values by replacing with 'unknown' before encoding
            feature_df[col] = feature_df[col].fillna('unknown').astype(str)
            feature_df[col] = le.fit_transform(feature_df[col])
            encoders[col] = le

    # Category and subcategory columns are label information, so exclude them
    # (they are redundant with the label_col)
    # Uncomment below if you want to use them as features:
    # label_columns_to_exclude = ['category', 'subcategory']
    # for col in label_columns_to_exclude:
    #     if col in feature_df.columns:
    #         feature_df = feature_df.drop(columns=[col])
    #         print(f"[load_bot_iot] Removed label column '{col}' (redundant with label_col).")

    # Encode any remaining string columns
    for col in feature_df.columns:
        if feature_df[col].dtype == 'object':
            print(f"[load_bot_iot] Encoding remaining string column '{col}' using Label Encoding.")
            le = LabelEncoder()
            feature_df[col] = le.fit_transform(feature_df[col].fillna('unknown').astype(str))

    # Convert all columns to numeric
    numeric_df = feature_df.apply(pd.to_numeric, errors="coerce")

    # Exclude columns that are all NaN (unusable)
    all_nan_cols = numeric_df.columns[numeric_df.isna().all()]
    if len(all_nan_cols) > 0:
        print(
            f"[load_bot_iot] Removed {len(all_nan_cols)} columns that are all NaN: "
            f"{list(all_nan_cols)}"
        )
        numeric_df = numeric_df.drop(columns=all_nan_cols)

    # Fill remaining NaN with column median, replace with 0 if median is NaN
    if numeric_df.isna().any().any():
        nan_rows = int(numeric_df.isna().any(axis=1).sum())
        print(
            f"[load_bot_iot] Filling {nan_rows} rows with NaN using median values."
        )
        medians = numeric_df.median(skipna=True)
        medians = medians.fillna(0.0)
        numeric_df = numeric_df.fillna(medians)

    if numeric_df.empty:
        raise ValueError(
            "[load_bot_iot] All feature columns are unusable. "
            "Please check if CSV contains usable features."
        )

    X = numeric_df.values

    # Integer encoding correction
    # (Map unique values to 0..C-1 even if not 0/1)
    uniq = np.unique(y)
    mapping = {v: i for i, v in enumerate(uniq)}
    y = np.vectorize(mapping.get)(y)

    if max_samples is not None and len(X) > max_samples:
        # Check if stratify is safe (each class needs at least 2 samples)
        unique, counts = np.unique(y, return_counts=True)
        min_class_count = counts.min()
        use_stratify = min_class_count >= 2
        
        X, _, y, _ = train_test_split(
            X, y,
            train_size=max_samples,
            stratify=y if use_stratify else None,
            random_state=random_state,
        )

    # Check if stratify is safe for final split
    unique, counts = np.unique(y, return_counts=True)
    min_class_count = counts.min()
    use_stratify = min_class_count >= 2
    
    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y if use_stratify else None,
        random_state=random_state,
    )

    # Mitigate imbalance: undersample majority (balance_ratio = minority:majority cap, e.g. 3.0 -> <=1:3)
    if balance_ratio is not None and balance_ratio > 0:
        unique_train = np.unique(y_train)
        if len(unique_train) == 2:
            rng = np.random.default_rng(random_state)
            counts = np.bincount(y_train.astype(int))
            minority_class = np.argmin(counts)
            majority_class = 1 - minority_class
            n_minority = counts[minority_class]
            n_majority = counts[majority_class]
            if n_majority > n_minority * balance_ratio:
                n_majority_target = int(n_minority * balance_ratio)
                majority_idx = np.where(y_train == majority_class)[0]
                keep_idx = rng.choice(majority_idx, size=min(n_majority_target, len(majority_idx)), replace=False)
                minority_idx = np.where(y_train == minority_class)[0]
                balanced_idx = np.concatenate([minority_idx, keep_idx])
                rng.shuffle(balanced_idx)
                x_train = x_train[balanced_idx]
                y_train = y_train[balanced_idx]
                print(f"[load_bot_iot] Balanced training set: majority {n_majority} -> {len(keep_idx)} (ratio<={balance_ratio}), total={len(y_train):,}")

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    # Feature scaling: improves NN training stability and accuracy (fit on train, transform on test)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype("float32")
    x_test = scaler.transform(x_test).astype("float32")
    print(f"[load_bot_iot] StandardScaler applied (fit on train).")

    # SMOTE: oversample minority class (train only)
    if use_smote and len(np.unique(y_train)) == 2:
        try:
            from imblearn.over_sampling import SMOTE
            counts_before = np.bincount(y_train.astype(int))
            k = max(1, min(5, int(counts_before.min()) - 1))
            smote = SMOTE(random_state=random_state, k_neighbors=k)
            x_train, y_train = smote.fit_resample(x_train, y_train)
            x_train = x_train.astype("float32")
            print(f"[load_bot_iot] SMOTE applied: train -> {len(y_train):,} (0={np.sum(y_train==0):,}, 1={np.sum(y_train==1):,})")
        except Exception as e:
            print(f"[load_bot_iot] SMOTE skipped: {e}")

    print(f"[load_bot_iot] Final feature shape: {x_train.shape[1]} features (includes IP addresses and categorical features)")

    return x_train, y_train, x_test, y_test


# -------------------------
# CIC-IDS2017 (CSV-based, multi-class classification)
# -------------------------

def load_cicids2017(
    data_path: str = "data/raw/Bot-IoT",
    max_samples: int = None,
    test_size: float = 0.2,
    random_state: int = 42,
    binary: bool = True,
    balance_ratio: float = None,
    use_smote: bool = False,  # SMOTE oversamples minority class (train only)
    **_,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIC-IDS2017 CSV files and return (x_train, y_train, x_test, y_test).
    
    Args:
        data_path: Directory containing .pcap_ISCX.csv files
        max_samples: Maximum total samples to use (for memory efficiency)
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        binary: If True, convert to BENIGN(0) vs ATTACK(1). If False, keep multi-class.
    
    Returns:
        x_train, y_train, x_test, y_test
    """
    data_dir = _ensure_dir(data_path)
    print(f"[load_cicids2017] data_path={data_path}")
    
    # Find all CIC-IDS2017 CSV files (*.pcap_ISCX.csv)
    csv_files = list(data_dir.glob("*.pcap_ISCX.csv"))
    
    if not csv_files:
        raise FileNotFoundError(
            f"No CIC-IDS2017 CSV files (*.pcap_ISCX.csv) found in {data_dir}. "
            "Please download the dataset from https://www.unb.ca/cic/datasets/ids-2017.html"
        )
    
    print(f"[load_cicids2017] Found {len(csv_files)} CSV files")
    
    # Load full files to preserve true class distribution (~55:45).
    # Random sample per file if max_samples set (avoids head-only BENIGN bias).
    if max_samples is not None:
        samples_per_file = max_samples // len(csv_files)
        print(f"[load_cicids2017] Target {samples_per_file} samples/file (random from full file)")
    else:
        samples_per_file = None

    rng = np.random.default_rng(random_state)
    dfs = []
    for csv_file in csv_files:
        try:
            try:
                df = pd.read_csv(csv_file, low_memory=False)
            except Exception:
                df = pd.read_csv(csv_file, engine="python", low_memory=False)
            if samples_per_file and len(df) > samples_per_file:
                idx = rng.choice(len(df), size=samples_per_file, replace=False)
                df = df.iloc[idx]
            dfs.append(df)
            print(f"[load_cicids2017] Loaded {len(df):,} rows from {csv_file.name}")
        except Exception as e:
            print(f"[load_cicids2017] Warning: Failed to load {csv_file.name}: {e}")
            continue
    
    if not dfs:
        raise ValueError("Failed to load any CIC-IDS2017 files")
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"[load_cicids2017] Total samples loaded: {len(df):,}")
    
    # Label column has leading space: ' Label'
    label_col = df.columns[-1]
    if label_col.strip() != "Label":
        print(f"[load_cicids2017] Warning: Expected last column to be 'Label', got '{label_col}'")
    
    # Extract labels
    y = df[label_col].values
    df = df.drop(columns=[label_col])
    
    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()
    
    # Handle infinity and NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Drop columns that are all NaN
    nan_cols = df.columns[df.isna().all()].tolist()
    if nan_cols:
        print(f"[load_cicids2017] Dropping {len(nan_cols)} all-NaN columns")
        df = df.drop(columns=nan_cols)
    
    # Fill remaining NaN with 0
    df = df.fillna(0)
    
    # Convert all columns to numeric (some might be strings)
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)
            except:
                # If conversion fails, drop the column
                print(f"[load_cicids2017] Dropping non-numeric column: {col}")
                df = df.drop(columns=[col])
    
    X = df.values.astype('float32')

    # Remove duplicates (CIC-IDS2017 has many duplicate rows)
    n_before = len(X)
    df_temp = pd.DataFrame(X)
    df_temp["_y"] = y
    df_dedup = df_temp.drop_duplicates()
    X = df_dedup.drop(columns=["_y"]).values.astype("float32")
    y = df_dedup["_y"].values
    n_after = len(X)
    print(f"[load_cicids2017] Removed {n_before - n_after:,} duplicates ({n_after:,} unique)")
    
    # Process labels
    unique_labels = np.unique(y)
    print(f"[load_cicids2017] Found {len(unique_labels)} unique labels")
    
    if binary:
        # Convert to binary: BENIGN=0, everything else=1
        y_binary = np.where(y == 'BENIGN', 0, 1)
        y = y_binary
        print(f"[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1")
        print(f"[load_cicids2017] Label distribution: BENIGN={np.sum(y==0)}, ATTACK={np.sum(y==1)}")
    else:
        # Multi-class: encode all labels
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        y = np.array([label_mapping[label] for label in y])
        print(f"[load_cicids2017] Multi-class mode: {len(unique_labels)} classes")
        for idx in range(len(unique_labels)):
            print(f"  Class {idx}: {np.sum(y==idx)} samples")
    
    # Final sampling if needed
    if max_samples is not None and len(X) > max_samples:
        unique, counts = np.unique(y, return_counts=True)
        min_class_count = counts.min()
        use_stratify = min_class_count >= 2
        
        X, _, y, _ = train_test_split(
            X, y,
            train_size=max_samples,
            stratify=y if use_stratify else None,
            random_state=random_state,
        )
        print(f"[load_cicids2017] Sampled down to {len(X):,} total samples")
    
    # Train/test split
    unique, counts = np.unique(y, return_counts=True)
    min_class_count = counts.min()
    use_stratify = min_class_count >= 2
    
    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y if use_stratify else None,
        random_state=random_state,
    )

    # Undersample majority for severe imbalance (binary only)
    if binary and balance_ratio is not None and balance_ratio > 0:
        rng = np.random.default_rng(random_state)
        counts = np.bincount(y_train.astype(int))
        minority_class = np.argmin(counts)
        majority_class = 1 - minority_class
        n_minority = counts[minority_class]
        n_majority = counts[majority_class]
        if n_majority > n_minority * balance_ratio:
            n_majority_target = int(n_minority * balance_ratio)
            majority_idx = np.where(y_train == majority_class)[0]
            keep_idx = rng.choice(majority_idx, size=n_majority_target, replace=False)
            minority_idx = np.where(y_train == minority_class)[0]
            balanced_idx = np.concatenate([minority_idx, keep_idx])
            rng.shuffle(balanced_idx)
            x_train = x_train[balanced_idx]
            y_train = y_train[balanced_idx]
            print(f"[load_cicids2017] Balanced training set: majority {n_majority} -> {n_majority_target} (ratio<={balance_ratio}), total={len(y_train):,}")

    # Feature scaling: improves NN training stability and accuracy (fit on train, transform on test)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype("float32")
    x_test = scaler.transform(x_test).astype("float32")
    print(f"[load_cicids2017] StandardScaler applied (fit on train).")

    # SMOTE: oversample minority class (ATTACK) (train only, test unchanged)
    if binary and use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            counts_before = np.bincount(y_train.astype(int))
            k = max(1, min(5, int(counts_before.min()) - 1))
            smote = SMOTE(random_state=random_state, k_neighbors=k)
            x_train, y_train = smote.fit_resample(x_train, y_train)
            x_train = x_train.astype("float32")
            print(f"[load_cicids2017] SMOTE applied: train -> {len(y_train):,} (BENIGN={np.sum(y_train==0):,}, ATTACK={np.sum(y_train==1):,})")
        except Exception as e:
            print(f"[load_cicids2017] SMOTE skipped: {e}")

    print(f"[load_cicids2017] Final feature shape: {x_train.shape[1]} features")
    print(f"[load_cicids2017] Train samples: {len(x_train):,}, Test samples: {len(x_test):,}")

    return x_train, y_train, x_test, y_test


# -------------------------
# TON_IoT (ToN_IoT) - UNSW Canberra IoT/IIoT Cybersecurity
# -------------------------

def load_ton_iot(
    data_path: str = "data/raw/TON_IoT",
    max_samples: int = None,
    test_size: float = 0.2,
    random_state: int = 42,
    label_col: str = "label",
    binary: bool = True,
    balance_ratio: float = None,
    use_smote: bool = False,
    **_,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load TON_IoT (ToN_IoT) CSV and return (x_train, y_train, x_test, y_test).

    TON_IoT: Telemetry, OS (Windows/Ubuntu), Network datasets from UNSW Canberra.
    Processed/Train_Test CSV have 'label' (normal vs attack) and optionally 'type' (attack sub-class).
    Academic use free (Dr Nour Moustafa, UNSW).

    Args:
        data_path: Directory containing CSV (e.g. Train_Test_datasets or Processed)
        label_col: Column name for label (default 'label')
        binary: If True, normal=0 / attack=1. If False, use 'type' for multi-class when present.
    """
    root = _ensure_dir(data_path)
    # Support both flat *.csv and subdirs like Train_Test_datasets/*.csv
    csv_files = sorted(root.glob("*.csv"))
    if not csv_files:
        for sub in ["Train_Test_datasets", "Processed_datasets", "processed"]:
            subdir = root / sub
            if subdir.exists():
                csv_files = sorted(subdir.glob("*.csv"))
                if csv_files:
                    break
    if not csv_files:
        raise FileNotFoundError(
            f"TON_IoT CSV files not found under {root}. "
            "Place CSV (e.g. from Train_Test_datasets) in data_path or data_path/Train_Test_datasets."
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
        dfs = [pd.read_csv(p, low_memory=False) for p in csv_files]
    df = pd.concat(dfs, axis=0, ignore_index=True)

    if label_col not in df.columns:
        for cand in ["label", "Label", "type", "Type", "attack"]:
            if cand in df.columns:
                label_col = cand
                print(f"[load_ton_iot] Using label column '{label_col}'")
                break
        else:
            raise KeyError(f"Label column not found. Available: {list(df.columns)[:10]}...")

    y_raw = df[label_col]
    feature_df = df.drop(columns=[label_col])
    if "type" in feature_df.columns and binary:
        feature_df = feature_df.drop(columns=["type"])

    # Map label to 0/1 for binary (support string or numeric)
    if binary:
        if pd.api.types.is_numeric_dtype(y_raw):
            y = y_raw.values.astype(np.int64)
            uniq = np.unique(y)
            if len(uniq) > 2:
                # Assume 0 = normal, others = attack
                y = np.where(y == uniq[0], 0, 1).astype(np.int64)
        else:
            y_str = y_raw.astype(str).str.lower()
            y = np.where(y_str.str.contains("normal|benign|0", regex=True), 0, 1).astype(np.int64)
        print(f"[load_ton_iot] Binary labels: 0={np.sum(y==0):,}, 1={np.sum(y==1):,}")
    else:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        print(f"[load_ton_iot] Multi-class: {len(le.classes_)} classes")

    # Numeric features only
    for col in list(feature_df.columns):
        if feature_df[col].dtype == "object" or not np.issubdtype(feature_df[col].dtype, np.number):
            try:
                feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
            except Exception:
                feature_df = feature_df.drop(columns=[col])
    numeric_df = feature_df.select_dtypes(include=[np.number])
    all_nan = numeric_df.columns[numeric_df.isna().all()]
    if len(all_nan) > 0:
        numeric_df = numeric_df.drop(columns=all_nan)
    medians = numeric_df.median(skipna=True).fillna(0)
    numeric_df = numeric_df.fillna(medians)
    X = numeric_df.values.astype("float32")

    if max_samples is not None and len(X) > max_samples:
        unique, counts = np.unique(y, return_counts=True)
        use_stratify = counts.min() >= 2
        X, _, y, _ = train_test_split(
            X, y, train_size=max_samples,
            stratify=y if use_stratify else None,
            random_state=random_state,
        )

    unique, counts = np.unique(y, return_counts=True)
    use_stratify = counts.min() >= 2
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        stratify=y if use_stratify else None,
        random_state=random_state,
    )

    # Undersample majority (binary)
    if binary and balance_ratio is not None and balance_ratio > 0 and len(unique) == 2:
        rng = np.random.default_rng(random_state)
        counts = np.bincount(y_train.astype(int))
        minority_class = np.argmin(counts)
        majority_class = 1 - minority_class
        n_minority, n_majority = counts[minority_class], counts[majority_class]
        if n_majority > n_minority * balance_ratio:
            n_majority_target = int(n_minority * balance_ratio)
            majority_idx = np.where(y_train == majority_class)[0]
            keep_idx = rng.choice(majority_idx, size=min(n_majority_target, len(majority_idx)), replace=False)
            minority_idx = np.where(y_train == minority_class)[0]
            balanced_idx = np.concatenate([minority_idx, keep_idx])
            rng.shuffle(balanced_idx)
            x_train = x_train[balanced_idx]
            y_train = y_train[balanced_idx]
            print(f"[load_ton_iot] Balanced: majority {n_majority} -> {len(keep_idx)} (ratio<={balance_ratio}), total={len(y_train):,}")

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype("float32")
    x_test = scaler.transform(x_test).astype("float32")
    print(f"[load_ton_iot] StandardScaler applied.")

    if binary and use_smote and len(np.unique(y_train)) == 2:
        try:
            from imblearn.over_sampling import SMOTE
            counts_before = np.bincount(y_train.astype(int))
            k = max(1, min(5, int(counts_before.min()) - 1))
            smote = SMOTE(random_state=random_state, k_neighbors=k)
            x_train, y_train = smote.fit_resample(x_train, y_train)
            x_train = x_train.astype("float32")
            print(f"[load_ton_iot] SMOTE applied: train -> {len(y_train):,} (0={np.sum(y_train==0):,}, 1={np.sum(y_train==1):,})")
        except Exception as e:
            print(f"[load_ton_iot] SMOTE skipped: {e}")

    print(f"[load_ton_iot] Features: {x_train.shape[1]}, Train: {len(y_train):,}, Test: {len(y_test):,}")
    return x_train, y_train, x_test, y_test


# -------------------------
# Public API
# -------------------------

def load_dataset(name: str, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Call actual loader with name from config and arguments."""

    # Map path -> data_path (works even if yaml uses path)
    if "path" in kwargs and "data_path" not in kwargs:
        kwargs["data_path"] = kwargs.pop("path")

    # Override data path from environment (e.g. on PSU server where data lives elsewhere)
    if name.lower() in ["cicids2017", "cic-ids-2017", "cic_ids_2017"]:
        env_path = os.environ.get("CICIDS2017_DATA_PATH")
        if env_path:
            kwargs["data_path"] = env_path
    elif name.lower() in ["bot_iot", "bot-iot", "botiot"]:
        env_path = os.environ.get("BOTIOT_DATA_PATH")
        if env_path:
            kwargs["data_path"] = env_path

    if name.lower() in ["mnist"]:
        return load_mnist(**kwargs)
    elif name.lower() in ["bot_iot", "bot-iot", "botiot"]:
        return load_bot_iot(**kwargs)
    elif name.lower() in ["cicids2017", "cic-ids-2017", "cic_ids_2017"]:
        return load_cicids2017(**kwargs)
    elif name.lower() in ["ton_iot", "toniot", "ton_iot"]:
        return load_ton_iot(**kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

