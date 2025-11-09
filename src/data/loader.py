# /src/data/loader.py
from __future__ import annotations
from typing import Tuple, Dict
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_bot_iot(
    data_path: str = "data/raw/Bot-IoT",
    *,
    max_samples: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bot-IoT 데이터셋 로드 및 전처리
    
    읽는 것: Bot-IoT CSV 파일들 (reduced_data_1.csv ~ reduced_data_4.csv)
    반환하는 것: (x_train, y_train, x_test, y_test)
        - x_train: 학습용 특징 데이터 (n_samples, n_features)
        - y_train: 학습용 레이블 (0=정상, 1=공격)
        - x_test: 테스트용 특징 데이터
        - y_test: 테스트용 레이블
    
    목적: 모델 학습을 위한 데이터 준비
    """
    data_path = Path(data_path)
    csv_files = sorted(data_path.glob("reduced_data_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"Bot-IoT CSV files not found in {data_path}")
    
    print(f"📂 Loading Bot-IoT data from {len(csv_files)} files...")
    
    # 모든 CSV 파일 읽기
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, low_memory=False)
        dfs.append(df)
        print(f"  Loaded {csv_file.name}: {len(df)} samples")
    
    # 합치기
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total samples: {len(df)}")
    
    # 샘플 제한 (메모리 절약용)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        print(f"  Limited to {max_samples} samples")
    
    # 특징과 레이블 분리
    # 레이블 컬럼: attack (0=정상, 1=공격)
    label_col = "attack"
    feature_cols = [col for col in df.columns if col not in [label_col, "category", "subcategory", "pkSeqID"]]
    
    # 숫자형이 아닌 컬럼 제외 (IP 주소 등)
    numeric_cols = []
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            numeric_cols.append(col)
    
    # x: 특징 데이터
    # y: 정답 데이터
    X = df[numeric_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.int32)
    
    print(f"  Features: {X.shape[1]} numeric columns")
    print(f"  Labels: {np.unique(y, return_counts=True)}")
    
    # Train/Test split
    # 데이터를 훈련 데이터와 테스트 데이터로 나눔
    # test_size: 테스트 데이터 비율 (0.2 = 20%)
    # random_state: 랜덤 시드 (42)
    # stratify: 레이블 분포를 유지하면서 나눔 (레이블 분포를 유지하면서 나눔)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    
    # Normalization (StandardScaler)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    return X_train, y_train, X_test, y_test


def load_dataset(
    name: str = "bot_iot",
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (x_train, y_train, x_test, y_test) for the requested dataset.
    """
    if name == "bot_iot":
        return load_bot_iot(**kwargs)
<<<<<<< HEAD
    raise ValueError(f"Unknown dataset: {name}")


=======
    if name == "placeholder_mnist":
        import tensorflow as tf

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = (x_train.astype("float32") / 255.0)[..., None]
        x_test = (x_test.astype("float32") / 255.0)[..., None]
        return x_train, y_train, x_test, y_test
    raise ValueError(f"Unknown dataset: {name}")

>>>>>>> ebb5a31 (Change settings)
def partition_non_iid(
    x: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    *,
    seed: int = 42,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
<<<<<<< HEAD
    Simple partitioning helper.
    Currently: shuffle indices and split into `num_clients` contiguous shards.
=======
    Stratified partitioning to ensure each client receives samples from each label.
    Returns {client_id: {'x':..., 'y':...}}.
>>>>>>> ebb5a31 (Change settings)
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")

    rng = np.random.default_rng(seed)
<<<<<<< HEAD
    indices = np.arange(len(y))
    rng.shuffle(indices)

    splits = np.array_split(indices, num_clients)
    clients: Dict[int, Dict[str, np.ndarray]] = {}
    for cid, idx in enumerate(splits):
        clients[cid] = {"x": x[idx], "y": y[idx]}
=======
    clients: Dict[int, Dict[str, np.ndarray]] = {cid: {"x": [], "y": []} for cid in range(num_clients)}

    unique_labels = np.unique(y)
    idx_by_lbl = {lbl: rng.permutation(np.where(y == lbl)[0]) for lbl in unique_labels}

    for lbl, indices in idx_by_lbl.items():
        splits = np.array_split(indices, num_clients)
        for cid, split in enumerate(splits):
            if split.size == 0:
                continue
            clients[cid]["x"].append(x[split])
            clients[cid]["y"].append(y[split])

    for cid in range(num_clients):
        if clients[cid]["x"]:
            clients[cid]["x"] = np.concatenate(clients[cid]["x"], axis=0)
            clients[cid]["y"] = np.concatenate(clients[cid]["y"], axis=0)
        else:
            clients[cid]["x"] = np.empty((0, x.shape[1]), dtype=x.dtype)
            clients[cid]["y"] = np.empty((0,), dtype=y.dtype)

>>>>>>> ebb5a31 (Change settings)
    return clients
