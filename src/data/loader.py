from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# -------------------------
# 공통 유틸
# -------------------------

def _ensure_dir(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {p.resolve()}")
    return p


def partition_non_iid(
    x: np.ndarray,
    y: np.ndarray,
    num_clients: int,
) -> List[Dict[str, np.ndarray]]:
    """라벨 분포를 최대한 유지하면서 num_clients개로 나누기."""
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
            # 혹시 비면 랜덤 샘플 조금 넣어줌
            ridx = rng.choice(len(x), size=len(x) // num_clients, replace=False)
            x_c = x[ridx]
            y_c = y[ridx]

        parts.append({"x": x_c, "y": y_c})

    return parts


# -------------------------
# MNIST (테스트용)
# -------------------------

def load_mnist(max_samples: int = None, **_) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # CNN용 채널 차원 추가
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    if max_samples is not None:
        x_train = x_train[:max_samples]
        y_train = y_train[:max_samples]

    return x_train, y_train, x_test, y_test


# -------------------------
# Bot-IoT (CSV 기반, 이진 분류 가정)
# -------------------------

def load_bot_iot(
    data_path: str = "data/raw/Bot-IoT",
    max_samples: int = None,
    test_size: float = 0.2,
    random_state: int = 42,
    label_col: str = "label",
    **_,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bot-IoT CSV들을 읽어서 (x_train, y_train, x_test, y_test) 반환.

    전제:
    - data_path 아래에 .csv 파일들이 있음
    - label_col 컬럼에 0/1 또는 정수 라벨이 들어 있음
      (0=정상, 1=공격 형태로 쓰면 됨)
    """

    root = _ensure_dir(data_path)
    csv_files = sorted([p for p in root.glob("*.csv")])
    if not csv_files:
        raise FileNotFoundError(f"Bot-IoT CSV 파일을 찾을 수 없습니다: {root.resolve()}")

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
            f"라벨 컬럼을 찾을 수 없습니다. candidates={label_candidates}. CSV에 라벨 컬럼 이름을 확인하세요."
        )

    if resolved_label_col != label_col:
        print(f"[load_bot_iot] 라벨 컬럼을 '{label_col}' 대신 '{resolved_label_col}'로 사용합니다.")

    label_col = resolved_label_col

    y = df[label_col].values
    X = df.drop(columns=[label_col]).values

    # 정수 인코딩 보정
    # (0/1이 아니어도 고유값을 0..C-1로 매핑)
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
# public API
# -------------------------

def load_dataset(name: str, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """config에서 온 name과 인자로 실제 로더 호출."""

    # path -> data_path로 매핑 (yaml이 path로 적혀 있어도 동작하도록)
    if "path" in kwargs and "data_path" not in kwargs:
        kwargs["data_path"] = kwargs.pop("path")

    if name.lower() in ["mnist"]:
        return load_mnist(**kwargs)
    elif name.lower() in ["bot_iot", "bot-iot", "botiot"]:
        return load_bot_iot(**kwargs)
    else:
        raise ValueError(f"지원하지 않는 데이터셋입니다: {name}")

