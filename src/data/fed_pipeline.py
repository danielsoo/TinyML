"""
Corrected federated data pipeline ("v13"), written after the audit in docs/AUDIT.md.

Differences from src/data/loader.py (+ partition_data), all aimed at keeping client
boundaries and the final test set intact:
  * every record keeps a stable id (source file, row number) and its metadata
    (TON_IoT: ts, src_ip, dst_ip, src_port; CIC-IDS2017: source file / capture day);
  * exact duplicates are removed on the feature vector; groups whose identical
    feature vectors carry different labels are dropped and counted (conflict_policy);
  * the final test split is drawn first (fresh split_seed, or a temporal split);
  * the remaining pool is assigned to clients BEFORE any fitted preprocessing;
  * each client holds out a real validation slice (never trained on) for threshold
    selection;
  * class balancing (majority undersampling) is done per client on its training slice;
  * no SMOTE (it interpolated label-encoded categories and mixed client records);
  * the scaler uses the mean / variance of the union of client training slices, i.e.
    the aggregate a federation can compute from per-client sums;
  * a manifest (ids per partition, hashes, counts, config) is returned for provenance.

Usage:
    fd = build_federated("ton_iot", data_path="data/raw/TON_IoT", num_clients=4, ...)
    fd.clients[k].x_train / y_train / x_val / y_val, fd.test.x / y, fd.manifest
"""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

TON_META_COLUMNS = ["ts", "src_ip", "dst_ip", "src_port"]
TON_MAX_CATEGORIES = 50


# --------------------------------------------------------------------------- raw loading
@dataclass
class RawData:
    X: np.ndarray                 # float32 features (unscaled)
    y: np.ndarray                 # 0 benign / 1 attack
    attack: np.ndarray            # attack family string ("BENIGN"/"normal" for benign)
    ids: np.ndarray               # "file:row"
    meta: pd.DataFrame            # per-record metadata (not model inputs)
    feature_names: List[str]
    categorical: List[str] = field(default_factory=list)  # label-encoded columns
    info: Dict[str, Any] = field(default_factory=dict)


def load_raw_cic(data_path: str) -> RawData:
    files = sorted(Path(data_path).glob("*.pcap_ISCX.csv"))
    if not files:
        raise FileNotFoundError(f"No *.pcap_ISCX.csv under {data_path}")
    dfs, ids, days = [], [], []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception:
            df = pd.read_csv(f, engine="python")
        dfs.append(df)
        ids.append(np.char.add(f"{f.name}:", np.arange(len(df)).astype(str)))
        days.append(np.full(len(df), f.name.split("-")[0]))
    df = pd.concat(dfs, ignore_index=True)
    label_col = df.columns[-1]
    attack = df[label_col].astype(str).str.strip().values
    df = df.drop(columns=[label_col])
    df.columns = df.columns.str.strip()
    n_inf = int(np.isinf(df.select_dtypes("number").values).sum())
    df = df.replace([np.inf, -np.inf], np.nan)
    nan_cols = df.columns[df.isna().all()].tolist()
    df = df.drop(columns=nan_cols)
    n_nan = int(df.isna().values.sum())
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(0)
    y = (attack != "BENIGN").astype(np.int64)
    meta = pd.DataFrame({"day": np.concatenate(days)})
    return RawData(df.values.astype("float32"), y, attack, np.concatenate(ids), meta,
                   list(df.columns), [],
                   {"inf_values_replaced_by_0": n_inf, "nan_values_replaced_by_0": n_nan,
                    "dropped_all_nan_columns": nan_cols, "files": [f.name for f in files]})


def load_raw_ton(data_path: str, text_features: bool = True) -> RawData:
    root = Path(data_path)
    files = sorted(p for p in root.rglob("*.csv") if "network" in p.name.lower()) or sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No TON_IoT CSV under {data_path}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
        parts = [pd.read_csv(p, low_memory=False) for p in files]
    ids = np.concatenate([np.char.add(f"{p.name}:", np.arange(len(d)).astype(str)) for p, d in zip(files, parts)])
    df = pd.concat(parts, ignore_index=True)
    y = (pd.to_numeric(df["label"], errors="coerce").values != 0).astype(np.int64)
    attack = df["type"].astype(str).values
    meta = df[[c for c in TON_META_COLUMNS if c in df.columns]].copy()
    feat = df.drop(columns=[c for c in ["label", "type"] + TON_META_COLUMNS if c in df.columns])
    encoded, dropped, extra = [], [], pd.DataFrame(index=feat.index)
    for col in list(feat.columns):
        if pd.api.types.is_numeric_dtype(feat[col]):
            continue
        as_num = pd.to_numeric(feat[col].replace("-", np.nan), errors="coerce")
        present = feat[col].replace("-", np.nan).notna()
        if present.sum() and as_num[present].notna().all():
            feat[col] = as_num
        elif feat[col].nunique(dropna=False) <= TON_MAX_CATEGORIES:
            feat[col] = pd.factorize(feat[col].astype(str), sort=True)[0]
            encoded.append(col)
        else:
            if text_features:
                text = feat[col].astype(str)
                missing = text.isin(["-", "", "nan"]).values
                extra[f"{col}_len"] = np.where(missing, 0, text.str.len())
                extra[f"{col}_digits"] = np.where(missing, 0, text.str.count(r"[0-9]"))
                extra[f"{col}_special"] = np.where(missing, 0, text.str.count(r"[^A-Za-z0-9]"))
            feat = feat.drop(columns=[col])
            dropped.append(col)
    feat = feat.apply(pd.to_numeric, errors="coerce").fillna(0)
    base_names = list(feat.columns)
    X = feat.values.astype("float32")
    if extra.shape[1]:
        X = np.hstack([X, extra.values.astype("float32")])
    return RawData(X, y, attack, ids, meta, base_names + list(extra.columns), encoded,
                   {"files": [p.name for p in files], "dropped_text": dropped,
                    "n_base_features": len(base_names)})


def load_raw(name: str, data_path: str, **kw) -> RawData:
    n = name.lower()
    if n in {"cicids2017", "cic-ids-2017", "cic_ids_2017"}:
        return load_raw_cic(data_path)
    if n in {"ton_iot", "toniot"}:
        return load_raw_ton(data_path, text_features=kw.get("text_features", True))
    raise ValueError(name)


# --------------------------------------------------------------------------- cleaning
def deduplicate(raw: RawData, key_cols: Optional[int] = None, conflict_policy: str = "drop") -> Dict[str, Any]:
    """Remove exact duplicate feature vectors (keyed on the first key_cols columns).
    conflict_policy: "drop" removes every record of a feature vector that occurs with
    both labels; "keep" keeps the first record per (vector, label)."""
    key = raw.X if key_cols is None else raw.X[:, :key_cols]
    # 64-bit row hash of the exact float32 vector (collisions are negligible at this scale)
    h = pd.util.hash_pandas_object(pd.DataFrame(np.ascontiguousarray(key)), index=False).values
    gid = pd.factorize(h)[0]
    g = pd.DataFrame({"gid": gid, "y": raw.y})
    nlab = g.groupby("gid")["y"].nunique()
    conflict_groups = set(nlab.index[nlab > 1])
    conflict_mask = np.isin(gid, list(conflict_groups)) if conflict_groups else np.zeros(len(gid), bool)
    first = ~pd.DataFrame({"gid": gid, "y": raw.y}).duplicated().values
    keep = first & ~conflict_mask if conflict_policy == "drop" else first
    stats = {"records": int(len(gid)), "unique_vectors": int(gid.max() + 1),
             "duplicate_records_removed": int((~first).sum()),
             "conflicting_label_vectors": int(len(conflict_groups)),
             "conflicting_label_records": int(conflict_mask.sum()),
             "conflict_policy": conflict_policy, "kept": int(keep.sum())}
    for a in ("X", "y", "attack", "ids"):
        setattr(raw, a, getattr(raw, a)[keep])
    raw.meta = raw.meta.iloc[np.where(keep)[0]].reset_index(drop=True)
    return stats


# --------------------------------------------------------------------------- splitting
def split_test(raw: RawData, mode: str, test_size: float, seed: int) -> np.ndarray:
    """Boolean mask of the final test records."""
    n = len(raw.y)
    if mode == "random":
        rng = np.random.default_rng(seed)
        test = np.zeros(n, bool)
        for c in (0, 1):
            idx = np.where(raw.y == c)[0]
            rng.shuffle(idx)
            test[idx[: int(round(test_size * len(idx)))]] = True
        return test
    if mode == "temporal":
        ts = pd.to_numeric(raw.meta["ts"], errors="coerce").values
        cut = np.nanquantile(ts, 1.0 - test_size)
        return ts > cut
    raise ValueError(mode)


def assign_clients(y: np.ndarray, num_clients: int, strategy: str, alpha: float, seed: int,
                   groups: Optional[np.ndarray] = None) -> np.ndarray:
    """Client id per record of the training pool (before any preprocessing)."""
    rng = np.random.default_rng(seed)
    cid = np.empty(len(y), dtype=np.int64)
    if strategy in {"label_balanced", "iid"}:
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0])
            for k, part in enumerate(np.array_split(idx, num_clients)):
                cid[part] = k
    elif strategy == "dirichlet":
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0])
            p = rng.dirichlet(np.repeat(alpha, num_clients))
            for k, part in enumerate(np.split(idx, (np.cumsum(p) * len(idx)).astype(int)[:-1])):
                cid[part] = k
    elif strategy == "host":
        # whole host groups go to one client (largest-first greedy balancing by record count)
        uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
        load = np.zeros(num_clients)
        owner = np.empty(len(uniq), dtype=np.int64)
        for g in np.argsort(-counts):
            owner[g] = int(np.argmin(load))
            load[owner[g]] += counts[g]
        cid = owner[inv]
    else:
        raise ValueError(strategy)
    return cid


def _hash_ids(ids: np.ndarray) -> str:
    h = hashlib.sha256()
    for s in np.sort(ids.astype(str)):
        h.update(s.encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


@dataclass
class Part:
    x: np.ndarray
    y: np.ndarray
    attack: np.ndarray
    ids: np.ndarray


@dataclass
class ClientData:
    train: Part
    val: Part


@dataclass
class FederatedData:
    clients: List[ClientData]
    test: Part
    test_meta: pd.DataFrame
    feature_names: List[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    manifest: Dict[str, Any]

    def pooled_val(self) -> Part:
        vs = [c.val for c in self.clients]
        return Part(np.concatenate([v.x for v in vs]), np.concatenate([v.y for v in vs]),
                    np.concatenate([v.attack for v in vs]), np.concatenate([v.ids for v in vs]))

    def save_manifest(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"test_ids": self.test.ids.astype(str), "scaler_mean": self.scaler_mean,
                  "scaler_scale": self.scaler_scale}
        for k, c in enumerate(self.clients):
            arrays[f"client{k}_train_ids"] = c.train.ids.astype(str)
            arrays[f"client{k}_val_ids"] = c.val.ids.astype(str)
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        path.with_suffix(".json").write_text(json.dumps(self.manifest, indent=2, default=str))


def build_federated(name: str, data_path: str, num_clients: int = 4,
                    partition_strategy: str = "label_balanced", dirichlet_alpha: float = 0.3,
                    split_mode: str = "random", test_size: float = 0.2, split_seed: int = 2027,
                    partition_seed: int = 42, val_frac: float = 0.1, balance_ratio: Optional[float] = 4.0,
                    conflict_policy: str = "drop", text_features: bool = True,
                    group_column: Optional[str] = None, **_) -> FederatedData:
    raw = load_raw(name, data_path, text_features=text_features)
    key_cols = raw.info.get("n_base_features")
    dedup = deduplicate(raw, key_cols=key_cols, conflict_policy=conflict_policy)

    test_mask = split_test(raw, split_mode, test_size, split_seed)
    pool = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]

    groups = raw.meta[group_column].astype(str).values[pool] if group_column else None
    cid = assign_clients(raw.y[pool], num_clients, partition_strategy, dirichlet_alpha, partition_seed, groups)

    rng = np.random.default_rng(partition_seed + 1)
    client_train_idx, client_val_idx, balance = [], [], []
    for k in range(num_clients):
        idx = pool[cid == k]
        # real validation slice, stratified, never used for training
        val = []
        for c in (0, 1):
            ic = rng.permutation(idx[raw.y[idx] == c])
            val.append(ic[: int(round(val_frac * len(ic)))])
        val = np.concatenate(val)
        tr = np.setdiff1d(idx, val, assume_unique=True)
        before = np.bincount(raw.y[tr], minlength=2)
        if balance_ratio and before.min() > 0:
            minority = int(np.argmin(before))
            maj = tr[raw.y[tr] != minority]
            cap = int(before[minority] * balance_ratio)
            if len(maj) > cap:
                maj = rng.choice(maj, size=cap, replace=False)
            tr = np.concatenate([tr[raw.y[tr] == minority], maj])
        tr = rng.permutation(tr)
        balance.append({"before": before.tolist(), "after": np.bincount(raw.y[tr], minlength=2).tolist()})
        client_train_idx.append(tr)
        client_val_idx.append(val)

    union = np.concatenate(client_train_idx)
    mean = raw.X[union].mean(axis=0).astype("float64")
    std = raw.X[union].std(axis=0).astype("float64")
    scale = np.where(std > 0, std, 1.0)

    def part(idx: np.ndarray) -> Part:
        x = ((raw.X[idx] - mean) / scale).astype("float32")
        return Part(x, raw.y[idx].astype(np.float32), raw.attack[idx], raw.ids[idx])

    clients = [ClientData(part(tr), part(va)) for tr, va in zip(client_train_idx, client_val_idx)]
    test = part(test_idx)

    def counts(idx):
        return {"n": int(len(idx)), "attack": int(raw.y[idx].sum()), "benign": int((raw.y[idx] == 0).sum())}

    manifest = {
        "pipeline": "v13", "dataset": name, "data_path": data_path,
        "config": {"num_clients": num_clients, "partition_strategy": partition_strategy,
                   "dirichlet_alpha": dirichlet_alpha, "split_mode": split_mode, "test_size": test_size,
                   "split_seed": split_seed, "partition_seed": partition_seed, "val_frac": val_frac,
                   "balance_ratio": balance_ratio, "conflict_policy": conflict_policy,
                   "text_features": text_features, "group_column": group_column, "smote": False,
                   "scaler": "mean/std of the union of client training slices (federated aggregate)"},
        "raw_info": raw.info, "dedup": dedup,
        "n_features": int(raw.X.shape[1]), "feature_names": raw.feature_names,
        "categorical_label_encoded": raw.categorical,
        "test": {**counts(test_idx), "hash": _hash_ids(raw.ids[test_idx]),
                 "attack_families": pd.Series(raw.attack[test_idx]).value_counts().to_dict()},
        "clients": [{"train": {**counts(tr), "hash": _hash_ids(raw.ids[tr])},
                     "val": {**counts(va), "hash": _hash_ids(raw.ids[va])},
                     "balance": b}
                    for tr, va, b in zip(client_train_idx, client_val_idx, balance)],
    }
    return FederatedData(clients, test, raw.meta.iloc[test_idx].reset_index(drop=True),
                         raw.feature_names, mean, scale, manifest)
