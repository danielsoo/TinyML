#!/usr/bin/env python3
"""
학습된 모델을 실제 상황처럼 정상:공격 = 9:1 테스트셋으로 시뮬레이션 평가합니다.
(테스트만 9:1로 맞추고, 모델은 그대로 사용)

Usage:
  python scripts/evaluate_9to1.py --config config/federated_local.yaml --model src/models/global_model.h5
  python scripts/evaluate_9to1.py --ratio 9
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on 9:1 (normal:attack) test set")
    parser.add_argument("--config", default="config/federated_local.yaml", help="Config YAML")
    parser.add_argument("--model", default="src/models/global_model.h5", help="Path to trained Keras model (.h5)")
    parser.add_argument("--ratio", type=float, default=9.0, help="Normal:Attack ratio (default 9 = 90%% normal, 10%% attack)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for test subsampling")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("name", "cicids2017")
    dataset_kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    if "path" in dataset_kwargs and "data_path" not in dataset_kwargs:
        dataset_kwargs["data_path"] = dataset_kwargs.pop("path")
    dataset_kwargs["balance_ratio"] = dataset_kwargs.get("balance_ratio")  # 학습용 균형은 그대로 (train만 영향)
    dataset_kwargs["binary"] = data_cfg.get("binary", True)

    from src.data.loader import load_dataset

    print("1. Loading dataset (same as training)...")
    x_train, y_train, x_test, y_test = load_dataset(dataset_name, **dataset_kwargs)
    n_test = len(y_test)
    n_normal = int(np.sum(y_test == 0))
    n_attack = int(np.sum(y_test == 1))
    print(f"   Test original: {n_test:,} (BENIGN={n_normal:,}, ATTACK={n_attack:,})")

    # 테스트셋을 정상:공격 = ratio:1 로 서브샘플 (공격은 전부 쓰고, 정상은 ratio*공격 수만큼만)
    rng = np.random.default_rng(args.seed)
    idx_attack = np.where(y_test == 1)[0]
    idx_normal = np.where(y_test == 0)[0]
    n_attack_keep = len(idx_attack)
    n_normal_target = int(n_attack_keep * args.ratio)
    if n_normal_target > len(idx_normal):
        n_normal_target = len(idx_normal)
        print(f"   Warning: Not enough normal samples for ratio {args.ratio}:1, using all normal -> actual ratio {len(idx_normal)/n_attack_keep:.2f}:1")
    keep_normal = rng.choice(idx_normal, size=n_normal_target, replace=False)
    eval_idx = np.concatenate([keep_normal, idx_attack])
    rng.shuffle(eval_idx)
    x_eval = x_test[eval_idx]
    y_eval = y_test[eval_idx]
    n_eval_normal = int(np.sum(y_eval == 0))
    n_eval_attack = int(np.sum(y_eval == 1))
    print(f"   Test 9:1 subset: {len(y_eval):,} (BENIGN={n_eval_normal:,}, ATTACK={n_eval_attack:,}) -> ratio {n_eval_normal/max(1,n_eval_attack):.1f}:1")

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\nModel not found: {model_path}")
        sys.exit(1)

    print(f"\n2. Loading model: {model_path}")
    import tensorflow as tf
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        if "Unknown" in str(e) or "custom" in str(e).lower():
            from src.models.nets import _focal_loss
            model = tf.keras.models.load_model(
                model_path, compile=False,
                custom_objects={"_focal_loss": _focal_loss, "loss_fn": _focal_loss()},
            )
        else:
            raise
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("\n3. Evaluating on 9:1 test set...")
    loss, acc = model.evaluate(x_eval, y_eval, verbose=1)
    y_prob = model.predict(x_eval, verbose=0)
    if y_prob.ndim > 1:
        y_prob = y_prob[:, -1]
    y_pred = (y_prob >= 0.5).astype(np.int32)

    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

    prec = precision_score(y_eval, y_pred, zero_division=0)
    rec = recall_score(y_eval, y_pred, zero_division=0)
    f1 = f1_score(y_eval, y_pred, zero_division=0)
    prec_per = precision_score(y_eval, y_pred, average=None, zero_division=0)
    rec_per = recall_score(y_eval, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_eval, y_pred)

    print("\n--- Result (9:1 simulation) ---")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Normal Recall (of actual normal, %% predicted as normal): {rec_per[0]:.4f}")
    print(f"  Normal Precision (of predicted normal, %% actually normal): {prec_per[0]:.4f}")
    print("  Confusion matrix (rows=true, cols=pred):")
    print("             pred_Normal  pred_Attack")
    print(f"    true_Normal    {cm[0,0]:>6}      {cm[0,1]:>6}")
    print(f"    true_Attack    {cm[1,0]:>6}      {cm[1,1]:>6}")
    print("\nDone.")


if __name__ == "__main__":
    main()
