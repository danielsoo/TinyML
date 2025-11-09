# /src/federated/client.py
from __future__ import annotations
from typing import Dict, Tuple, Optional, Any
import argparse
import numpy as np
import flwr as fl
import yaml
from pathlib import Path
from src.data.loader import load_dataset, partition_non_iid
from src.models.nets import make_small_cnn, make_mlp
from flwr.common import parameters_to_ndarrays, Context
from src.tinyml.export_tflite import export_tflite

CFG = yaml.safe_load(Path("config/federated.yaml").read_text())
SIM_STATE: Dict[str, Any]

class SaveModelStrategy(fl.server.strategy.FedAvg):
    """FedAvg strategy that keeps the latest aggregated parameters."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latest_parameters: Optional[fl.common.Parameters] = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters
        return aggregated_parameters, aggregated_metrics

class KerasClient(fl.client.NumPyClient):
    def __init__(self, x_train, y_train, x_test, y_test, num_classes: int):
        input_shape = x_train.shape[1:]
        
        # 데이터 형태에 따라 모델 선택
        # 2D/3D shape (이미지): CNN 사용
        # 1D shape (tabular): MLP 사용
        if len(input_shape) > 1:
            self.model = make_small_cnn(input_shape=input_shape, num_classes=num_classes)
        else:
            # Tabular 데이터 (Bot-IoT 등)
            self.model = make_mlp(input_shape=input_shape, num_classes=num_classes)
        
        self.num_classes = num_classes
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train, self.y_train,
            epochs=int(config["local_epochs"]),
            batch_size=int(config["batch_size"]),
            verbose=0,
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
<<<<<<< HEAD
        # 예측 결과 계산 (상세 통계용)
=======
        y_true = self.y_test.astype(int)
>>>>>>> ebb5a31 (Change settings)
        y_prob = self.model.predict(self.x_test, verbose=0)
        if y_prob.ndim == 2 and y_prob.shape[1] == 1:
            y_prob = y_prob.ravel()
        
<<<<<<< HEAD
        if self.num_classes == 2:
            if y_prob.ndim > 1:
                # 안전장치 - sigmoid 출력이 2D일 경우 첫 번째 컬럼만 사용
=======
        if self.num_classes <= 2:
            # 이진 분류
            if y_prob.ndim > 1:
>>>>>>> ebb5a31 (Change settings)
                y_prob_flat = y_prob[:, 0]
            else:
                y_prob_flat = y_prob
            y_pred_classes = (y_prob_flat >= 0.5).astype(int)
        else:
<<<<<<< HEAD
            # 다중 분류
            if y_prob.ndim == 1:
                raise ValueError("Expected 2D probabilities for multi-class output.")
=======
            if y_prob.ndim == 1:
                raise ValueError("Expected 2D probability array for multi-class output.")
>>>>>>> ebb5a31 (Change settings)
            y_pred_classes = np.argmax(y_prob, axis=1)
        
        y_true = self.y_test.astype(int)
        # 상세 통계
        total = len(y_true)
        attack_actual = int(np.sum(y_true == 1))
        normal_actual = int(np.sum(y_true == 0))
        attack_predicted = int(np.sum(y_pred_classes == 1))
        normal_predicted = int(np.sum(y_pred_classes == 0))
        
        # Confusion matrix 계산
        true_positives = int(np.sum((y_true == 1) & (y_pred_classes == 1)))
        true_negatives = int(np.sum((y_true == 0) & (y_pred_classes == 0)))
        false_positives = int(np.sum((y_true == 0) & (y_pred_classes == 1)))
        false_negatives = int(np.sum((y_true == 1) & (y_pred_classes == 0)))
        
        metrics = {
            "accuracy": float(acc),
            "loss": float(loss),
            "total_samples": total,
            "actual_attack": attack_actual,
            "actual_normal": normal_actual,
            "predicted_attack": attack_predicted,
            "predicted_normal": normal_predicted,
            "true_positives": true_positives,
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }
        
        return float(loss), total, metrics

def simulate_clients() -> Dict[str, Any]:
<<<<<<< HEAD
    data_cfg = CFG["data"]
    test_size = 1.0 - data_cfg.get("train_split", 0.8)
    x_train, y_train, x_test, y_test = load_dataset(
        data_cfg["name"],
        data_path=data_cfg.get("path", "data/raw/Bot-IoT"),
        max_samples=data_cfg.get("max_samples"),
        test_size=test_size,
        random_state=data_cfg.get("random_state", 42),
    )
    parts = partition_non_iid(
        x_train,
        y_train,
        num_clients=data_cfg["num_clients"],
        seed=data_cfg.get("random_state", 42),
    )
=======
    x_train, y_train, x_test, y_test = load_dataset(CFG["data"]["name"])
    parts = partition_non_iid(x_train, y_train, num_clients=CFG["data"]["num_clients"])
    unique_labels = np.unique(y_train)
    is_binary = set(unique_labels).issubset({0, 1})
    num_classes = 2 if is_binary else len(unique_labels)
>>>>>>> ebb5a31 (Change settings)
    metadata = {
        "num_classes": int(num_classes),
        "input_shape": x_train.shape[1:],
        "is_binary": is_binary,
    }
    return {
        "evaluation": (x_test, y_test),
        "partitions": parts,
        "metadata": metadata,
    }

def client_fn(context: Context):
    """Create a client instance for a given Flower context.

    Flower 버전에 따라 context 안에서 cid를 주는 방식이 달라져서,
    방어적으로 꺼내고, 우리 쪽 파티션 개수 안으로 매핑해서 사용한다.
    """

    # 시뮬레이션 상태에서 데이터/메타데이터 가져오기
    parts = SIM_STATE["partitions"]
    x_test, y_test = SIM_STATE["evaluation"]
    num_classes = SIM_STATE["metadata"]["num_classes"]

    # 1) 여러 후보 위치에서 cid 시도
    cid_value = getattr(context, "client_id", None)

    if cid_value is None:
        cid_value = getattr(context, "node_id", None)

    if cid_value is None and hasattr(context, "node_config"):
        cid_value = (
            context.node_config.get("cid")
            or context.node_config.get("client_id")
            or context.node_config.get("partition-id")
        )

    if cid_value is None:
        props = getattr(context, "properties", None)
        if isinstance(props, dict):
            cid_value = (
                props.get("cid")
                or props.get("client_id")
                or props.get("partition-id")
            )

    # 2) cid를 정수로 변환 (안 되면 0으로)
    cid_int = 0
    if cid_value is not None:
        try:
            cid_int = int(cid_value)
        except (TypeError, ValueError):
            if isinstance(cid_value, str):
                digits = "".join(ch for ch in cid_value if ch.isdigit())
                if digits:
                    cid_int = int(digits)
            # 그래도 안 되면 그냥 0 유지

    # 3) 현재 파티션 개수 범위 안으로 매핑
    num_clients = len(parts)
    if num_clients == 0:
        raise RuntimeError("No client partitions available in SIM_STATE['partitions'].")

    cid_int = cid_int % num_clients  # 핵심: 이상한 큰 숫자도 0~num_clients-1로 변환

    # 4) 해당 파티션으로 KerasClient 생성
    data = parts[cid_int]
    client = KerasClient(
        data["x"],
        data["y"],
        x_test,
        y_test,
        num_classes=num_classes,
    )

    return client.to_client()

def start_simulation(save_path: Optional[str] = None):
    def evaluate_metrics_aggregation_fn(results):
        """평가 결과를 집계하고 상세 정보 출력"""
        if not results:
            return {}
        
        # 모든 클라이언트의 메트릭 집계
        all_metrics = [m[1] for m in results]
        
        aggregated = {
            "accuracy": float(np.mean([m["accuracy"] for m in all_metrics])),
            "loss": float(np.mean([m["loss"] for m in all_metrics])),
        }
        
        # 상세 통계 (첫 번째 클라이언트 기준)
        if all_metrics and "total_samples" in all_metrics[0]:
            first_metrics = all_metrics[0]
            aggregated.update({
                "total_samples": first_metrics.get("total_samples", 0),
                "actual_attack": first_metrics.get("actual_attack", 0),
                "actual_normal": first_metrics.get("actual_normal", 0),
                "predicted_attack": first_metrics.get("predicted_attack", 0),
                "predicted_normal": first_metrics.get("predicted_normal", 0),
                "true_positives": first_metrics.get("true_positives", 0),
                "true_negatives": first_metrics.get("true_negatives", 0),
                "false_positives": first_metrics.get("false_positives", 0),
                "false_negatives": first_metrics.get("false_negatives", 0),
            })
            
            # 상세 정보 출력
            print("\n" + "="*60)
            print("📊 Evaluation Summary")
            print("="*60)
            print(f"Accuracy: {aggregated['accuracy']:.4f} ({aggregated['accuracy']*100:.2f}%)")
            print(f"Loss: {aggregated['loss']:.4f}")
            print(f"\n📈 Ground Truth:")
            print(f"  - Attack samples: {aggregated['actual_attack']}")
            print(f"  - Normal samples: {aggregated['actual_normal']}")
            print(f"  - Total samples: {aggregated['total_samples']}")
            print(f"\n🔮 Predictions:")
            print(f"  - Predicted attack: {aggregated['predicted_attack']}")
            print(f"  - Predicted normal: {aggregated['predicted_normal']}")
            print(f"\n✅ Confusion Matrix:")
            print(f"  - True Positives (TP): {aggregated['true_positives']}")
            print(f"  - True Negatives (TN): {aggregated['true_negatives']}")
            print(f"  - False Positives (FP): {aggregated['false_positives']}")
            print(f"  - False Negatives (FN): {aggregated['false_negatives']}")
            
            # Derive precision/recall metrics
            tp = aggregated['true_positives']
            fp = aggregated['false_positives']
            fn = aggregated['false_negatives']
            
            if tp + fp > 0:
                precision = tp / (tp + fp)
                print(f"\n📏 Metrics:")
                print(f"  - Precision: {precision:.4f} ({precision*100:.2f}%)")
            else:
                precision = None
            
            if tp + fn > 0:
                recall = tp / (tp + fn)
                if precision is None:
                    print(f"\n📏 Metrics:")
                print(f"  - Recall: {recall:.4f} ({recall*100:.2f}%)")
            else:
                recall = None
            
            if precision is not None and recall is not None and (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                print(f"  - F1-Score: {f1:.4f} ({f1*100:.2f}%)")
            
            print("="*60 + "\n")
        
        return aggregated
    
    def fit_config_fn(server_round: int):
        return {
            "local_epochs": CFG["client"]["local_epochs"],
            "batch_size": CFG["client"]["batch_size"],
        }

    strategy = SaveModelStrategy(
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        on_fit_config_fn=fit_config_fn,
    )
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=CFG["data"]["num_clients"],
        config=fl.server.ServerConfig(num_rounds=CFG["server"]["rounds"]),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )

    if save_path and strategy.latest_parameters is not None:
        metadata = SIM_STATE["metadata"]
        input_shape = metadata["input_shape"]
        num_classes = metadata["num_classes"]

        if len(input_shape) > 1:
            model = make_small_cnn(input_shape=input_shape, num_classes=num_classes)
        else:
            model = make_mlp(input_shape=input_shape, num_classes=num_classes)

        weights = parameters_to_ndarrays(strategy.latest_parameters)
        model.set_weights(weights)

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.suffix == ".tflite":
            export_tflite(model, str(save_path))
        elif save_path.suffix == ".h5":
            model.save(str(save_path))
            print(f"✅ Saved global model to {save_path}")
        else:
            np.savez(str(save_path), *weights)
            print(f"✅ Saved raw weights (NumPy .npz) to {save_path}")

    return history

def main():
    parser = argparse.ArgumentParser(description="Flower Federated Simulation")
    parser.add_argument("--save-model", type=str, default=None, help="Path to export the aggregated global model (.h5, .tflite, or .npz).")
    args = parser.parse_args()

    global SIM_STATE
    SIM_STATE = simulate_clients()
    start_simulation(save_path=args.save_model)


if __name__ == "__main__":
    main()
