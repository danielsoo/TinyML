# /src/federated/client.py
from __future__ import annotations
from typing import Dict, Tuple, Optional
import argparse
import numpy as np
import flwr as fl
import yaml
from pathlib import Path
from src.data.loader import load_dataset, partition_non_iid
from src.models.nets import make_small_cnn, make_mlp
from flwr.common import parameters_to_ndarrays
from src.tinyml.export_tflite import export_tflite

CFG = yaml.safe_load(Path("config/federated.yaml").read_text())
SIM_STATE: Tuple[Tuple[np.ndarray, ...], Dict[int, Dict[str, np.ndarray]]]

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
    def __init__(self, x_train, y_train, x_test, y_test):
        input_shape = x_train.shape[1:]
        num_classes = len(np.unique(y_train))
        
        # 데이터 형태에 따라 모델 선택
        # 2D/3D shape (이미지): CNN 사용
        # 1D shape (tabular): MLP 사용
        if len(input_shape) > 1:
            self.model = make_small_cnn(input_shape=input_shape, num_classes=num_classes)
        else:
            # Tabular 데이터 (Bot-IoT 등)
            self.model = make_mlp(input_shape=input_shape, num_classes=num_classes)
        
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
        
        # 예측 결과 계산 (상세 통계용)
        y_pred = self.model.predict(self.x_test, verbose=0)
        
        # 이진 분류인 경우 (출력 shape이 (None, 1))
        if y_pred.shape[1] == 1:
            y_pred_classes = (y_pred > 0.5).astype(int).flatten()
        else:
            # 다중 분류
            y_pred_classes = np.argmax(y_pred, axis=1)
        
        # 상세 통계
        total = len(self.y_test)
        attack_actual = int(np.sum(self.y_test == 1))
        normal_actual = int(np.sum(self.y_test == 0))
        attack_predicted = int(np.sum(y_pred_classes == 1))
        normal_predicted = int(np.sum(y_pred_classes == 0))
        
        # Confusion matrix 계산
        true_positives = int(np.sum((self.y_test == 1) & (y_pred_classes == 1)))
        true_negatives = int(np.sum((self.y_test == 0) & (y_pred_classes == 0)))
        false_positives = int(np.sum((self.y_test == 0) & (y_pred_classes == 1)))
        false_negatives = int(np.sum((self.y_test == 1) & (y_pred_classes == 0)))
        
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

def simulate_clients() -> Tuple[Tuple[np.ndarray, ...], Dict[int, Dict[str, np.ndarray]]]:
    x_train, y_train, x_test, y_test = load_dataset(CFG["data"]["name"])
    parts = partition_non_iid(x_train, y_train, num_clients=CFG["data"]["num_clients"])
    return (x_test, y_test), parts

def client_fn(cid: str):
    (x_test, y_test), parts = SIM_STATE
    cid_int = int(cid)
    data = parts[cid_int]
    return KerasClient(data["x"], data["y"], x_test, y_test)

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
    
    strategy = SaveModelStrategy(
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
    )
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=CFG["data"]["num_clients"],
        config=fl.server.ServerConfig(num_rounds=CFG["server"]["rounds"]),
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )

    if save_path and strategy.latest_parameters is not None:
        (x_test, y_test), parts = SIM_STATE
        sample_client = next(iter(parts.values()))
        input_shape = sample_client["x"].shape[1:]
        num_classes = len(np.unique(sample_client["y"]))

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
