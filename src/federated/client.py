from pathlib import Path
from typing import Dict, Any, Tuple

import flwr as fl
import numpy as np
import yaml

from src.data.loader import load_dataset, partition_non_iid
from src.models import nets


# ------------------------------------------------
# 설정 불러오기
# ------------------------------------------------

CFG_PATH = Path("config/federated.yaml")
if not CFG_PATH.exists():
    raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {CFG_PATH.resolve()}")

with CFG_PATH.open() as f:
    CFG = yaml.safe_load(f)


def _build_model(model_name: str, input_shape: Tuple[int, ...], num_classes: int):
    """모델 이름에 따라 생성. nets 모듈에 get_model이 없으면 기본 함수로 대체."""
    if hasattr(nets, "get_model"):
        return nets.get_model(model_name, input_shape, num_classes)

    name = (model_name or "mlp").lower()
    if name in ["cnn", "small_cnn"] and hasattr(nets, "make_small_cnn"):
        return nets.make_small_cnn(input_shape, num_classes)
    if hasattr(nets, "make_mlp"):
        return nets.make_mlp(input_shape, num_classes)

    raise AttributeError(
        "모델 생성 함수를 찾을 수 없습니다. 'get_model' 또는 'make_mlp' 정의를 확인하세요."
    )


# ------------------------------------------------
# Flower NumPyClient
# ------------------------------------------------

class KerasClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, cid: int):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.cid = cid

    def get_parameters(self, config: Dict[str, Any]):
        return self.model.get_weights()

    def fit(self, parameters, config: Dict[str, Any]):
        self.model.set_weights(parameters)

        epochs = int(config.get("local_epochs", 1))
        batch_size = int(config.get("batch_size", 32))

        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )

        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config: Dict[str, Any]):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return float(loss), len(self.x_test), {"accuracy": float(acc)}


# ------------------------------------------------
# 시뮬레이션 준비
# ------------------------------------------------

def simulate_clients():
    data_cfg = CFG.get("data", {})
    fed_cfg = CFG.get("federated", {})
    model_cfg = CFG.get("model", {})

    dataset_name = data_cfg.get("name", "bot_iot")

    # federated.yaml에서 name, num_clients 제외한 나머지는 그대로 데이터 로더에 전달
    dataset_kwargs = {
        k: v
        for k, v in data_cfg.items()
        if k not in {"name", "num_clients"}
    }

    num_clients = int(data_cfg.get("num_clients", 1))

    try:
        x_train, y_train, x_test, y_test = load_dataset(dataset_name, **dataset_kwargs)
    except FileNotFoundError as err:
        data_path = dataset_kwargs.get("data_path") or dataset_kwargs.get("path")
        hint = f" (확인한 경로: {data_path})" if data_path else ""
        raise FileNotFoundError(
            f"데이터셋 '{dataset_name}'을(를) 불러오지 못했습니다{hint}. "
            f"Colab이라면 해당 경로에 CSV 파일이 있는지 확인해 주세요."
        ) from err

    # num_classes 계산
    unique_labels = np.unique(np.concatenate([y_train, y_test]))
    num_classes = len(unique_labels)

    # 0/1 이진 분류인데 실수로 한쪽만 있는 경우 방어 코드
    if num_classes == 1 and 0 in unique_labels:
        num_classes = 2

    # 입력 shape 정리 (2D면 MLP용)
    if x_train.ndim == 2:
        input_shape = (x_train.shape[1],)
    else:
        input_shape = x_train.shape[1:]

    # 데이터 파티션 나누기
    train_parts = partition_non_iid(x_train, y_train, num_clients)
    test_parts = partition_non_iid(x_test, y_test, num_clients)

    model_name = model_cfg.get("name", "mlp")

    # client_fn 안에서 쓸 상태
    state = {
        "num_clients": num_clients,
        "input_shape": input_shape,
        "num_classes": num_classes,
        "model_name": model_name,
        "train_parts": train_parts,
        "test_parts": test_parts,
    }

    def client_fn(context: fl.common.Context) -> fl.client.Client:
        # Flower/Ray가 주는 cid를 안전하게 인덱스로 매핑
        raw_cid = None

        # VCE / 새로운 API
        if hasattr(context, "node_config") and isinstance(context.node_config, dict):
            raw_cid = context.node_config.get("cid", None)

        # 레거시
        if raw_cid is None and hasattr(context, "cid"):
            raw_cid = context.cid

        try:
            cid_int = int(raw_cid)
        except (TypeError, ValueError):
            cid_int = hash(str(raw_cid))

        idx = cid_int % state["num_clients"]

        part_tr = state["train_parts"][idx]
        part_te = state["test_parts"][idx]

        model = _build_model(
            state["model_name"],
            state["input_shape"],
            state["num_classes"],
        )

        return KerasClient(
            model,
            part_tr["x"],
            part_tr["y"],
            part_te["x"],
            part_te["y"],
            cid=idx,
        )

    return state, client_fn


# ------------------------------------------------
# 실행
# ------------------------------------------------

def main(save_path: str = "src/models/global_model.h5"):
    (state, client_fn) = simulate_clients()
    fed_cfg = CFG.get("federated", {})

    num_rounds = int(fed_cfg.get("num_rounds", 3))
    batch_size = int(fed_cfg.get("batch_size", 32))
    local_epochs = int(fed_cfg.get("local_epochs", 1))

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=fed_cfg.get("fraction_fit", 1.0),
        fraction_evaluate=fed_cfg.get("fraction_evaluate", 1.0),
        min_fit_clients=fed_cfg.get("min_fit_clients", state["num_clients"]),
        min_evaluate_clients=fed_cfg.get("min_evaluate_clients", state["num_clients"]),
        min_available_clients=fed_cfg.get("min_available_clients", state["num_clients"]),
        on_fit_config_fn=lambda rnd: {
            "batch_size": batch_size,
            "local_epochs": local_epochs,
        },
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=state["num_clients"],
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # 글로벌 모델 저장
    # 마지막 round weight 사용
    # start_simulation이 strategy.parameters 에 최종값 채워놨다고 가정
    global_model = _build_model(
        state["model_name"],
        state["input_shape"],
        state["num_classes"],
    )
    if strategy.parameters is not None:
        from flwr.common import parameters_to_ndarrays

        weights = parameters_to_ndarrays(strategy.parameters)
        global_model.set_weights(weights)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        global_model.save(save_path)
        print(f"✅ Saved global model to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-model",
        type=str,
        default="src/models/global_model.h5",
        help="Path to save the global model",
    )
    args = parser.parse_args()
    main(save_path=args.save_model)

