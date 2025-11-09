# /src/federated/server.py
from __future__ import annotations
import flwr as fl
import yaml
from pathlib import Path

CFG = yaml.safe_load(Path("config/federated.yaml").read_text())

def on_fit_config(server_round: int):
    return {
        "local_epochs": CFG["client"]["local_epochs"],
        "batch_size": CFG["client"]["batch_size"],
    }

def main():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        # 모든 클라이언트에서 훈련
        fraction_evaluate=1.0,
        # 모든 클라이언트에서 평가
        min_fit_clients=CFG["server"]["min_available_clients"],
        # 최소 클라이언트 수
        min_available_clients=CFG["server"]["min_available_clients"],
        # 훈련 설정 함수
        on_fit_config_fn=on_fit_config,
        # 평가 설정 함수
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=CFG["server"]["rounds"]),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
