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
        fraction_evaluate=1.0,
        min_fit_clients=CFG["server"]["min_available_clients"],
        min_available_clients=CFG["server"]["min_available_clients"],
        on_fit_config_fn=on_fit_config,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=CFG["server"]["rounds"]),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
