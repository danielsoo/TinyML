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
        # Train on all clients
        fraction_evaluate=1.0,
        # Evaluate on all clients
        min_fit_clients=CFG["server"]["min_available_clients"],
        # Minimum number of clients
        min_available_clients=CFG["server"]["min_available_clients"],
        # Training configuration function
        on_fit_config_fn=on_fit_config,
        # Evaluation configuration function
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=CFG["server"]["rounds"]),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
