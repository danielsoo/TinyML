# /src/federated/server.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import flwr as fl
import numpy as np
import yaml
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

# Import quantization utilities
try:
    import tensorflow_model_optimization as tfmot
    HAS_TFMOT = True
except ImportError:
    HAS_TFMOT = False

from src.modelcompression.quantization import (
    QuantizationParams,
    calculate_quantization_params,
    quantize_array,
    dequantize_array,
)


def load_server_config(config_path: str = None) -> dict:
    """Load server configuration from YAML file."""
    if config_path is None:
        config_path = os.getenv("FEDERATED_CONFIG", "config/federated_local_sky.yaml")

    cfg_path = Path(config_path)
    if not cfg_path.exists():
        # Fallback paths
        for fallback in ["config/federated_local.yaml", "config/federated.yaml"]:
            if Path(fallback).exists():
                cfg_path = Path(fallback)
                break

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    return yaml.safe_load(cfg_path.read_text(encoding='utf-8'))


# Default config
CFG = None


def _get_config():
    global CFG
    if CFG is None:
        CFG = load_server_config()
    return CFG


class QATAwareStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg strategy with QAT (Quantization-Aware Training) support.

    When clients use QAT, their weights may need special handling:
    - Dequantize received weights for aggregation (if quantized)
    - Aggregate in float32 space
    - Re-quantize before sending back (optional)

    This strategy also stores the latest parameters for model saving.
    """

    def __init__(
        self,
        use_qat: bool = False,
        use_dequantize_aggregate: bool = True,
        quantize_before_send: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.use_qat = use_qat
        self.use_dequantize_aggregate = use_dequantize_aggregate
        self.quantize_before_send = quantize_before_send
        self.latest_parameters: Optional[Parameters] = None
        self.quant_params_cache: Dict[int, List[QuantizationParams]] = {}

        if use_qat:
            print("[Server] QAT-aware aggregation enabled")
            if use_dequantize_aggregate:
                print("[Server] Will dequantize client weights before aggregation")
            if quantize_before_send:
                print("[Server] Will quantize weights before sending to clients")

    def _dequantize_weights(
        self,
        weights: List[np.ndarray],
        client_id: int = 0
    ) -> List[np.ndarray]:
        """
        Dequantize int8 weights back to float32 for aggregation.

        For QAT models, weights are typically already float32 with fake quantization,
        but this handles cases where actual int8 weights are sent.
        """
        dequantized = []
        quant_params_list = []

        for i, w in enumerate(weights):
            if w.dtype == np.int8:
                # Actual quantized weights - need to dequantize
                params = calculate_quantization_params(w.astype(np.float32))
                quant_params_list.append(params)
                dequantized.append(dequantize_array(w, params))
            else:
                # Already float32 (QAT fake quantization)
                quant_params_list.append(None)
                dequantized.append(w)

        # Cache params for potential re-quantization
        self.quant_params_cache[client_id] = quant_params_list

        return dequantized

    def _quantize_weights(
        self,
        weights: List[np.ndarray],
        symmetric: bool = True
    ) -> Tuple[List[np.ndarray], List[QuantizationParams]]:
        """
        Quantize float32 weights to int8 for efficient transmission.
        """
        quantized = []
        quant_params = []

        for w in weights:
            params = calculate_quantization_params(w, symmetric=symmetric)
            q_w = quantize_array(w, params)
            quantized.append(q_w)
            quant_params.append(params)

        return quantized, quant_params

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes] | BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate client weights with optional dequantization for QAT.

        Process:
        1. Extract weights from each client
        2. If QAT + dequantize: convert int8 -> float32
        3. Perform weighted average (FedAvg)
        4. Store latest parameters for model saving
        """
        if not results:
            return None, {}

        # Log client contributions periodically
        if server_round % 5 == 0 or server_round == 1:
            total_examples = sum(fit_res.num_examples for _, fit_res in results)
            print(f"\n[Round {server_round}] Aggregating {len(results)} clients:")
            for client_proxy, fit_res in results:
                weight = fit_res.num_examples / total_examples
                print(f"  Client: {fit_res.num_examples:,} samples (weight: {weight:.4f})")

        # If using QAT with dequantization, process weights
        if self.use_qat and self.use_dequantize_aggregate:
            processed_results = []
            for idx, (client_proxy, fit_res) in enumerate(results):
                weights = parameters_to_ndarrays(fit_res.parameters)

                # Dequantize if needed
                dequantized_weights = self._dequantize_weights(weights, client_id=idx)

                # Create new FitRes with dequantized weights
                new_params = ndarrays_to_parameters(dequantized_weights)
                new_fit_res = FitRes(
                    status=fit_res.status,
                    parameters=new_params,
                    num_examples=fit_res.num_examples,
                    metrics=fit_res.metrics,
                )
                processed_results.append((client_proxy, new_fit_res))

            results = processed_results

        # Call parent FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Store latest parameters for saving (keep float32 version)
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters

            if server_round % 5 == 0:
                weights = parameters_to_ndarrays(aggregated_parameters)
                total_params = sum(w.size for w in weights)
                total_bytes = sum(w.nbytes for w in weights)
                print(f"  Aggregated model: {total_params:,} params, {total_bytes/1024:.2f} KB")

            # Quantize weights before sending to clients if QAT is enabled
            if self.use_qat and self.quantize_before_send:
                weights = parameters_to_ndarrays(aggregated_parameters)
                quantized_weights, _ = self._quantize_weights(weights)
                aggregated_parameters = ndarrays_to_parameters(quantized_weights)

                if server_round % 5 == 0:
                    q_bytes = sum(w.nbytes for w in quantized_weights)
                    print(f"  Quantized for clients: {q_bytes/1024:.2f} KB ({total_bytes/q_bytes:.1f}x compression)")

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, Any]],
        failures: List[Tuple[ClientProxy, Any] | BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics from clients."""
        if not results:
            return None, {}

        # Weighted average of metrics
        total_examples = sum(eval_res.num_examples for _, eval_res in results)

        aggregated_loss = sum(
            eval_res.loss * eval_res.num_examples
            for _, eval_res in results
        ) / total_examples

        # Aggregate other metrics if available
        aggregated_metrics = {"loss": aggregated_loss}

        # Check for additional metrics in client results
        sample_metrics = results[0][1].metrics if results[0][1].metrics else {}
        for key in sample_metrics:
            if key != "loss":
                weighted_sum = sum(
                    eval_res.metrics.get(key, 0) * eval_res.num_examples
                    for _, eval_res in results
                )
                aggregated_metrics[key] = weighted_sum / total_examples

        return aggregated_loss, aggregated_metrics


class QATFedAvgMStrategy(QATAwareStrategy):
    """
    QAT-aware FedAvgM (with server momentum) strategy.

    FedAvgM applies momentum on the server side to stabilize training
    and reduce client drift in non-IID settings.
    """

    def __init__(
        self,
        server_momentum: float = 0.9,
        server_learning_rate: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.server_momentum = server_momentum
        self.server_learning_rate = server_learning_rate
        self.momentum_buffer: Optional[List[np.ndarray]] = None

        print(f"[Server] FedAvgM: momentum={server_momentum}, lr={server_learning_rate}")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes] | BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate with server-side momentum."""

        # Get aggregated parameters from parent
        aggregated_parameters, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is None:
            return None, metrics

        # Apply server momentum
        new_weights = parameters_to_ndarrays(aggregated_parameters)

        if self.momentum_buffer is None:
            # Initialize momentum buffer
            self.momentum_buffer = [np.zeros_like(w) for w in new_weights]

        # Compute pseudo-gradient (difference from current)
        if self.latest_parameters is not None:
            old_weights = parameters_to_ndarrays(self.latest_parameters)

            # Update momentum buffer and apply
            for i in range(len(new_weights)):
                delta = new_weights[i] - old_weights[i]
                self.momentum_buffer[i] = (
                    self.server_momentum * self.momentum_buffer[i] + delta
                )
                new_weights[i] = (
                    old_weights[i] + self.server_learning_rate * self.momentum_buffer[i]
                )

        # Update latest parameters
        self.latest_parameters = ndarrays_to_parameters(new_weights)

        return self.latest_parameters, metrics


def on_fit_config(server_round: int) -> Dict[str, Any]:
    """Generate fit configuration for clients."""
    cfg = _get_config()
    fed_cfg = cfg.get("federated", {})

    return {
        "local_epochs": fed_cfg.get("local_epochs", 2),
        "batch_size": fed_cfg.get("batch_size", 128),
        "server_round": server_round,
        "use_callbacks": fed_cfg.get("use_callbacks", False),
        "learning_rate": fed_cfg.get("learning_rate", 0.001),
        "lr_decay": fed_cfg.get("lr_decay", 1.0),
    }


def on_evaluate_config(server_round: int) -> Dict[str, Any]:
    """Generate evaluate configuration for clients."""
    return {"server_round": server_round}


def create_strategy(config: dict = None) -> fl.server.strategy.Strategy:
    """
    Create FL strategy based on configuration.

    Selects QATFedAvgMStrategy if QAT is enabled and FedAvgM is available,
    otherwise falls back to QATAwareStrategy (FedAvg).
    """
    if config is None:
        config = _get_config()

    fed_cfg = config.get("federated", {})

    use_qat = fed_cfg.get("use_qat", False)
    use_momentum = fed_cfg.get("server_momentum", 0) > 0

    strategy_kwargs = {
        "fraction_fit": fed_cfg.get("fraction_fit", 1.0),
        "fraction_evaluate": fed_cfg.get("fraction_evaluate", 1.0),
        "min_fit_clients": fed_cfg.get("min_fit_clients", 1),
        "min_evaluate_clients": fed_cfg.get("min_evaluate_clients", 1),
        "min_available_clients": fed_cfg.get("min_available_clients", 1),
        "on_fit_config_fn": on_fit_config,
        "on_evaluate_config_fn": on_evaluate_config,
        "use_qat": use_qat,
        "use_dequantize_aggregate": use_qat,  # Dequantize client weights when QAT is used
        "quantize_before_send": use_qat,  # Quantize before sending to clients when QAT is used
    }

    if use_momentum:
        strategy = QATFedAvgMStrategy(
            server_momentum=fed_cfg.get("server_momentum", 0.9),
            server_learning_rate=fed_cfg.get("server_learning_rate", 1.0),
            **strategy_kwargs
        )
    else:
        strategy = QATAwareStrategy(**strategy_kwargs)

    return strategy


def main(config_path: str = None):
    """Start FL server with QAT-aware strategy."""
    global CFG
    CFG = load_server_config(config_path)

    fed_cfg = CFG.get("federated", {})
    num_rounds = fed_cfg.get("num_rounds", 10)

    print("\n" + "="*60)
    print("  Federated Learning Server (QAT-Aware)")
    print("="*60)
    print(f"Rounds: {num_rounds}")
    print(f"QAT enabled: {fed_cfg.get('use_qat', False)}")
    print(f"Server momentum: {fed_cfg.get('server_momentum', 0)}")
    print("="*60 + "\n")

    strategy = create_strategy(CFG)

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FL Server with QAT support")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    args = parser.parse_args()

    main(config_path=args.config)
