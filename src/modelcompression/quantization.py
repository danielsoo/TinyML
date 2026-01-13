"""
Post-training quantization implementation for TensorFlow/Keras models.
Reduces model size by converting 32-bit float weights to 8-bit integers.
"""
import numpy as np
from tensorflow import keras
from typing import Tuple, Dict, Optional, List


class QuantizationParams:
    """
    Stores quantization parameters for converting between float and int8.

    Quantization formula:
        quantized_value = round(float_value / scale) + zero_point
        float_value = (quantized_value - zero_point) * scale

    Attributes:
        scale: Scaling factor for quantization
        zero_point: Zero point offset (typically 0 for symmetric quantization)
        min_val: Minimum value in the original float data
        max_val: Maximum value in the original float data
    """

    def __init__(self, scale: float, zero_point: int, min_val: float, max_val: float):
        self.scale = scale
        self.zero_point = zero_point
        self.min_val = min_val
        self.max_val = max_val

    def __repr__(self):
        return f"QuantizationParams(scale={self.scale:.6f}, zero_point={self.zero_point})"


def calculate_quantization_params(
    data: np.ndarray,
    num_bits: int = 8,
    symmetric: bool = True
) -> QuantizationParams:
    """
    Calculate quantization parameters for a given data array.

    Args:
        data: Input float array to be quantized
        num_bits: Number of bits for quantization (default: 8 for int8)
        symmetric: If True, use symmetric quantization around zero
                   If False, use asymmetric quantization

    Returns:
        QuantizationParams object containing scale and zero_point

    Explanation:
        Symmetric quantization: Maps float range [-abs_max, abs_max] to [-127, 127]
            - Better for weights that are symmetric around zero
            - Zero point is always 0
            - scale = max(abs(min), abs(max)) / 127

        Asymmetric quantization: Maps [min, max] to [-128, 127]
            - Better utilizes the int8 range for asymmetric data
            - Zero point can be non-zero
            - scale = (max - min) / 255
    """
    min_val = float(np.min(data))
    max_val = float(np.max(data))

    if symmetric:
        # Symmetric quantization: center around zero
        # Map [-abs_max, abs_max] to [-127, 127] (leaving -128 unused for symmetry)
        abs_max = max(abs(min_val), abs(max_val))

        # Avoid division by zero
        if abs_max == 0:
            abs_max = 1.0

        scale = abs_max / 127.0
        zero_point = 0
    else:
        # Asymmetric quantization: use full int8 range
        # Map [min_val, max_val] to [-128, 127]
        value_range = max_val - min_val

        # Avoid division by zero
        if value_range == 0:
            value_range = 1.0

        scale = value_range / 255.0
        zero_point = int(round(-min_val / scale - 128))

    return QuantizationParams(scale, zero_point, min_val, max_val)


def quantize_array(
    data: np.ndarray,
    params: QuantizationParams,
    num_bits: int = 8
) -> np.ndarray:
    """
    Quantize a float array to int8 using given quantization parameters.

    Args:
        data: Input float array
        params: Quantization parameters (scale, zero_point)
        num_bits: Number of bits (default: 8)

    Returns:
        Quantized int8 array

    Process:
        1. Divide float values by scale
        2. Round to nearest integer
        3. Add zero_point offset
        4. Clip to int8 range [-128, 127]
    """
    # Quantize: divide by scale, round, add zero point
    quantized = np.round(data / params.scale) + params.zero_point

    # Clip to int8 range
    quantized = np.clip(quantized, -128, 127)

    return quantized.astype(np.int8)


def dequantize_array(
    quantized_data: np.ndarray,
    params: QuantizationParams
) -> np.ndarray:
    """
    Dequantize an int8 array back to float32 using quantization parameters.

    Args:
        quantized_data: Quantized int8 array
        params: Quantization parameters used during quantization

    Returns:
        Dequantized float32 array

    Process:
        1. Subtract zero_point offset
        2. Multiply by scale
        3. Convert back to float32
    """
    # Dequantize: subtract zero point, multiply by scale
    dequantized = (quantized_data.astype(np.float32) - params.zero_point) * params.scale
    return dequantized


class QuantizedLayer:
    """
    Stores a quantized layer's weights and quantization parameters.

    Attributes:
        layer_name: Name of the original layer
        layer_type: Type of layer (Dense, Conv2D, etc.)
        quantized_weights: List of quantized weight arrays (weights, bias, etc.)
        quant_params: List of QuantizationParams for each weight array
        layer_config: Configuration dict to reconstruct the layer
    """

    def __init__(
        self,
        layer_name: str,
        layer_type: str,
        quantized_weights: List[np.ndarray],
        quant_params: List[QuantizationParams],
        layer_config: dict
    ):
        self.layer_name = layer_name
        self.layer_type = layer_type
        self.quantized_weights = quantized_weights
        self.quant_params = quant_params
        self.layer_config = layer_config


def quantize_layer_weights(
    layer: keras.layers.Layer,
    symmetric: bool = True,
    verbose: bool = False
) -> Optional[QuantizedLayer]:
    """
    Quantize weights of a single layer.

    Args:
        layer: Keras layer to quantize
        symmetric: Use symmetric quantization
        verbose: Print quantization info

    Returns:
        QuantizedLayer object or None if layer has no weights

    Explanation:
        - Extracts all weights from the layer (kernel, bias, etc.)
        - Calculates quantization parameters for each weight array
        - Quantizes each weight array to int8
        - Stores quantization params for later dequantization
    """
    weights = layer.get_weights()

    if len(weights) == 0:
        return None

    quantized_weights = []
    quant_params_list = []

    for i, weight in enumerate(weights):
        # Calculate quantization parameters
        params = calculate_quantization_params(weight, symmetric=symmetric)

        # Quantize the weight array
        quantized = quantize_array(weight, params)

        quantized_weights.append(quantized)
        quant_params_list.append(params)

        if verbose:
            original_size = weight.nbytes / 1024
            quantized_size = quantized.nbytes / 1024
            compression_ratio = original_size / quantized_size
            print(f"  Weight {i}: {weight.shape} | "
                  f"{original_size:.2f}KB -> {quantized_size:.2f}KB | "
                  f"{compression_ratio:.1f}x compression | "
                  f"{params}")

    return QuantizedLayer(
        layer_name=layer.name,
        layer_type=layer.__class__.__name__,
        quantized_weights=quantized_weights,
        quant_params=quant_params_list,
        layer_config=layer.get_config()
    )


def quantize_model(
    model: keras.Model,
    symmetric: bool = True,
    verbose: bool = True
) -> Dict[str, QuantizedLayer]:
    """
    Quantize all weights in a Keras model to int8.

    Args:
        model: Input Keras model with float32 weights
        symmetric: Use symmetric quantization (better for weights)
        verbose: Print detailed quantization information

    Returns:
        Dictionary mapping layer names to QuantizedLayer objects

    Process:
        1. Iterate through all layers in the model
        2. For layers with trainable weights (Dense, Conv2D, etc.):
           - Calculate quantization parameters per weight array
           - Quantize weights from float32 to int8
           - Store quantization params for dequantization
        3. Skip layers without weights (Activation, Pooling, etc.)

    Benefits:
        - Reduces model size by ~4x (32-bit float -> 8-bit int)
        - Faster inference on hardware with int8 acceleration
        - Minimal accuracy loss (typically < 1% for vision models)
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"  Quantizing Model to INT8")
        print(f"{'='*70}")
        print(f"Quantization type: {'Symmetric' if symmetric else 'Asymmetric'}\n")

    quantized_layers = {}

    for layer in model.layers:
        # Only quantize layers with weights
        if len(layer.get_weights()) > 0:
            if verbose:
                print(f"Layer: {layer.name} ({layer.__class__.__name__})")

            quantized_layer = quantize_layer_weights(layer, symmetric, verbose)

            if quantized_layer:
                quantized_layers[layer.name] = quantized_layer

            if verbose:
                print()

    if verbose:
        print(f"{'='*70}")
        print(f"  Quantization Complete!")
        print(f"  Quantized {len(quantized_layers)} layers")
        print(f"{'='*70}\n")

    return quantized_layers


def dequantize_model(
    model: keras.Model,
    quantized_layers: Dict[str, QuantizedLayer]
) -> keras.Model:
    """
    Dequantize a model by restoring float32 weights from quantized int8 weights.

    Args:
        model: Keras model (can be a new instance or the original)
        quantized_layers: Dictionary of quantized layers from quantize_model()

    Returns:
        Model with dequantized float32 weights

    Purpose:
        - For inference: Converts int8 weights back to float32 for standard inference
        - For evaluation: Measure accuracy loss from quantization
        - Note: Real deployment would use int8 weights directly for efficiency
    """
    for layer in model.layers:
        if layer.name in quantized_layers:
            quantized_layer = quantized_layers[layer.name]

            # Dequantize all weights
            dequantized_weights = []
            for quantized_w, params in zip(quantized_layer.quantized_weights,
                                           quantized_layer.quant_params):
                dequantized = dequantize_array(quantized_w, params)
                dequantized_weights.append(dequantized)

            # Set the dequantized weights back to the layer
            layer.set_weights(dequantized_weights)

    return model


def calculate_quantization_error(
    original_weights: np.ndarray,
    quantized_weights: np.ndarray,
    params: QuantizationParams
) -> Dict[str, float]:
    """
    Calculate error metrics between original and quantized weights.

    Args:
        original_weights: Original float32 weights
        quantized_weights: Quantized int8 weights
        params: Quantization parameters

    Returns:
        Dictionary with error metrics: MSE, MAE, max_error, SNR

    Metrics:
        - MSE: Mean Squared Error
        - MAE: Mean Absolute Error
        - max_error: Maximum absolute difference
        - SNR: Signal-to-Noise Ratio in dB (higher is better)
    """
    # Dequantize to compare
    dequantized = dequantize_array(quantized_weights, params)

    # Calculate errors
    diff = original_weights - dequantized
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    max_error = np.max(np.abs(diff))

    # Signal-to-Noise Ratio
    signal_power = np.mean(original_weights ** 2)
    noise_power = mse
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

    return {
        'mse': mse,
        'mae': mae,
        'max_error': max_error,
        'snr_db': snr_db
    }


def compare_model_sizes(
    original_model: keras.Model,
    quantized_layers: Dict[str, QuantizedLayer]
) -> None:
    """
    Print comparison of model sizes before and after quantization.

    Args:
        original_model: Original float32 model
        quantized_layers: Dictionary of quantized layers

    Output:
        Prints detailed size comparison and compression ratio
    """
    # Calculate original model size
    original_size = 0
    for layer in original_model.layers:
        for weight in layer.get_weights():
            original_size += weight.nbytes

    # Calculate quantized model size
    quantized_size = 0
    for quantized_layer in quantized_layers.values():
        for weight in quantized_layer.quantized_weights:
            quantized_size += weight.nbytes
        # Add overhead for storing quantization params
        # (scale and zero_point for each weight array)
        quantized_size += len(quantized_layer.quant_params) * (4 + 4)  # float + int

    original_size_kb = original_size / 1024
    quantized_size_kb = quantized_size / 1024
    compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
    reduction_percent = (1 - quantized_size / original_size) * 100 if original_size > 0 else 0

    print(f"\n{'='*70}")
    print(f"  Model Size Comparison")
    print(f"{'='*70}")
    print(f"\nOriginal Model (float32):")
    print(f"  Size: {original_size_kb:.2f} KB")
    print(f"\nQuantized Model (int8):")
    print(f"  Size: {quantized_size_kb:.2f} KB")
    print(f"\nCompression:")
    print(f"  Ratio: {compression_ratio:.2f}x")
    print(f"  Reduction: {reduction_percent:.1f}%")
    print(f"{'='*70}\n")


def evaluate_quantization_accuracy(
    original_model: keras.Model,
    quantized_layers: Dict[str, QuantizedLayer],
    x_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True
) -> Tuple[float, float]:
    """
    Evaluate accuracy of original vs quantized model.

    Args:
        original_model: Original float32 model
        quantized_layers: Dictionary of quantized layers
        x_test: Test data
        y_test: Test labels
        verbose: Print comparison results

    Returns:
        Tuple of (original_accuracy, quantized_accuracy)

    Process:
        1. Evaluate original model
        2. Create copy of model and dequantize weights
        3. Evaluate quantized model
        4. Compare accuracy loss
    """
    # Evaluate original model
    original_loss, original_acc = original_model.evaluate(x_test, y_test, verbose=0)

    # Create a copy and dequantize
    quantized_model = keras.models.clone_model(original_model)
    quantized_model.set_weights(original_model.get_weights())

    # Compile with fresh instances to avoid state issues
    # Get loss as string
    if hasattr(original_model.loss, '__name__'):
        loss = original_model.loss.__name__
    elif hasattr(original_model.loss, 'name'):
        loss = original_model.loss.name
    else:
        loss_str = str(original_model.loss)
        if 'binary' in loss_str.lower():
            loss = 'binary_crossentropy'
        elif 'categorical' in loss_str.lower():
            loss = 'sparse_categorical_crossentropy' if 'sparse' in loss_str.lower() else 'categorical_crossentropy'
        else:
            loss = 'binary_crossentropy'

    # Get optimizer as string
    optimizer_name = original_model.optimizer.__class__.__name__.lower()
    if optimizer_name == 'adam':
        optimizer = 'adam'
    elif optimizer_name == 'sgd':
        optimizer = 'sgd'
    elif optimizer_name == 'rmsprop':
        optimizer = 'rmsprop'
    else:
        optimizer = 'adam'

    quantized_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    quantized_model = dequantize_model(quantized_model, quantized_layers)

    # Evaluate quantized model
    quantized_loss, quantized_acc = quantized_model.evaluate(x_test, y_test, verbose=0)

    if verbose:
        accuracy_drop = (original_acc - quantized_acc) * 100

        print(f"\n{'='*70}")
        print(f"  Accuracy Comparison")
        print(f"{'='*70}")
        print(f"\nOriginal Model:")
        print(f"  Accuracy: {original_acc*100:.2f}%")
        print(f"  Loss: {original_loss:.4f}")
        print(f"\nQuantized Model:")
        print(f"  Accuracy: {quantized_acc*100:.2f}%")
        print(f"  Loss: {quantized_loss:.4f}")
        print(f"\nAccuracy Drop: {accuracy_drop:.2f}%")
        print(f"{'='*70}\n")

    return original_acc, quantized_acc
