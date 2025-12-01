"""
Structured pruning implementation for TensorFlow/Keras models.
Removes entire neurons/filters based on magnitude criteria.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, List, Optional
from pathlib import Path


def calculate_neuron_importance(weights: np.ndarray) -> np.ndarray:
    """
    Calculate importance scores for neurons based on L2 norm of weights.

    Args:
        weights: Weight matrix of shape (input_dim, output_dim) for Dense layers
                 or (kernel_h, kernel_w, in_channels, out_channels) for Conv2D

    Returns:
        Importance scores for each output neuron/filter
    """
    if len(weights.shape) == 2:  # Dense layer
        # L2 norm of incoming weights for each neuron
        importance = np.linalg.norm(weights, axis=0)
    elif len(weights.shape) == 4:  # Conv2D layer
        # L2 norm of filter weights
        importance = np.linalg.norm(weights, axis=(0, 1, 2))
    else:
        raise ValueError(f"Unsupported weight shape: {weights.shape}")

    return importance


def prune_dense_layer(
    weights: np.ndarray,
    bias: Optional[np.ndarray],
    pruning_ratio: float
) -> Tuple[np.ndarray, Optional[np.ndarray], List[int]]:
    """
    Prune neurons from a Dense layer based on magnitude.

    Args:
        weights: Weight matrix (input_dim, output_dim)
        bias: Bias vector (output_dim,) or None
        pruning_ratio: Fraction of neurons to remove (0.0 to 1.0)

    Returns:
        Tuple of (pruned_weights, pruned_bias, kept_indices)
    """
    importance = calculate_neuron_importance(weights)
    num_neurons = len(importance)
    num_to_keep = max(1, int(num_neurons * (1 - pruning_ratio)))

    # Keep top-k neurons
    kept_indices = np.argsort(importance)[-num_to_keep:]
    kept_indices = sorted(kept_indices)

    pruned_weights = weights[:, kept_indices]
    pruned_bias = bias[kept_indices] if bias is not None else None

    return pruned_weights, pruned_bias, kept_indices


def prune_conv2d_layer(
    weights: np.ndarray,
    bias: Optional[np.ndarray],
    pruning_ratio: float
) -> Tuple[np.ndarray, Optional[np.ndarray], List[int]]:
    """
    Prune filters from a Conv2D layer based on magnitude.

    Args:
        weights: Weight tensor (kernel_h, kernel_w, in_channels, out_channels)
        bias: Bias vector (out_channels,) or None
        pruning_ratio: Fraction of filters to remove (0.0 to 1.0)

    Returns:
        Tuple of (pruned_weights, pruned_bias, kept_indices)
    """
    importance = calculate_neuron_importance(weights)
    num_filters = len(importance)
    num_to_keep = max(1, int(num_filters * (1 - pruning_ratio)))

    # Keep top-k filters
    kept_indices = np.argsort(importance)[-num_to_keep:]
    kept_indices = sorted(kept_indices)

    pruned_weights = weights[:, :, :, kept_indices]
    pruned_bias = bias[kept_indices] if bias is not None else None

    return pruned_weights, pruned_bias, kept_indices


def apply_structured_pruning(
    model: keras.Model,
    pruning_ratio: float = 0.5,
    skip_last_layer: bool = True,
    verbose: bool = True
) -> keras.Model:
    """
    Apply structured pruning to a Keras model.

    This removes entire neurons/filters from Dense and Conv2D layers,
    resulting in a smaller model architecture.

    Args:
        model: Input Keras model
        pruning_ratio: Fraction of neurons/filters to remove per layer (0.0 to 1.0)
        skip_last_layer: If True, don't prune the output layer
        verbose: Print pruning progress

    Returns:
        New pruned model with reduced architecture
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"=' Applying Structured Pruning (ratio={pruning_ratio:.1%})")
        print(f"{'='*60}\n")

    # Track which neurons/filters to keep at each layer
    kept_indices_map = {}

    # First pass: determine which neurons/filters to keep
    for i, layer in enumerate(model.layers):
        if isinstance(layer, layers.Dense):
            if skip_last_layer and i == len(model.layers) - 1:
                continue  # Skip output layer

            weights = layer.get_weights()
            if len(weights) >= 1:
                w = weights[0]
                bias = weights[1] if len(weights) > 1 else None
                _, _, kept_indices = prune_dense_layer(w, bias, pruning_ratio)
                kept_indices_map[i] = kept_indices

                if verbose:
                    print(f"Layer {i} ({layer.name}): Dense {w.shape[1]} � {len(kept_indices)} neurons")

        elif isinstance(layer, layers.Conv2D):
            weights = layer.get_weights()
            if len(weights) >= 1:
                w = weights[0]
                bias = weights[1] if len(weights) > 1 else None
                _, _, kept_indices = prune_conv2d_layer(w, bias, pruning_ratio)
                kept_indices_map[i] = kept_indices

                if verbose:
                    print(f"Layer {i} ({layer.name}): Conv2D {w.shape[3]} � {len(kept_indices)} filters")

    # Second pass: build new model with pruned architecture
    # Get input shape from the model
    input_shape = model.input_shape[1:]  # Remove batch dimension
    new_layers = []
    prev_kept_indices = None

    # Add input layer
    new_layers.append(keras.Input(shape=input_shape))

    for i, layer in enumerate(model.layers):
        if isinstance(layer, layers.Dense):
            weights = layer.get_weights()
            w = weights[0]
            bias = weights[1] if len(weights) > 1 else None

            # Prune input dimension based on previous layer
            if prev_kept_indices is not None:
                w = w[prev_kept_indices, :]

            # Prune output dimension if this layer is being pruned
            if i in kept_indices_map:
                w, bias, kept_indices = prune_dense_layer(w, bias, pruning_ratio)
                prev_kept_indices = kept_indices
            else:
                prev_kept_indices = None

            # Create new layer
            new_layer = layers.Dense(
                w.shape[1],
                activation=layer.activation,
                name=f"{layer.name}_pruned"
            )
            new_layers.append(new_layer)

        elif isinstance(layer, layers.Conv2D):
            weights = layer.get_weights()
            w = weights[0]
            bias = weights[1] if len(weights) > 1 else None

            # Prune input channels based on previous Conv2D layer
            if prev_kept_indices is not None and len(w.shape) == 4:
                w = w[:, :, prev_kept_indices, :]

            # Prune output filters if this layer is being pruned
            if i in kept_indices_map:
                w, bias, kept_indices = prune_conv2d_layer(w, bias, pruning_ratio)
                prev_kept_indices = kept_indices
            else:
                prev_kept_indices = None

            # Create new layer
            new_layer = layers.Conv2D(
                w.shape[3],
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                activation=layer.activation,
                name=f"{layer.name}_pruned"
            )
            new_layers.append(new_layer)

        elif isinstance(layer, (layers.MaxPooling2D, layers.Flatten, layers.Dropout)):
            # Pass-through layers
            new_layer = layer.__class__.from_config(layer.get_config())
            new_layers.append(new_layer)
            # Pooling/Flatten/Dropout don't change the kept_indices

        elif isinstance(layer, layers.InputLayer):
            # Skip, already handled
            pass

        else:
            # Copy other layers as-is
            new_layer = layer.__class__.from_config(layer.get_config())
            new_layers.append(new_layer)

    # Build the new model
    x = new_layers[0]
    for layer in new_layers[1:]:
        x = layer(x)

    pruned_model = keras.Model(inputs=new_layers[0], outputs=x)

    # Third pass: copy weights to the new model
    # Start at index 1 because index 0 is the Input layer
    new_layer_idx = 1
    prev_kept_indices = None

    for i, layer in enumerate(model.layers):
        if isinstance(layer, layers.InputLayer):
            continue

        if isinstance(layer, layers.Dense):
            weights = layer.get_weights()
            w = weights[0]
            bias = weights[1] if len(weights) > 1 else None

            # Apply pruning based on previous layer
            if prev_kept_indices is not None:
                w = w[prev_kept_indices, :]

            # Apply pruning to current layer
            if i in kept_indices_map:
                w, bias, kept_indices = prune_dense_layer(w, bias, pruning_ratio)
                prev_kept_indices = kept_indices
            else:
                prev_kept_indices = None

            # Set weights
            if bias is not None:
                pruned_model.layers[new_layer_idx].set_weights([w, bias])
            else:
                pruned_model.layers[new_layer_idx].set_weights([w])
            new_layer_idx += 1

        elif isinstance(layer, layers.Conv2D):
            weights = layer.get_weights()
            w = weights[0]
            bias = weights[1] if len(weights) > 1 else None

            # Apply pruning based on previous layer
            if prev_kept_indices is not None and len(w.shape) == 4:
                w = w[:, :, prev_kept_indices, :]

            # Apply pruning to current layer
            if i in kept_indices_map:
                w, bias, kept_indices = prune_conv2d_layer(w, bias, pruning_ratio)
                prev_kept_indices = kept_indices
            else:
                prev_kept_indices = None

            # Set weights
            if bias is not None:
                pruned_model.layers[new_layer_idx].set_weights([w, bias])
            else:
                pruned_model.layers[new_layer_idx].set_weights([w])
            new_layer_idx += 1

        else:
            # Other layers (pooling, flatten, etc.)
            new_layer_idx += 1

    # Compile the pruned model with the same configuration as the original
    # Use fresh instances to avoid any state issues

    # Get loss name as string
    if hasattr(model.loss, '__name__'):
        loss = model.loss.__name__
    elif hasattr(model.loss, 'name'):
        loss = model.loss.name
    else:
        # Fallback based on common patterns
        loss_str = str(model.loss)
        if 'binary' in loss_str.lower():
            loss = 'binary_crossentropy'
        elif 'categorical' in loss_str.lower():
            loss = 'sparse_categorical_crossentropy' if 'sparse' in loss_str.lower() else 'categorical_crossentropy'
        else:
            loss = 'binary_crossentropy'  # Default fallback

    # Get optimizer as string
    optimizer_name = model.optimizer.__class__.__name__.lower()
    if optimizer_name == 'adam':
        optimizer = 'adam'
    elif optimizer_name == 'sgd':
        optimizer = 'sgd'
    elif optimizer_name == 'rmsprop':
        optimizer = 'rmsprop'
    else:
        optimizer = 'adam'  # Default fallback

    pruned_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )

    if verbose:
        print(f"\n Pruning complete!\n")

    return pruned_model


def get_model_size(model: keras.Model) -> Tuple[int, float]:
    """
    Calculate the number of parameters and approximate size of a model.

    Args:
        model: Keras model

    Returns:
        Tuple of (num_parameters, size_in_kb)
    """
    num_params = model.count_params()
    # Approximate size: 4 bytes per float32 parameter
    size_kb = (num_params * 4) / 1024
    return num_params, size_kb


def compare_models(original: keras.Model, pruned: keras.Model) -> None:
    """
    Print comparison statistics between original and pruned models.

    Args:
        original: Original model
        pruned: Pruned model
    """
    orig_params, orig_size = get_model_size(original)
    pruned_params, pruned_size = get_model_size(pruned)

    reduction_params = (1 - pruned_params / orig_params) * 100
    reduction_size = (1 - pruned_size / orig_size) * 100

    print(f"\n{'='*60}")
    print(f"=� Model Comparison")
    print(f"{'='*60}")
    print(f"\nOriginal Model:")
    print(f"  - Parameters: {orig_params:,}")
    print(f"  - Size: {orig_size:.2f} KB")

    print(f"\nPruned Model:")
    print(f"  - Parameters: {pruned_params:,}")
    print(f"  - Size: {pruned_size:.2f} KB")

    print(f"\nReduction:")
    print(f"  - Parameters: {reduction_params:.1f}%")
    print(f"  - Size: {reduction_size:.1f}%")
    print(f"{'='*60}\n")


def fine_tune_pruned_model(
    pruned_model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 5,
    batch_size: int = 128,
    learning_rate: float = 0.0001,
    verbose: bool = True
) -> keras.Model:
    """
    Fine-tune a pruned model to recover accuracy.

    Args:
        pruned_model: Pruned model to fine-tune
        x_train: Training data
        y_train: Training labels
        x_val: Validation data
        y_val: Validation labels
        epochs: Number of fine-tuning epochs
        batch_size: Batch size for training
        learning_rate: Learning rate (typically lower than initial training)
        verbose: Print training progress

    Returns:
        Fine-tuned model
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"= Fine-tuning Pruned Model")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}\n")

    # Use lower learning rate for fine-tuning
    # Get loss as string to avoid state issues
    if hasattr(pruned_model.loss, '__name__'):
        loss = pruned_model.loss.__name__
    elif hasattr(pruned_model.loss, 'name'):
        loss = pruned_model.loss.name
    else:
        loss_str = str(pruned_model.loss)
        if 'binary' in loss_str.lower():
            loss = 'binary_crossentropy'
        elif 'categorical' in loss_str.lower():
            loss = 'sparse_categorical_crossentropy' if 'sparse' in loss_str.lower() else 'categorical_crossentropy'
        else:
            loss = 'binary_crossentropy'

    pruned_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['accuracy']
    )

    pruned_model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        verbose=1 if verbose else 0
    )

    if verbose:
        print(f"\n Fine-tuning complete!\n")

    return pruned_model
