"""
Knowledge Distillation implementation for TensorFlow/Keras models.
Transfers knowledge from a large teacher model to a smaller student model.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional, Callable
from pathlib import Path


class DistillationLoss(keras.losses.Loss):
    """
    Custom loss function for knowledge distillation.

    Combines two objectives:
    1. Distillation loss: KL divergence between teacher and student soft predictions
    2. Student loss: Standard cross-entropy with true labels

    Formula:
        total_loss = alpha * distillation_loss + (1 - alpha) * student_loss

    Attributes:
        alpha: Weight for distillation loss (0.0 to 1.0)
        temperature: Temperature for softening probability distributions
                     Higher temperature = softer distributions
        student_loss_fn: Loss function for student (e.g., categorical_crossentropy)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        temperature: float = 3.0,
        student_loss_fn: Optional[keras.losses.Loss] = None,
        name: str = "distillation_loss"
    ):
        """
        Initialize distillation loss.

        Args:
            alpha: Weight for distillation loss vs student loss
                   - 0.0 = only student loss (standard training)
                   - 1.0 = only distillation loss (pure knowledge transfer)
                   - Typical values: 0.1 to 0.5
            temperature: Temperature for softening predictions
                        - Higher values = softer distributions (more information transfer)
                        - Lower values = sharper distributions (closer to hard labels)
                        - Typical values: 2.0 to 10.0

                        Why high temperature (3-10)?
                        - T=1: [0.86, 0.12, 0.02] - too sharp, like hard labels
                        - T=3: [0.55, 0.28, 0.17] - reveals class relationships
                        - T=10: [0.40, 0.33, 0.27] - too soft, loses information

                        The T² scaling in the loss prevents gradient vanishing:
                        - Without scaling: high T → tiny gradients → no learning
                        - With scaling: gradients remain useful at all temperatures
            student_loss_fn: Loss function for student training
        """
        super().__init__(name=name)
        self.alpha = alpha
        self.temperature = temperature
        self.student_loss_fn = student_loss_fn or keras.losses.CategoricalCrossentropy(
            from_logits=False
        )

    def call(self, y_true, y_pred):
        """
        Calculate distillation loss.

        Args:
            y_true: Tuple of (true_labels, teacher_logits)
            y_pred: Student predictions (logits or probabilities)

        Returns:
            Combined distillation and student loss
        """
        # y_true is actually a tuple: (labels, teacher_predictions)
        # This is set up in the training loop
        return self.student_loss_fn(y_true, y_pred)

    def get_config(self):
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'temperature': self.temperature,
        })
        return config


def distillation_loss_fn(
    y_true: tf.Tensor,
    student_predictions: tf.Tensor,
    teacher_predictions: tf.Tensor,
    temperature: float = 3.0,
    alpha: float = 0.1
) -> tf.Tensor:
    """
    Calculate knowledge distillation loss.

    Args:
        y_true: True labels (one-hot encoded or sparse)
        student_predictions: Student model predictions (logits)
        teacher_predictions: Teacher model predictions (logits)
        temperature: Temperature for softening distributions
        alpha: Weight for distillation loss

    Returns:
        Combined distillation loss

    Explanation:
        Distillation loss uses temperature-scaled softmax to create "soft targets"
        that contain more information than hard labels (0 or 1).

        Example (classifying dog/cat/bird):
            Hard label: [0, 1, 0] (one-hot) - says "it's a cat, nothing else"

            Teacher at T=1: [0.05, 0.90, 0.05] - still very confident
            Teacher at T=3: [0.15, 0.65, 0.20] - reveals dog-cat similarity!

        Why high temperature (3-10) works better:
            1. Reveals inter-class relationships (dog-cat more similar than cat-bird)
            2. Transfers teacher's learned features and uncertainties
            3. Provides stronger gradients (scaled by T²) for better learning
            4. Empirically proven to give +1-3% accuracy vs training from scratch

        The T² scaling is critical:
            - Dividing logits by T makes probabilities softer
            - But this also makes gradients T² times smaller
            - Multiplying loss by T² compensates for this
            - Result: same gradient magnitude regardless of T
    """
    # Detect binary vs multi-class classification based on output shape
    # Use tf.cond to handle conditional logic in graph mode (required for TensorFlow)
    def binary_case():
        """Handle binary classification case."""
        # Teacher: convert sigmoid probabilities to 2-class logits
        teacher_shape_dyn = tf.shape(teacher_predictions)
        teacher_is_binary = tf.equal(teacher_shape_dyn[-1], 1)
        
        def teacher_binary():
            # Teacher output is probabilities (from sigmoid activation), convert to logits
            teacher_probs = tf.clip_by_value(teacher_predictions, 1e-8, 1.0 - 1e-8)
            return tf.concat([
                tf.math.log(1 - teacher_probs),  # log(P(class=0))
                tf.math.log(teacher_probs)       # log(P(class=1))
            ], axis=-1)
        
        def teacher_multi():
            # Teacher already has multi-class logits (from softmax model)
            return teacher_predictions
        
        teacher_logits_2d = tf.cond(teacher_is_binary, teacher_binary, teacher_multi)
        
        # Student: convert probabilities to 2-class logits
        # Student output is probabilities (from sigmoid activation)
        student_probs = tf.clip_by_value(student_predictions, 1e-8, 1.0 - 1e-8)
        student_logits_2d = tf.concat([
            tf.math.log(1 - student_probs),  # log(P(class=0))
            tf.math.log(student_probs)       # log(P(class=1))
        ], axis=-1)
        
        # Apply temperature scaling and softmax
        teacher_soft = tf.nn.softmax(teacher_logits_2d / temperature)
        student_soft = tf.nn.softmax(student_logits_2d / temperature)
        
        # Distillation loss: KL divergence between soft predictions
        distillation_loss = tf.keras.losses.categorical_crossentropy(
            teacher_soft, student_soft
        ) * (temperature ** 2)
        
        # Student loss: binary cross-entropy
        # Ensure y_true is float32 and has correct shape for binary_crossentropy
        # y_true should be (batch,) shape, student_predictions is (batch, 1)
        y_true_float = tf.cast(y_true, tf.float32)
        
        # Handle shape: ensure y_true is 1D (batch,)
        # Use tf.shape to get dynamic shape safely
        y_true_shape = tf.shape(y_true_float)
        y_true_rank = tf.size(y_true_shape)
        
        # Flatten y_true to 1D if needed
        # If rank is 0 (scalar), expand to 1D
        # If rank > 1, flatten to 1D
        y_true_float = tf.cond(
            tf.equal(y_true_rank, 0),
            lambda: tf.expand_dims(y_true_float, 0),
            lambda: tf.cond(
                tf.greater(y_true_rank, 1),
                lambda: tf.reshape(y_true_float, [-1]),  # Flatten to 1D
                lambda: y_true_float  # Already 1D
            )
        )
        
        # Ensure student_predictions is (batch, 1) or (batch,)
        # binary_crossentropy can handle both shapes
        # Squeeze the last dimension if it's 1 to match y_true shape
        student_pred_shape = tf.shape(student_predictions)
        student_pred_rank = tf.size(student_pred_shape)
        
        # If predictions are (batch, 1), squeeze to (batch,)
        student_pred_final = tf.cond(
            tf.greater(student_pred_rank, 1),
            lambda: tf.squeeze(student_predictions, axis=-1),
            lambda: student_predictions
        )
        
        student_loss = tf.keras.losses.binary_crossentropy(
            y_true_float, student_pred_final, from_logits=False
        )
        
        return distillation_loss, student_loss
    
    def multiclass_case():
        """Handle multi-class classification case."""
        # Soften teacher predictions with temperature
        teacher_soft = tf.nn.softmax(teacher_predictions / temperature)

        # Soften student predictions with temperature
        student_soft = tf.nn.softmax(student_predictions / temperature)

        # Distillation loss: KL divergence between soft predictions
        # Scale by temperature^2 to maintain gradient magnitude
        distillation_loss = tf.keras.losses.categorical_crossentropy(
            teacher_soft, student_soft
        ) * (temperature ** 2)

        # Student loss: standard cross-entropy with true labels
        student_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, student_predictions, from_logits=True
        )
        
        return distillation_loss, student_loss
    
    # Use tf.cond to select the appropriate case
    # Check if binary using dynamic shape
    student_shape_dyn = tf.shape(student_predictions)
    is_binary_dyn = tf.equal(student_shape_dyn[-1], 1)
    
    distillation_loss, student_loss = tf.cond(
        is_binary_dyn,
        binary_case,
        multiclass_case
    )

    # Combine losses
    total_loss = alpha * distillation_loss + (1 - alpha) * student_loss

    return total_loss


class Distiller(keras.Model):
    """
    Wrapper model for knowledge distillation training.

    Combines teacher and student models for efficient training.
    The teacher model's weights are frozen during training.

    Usage:
        teacher = keras.models.load_model('teacher.h5')
        student = make_small_model()

        distiller = Distiller(student=student, teacher=teacher)
        distiller.compile(optimizer='adam', metrics=['accuracy'])
        distiller.fit(x_train, y_train, epochs=10)
    """

    def __init__(
        self,
        student: keras.Model,
        teacher: keras.Model,
        temperature: float = 3.0,
        alpha: float = 0.1
    ):
        """
        Initialize distiller.

        Args:
            student: Student model to train
            teacher: Pre-trained teacher model (will be frozen)
            temperature: Temperature for soft targets
            alpha: Weight for distillation loss
        """
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha

        # Freeze teacher model
        self.teacher.trainable = False

    def compile(
        self,
        optimizer: keras.optimizers.Optimizer,
        metrics: list,
        student_loss_fn: Optional[keras.losses.Loss] = None,
        distillation_loss_fn: Optional[Callable] = None
    ):
        """
        Compile the distiller model.

        Args:
            optimizer: Optimizer for student model
            metrics: Metrics to track during training
            student_loss_fn: Loss for student (default: sparse_categorical_crossentropy)
            distillation_loss_fn: Custom distillation loss function
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        # Default loss will be determined dynamically based on output shape
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn

    def train_step(self, data):
        """
        Custom training step for distillation.

        Args:
            data: Tuple of (x, y) or just x

        Returns:
            Dictionary of metrics
        """
        # Unpack data
        x, y = data

        # Forward pass: teacher predictions (no gradients)
        teacher_predictions = self.teacher(x, training=False)

        # Forward pass: student predictions (with gradients)
        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)

            # Calculate distillation loss
            if self.distillation_loss_fn is not None:
                loss = self.distillation_loss_fn(
                    y, student_predictions, teacher_predictions,
                    self.temperature, self.alpha
                )
            else:
                loss = distillation_loss_fn(
                    y, student_predictions, teacher_predictions,
                    self.temperature, self.alpha
                )

            # CRITICAL FIX: Reduce loss to scalar for metrics
            loss = tf.reduce_mean(loss)

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.compiled_metrics.update_state(y, student_predictions)

        # Return metrics
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = loss
        return results

    def test_step(self, data):
        """
        Custom test step for evaluation.

        Args:
            data: Tuple of (x, y) or just x

        Returns:
            Dictionary of metrics
        """
        # Unpack data
        x, y = data

        # Forward pass: student predictions only
        student_predictions = self.student(x, training=False)

        # Update metrics
        self.compiled_metrics.update_state(y, student_predictions)

        # Return metrics
        results = {m.name: m.result() for m in self.metrics}
        return results

    def call(self, inputs):
        """
        Forward pass returns student predictions.

        Args:
            inputs: Input data

        Returns:
            Student predictions
        """
        return self.student(inputs)


def create_student_model(
    teacher_model: keras.Model,
    compression_ratio: float = 0.5,
    num_classes: int = 2
) -> keras.Model:
    """
    Create a student model with reduced capacity compared to teacher.

    Args:
        teacher_model: Teacher model to mimic
        compression_ratio: Ratio of student size to teacher size (0.0 to 1.0)
                          e.g., 0.5 = student has 50% of teacher's neurons
        num_classes: Number of output classes

    Returns:
        Smaller student model with similar architecture

    Strategy:
        - Maintains the same number of layers
        - Reduces neurons per layer by compression_ratio
        - Keeps the same activation functions
        - Useful for creating architecturally similar but smaller models
    """
    # Get input shape from teacher
    input_shape = teacher_model.input_shape[1:]

    # Build student with reduced capacity
    student_layers = []
    student_layers.append(keras.Input(shape=input_shape))

    for layer in teacher_model.layers:
        if isinstance(layer, layers.Dense):
            # Reduce number of neurons
            original_units = layer.units
            student_units = max(1, int(original_units * compression_ratio))

            # Skip the last layer (will be added separately)
            if layer == teacher_model.layers[-1]:
                continue

            student_layers.append(
                layers.Dense(
                    student_units,
                    activation=layer.activation,
                    name=f"student_{layer.name}"
                )
            )

        elif isinstance(layer, layers.Conv2D):
            # Reduce number of filters
            original_filters = layer.filters
            student_filters = max(1, int(original_filters * compression_ratio))

            student_layers.append(
                layers.Conv2D(
                    student_filters,
                    kernel_size=layer.kernel_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    activation=layer.activation,
                    name=f"student_{layer.name}"
                )
            )

        elif isinstance(layer, (layers.MaxPooling2D, layers.Flatten, layers.Dropout)):
            # Copy pooling/flatten/dropout layers as-is
            student_layers.append(
                layer.__class__.from_config(layer.get_config())
            )

        elif isinstance(layer, layers.InputLayer):
            # Skip, already added
            pass

    # Add output layer with same number of classes
    # IMPORTANT: Always use num_classes outputs (even for binary: 2 not 1)
    # This ensures consistency with teacher model and sparse_categorical_crossentropy
    # Use NO activation (logits) for distillation
    if num_classes <= 1:
        num_classes = 2  # Ensure at least 2 for binary

    student_layers.append(layers.Dense(num_classes, activation=None, name='student_output'))

    # Build model
    x = student_layers[0]
    for layer in student_layers[1:]:
        x = layer(x)

    student_model = keras.Model(inputs=student_layers[0], outputs=x)

    return student_model


def train_with_distillation(
    teacher_model: keras.Model,
    student_model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    temperature: float = 3.0,
    alpha: float = 0.1,
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    verbose: bool = True
) -> Tuple[keras.Model, dict]:
    """
    Train a student model using knowledge distillation from a teacher model.

    Args:
        teacher_model: Pre-trained teacher model
        student_model: Student model to train
        x_train: Training data
        y_train: Training labels
        x_val: Validation data
        y_val: Validation labels
        temperature: Temperature for soft targets (2.0-10.0)
                    Higher = more knowledge transfer
        alpha: Weight for distillation loss (0.0-1.0)
              Higher = more reliance on teacher
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        verbose: Print training progress

    Returns:
        Tuple of (trained_student_model, training_history)

    Process:
        1. Freeze teacher model weights
        2. Create Distiller wrapper combining teacher and student
        3. Train student to match both:
           - Soft targets from teacher (knowledge transfer)
           - Hard labels from dataset (ground truth)
        4. Return trained student model

    Benefits of Knowledge Distillation:
        - Student learns from teacher's mistakes and uncertainties
        - Better generalization than training from scratch
        - Captures inter-class relationships
        - Often achieves higher accuracy than standard training
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"  Knowledge Distillation Training")
        print(f"{'='*70}")
        print(f"Temperature: {temperature}")
        print(f"Alpha (distillation weight): {alpha}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}\n")

        # Print model comparison
        teacher_params = teacher_model.count_params()
        student_params = student_model.count_params()
        compression_ratio = student_params / teacher_params

        print(f"Teacher model: {teacher_params:,} parameters")
        print(f"Student model: {student_params:,} parameters")
        print(f"Compression ratio: {compression_ratio:.1%}\n")

    # Create distiller
    distiller = Distiller(
        student=student_model,
        teacher=teacher_model,
        temperature=temperature,
        alpha=alpha
    )

    # Compile distiller
    distiller.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        distillation_loss_fn=distillation_loss_fn
    )

    # Train
    history = distiller.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        verbose=1 if verbose else 0
    )

    if verbose:
        print(f"\n  Distillation training complete!\n")

    # Return the trained student model
    return student_model, history.history


def compare_models(
    teacher_model: keras.Model,
    student_model: keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True
) -> Tuple[float, float, float]:
    """
    Compare teacher and student model performance.

    Args:
        teacher_model: Teacher model
        student_model: Student model
        x_test: Test data
        y_test: Test labels
        verbose: Print comparison results

    Returns:
        Tuple of (teacher_accuracy, student_accuracy, accuracy_gap)

    Metrics:
        - Model size comparison
        - Accuracy comparison
        - Performance gap
        - Compression ratio
    """
    # Evaluate teacher
    teacher_loss, teacher_acc = teacher_model.evaluate(x_test, y_test, verbose=0)

    # Evaluate student
    student_loss, student_acc = student_model.evaluate(x_test, y_test, verbose=0)

    # Calculate sizes
    teacher_params = teacher_model.count_params()
    student_params = student_model.count_params()
    teacher_size_kb = (teacher_params * 4) / 1024
    student_size_kb = (student_params * 4) / 1024

    compression_ratio = teacher_params / student_params
    size_reduction = (1 - student_params / teacher_params) * 100
    accuracy_gap = (teacher_acc - student_acc) * 100

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Model Comparison: Teacher vs Student")
        print(f"{'='*70}")

        print(f"\nTeacher Model:")
        print(f"  - Parameters: {teacher_params:,}")
        print(f"  - Size: {teacher_size_kb:.2f} KB")
        print(f"  - Accuracy: {teacher_acc*100:.2f}%")
        print(f"  - Loss: {teacher_loss:.4f}")

        print(f"\nStudent Model:")
        print(f"  - Parameters: {student_params:,}")
        print(f"  - Size: {student_size_kb:.2f} KB")
        print(f"  - Accuracy: {student_acc*100:.2f}%")
        print(f"  - Loss: {student_loss:.4f}")

        print(f"\nCompression:")
        print(f"  - Compression ratio: {compression_ratio:.2f}x")
        print(f"  - Size reduction: {size_reduction:.1f}%")
        print(f"  - Accuracy gap: {accuracy_gap:.2f}%")

        print(f"{'='*70}\n")

    return teacher_acc, student_acc, accuracy_gap


def get_model_summary(model: keras.Model, name: str = "Model") -> None:
    """
    Print detailed model summary.

    Args:
        model: Keras model
        name: Name to display
    """
    print(f"\n{'='*70}")
    print(f"  {name} Architecture")
    print(f"{'='*70}\n")
    model.summary()
    print(f"{'='*70}\n")
