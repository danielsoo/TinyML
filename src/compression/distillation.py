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

    Both teacher and student output num_classes dimensions:
    - Teacher: softmax probabilities, shape (batch, num_classes)
    - Student: raw logits, shape (batch, num_classes)

    Loss = alpha * KL(soft_teacher, soft_student) * T²
         + (1 - alpha) * sparse_categorical_crossentropy(y_true, student_logits)

    Args:
        y_true: True labels (sparse integer labels)
        student_predictions: Student model logits (batch, num_classes)
        teacher_predictions: Teacher model softmax probabilities (batch, num_classes)
        temperature: Temperature for softening distributions
        alpha: Weight for distillation loss

    Returns:
        Combined distillation loss (batch,)
    """
    # Convert teacher probabilities to logits for temperature scaling
    teacher_probs_clipped = tf.clip_by_value(teacher_predictions, 1e-8, 1.0)
    teacher_logits = tf.math.log(teacher_probs_clipped)

    # Apply temperature scaling and softmax to both
    teacher_soft = tf.nn.softmax(teacher_logits / temperature) # for loss function
    student_soft = tf.nn.softmax(student_predictions / temperature) 

    # Distillation loss: KL divergence (via cross-entropy) scaled by T²
    distillation_loss = tf.keras.losses.categorical_crossentropy(
        teacher_soft, student_soft
    ) * (temperature ** 2)

    # Student loss: standard cross-entropy with true labels
    student_loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, student_predictions, from_logits=True
    )

    # Combine: alpha * distillation + (1-alpha) * student
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
    Create a student model as a smaller version of the teacher's MLP architecture.

    The student mirrors the teacher's structure (same number of hidden layers)
    but with fewer neurons per layer, scaled by compression_ratio.

    Teacher (256 → 128 → 64 → 2):
    Student (128 →  64 → 32 → 2) at compression_ratio=0.5

    Args:
        teacher_model: Teacher model to compress
        compression_ratio: Fraction of neurons to keep (0.5 = half the neurons)
        num_classes: Number of output classes (always >= 2)

    Returns:
        Student model outputting logits (no activation on final layer)
    """
    if num_classes <= 1:
        num_classes = 2

    input_shape = teacher_model.input_shape[1:]

    # Extract hidden layer sizes from teacher (skip output layer)
    teacher_hidden_units = []
    for layer in teacher_model.layers:
        if isinstance(layer, layers.Dense):
            # Skip the output layer (last Dense layer)
            if layer == teacher_model.layers[-1]:
                continue
            teacher_hidden_units.append(layer.units)

    # Build student with reduced hidden layers
    student_hidden_units = [max(1, int(u * compression_ratio)) for u in teacher_hidden_units]

    # Build sequential model
    model = keras.Sequential(name="student_model")
    model.add(keras.Input(shape=input_shape))

    for i, units in enumerate(student_hidden_units):
        model.add(layers.Dense(units, activation="relu", name=f"student_dense_{i}"))

    # Output layer: num_classes outputs, NO activation (logits for distillation)
    model.add(layers.Dense(num_classes, activation=None, name="student_output"))

    return model


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
