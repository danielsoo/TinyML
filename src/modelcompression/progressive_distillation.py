"""
Progressive Knowledge Distillation for TensorFlow/Keras models.

Implements multi-stage distillation:
- Stage 1: Teacher → Student-1
- Stage 2: Teacher + Student-1 → Student-2 (ensemble distillation)

This progressive approach:
1. Creates intermediate models for smoother knowledge transfer
2. Combines multiple teachers (original + intermediate) for richer guidance
3. Achieves better compression with minimal accuracy loss
"""
import os
import json
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Union
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.modelcompression.distillation import (
    distillation_loss_fn,
    Distiller,
    create_student_model,
)


@dataclass
class DistillationStageResult:
    """Results from a single distillation stage."""
    stage: int
    student_name: str
    teacher_names: List[str]
    epochs: int
    final_loss: float
    final_accuracy: float
    model_params: int
    model_size_kb: float
    compression_ratio: float
    history: Dict[str, List[float]]


def ensemble_distillation_loss_fn(
    y_true: tf.Tensor,
    student_predictions: tf.Tensor,
    teacher_predictions_list: List[tf.Tensor],
    teacher_weights: List[float],
    temperature: float = 3.0,
    alpha: float = 0.1
) -> tf.Tensor:
    """
    Calculate ensemble distillation loss from multiple teachers.

    Args:
        y_true: True labels (sparse or one-hot)
        student_predictions: Student model predictions (logits)
        teacher_predictions_list: List of teacher predictions
        teacher_weights: Weights for each teacher (should sum to 1.0)
        temperature: Temperature for softening distributions
        alpha: Weight for distillation loss vs student loss

    Returns:
        Combined loss from all teachers
    """
    # Normalize teacher weights
    total_weight = sum(teacher_weights)
    normalized_weights = [w / total_weight for w in teacher_weights]

    # Calculate weighted average of teacher soft predictions
    def get_soft_predictions(predictions):
        """Convert predictions to soft targets.

        Handles both binary (1 sigmoid output) and multi-class (2+ softmax outputs).
        For binary teachers with 1 output, converts to 2-class probabilities.
        """
        num_outputs = predictions.shape[-1]

        # Static shape check (works for known shapes at graph build time)
        if num_outputs is not None and num_outputs == 1:
            # Binary teacher with sigmoid output (1 unit)
            p = tf.squeeze(predictions, axis=-1)
            eps = 1e-7
            logits_2 = tf.stack([tf.math.log(1 - p + eps), tf.math.log(p + eps)], axis=-1)
            return tf.nn.softmax(logits_2 / temperature)
        else:
            # Multi-class or binary with 2 softmax outputs
            return tf.nn.softmax(predictions / temperature)

    # Weighted ensemble of teacher soft predictions
    ensemble_soft = None
    for teacher_preds, weight in zip(teacher_predictions_list, normalized_weights):
        teacher_soft = get_soft_predictions(teacher_preds)
        if ensemble_soft is None:
            ensemble_soft = weight * teacher_soft
        else:
            ensemble_soft = ensemble_soft + weight * teacher_soft

    # Student soft predictions
    student_soft = tf.nn.softmax(student_predictions / temperature)

    # Distillation loss: KL divergence with ensemble teacher
    distillation_loss = tf.keras.losses.categorical_crossentropy(
        ensemble_soft, student_soft
    ) * (temperature ** 2)

    # Student loss: standard cross-entropy with true labels
    student_loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, student_predictions, from_logits=True
    )

    # Combine losses
    total_loss = alpha * distillation_loss + (1 - alpha) * student_loss

    return total_loss


class EnsembleDistiller(keras.Model):
    """
    Distiller that learns from multiple teacher models.

    Used in progressive distillation where later students learn
    from both the original teacher and intermediate students.
    """

    def __init__(
        self,
        student: keras.Model,
        teachers: List[keras.Model],
        teacher_weights: Optional[List[float]] = None,
        temperature: float = 3.0,
        alpha: float = 0.1
    ):
        """
        Initialize ensemble distiller.

        Args:
            student: Student model to train
            teachers: List of pre-trained teacher models
            teacher_weights: Weights for each teacher (default: equal weights)
            temperature: Temperature for soft targets
            alpha: Weight for distillation loss
        """
        super().__init__()
        self.student = student
        self.teachers = teachers
        self.teacher_weights = teacher_weights or [1.0 / len(teachers)] * len(teachers)
        self.temperature = temperature
        self.alpha = alpha

        # Freeze all teacher models
        for teacher in self.teachers:
            teacher.trainable = False

    def compile(
        self,
        optimizer: keras.optimizers.Optimizer,
        metrics: list,
        student_loss_fn: Optional[keras.losses.Loss] = None,
    ):
        """Compile the ensemble distiller model."""
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn or keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )

    def train_step(self, data):
        """Custom training step for ensemble distillation."""
        x, y = data

        # Get predictions from all teachers (no gradients)
        teacher_predictions = [teacher(x, training=False) for teacher in self.teachers]

        # Forward pass: student predictions (with gradients)
        with tf.GradientTape() as tape:
            student_predictions = self.student(x, training=True)

            # Calculate ensemble distillation loss
            loss = ensemble_distillation_loss_fn(
                y, student_predictions, teacher_predictions,
                self.teacher_weights, self.temperature, self.alpha
            )
            # Reduce to scalar (mean over batch)
            loss = tf.reduce_mean(loss)

        # Compute gradients and update weights
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.compiled_metrics.update_state(y, student_predictions)

        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = loss
        return results

    def test_step(self, data):
        """Custom test step for evaluation."""
        x, y = data
        student_predictions = self.student(x, training=False)
        self.compiled_metrics.update_state(y, student_predictions)
        results = {m.name: m.result() for m in self.metrics}
        return results

    def call(self, inputs):
        """Forward pass returns student predictions."""
        return self.student(inputs)


def create_progressive_student(
    input_shape: Tuple[int, ...],
    num_classes: int,
    hidden_units: List[int],
    dropout_rate: float = 0.2,
    use_batch_norm: bool = False,
    name: str = "student"
) -> keras.Model:
    """
    Create a student model with specified architecture.

    Args:
        input_shape: Input shape (excluding batch dimension)
        num_classes: Number of output classes
        hidden_units: List of hidden layer sizes, e.g., [256, 128]
        dropout_rate: Dropout rate between layers
        use_batch_norm: Whether to use batch normalization
        name: Model name

    Returns:
        Compiled student model
    """
    model = keras.Sequential(name=name)
    model.add(keras.Input(shape=input_shape))

    for i, units in enumerate(hidden_units):
        model.add(layers.Dense(units, activation='relu', name=f'{name}_dense_{i}'))
        if use_batch_norm:
            model.add(layers.BatchNormalization(name=f'{name}_bn_{i}'))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate, name=f'{name}_dropout_{i}'))

    # Output layer
    # Always use 2 softmax outputs for binary classification so sparse_categorical_crossentropy works
    # (sigmoid with 1 output would require binary_crossentropy which doesn't match distillation loss)
    if num_classes <= 2:
        model.add(layers.Dense(2, activation='softmax', name=f'{name}_output'))
    else:
        model.add(layers.Dense(num_classes, activation='softmax', name=f'{name}_output'))

    return model


class ProgressiveDistillation:
    """
    Progressive Knowledge Distillation Pipeline.

    Implements multi-stage distillation with configurable stages:
    - Stage 1: Teacher → Student-1 (e.g., 512 → 256)
    - Stage 2: Teacher + Student-1 → Student-2 (e.g., 256 → 128)
    - Stage N: All previous models → Student-N (ensemble distillation)

    Example: 512 → 256 → 128
        Stage 1: Teacher(512) → Student-1(256)
        Stage 2: Teacher(512) + Student-1(256) → Student-2(128)

    Benefits:
    - Smoother knowledge transfer through intermediate models
    - Ensemble learning from multiple teachers
    - Better final model accuracy at high compression ratios
    """

    def __init__(
        self,
        teacher_model: keras.Model,
        num_classes: int = 2,
        temperature: float = 3.0,
        alpha: float = 0.3,
        verbose: bool = True
    ):
        """
        Initialize progressive distillation pipeline.

        Args:
            teacher_model: Pre-trained teacher model
            num_classes: Number of output classes
            temperature: Temperature for soft targets
            alpha: Weight for distillation loss (higher = more teacher influence)
            verbose: Print progress information
        """
        self.teacher = teacher_model
        self.num_classes = num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.verbose = verbose

        # Store intermediate models and results
        self.students: List[keras.Model] = []
        self.stage_results: List[DistillationStageResult] = []

        # Freeze teacher
        self.teacher.trainable = False

        # Get input shape
        self.input_shape = teacher_model.input_shape[1:]

        if verbose:
            teacher_params = teacher_model.count_params()
            print(f"\n{'='*70}")
            print("  Progressive Knowledge Distillation")
            print(f"{'='*70}")
            print(f"Teacher model: {teacher_params:,} parameters")
            print(f"Input shape: {self.input_shape}")
            print(f"Num classes: {num_classes}")
            print(f"Temperature: {temperature}")
            print(f"Alpha: {alpha}")
            print(f"{'='*70}\n")

    def _get_model_info(self, model: keras.Model) -> Dict[str, Any]:
        """Get model size and parameter count."""
        total_params = model.count_params()
        model_size_bytes = sum(w.nbytes for w in model.get_weights())
        return {
            'params': total_params,
            'size_kb': model_size_bytes / 1024,
            'size_mb': model_size_bytes / (1024 * 1024),
        }

    def _compile_model(self, model: keras.Model, learning_rate: float = 0.001):
        """Compile a model for training."""
        # Always use sparse_categorical_crossentropy since students have 2+ softmax outputs
        # (even for binary classification, we use 2 softmax outputs instead of 1 sigmoid)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def distill_stage(
        self,
        stage_num: int,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        student_units: List[int],
        teacher_weights: Optional[List[float]] = None,
        epochs: int = 10,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        dropout_rate: float = 0.2,
    ) -> Tuple[keras.Model, DistillationStageResult]:
        """
        Generic distillation stage that can handle any stage number.

        For stage 1: Uses only the original teacher
        For stage 2+: Uses ensemble of teacher + all previous students

        Args:
            stage_num: Stage number (1, 2, 3, ...)
            x_train, y_train: Training data
            x_val, y_val: Validation data
            student_units: Hidden layer sizes for this student, e.g., [256, 128]
            teacher_weights: Weights for each teacher in ensemble (None = equal weights)
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            dropout_rate: Dropout rate for student model

        Returns:
            Trained student model and stage results
        """
        student_name = f"Student-{stage_num}"

        if self.verbose:
            print(f"\n{'='*70}")
            if stage_num == 1:
                print(f"  Stage {stage_num}: Teacher → {student_name}")
            else:
                teachers_str = "Teacher + " + " + ".join([f"Student-{i}" for i in range(1, stage_num)])
                print(f"  Stage {stage_num}: {teachers_str} → {student_name} (Ensemble)")
            print(f"{'='*70}\n")

        # Create student model
        student = create_progressive_student(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            hidden_units=student_units,
            dropout_rate=dropout_rate,
            name=f"student_{stage_num}"
        )

        student_info = self._get_model_info(student)
        teacher_info = self._get_model_info(self.teacher)

        if self.verbose:
            print(f"{student_name} architecture: {student_units}")
            print(f"{student_name} parameters: {student_info['params']:,}")
            print(f"Compression from Teacher: {teacher_info['params']/student_info['params']:.2f}x\n")

        # Build list of teachers for this stage
        if stage_num == 1:
            # Stage 1: Only use original teacher
            teachers = [self.teacher]
            teacher_names = ["Teacher"]
        else:
            # Stage 2+: Use teacher + all previous students
            teachers = [self.teacher] + self.students
            teacher_names = ["Teacher"] + [f"Student-{i}" for i in range(1, stage_num)]

        # Set default teacher weights if not provided
        if teacher_weights is None:
            # Default: teacher gets 50%, rest split equally among students
            if len(teachers) == 1:
                teacher_weights = [1.0]
            else:
                teacher_weight = 0.5
                student_weight = 0.5 / (len(teachers) - 1)
                teacher_weights = [teacher_weight] + [student_weight] * (len(teachers) - 1)

        if self.verbose and len(teachers) > 1:
            print("Teacher weights:")
            for name, weight in zip(teacher_names, teacher_weights):
                print(f"  - {name}: {weight:.2f}")
            print()

        # Create appropriate distiller
        if stage_num == 1:
            # Single teacher distillation
            distiller = Distiller(
                student=student,
                teacher=self.teacher,
                temperature=self.temperature,
                alpha=self.alpha
            )
        else:
            # Ensemble distillation
            distiller = EnsembleDistiller(
                student=student,
                teachers=teachers,
                teacher_weights=teacher_weights,
                temperature=self.temperature,
                alpha=self.alpha
            )

        distiller.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['accuracy']
        )

        # Train
        if self.verbose:
            print(f"Training {student_name} with {'ensemble ' if stage_num > 1 else ''}distillation...")

        history = distiller.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1 if self.verbose else 0
        )

        # Compile student for standalone use
        self._compile_model(student, learning_rate)

        # Evaluate
        loss, acc = student.evaluate(x_val, y_val, verbose=0)

        # Store results
        result = DistillationStageResult(
            stage=stage_num,
            student_name=student_name,
            teacher_names=teacher_names,
            epochs=epochs,
            final_loss=float(loss),
            final_accuracy=float(acc),
            model_params=student_info['params'],
            model_size_kb=student_info['size_kb'],
            compression_ratio=teacher_info['params'] / student_info['params'],
            history={k: [float(v) for v in vals] for k, vals in history.history.items()}
        )

        self.students.append(student)
        self.stage_results.append(result)

        if self.verbose:
            print(f"\nStage {stage_num} Complete:")
            print(f"  - Accuracy: {acc:.4f} ({acc*100:.2f}%)")
            print(f"  - Loss: {loss:.4f}")
            print(f"  - Size: {student_info['size_kb']:.2f} KB")
            print(f"  - Compression: {result.compression_ratio:.2f}x")

        return student, result

    def distill_stage1(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        student1_units: List[int] = None,
        epochs: int = 20,
        batch_size: int = 128,
        learning_rate: float = 0.001,
    ) -> Tuple[keras.Model, DistillationStageResult]:
        """
        Stage 1: Distill knowledge from Teacher → Student-1.

        Args:
            x_train, y_train: Training data
            x_val, y_val: Validation data
            student1_units: Hidden layer sizes for Student-1
                           Default: half of teacher's hidden units
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate

        Returns:
            Trained Student-1 model and stage results
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print("  Stage 1: Teacher → Student-1")
            print(f"{'='*70}\n")

        # Determine student architecture
        if student1_units is None:
            # Default: reduce teacher units by half
            teacher_info = self._get_model_info(self.teacher)
            # Estimate based on teacher size
            student1_units = [256, 128]  # Default intermediate size

        # Create Student-1
        student1 = create_progressive_student(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            hidden_units=student1_units,
            dropout_rate=0.2,
            name="student_1"
        )

        student1_info = self._get_model_info(student1)
        teacher_info = self._get_model_info(self.teacher)

        if self.verbose:
            print(f"Student-1 architecture: {student1_units}")
            print(f"Student-1 parameters: {student1_info['params']:,}")
            print(f"Compression: {teacher_info['params']/student1_info['params']:.2f}x\n")

        # Create distiller
        distiller = Distiller(
            student=student1,
            teacher=self.teacher,
            temperature=self.temperature,
            alpha=self.alpha
        )

        distiller.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['accuracy']
        )

        # Train with distillation
        if self.verbose:
            print("Training Student-1 with knowledge distillation...")

        history = distiller.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1 if self.verbose else 0
        )

        # Compile student for standalone use
        self._compile_model(student1, learning_rate)

        # Evaluate
        loss, acc = student1.evaluate(x_val, y_val, verbose=0)

        # Store results
        result = DistillationStageResult(
            stage=1,
            student_name="Student-1",
            teacher_names=["Teacher"],
            epochs=epochs,
            final_loss=float(loss),
            final_accuracy=float(acc),
            model_params=student1_info['params'],
            model_size_kb=student1_info['size_kb'],
            compression_ratio=teacher_info['params'] / student1_info['params'],
            history={k: [float(v) for v in vals] for k, vals in history.history.items()}
        )

        self.students.append(student1)
        self.stage_results.append(result)

        if self.verbose:
            print(f"\nStage 1 Complete:")
            print(f"  - Accuracy: {acc:.4f} ({acc*100:.2f}%)")
            print(f"  - Loss: {loss:.4f}")
            print(f"  - Size: {student1_info['size_kb']:.2f} KB")
            print(f"  - Compression: {result.compression_ratio:.2f}x")

        return student1, result

    def distill_stage2(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        student2_units: List[int] = None,
        teacher_weight: float = 0.6,
        student1_weight: float = 0.4,
        epochs: int = 20,
        batch_size: int = 128,
        learning_rate: float = 0.001,
    ) -> Tuple[keras.Model, DistillationStageResult]:
        """
        Stage 2: Distill knowledge from Teacher + Student-1 → Student-2.

        Uses ensemble distillation where Student-2 learns from both
        the original Teacher and the intermediate Student-1.

        Args:
            x_train, y_train: Training data
            x_val, y_val: Validation data
            student2_units: Hidden layer sizes for Student-2
                           Default: half of Student-1's units
            teacher_weight: Weight for Teacher's contribution (0-1)
            student1_weight: Weight for Student-1's contribution (0-1)
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate

        Returns:
            Trained Student-2 model and stage results
        """
        if len(self.students) < 1:
            raise ValueError("Must run distill_stage1() before distill_stage2()")

        student1 = self.students[0]

        if self.verbose:
            print(f"\n{'='*70}")
            print("  Stage 2: Teacher + Student-1 → Student-2 (Ensemble)")
            print(f"{'='*70}\n")
            print(f"Teacher weight: {teacher_weight:.2f}")
            print(f"Student-1 weight: {student1_weight:.2f}")

        # Determine student architecture
        if student2_units is None:
            # Default: smaller than Student-1
            student2_units = [128, 64]

        # Create Student-2
        student2 = create_progressive_student(
            input_shape=self.input_shape,
            num_classes=self.num_classes,
            hidden_units=student2_units,
            dropout_rate=0.15,
            name="student_2"
        )

        student2_info = self._get_model_info(student2)
        teacher_info = self._get_model_info(self.teacher)

        if self.verbose:
            print(f"Student-2 architecture: {student2_units}")
            print(f"Student-2 parameters: {student2_info['params']:,}")
            print(f"Compression from Teacher: {teacher_info['params']/student2_info['params']:.2f}x\n")

        # Create ensemble distiller
        ensemble_distiller = EnsembleDistiller(
            student=student2,
            teachers=[self.teacher, student1],
            teacher_weights=[teacher_weight, student1_weight],
            temperature=self.temperature,
            alpha=self.alpha
        )

        ensemble_distiller.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=['accuracy']
        )

        # Train with ensemble distillation
        if self.verbose:
            print("Training Student-2 with ensemble distillation...")

        history = ensemble_distiller.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1 if self.verbose else 0
        )

        # Compile student for standalone use
        self._compile_model(student2, learning_rate)

        # Evaluate
        loss, acc = student2.evaluate(x_val, y_val, verbose=0)

        # Store results
        result = DistillationStageResult(
            stage=2,
            student_name="Student-2",
            teacher_names=["Teacher", "Student-1"],
            epochs=epochs,
            final_loss=float(loss),
            final_accuracy=float(acc),
            model_params=student2_info['params'],
            model_size_kb=student2_info['size_kb'],
            compression_ratio=teacher_info['params'] / student2_info['params'],
            history={k: [float(v) for v in vals] for k, vals in history.history.items()}
        )

        self.students.append(student2)
        self.stage_results.append(result)

        if self.verbose:
            print(f"\nStage 2 Complete:")
            print(f"  - Accuracy: {acc:.4f} ({acc*100:.2f}%)")
            print(f"  - Loss: {loss:.4f}")
            print(f"  - Size: {student2_info['size_kb']:.2f} KB")
            print(f"  - Compression: {result.compression_ratio:.2f}x")

        return student2, result

    def run_n_stage_pipeline(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        stage_units: List[List[int]],
        epochs_per_stage: Union[int, List[int]] = 10,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        dropout_rates: Optional[List[float]] = None,
    ) -> Tuple[List[keras.Model], List[DistillationStageResult]]:
        """
        Run N-stage progressive distillation pipeline.

        Example for 512 → 256 → 128:
            stage_units = [[512, 256], [256, 128], [128, 64]]
            This creates:
              Stage 1: Teacher → Student-1 (512, 256 hidden units)
              Stage 2: Teacher + Student-1 → Student-2 (256, 128 hidden units)
              Stage 3: Teacher + Student-1 + Student-2 → Student-3 (128, 64 hidden units)

        Args:
            x_train, y_train: Training data
            x_val, y_val: Validation data
            stage_units: List of hidden unit configurations for each stage
                         e.g., [[512, 256], [256, 128], [128, 64]] for 3 stages
            epochs_per_stage: Epochs for each stage (int for all same, or list)
            batch_size: Batch size for training
            learning_rate: Learning rate
            dropout_rates: Dropout rate for each stage (default: 0.2 for all)

        Returns:
            Tuple of (list of all students, list of stage results)
        """
        num_stages = len(stage_units)

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  Running {num_stages}-Stage Progressive Distillation")
            print(f"{'='*70}")
            for i, units in enumerate(stage_units):
                print(f"  Stage {i+1}: {units}")
            print(f"{'='*70}\n")

        # Handle epochs as int or list
        if isinstance(epochs_per_stage, int):
            epochs_list = [epochs_per_stage] * num_stages
        else:
            epochs_list = epochs_per_stage
            if len(epochs_list) < num_stages:
                epochs_list = epochs_list + [epochs_list[-1]] * (num_stages - len(epochs_list))

        # Handle dropout rates
        if dropout_rates is None:
            dropout_rates = [0.2] * num_stages
        elif len(dropout_rates) < num_stages:
            dropout_rates = dropout_rates + [dropout_rates[-1]] * (num_stages - len(dropout_rates))

        # Run each stage
        for stage_num in range(1, num_stages + 1):
            idx = stage_num - 1
            self.distill_stage(
                stage_num=stage_num,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                student_units=stage_units[idx],
                epochs=epochs_list[idx],
                batch_size=batch_size,
                learning_rate=learning_rate,
                dropout_rate=dropout_rates[idx],
            )

        # Print summary
        if self.verbose:
            self.print_summary()

        return self.students, self.stage_results

    def run_full_pipeline(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        student1_units: List[int] = None,
        student2_units: List[int] = None,
        epochs_stage1: int = 10,
        epochs_stage2: int = 10,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        teacher_weight: float = 0.6,
        student1_weight: float = 0.4,
    ) -> Tuple[keras.Model, keras.Model, List[DistillationStageResult]]:
        """
        Run the full 2-stage progressive distillation pipeline (legacy method).

        For N-stage distillation, use run_n_stage_pipeline() instead.

        Args:
            x_train, y_train: Training data
            x_val, y_val: Validation data
            student1_units: Hidden units for Student-1
            student2_units: Hidden units for Student-2
            epochs_stage1: Epochs for Stage 1
            epochs_stage2: Epochs for Stage 2
            batch_size: Batch size for training
            learning_rate: Learning rate
            teacher_weight: Teacher weight in Stage 2
            student1_weight: Student-1 weight in Stage 2

        Returns:
            Tuple of (Student-1, Student-2, list of stage results)
        """
        # Stage 1: Teacher → Student-1
        student1, result1 = self.distill_stage1(
            x_train, y_train, x_val, y_val,
            student1_units=student1_units,
            epochs=epochs_stage1,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        # Stage 2: Teacher + Student-1 → Student-2
        student2, result2 = self.distill_stage2(
            x_train, y_train, x_val, y_val,
            student2_units=student2_units,
            teacher_weight=teacher_weight,
            student1_weight=student1_weight,
            epochs=epochs_stage2,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        # Print summary
        if self.verbose:
            self.print_summary()

        return student1, student2, self.stage_results

    def print_summary(self):
        """Print summary of all distillation stages."""
        print(f"\n{'='*70}")
        print("  Progressive Distillation Summary")
        print(f"{'='*70}\n")

        teacher_info = self._get_model_info(self.teacher)
        print(f"Teacher: {teacher_info['params']:,} params, {teacher_info['size_kb']:.2f} KB\n")

        for result in self.stage_results:
            print(f"Stage {result.stage}: {result.student_name}")
            print(f"  Teachers: {', '.join(result.teacher_names)}")
            print(f"  Parameters: {result.model_params:,}")
            print(f"  Size: {result.model_size_kb:.2f} KB")
            print(f"  Compression: {result.compression_ratio:.2f}x")
            print(f"  Accuracy: {result.final_accuracy:.4f} ({result.final_accuracy*100:.2f}%)")
            print(f"  Loss: {result.final_loss:.4f}")
            print()

        # Final comparison
        if len(self.stage_results) >= 1:
            final_result = self.stage_results[-1]
            final_name = f"Student-{len(self.stage_results)}"
            print(f"Final Model ({final_name}):")
            print(f"  - Total compression: {final_result.compression_ratio:.2f}x")
            print(f"  - Size reduction: {(1 - 1/final_result.compression_ratio)*100:.1f}%")
            print(f"  - Final accuracy: {final_result.final_accuracy*100:.2f}%")

            # Show progression
            if len(self.stage_results) > 1:
                print(f"\nAccuracy progression:")
                for result in self.stage_results:
                    print(f"  Stage {result.stage}: {result.final_accuracy*100:.2f}%")

        print(f"{'='*70}\n")

    def save_results(self, output_dir: str):
        """Save models and results to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save models
        for i, student in enumerate(self.students):
            model_path = output_path / f"student_{i+1}.h5"
            student.save(model_path)
            print(f"Saved: {model_path}")

        # Save results as JSON
        results_data = {
            'teacher_params': self.teacher.count_params(),
            'temperature': self.temperature,
            'alpha': self.alpha,
            'stages': [
                {
                    'stage': r.stage,
                    'student_name': r.student_name,
                    'teacher_names': r.teacher_names,
                    'epochs': r.epochs,
                    'final_loss': r.final_loss,
                    'final_accuracy': r.final_accuracy,
                    'model_params': r.model_params,
                    'model_size_kb': r.model_size_kb,
                    'compression_ratio': r.compression_ratio,
                    'history': r.history
                }
                for r in self.stage_results
            ]
        }

        results_path = output_path / "progressive_distillation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"Saved: {results_path}")


def run_progressive_distillation(
    teacher_model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int = 2,
    student1_units: List[int] = None,
    student2_units: List[int] = None,
    temperature: float = 3.0,
    alpha: float = 0.3,
    epochs_stage1: int = 10,
    epochs_stage2: int = 10,
    batch_size: int = 128,
    output_dir: str = None,
) -> Tuple[keras.Model, keras.Model, List[DistillationStageResult]]:
    """
    Convenience function to run 2-stage progressive distillation.

    For N-stage distillation (e.g., 512 → 256 → 128), use
    run_n_stage_distillation() instead.

    Args:
        teacher_model: Pre-trained teacher model
        x_train, y_train: Training data
        x_val, y_val: Validation data
        num_classes: Number of output classes
        student1_units: Hidden units for Student-1 (default: [256, 128])
        student2_units: Hidden units for Student-2 (default: [128, 64])
        temperature: Distillation temperature
        alpha: Distillation loss weight
        epochs_stage1: Epochs for Stage 1
        epochs_stage2: Epochs for Stage 2
        batch_size: Batch size
        output_dir: Directory to save results (optional)

    Returns:
        Tuple of (Student-1, Student-2, stage results)
    """
    # Set defaults
    if student1_units is None:
        student1_units = [256, 128]
    if student2_units is None:
        student2_units = [128, 64]

    # Create pipeline
    pipeline = ProgressiveDistillation(
        teacher_model=teacher_model,
        num_classes=num_classes,
        temperature=temperature,
        alpha=alpha,
        verbose=True
    )

    # Run full pipeline
    student1, student2, results = pipeline.run_full_pipeline(
        x_train, y_train, x_val, y_val,
        student1_units=student1_units,
        student2_units=student2_units,
        epochs_stage1=epochs_stage1,
        epochs_stage2=epochs_stage2,
        batch_size=batch_size
    )

    # Save results if output directory specified
    if output_dir:
        pipeline.save_results(output_dir)

    return student1, student2, results


def run_n_stage_distillation(
    teacher_model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    stage_units: List[List[int]],
    num_classes: int = 2,
    temperature: float = 3.0,
    alpha: float = 0.3,
    epochs_per_stage: Union[int, List[int]] = 10,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    output_dir: str = None,
) -> Tuple[List[keras.Model], List[DistillationStageResult]]:
    """
    Run N-stage progressive distillation.

    Example for Teacher(512) → 256 → 128:
        stage_units = [[256, 128], [128, 64]]

        This creates:
          Stage 1: Teacher → Student-1 (256, 128 hidden units)
          Stage 2: Teacher + Student-1 → Student-2 (128, 64 hidden units)

    Example for 512 → 256 → 128 → 64 (3 stages):
        stage_units = [[256, 128], [128, 64], [64, 32]]

    Args:
        teacher_model: Pre-trained teacher model (e.g., with 512 hidden units)
        x_train, y_train: Training data
        x_val, y_val: Validation data
        stage_units: List of hidden unit configurations for each stage
        num_classes: Number of output classes
        temperature: Distillation temperature
        alpha: Distillation loss weight
        epochs_per_stage: Epochs for each stage (int or list)
        batch_size: Batch size
        learning_rate: Learning rate
        output_dir: Directory to save results (optional)

    Returns:
        Tuple of (list of all students, stage results)
    """
    # Create pipeline
    pipeline = ProgressiveDistillation(
        teacher_model=teacher_model,
        num_classes=num_classes,
        temperature=temperature,
        alpha=alpha,
        verbose=True
    )

    # Run N-stage pipeline
    students, results = pipeline.run_n_stage_pipeline(
        x_train, y_train, x_val, y_val,
        stage_units=stage_units,
        epochs_per_stage=epochs_per_stage,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    # Save results if output directory specified
    if output_dir:
        pipeline.save_results(output_dir)

    return students, results


if __name__ == "__main__":
    """Example usage of progressive distillation."""
    import argparse

    parser = argparse.ArgumentParser(description="Progressive Knowledge Distillation")
    parser.add_argument("--teacher", type=str, required=True, help="Path to teacher model")
    parser.add_argument("--output", type=str, default="models/distilled", help="Output directory")
    parser.add_argument("--epochs1", type=int, default=20, help="Epochs for Stage 1")
    parser.add_argument("--epochs2", type=int, default=20, help="Epochs for Stage 2")
    parser.add_argument("--temperature", type=float, default=3.0, help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.3, help="Distillation loss weight")

    args = parser.parse_args()

    # Load teacher model
    print(f"Loading teacher model from: {args.teacher}")
    teacher = keras.models.load_model(args.teacher)

    # For demo, create synthetic data
    print("Note: This demo uses synthetic data. Replace with real data for actual use.")
    input_shape = teacher.input_shape[1:]
    x_train = np.random.randn(1000, *input_shape).astype(np.float32)
    y_train = np.random.randint(0, 2, 1000)
    x_val = np.random.randn(200, *input_shape).astype(np.float32)
    y_val = np.random.randint(0, 2, 200)

    # Run progressive distillation
    student1, student2, results = run_progressive_distillation(
        teacher_model=teacher,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        num_classes=2,
        temperature=args.temperature,
        alpha=args.alpha,
        epochs_stage1=args.epochs1,
        epochs_stage2=args.epochs2,
        output_dir=args.output
    )

    print("\nProgressive distillation complete!")
    print(f"Student-1 saved to: {args.output}/student_1.h5")
    print(f"Student-2 saved to: {args.output}/student_2.h5")
