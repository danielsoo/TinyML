#!/usr/bin/env python3
"""
Quick GPU check: TensorFlow가 GPU를 보는지, VRAM 사용 가능한지 확인합니다.

Usage (프로젝트 루트에서):
  python scripts/check_gpu.py

vast.ai 등 원격에서 GPU 인스턴스인지 확인할 때 실행하면 됩니다.
"""

import sys


def main():
    print("=" * 60)
    print("  GPU check (TensorFlow)")
    print("=" * 60)

    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not installed. pip install tensorflow")
        return 1

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU devices found. Training will use CPU.")
        print("  On vast.ai: rent an instance with GPU (e.g. RTX 4090).")
        print("  On Colab: Runtime → Change runtime type → GPU")
        print(f"\nTensorFlow version: {tf.__version__}")
        return 0

    # VRAM 메모리 증가 방식 설정 (권장)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

    print(f"GPU devices: {len(gpus)}")
    for g in gpus:
        print(f"  - {g.name}")
    print("VRAM will be used for training (memory_growth enabled).")
    print(f"\nTensorFlow version: {tf.__version__}")

    # 실제로 GPU에서 연산 되는지 간단히 확인 (선택)
    try:
        with tf.device("/GPU:0"):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0], [1.0]])
            c = tf.linalg.matmul(a, b)
        print("Quick test: matmul on GPU:0 OK")
    except Exception as e:
        print(f"Quick test on GPU:0 failed: {e}")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
