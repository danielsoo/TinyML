#!/usr/bin/env python3
"""
TFLite 모델을 엣지(Raspberry Pi, ESP32)에서 쓸 수 있는지 로컬에서 검증합니다.
입력 차원을 자동으로 읽어서, IDS(78차원) 등 어떤 모델이든 추론 + 지연 시간을 측정합니다.

Usage:
  # 기본: models/tflite/saved_model_qat_ptq.tflite
  python scripts/run_on_edge_test.py

  # 다른 모델 지정
  python scripts/run_on_edge_test.py --model models/tflite/saved_model_no_qat_ptq.tflite

  # 반복 횟수 (지연 시간 통계용)
  python scripts/run_on_edge_test.py --iters 100
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

# project root
ROOT = Path(__file__).resolve().parent.parent


def run_edge_test(model_path: str, iters: int = 50, warmup: int = 5) -> bool:
    try:
        import tensorflow as tf
    except ImportError:
        print("❌ tensorflow가 필요합니다: pip install tensorflow")
        return False

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ 모델 없음: {model_path}")
        return False

    print("=" * 60)
    print("TFLite 엣지 실행 테스트 (Raspberry Pi / ESP 검증용)")
    print("=" * 60)
    print(f"모델: {model_path}")
    print(f"크기: {model_path.stat().st_size / 1024:.2f} KB\n")

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    inp = input_details[0]
    shape = inp["shape"]
    dtype = inp["dtype"]
    # 배치 제외한 입력 차원 (예: [78] 또는 [1, 78])
    input_dim = shape[-1] if len(shape) > 1 else shape[0]
    print(f"입력 shape: {shape} (특성 수: {input_dim})")
    print(f"출력 shape: {output_details[0]['shape']}\n")

    # 더미 입력 (실제 배포 시에는 센서/패킷 특징으로 채움)
    dummy = np.random.randn(*shape).astype(dtype)

    # Warmup
    for _ in range(warmup):
        interpreter.set_tensor(inp["index"], dummy)
        interpreter.invoke()

    # 지연 시간 측정
    latencies_ms = []
    for _ in range(iters):
        interpreter.set_tensor(inp["index"], dummy)
        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

    interpreter.get_tensor(output_details[0]["index"])  # 출력 일단 읽기

    avg_ms = np.mean(latencies_ms)
    min_ms = np.min(latencies_ms)
    max_ms = np.max(latencies_ms)
    p50 = np.percentile(latencies_ms, 50)
    p99 = np.percentile(latencies_ms, 99)

    print("추론 지연 (로컬 CPU 기준)")
    print("-" * 40)
    print(f"  평균:   {avg_ms:.3f} ms")
    print(f"  최소:   {min_ms:.3f} ms")
    print(f"  최대:   {max_ms:.3f} ms")
    print(f"  P50:    {p50:.3f} ms")
    print(f"  P99:    {p99:.3f} ms")
    print()
    print("✅ 이 모델은 TFLite 인터프리터에서 정상 동작합니다.")
    print()
    print("=" * 60)
    print("라즈베리 파이에서 확인하는 방법")
    print("=" * 60)
    print("1. .tflite 파일을 RPi로 복사 (scp 등)")
    print("2. RPi에서: pip install tflite-runtime  (또는 tensorflow)")
    print("3. 아래와 같이 Python으로 추론:")
    print("   import tflite_runtime.interpreter as tflite")
    print("   interp = tflite.Interpreter(model_path='saved_model_qat_ptq.tflite')")
    print("   interp.allocate_tensors()")
    print("   # input shape에 맞게 배열 채운 뒤 set_tensor → invoke → get_tensor")
    print()
    print("=" * 60)
    print("ESP32에서 확인하는 방법")
    print("=" * 60)
    print("1. TFLite → C 배열 변환:")
    print("   python scripts/deploy_microcontroller.py --model models/tflite/saved_model_qat_ptq.tflite")
    print("2. 생성된 model_data.c / model_data.h 를 ESP32 프로젝트에 포함")
    print("3. TensorFlow Lite Micro 지원 op만 사용했는지 확인 (일부 op는 ESP에서 미지원)")
    print("   자세한 절차: docs/MICROCONTROLLER_DEPLOYMENT.md")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="TFLite 엣지 실행 테스트 (RPi/ESP 검증)")
    parser.add_argument(
        "--model",
        type=str,
        default=str(ROOT / "models" / "tflite" / "saved_model_qat_ptq.tflite"),
        help="TFLite 모델 경로",
    )
    parser.add_argument("--iters", type=int, default=50, help="추론 반복 횟수")
    parser.add_argument("--warmup", type=int, default=5, help="웜업 횟수")
    args = parser.parse_args()

    ok = run_edge_test(args.model, iters=args.iters, warmup=args.warmup)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
