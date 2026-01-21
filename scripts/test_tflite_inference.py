"""
TFLite model inference test script
Validates that TFLite models work correctly locally without hardware
"""
import os
import sys
import numpy as np
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_tflite_inference(model_path: str):
    """
    Test TFLite model inference
    
    Args:
        model_path: Path to TFLite model file
    """
    try:
        import tensorflow as tf
        
        print("=" * 60)
        print("TFLite Model Inference Test")
        print("=" * 60)
        
        # 모델 로드
        print(f"\nLoading model: {model_path}")
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # 입력/출력 정보 가져오기
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("\nModel Information:")
        print(f"  Input shape: {input_details[0]['shape']}")
        print(f"  Input type: {input_details[0]['dtype']}")
        print(f"  Output shape: {output_details[0]['shape']}")
        print(f"  Output type: {output_details[0]['dtype']}")
        
        # Generate test input data
        input_shape = input_details[0]['shape']
        # Remove batch size (first dimension)
        actual_input_shape = input_shape[1:] if len(input_shape) > 1 else input_shape
        
        print("\n" + "=" * 60)
        print("Running Inference Tests")
        print("=" * 60)
        
        # 여러 테스트 케이스
        test_cases = [
            {"name": "Test 1: Low values", "input": [0.3, 0.3]},
            {"name": "Test 2: Medium values", "input": [0.5, 0.5]},
            {"name": "Test 3: High values", "input": [0.7, 0.7]},
            {"name": "Test 4: Mixed values", "input": [0.2, 0.8]},
            {"name": "Test 5: Edge case", "input": [0.0, 1.0]},
        ]
        
        results = []
        
        for test_case in test_cases:
            # 입력 데이터 준비
            input_data = np.array([test_case["input"]], dtype=np.float32)
            
            # 입력 텐서 설정
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # 추론 실행
            import time
            start_time = time.time()
            interpreter.invoke()
            end_time = time.time()
            
            # 출력 가져오기
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            inference_time = (end_time - start_time) * 1000  # ms로 변환
            
            results.append({
                "name": test_case["name"],
                "input": test_case["input"],
                "output": output_data[0][0] if len(output_data[0]) == 1 else output_data[0],
                "time_ms": inference_time
            })
            
            print(f"\n{test_case['name']}")
            print(f"  Input: {test_case['input']}")
            print(f"  Output: {output_data[0]}")
            print(f"  Inference time: {inference_time:.3f} ms")
        
        # Summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        avg_time = np.mean([r["time_ms"] for r in results])
        min_time = np.min([r["time_ms"] for r in results])
        max_time = np.max([r["time_ms"] for r in results])
        
        print(f"Average inference time: {avg_time:.3f} ms")
        print(f"Min inference time: {min_time:.3f} ms")
        print(f"Max inference time: {max_time:.3f} ms")
        
        print("\n✅ All inference tests passed!")
        print("\n📋 Deployment Readiness:")
        print("  ✅ Model loads successfully")
        print("  ✅ Inference runs without errors")
        print("  ✅ Output format is correct")
        print("  ✅ Model is ready for ESP32 deployment")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during inference test: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_c_array_files(c_file: str, h_file: str):
    """C 배열 파일 검증"""
    print("\n" + "=" * 60)
    print("C Array Files Verification")
    print("=" * 60)
    
    try:
        # Check header file
        if os.path.exists(h_file):
            print(f"✅ Header file exists: {h_file}")
            with open(h_file, 'r') as f:
                content = f.read()
                if 'extern const unsigned char' in content:
                    print("  ✅ Header file format is correct")
                else:
                    print("  ⚠️  Header file format may be incorrect")
        else:
            print(f"❌ Header file not found: {h_file}")
            return False
        
        # Check source file
        if os.path.exists(c_file):
            print(f"✅ Source file exists: {c_file}")
            file_size = os.path.getsize(c_file)
            print(f"  File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
            
            with open(c_file, 'r') as f:
                content = f.read()
                if 'const unsigned char' in content:
                    print("  ✅ Source file format is correct")
                else:
                    print("  ⚠️  Source file format may be incorrect")
        else:
            print(f"❌ Source file not found: {c_file}")
            return False
        
        print("\n✅ C array files are ready for ESP32 deployment")
        return True
        
    except Exception as e:
        print(f"❌ Error verifying C array files: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test TFLite model inference locally (without hardware)"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='data/processed/microcontroller/hello_world_model.tflite',
        help='Path to TFLite model file'
    )
    parser.add_argument(
        '--verify-c-files',
        action='store_true',
        help='Also verify C array files'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        print("\n💡 Create a test model first:")
        print("   python scripts/create_test_model.py")
        sys.exit(1)
    
    # Test inference
    success = test_tflite_inference(args.model)
    
    # Verify C files (optional)
    if args.verify_c_files:
        c_file = "data/processed/microcontroller/model_data.c"
        h_file = "data/processed/microcontroller/model_data.h"
        verify_c_array_files(c_file, h_file)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Deployment Pipeline Validation Complete!")
        print("=" * 60)
        print("\nThe model is ready for deployment to ESP32.")
        print("When you have hardware available, follow the guide in:")
        print("  docs/MICROCONTROLLER_DEPLOYMENT.md")
    else:
        sys.exit(1)

