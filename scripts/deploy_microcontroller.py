"""
Microcontroller deployment script
Utility for deploying TFLite models to ESP32
"""
import argparse
import os
import sys
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def convert_to_c_array(model_path: str, output_path: str):
    """
    Convert TFLite model to C array
    
    Args:
        model_path: Path to TFLite model file
        output_path: Output C file path
    """
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        with open(model_path, 'rb') as f:
            model_data = f.read()
        
        # C 배열 이름 생성 (파일명 기반)
        array_name = Path(model_path).stem.replace('-', '_').replace('.', '_')
        
        # C 헤더 파일 생성
        header_path = output_path.replace('.c', '.h')
        header_content = f"""#ifndef {array_name.upper()}_H
#define {array_name.upper()}_H

// Auto-generated from {Path(model_path).name}
// Model size: {len(model_data)} bytes

extern const unsigned char {array_name}[];
extern const unsigned int {array_name}_len;

#endif
"""
        
        # Create C source file
        source_content = f"""// Auto-generated from {Path(model_path).name}
#include "{Path(header_path).name}"

const unsigned char {array_name}[] = {{
"""
        
        # Convert bytes to hexadecimal array (12 per line)
        for i, byte in enumerate(model_data):
            if i % 12 == 0:
                source_content += "\n  "
            source_content += f"0x{byte:02x},"
        
        source_content = source_content.rstrip(',') + "\n};\n\n"
        source_content += f"const unsigned int {array_name}_len = {len(model_data)};\n"
        
        # 파일 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(header_path, 'w') as f:
            f.write(header_content)
        
        with open(output_path, 'w') as f:
            f.write(source_content)
        
        print(f"✅ Converted to C array:")
        print(f"   - Header: {header_path}")
        print(f"   - Source: {output_path}")
        print(f"   - Array name: {array_name}")
        print(f"   - Size: {len(model_data):,} bytes ({len(model_data)/1024:.2f} KB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting to C array: {e}")
        return False


def check_tflite_model(model_path: str):
    """TFLite 모델 정보 확인"""
    try:
        import tensorflow as tf
        
        # 모델 로드
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # 입력/출력 정보
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # 파일 크기
        file_size = os.path.getsize(model_path)
        
        print("\n" + "=" * 60)
        print("TFLite Model Information")
        print("=" * 60)
        print(f"File size: {file_size:,} bytes ({file_size/1024:.2f} KB)")
        print(f"\nInput:")
        print(f"  Shape: {input_details[0]['shape']}")
        print(f"  Type: {input_details[0]['dtype']}")
        print(f"\nOutput:")
        print(f"  Shape: {output_details[0]['shape']}")
        print(f"  Type: {output_details[0]['dtype']}")
        print("=" * 60)
        
        return {
            'file_size': file_size,
            'input_shape': input_details[0]['shape'],
            'input_dtype': input_details[0]['dtype'],
            'output_shape': output_details[0]['shape'],
            'output_dtype': output_details[0]['dtype'],
        }
    except Exception as e:
        print(f"❌ Error checking TFLite model: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert TFLite model to C array for microcontroller deployment"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='data/processed/microcontroller/hello_world_model.tflite',
        help='Path to TFLite model file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/microcontroller/model_data.c',
        help='Output C file path'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check model info, do not convert'
    )
    
    args = parser.parse_args()
    
    # Check model information
    info = check_tflite_model(args.model)
    
    if not info:
        sys.exit(1)
    
    # Convert to C array
    if not args.check_only:
        success = convert_to_c_array(args.model, args.output)
        if success:
            print("\n✅ Conversion complete!")
            print(f"\n📋 Next steps:")
            print(f"   1. Copy {args.output} to your ESP32 project")
            print(f"   2. Copy {args.output.replace('.c', '.h')} to your ESP32 project")
            print(f"   3. Include the header in your main.cpp")
        else:
            sys.exit(1)

