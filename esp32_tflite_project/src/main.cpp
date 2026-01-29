/*
 * ESP32 TensorFlow Lite Micro Hello World Example
 * 
 * This example loads a simple ML model and runs inference on ESP32.
 * Model: Hello World (2 inputs -> 1 output)
 */

#include <Arduino.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

// 전역 변수
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// 메모리 버퍼 (모델 크기에 따라 조정)
// Hello World 모델은 작으므로 2000 bytes면 충분
constexpr int kTensorArenaSize = 2000;
uint8_t tensor_arena[kTensorArenaSize];

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n========================================");
  Serial.println("ESP32 TensorFlow Lite Micro Test");
  Serial.println("========================================\n");
  
  // 모델 로드
  Serial.println("Loading TFLite model...");
  model = tflite::GetModel(hello_world_model);
  
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("Model schema version ");
    Serial.print(model->version());
    Serial.print(" not supported. Supported version is ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    return;
  }
  
  Serial.println("Model loaded successfully!");
  Serial.print("Model size: ");
  Serial.print(hello_world_model_len);
  Serial.println(" bytes\n");
  
  // 인터프리터 생성
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize,
      &micro_error_reporter);
  interpreter = &static_interpreter;
  
  // 텐서 할당
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed!");
    return;
  }
  
  // 입력/출력 텐서 가져오기
  input = interpreter->input(0);
  output = interpreter->output(0);
  
  Serial.println("Model initialized successfully!");
  Serial.print("Input shape: [");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    if (i < input->dims->size - 1) Serial.print(", ");
  }
  Serial.println("]");
  
  Serial.print("Output shape: [");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);
    if (i < output->dims->size - 1) Serial.print(", ");
  }
  Serial.println("]\n");
  
  Serial.println("Ready for inference!");
  Serial.println("========================================\n");
}

void loop() {
  // 테스트 입력 데이터
  // 입력: [0.5, 0.5]
  float test_input[2] = {0.5, 0.5};
  
  Serial.println("Running inference...");
  Serial.print("Input: [");
  Serial.print(test_input[0], 2);
  Serial.print(", ");
  Serial.print(test_input[1], 2);
  Serial.println("]");
  
  // 입력 데이터 복사
  for (int i = 0; i < input->dims->data[1]; i++) {
    input->data.f[i] = test_input[i];
  }
  
  // 추론 실행
  unsigned long start_time = micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long end_time = micros();
  
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke() failed!");
    return;
  }
  
  // 결과 출력
  Serial.print("Output: ");
  Serial.println(output->data.f[0], 6);
  Serial.print("Inference time: ");
  Serial.print(end_time - start_time);
  Serial.println(" microseconds");
  Serial.println("----------------------------------------\n");
  
  delay(2000);  // 2초마다 반복
}

