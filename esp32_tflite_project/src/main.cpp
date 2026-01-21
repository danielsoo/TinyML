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

// Global variables
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Memory buffer (adjust based on model size)
// Hello World model is small, so 2000 bytes is sufficient
constexpr int kTensorArenaSize = 2000;
uint8_t tensor_arena[kTensorArenaSize];

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n========================================");
  Serial.println("ESP32 TensorFlow Lite Micro Test");
  Serial.println("========================================\n");
  
  // Load model
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
  
  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize,
      &micro_error_reporter);
  interpreter = &static_interpreter;
  
  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed!");
    return;
  }
  
  // Get input/output tensors
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
  // Test input data
  // Input: [0.5, 0.5]
  float test_input[2] = {0.5, 0.5};
  
  Serial.println("Running inference...");
  Serial.print("Input: [");
  Serial.print(test_input[0], 2);
  Serial.print(", ");
  Serial.print(test_input[1], 2);
  Serial.println("]");
  
  // Copy input data
  for (int i = 0; i < input->dims->data[1]; i++) {
    input->data.f[i] = test_input[i];
  }
  
  // Run inference
  unsigned long start_time = micros();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long end_time = micros();
  
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke() failed!");
    return;
  }
  
  // Print results
  Serial.print("Output: ");
  Serial.println(output->data.f[0], 6);
  Serial.print("Inference time: ");
  Serial.print(end_time - start_time);
  Serial.println(" microseconds");
  Serial.println("----------------------------------------\n");
  
  delay(2000);  // Repeat every 2 seconds
}

