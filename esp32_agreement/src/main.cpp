/*
 * E7 device agreement (Dr. Saha's guide): the host streams test inputs over serial, the device
 * runs each embedded model and returns the exact float32 output bits plus per-stage timings.
 * Host side: scripts/device_agreement.py run --port <COM> --bundle <dir>.
 *
 * Protocol (host -> device, binary):
 *   'S' <u8 model index>          select a model (interpreter re-created in the shared arena)
 *   'X' <dim x float32, LE>       one input vector for the selected model
 * Device -> host (text lines):
 *   READY models=<n> arena=<bytes>
 *   MODEL idx=<i> name=<s> bytes=<n>
 *   SELECTED idx=<i> name=<s> dim=<d> arena_used=<b> in_type=<t> in_scale=<f> in_zp=<z> out_type=<t>
 *   Y <output float32 bits, hex> <quantize_us> <invoke_us> <dequantize_us>
 *   ERROR <message>
 */

#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>
#include <math.h>
#include <new>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

namespace {
constexpr int kTensorArenaSize = 48 * 1024;
constexpr int kMaxDim = 256;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
alignas(tflite::MicroInterpreter) uint8_t interpreter_buf[sizeof(tflite::MicroInterpreter)];
tflite::MicroInterpreter* interpreter = nullptr;
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;
float xbuf[kMaxDim];
int dim = 0;

const char* type_name(TfLiteType t) {
  switch (t) {
    case kTfLiteFloat32: return "float32";
    case kTfLiteInt8: return "int8";
    case kTfLiteInt16: return "int16";
    case kTfLiteUInt8: return "uint8";
    default: return "other";
  }
}

void select_model(int idx) {
  if (interpreter) {
    interpreter->~MicroInterpreter();
    interpreter = nullptr;
  }
  dim = 0;
  if (idx < 0 || idx >= kNumModels) {
    Serial.printf("ERROR bad model index %d\n", idx);
    return;
  }
  const tflite::Model* model = tflite::GetModel(kModels[idx].data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("ERROR model=%s schema %d\n", kModels[idx].name, static_cast<int>(model->version()));
    return;
  }
  interpreter = new (interpreter_buf) tflite::MicroInterpreter(
      model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.printf("ERROR model=%s AllocateTensors failed (arena %d B or unsupported op)\n",
                  kModels[idx].name, kTensorArenaSize);
    interpreter->~MicroInterpreter();
    interpreter = nullptr;
    return;
  }
  TfLiteTensor* in = interpreter->input(0);
  TfLiteTensor* out = interpreter->output(0);
  dim = in->dims->data[in->dims->size - 1];
  if (dim > kMaxDim) {
    Serial.printf("ERROR model=%s input dim %d > %d\n", kModels[idx].name, dim, kMaxDim);
    dim = 0;
    return;
  }
  Serial.printf("SELECTED idx=%d name=%s dim=%d arena_used=%u in_type=%s in_scale=%.9g in_zp=%d out_type=%s\n",
                idx, kModels[idx].name, dim, static_cast<unsigned>(interpreter->arena_used_bytes()),
                type_name(in->type), in->params.scale, in->params.zero_point, type_name(out->type));
}

void run_vector() {
  TfLiteTensor* in = interpreter->input(0);
  TfLiteTensor* out = interpreter->output(0);
  const unsigned long t0 = micros();
  if (in->type == kTfLiteFloat32) {  // float-input models quantize inside the graph
    for (int i = 0; i < dim; ++i) in->data.f[i] = xbuf[i];
  } else if (in->type == kTfLiteInt8) {
    const float scale = in->params.scale;
    const int zp = in->params.zero_point;
    for (int i = 0; i < dim; ++i) {
      const int q = static_cast<int>(lroundf(xbuf[i] / scale)) + zp;
      in->data.int8[i] = static_cast<int8_t>(q < -128 ? -128 : (q > 127 ? 127 : q));
    }
  }
  const unsigned long t1 = micros();
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR invoke failed");
    return;
  }
  const unsigned long t2 = micros();
  float y = NAN;
  if (out->type == kTfLiteFloat32) y = out->data.f[0];
  else if (out->type == kTfLiteInt8) y = (out->data.int8[0] - out->params.zero_point) * out->params.scale;
  const unsigned long t3 = micros();
  uint32_t bits;
  memcpy(&bits, &y, sizeof(bits));
  Serial.printf("Y %08lx %lu %lu %lu\n", static_cast<unsigned long>(bits), t1 - t0, t2 - t1, t3 - t2);
}
}  // namespace

void setup() {
  Serial.setRxBufferSize(4096);
  Serial.begin(921600);
  delay(1500);
  Serial.printf("DEVICE chip=%s cores=%d cpu_mhz=%u flash_bytes=%u sdk=%s\n",
                ESP.getChipModel(), static_cast<int>(ESP.getChipCores()),
                static_cast<unsigned>(ESP.getCpuFreqMHz()),
                static_cast<unsigned>(ESP.getFlashChipSize()), ESP.getSdkVersion());
  for (int i = 0; i < kNumModels; ++i)
    Serial.printf("MODEL idx=%d name=%s bytes=%u\n", i, kModels[i].name, kModels[i].len);
  Serial.printf("READY models=%d arena=%d\n", kNumModels, kTensorArenaSize);
}

void loop() {
  if (!Serial.available()) return;
  const int cmd = Serial.read();
  if (cmd == 'S') {
    uint8_t idx;
    if (Serial.readBytes(&idx, 1) == 1) select_model(idx);
  } else if (cmd == 'X') {
    if (!interpreter || dim == 0) {
      Serial.println("ERROR no model selected");
      return;
    }
    const size_t n = Serial.readBytes(reinterpret_cast<uint8_t*>(xbuf), dim * sizeof(float));
    if (n != dim * sizeof(float)) {
      Serial.printf("ERROR short frame %u/%u\n", static_cast<unsigned>(n), static_cast<unsigned>(dim * sizeof(float)));
      return;
    }
    run_vector();
  } else if (cmd == 'I') {
    for (int i = 0; i < kNumModels; ++i)
      Serial.printf("MODEL idx=%d name=%s bytes=%u\n", i, kModels[i].name, kModels[i].len);
    Serial.printf("READY models=%d arena=%d\n", kNumModels, kTensorArenaSize);
  }
}
