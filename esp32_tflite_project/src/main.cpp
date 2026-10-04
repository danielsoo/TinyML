/*
 * ESP32 TensorFlow Lite Micro — CIC-IDS2017 IDS on-device benchmark (78-dim input).
 *
 * Benchmarks both models from the paper's Table 3 in one flash:
 *   compressed : models/ids_compressed_int8.tflite (prune 50% -> client FT -> QAT, INT8, 67,008 B)
 *   baseline   : models/ids_baseline_fp32.tflite   (FP32, 821,792 B)   -> 12.26x
 * Models are embedded automatically by gen_model_data.py at build time.
 *
 * Serial output (parsed by scripts/collect_esp32_benchmark.py):
 *   DEVICE chip=<model> cores=<n> cpu_mhz=<f> flash_bytes=<n> sdk=<ver>
 *   MODEL name=<name> bytes=<n> input_type=<t> output_type=<t>
 *   PARITY model=<name> vec=<i> device=<y> host=<y> abs_diff=<d>
 *   PARITY_SUMMARY model=<name> max_abs_diff=<d> label_agree=<k>/<n>
 *   BENCHMARK model=<name> latency_us=<us> arena_used=<bytes> input_dim=78 run=<i>
 *   BENCHMARK_DONE
 */

#include <Arduino.h>
#include <TensorFlowLite_ESP32.h>
#include <math.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"
#include "test_vectors.h"

namespace {
constexpr int kWarmupRuns = 5;
constexpr int kBenchmarkRuns = 100;
constexpr float kDecisionThreshold = 0.5f;  // only used for the label-agreement check
// Host build measured 2-4 KB arena use for both models; increase if AllocateTensors fails.
constexpr int kTensorArenaSize = 16 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;

const char* type_name(TfLiteType t) {
  switch (t) {
    case kTfLiteFloat32: return "float32";
    case kTfLiteInt8: return "int8";
    case kTfLiteUInt8: return "uint8";
    default: return "other";
  }
}

bool set_input(TfLiteTensor* tensor, const float* x) {
  if (tensor->type == kTfLiteFloat32) {
    for (int i = 0; i < kTestInputDim; ++i) tensor->data.f[i] = x[i];
    return true;
  }
  if (tensor->type == kTfLiteInt8) {
    const float scale = tensor->params.scale;
    const int zp = tensor->params.zero_point;
    for (int i = 0; i < kTestInputDim; ++i) {
      int q = static_cast<int>(lroundf(x[i] / scale)) + zp;
      tensor->data.int8[i] = static_cast<int8_t>(q < -128 ? -128 : (q > 127 ? 127 : q));
    }
    return true;
  }
  return false;
}

float get_output(const TfLiteTensor* tensor) {
  if (tensor->type == kTfLiteFloat32) return tensor->data.f[0];
  if (tensor->type == kTfLiteInt8) {
    return (tensor->data.int8[0] - tensor->params.zero_point) * tensor->params.scale;
  }
  return NAN;
}

void benchmark_model(const char* name, const unsigned char* model_data,
                     unsigned int model_len, const float* expected) {
  const tflite::Model* model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("ERROR model=%s unsupported schema version %d\n", name,
                  static_cast<int>(model->version()));
    return;
  }

  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       kTensorArenaSize, &micro_error_reporter);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    Serial.printf("ERROR model=%s AllocateTensors failed (arena=%d bytes)\n", name,
                  kTensorArenaSize);
    return;
  }
  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  Serial.printf("MODEL name=%s bytes=%u input_type=%s output_type=%s\n", name,
                model_len, type_name(input->type), type_name(output->type));

  // Parity: device output vs. host TFLite interpreter output on the same inputs.
  float max_diff = 0.0f;
  int label_agree = 0;
  for (int v = 0; v < kNumTestVectors; ++v) {
    if (!set_input(input, kTestInputs[v]) || interpreter.Invoke() != kTfLiteOk) {
      Serial.printf("ERROR model=%s Invoke failed on vec %d\n", name, v);
      return;
    }
    const float y = get_output(output);
    const float diff = fabsf(y - expected[v]);
    if (diff > max_diff) max_diff = diff;
    if ((y >= kDecisionThreshold) == (expected[v] >= kDecisionThreshold)) ++label_agree;
    Serial.printf("PARITY model=%s vec=%d device=%.6f host=%.6f abs_diff=%.6f\n", name,
                  v, y, expected[v], diff);
  }
  Serial.printf("PARITY_SUMMARY model=%s max_abs_diff=%.6f label_agree=%d/%d\n", name,
                max_diff, label_agree, kNumTestVectors);

  for (int run = 0; run < kWarmupRuns; ++run) {
    set_input(input, kTestInputs[run % kNumTestVectors]);
    interpreter.Invoke();
  }

  for (int run = 0; run < kBenchmarkRuns; ++run) {
    set_input(input, kTestInputs[run % kNumTestVectors]);
    const unsigned long t0 = micros();
    const TfLiteStatus status = interpreter.Invoke();
    const unsigned long t1 = micros();
    if (status != kTfLiteOk) {
      Serial.printf("ERROR model=%s Invoke failed on run %d\n", name, run + 1);
      return;
    }
    Serial.printf("BENCHMARK model=%s latency_us=%lu arena_used=%u input_dim=%d run=%d\n",
                  name, t1 - t0, static_cast<unsigned>(interpreter.arena_used_bytes()),
                  kTestInputDim, run + 1);
    delay(5);
  }
}
}  // namespace

void setup() {
  Serial.begin(115200);
  delay(1500);

  Serial.println("\n========================================");
  Serial.println("ESP32 IDS TFLite Micro Benchmark");
  Serial.println("========================================");
  Serial.printf("DEVICE chip=%s cores=%d cpu_mhz=%u flash_bytes=%u sdk=%s\n",
                ESP.getChipModel(), static_cast<int>(ESP.getChipCores()),
                static_cast<unsigned>(ESP.getCpuFreqMHz()),
                static_cast<unsigned>(ESP.getFlashChipSize()), ESP.getSdkVersion());

  benchmark_model("compressed", g_model_compressed, g_model_compressed_len,
                  kExpected_compressed);
  benchmark_model("baseline", g_model_baseline, g_model_baseline_len, kExpected_baseline);

  Serial.println("BENCHMARK_DONE");
}

void loop() {
  delay(10000);
}
