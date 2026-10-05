/*
 * ESP32 TensorFlow Lite Micro — federated IDS on-device benchmark (paper 6.1 / Appendix B).
 *
 * Benchmarks the paper's two deployment models and the FP32 federated models they came from:
 *   cic_deploy : models/cic_deploy_int8.tflite (CIC-IDS2017, 78 inputs, INT8, 114,760 B)
 *   cic_fp32   : models/cic_fp32.tflite        (CIC-IDS2017, FP32, 821,696 B)
 *   ton_deploy : models/ton_deploy_int8.tflite (TON_IoT, 43 inputs, INT8, 58,048 B)
 *   ton_fp32   : models/ton_fp32.tflite        (TON_IoT, FP32, 750,016 B)
 * Models are embedded automatically by gen_model_data.py at build time; inputs, host outputs
 * and decision thresholds come from include/test_vectors.h (scripts/prepare_esp32_benchmark.py).
 *
 * Serial output (parsed by scripts/collect_esp32_benchmark.py):
 *   DEVICE chip=<model> cores=<n> cpu_mhz=<f> flash_bytes=<n> sdk=<ver>
 *   MODEL name=<name> bytes=<n> input_type=<t> output_type=<t>
 *   PARITY model=<name> vec=<i> device=<y> host=<y> abs_diff=<d>
 *   PARITY_SUMMARY model=<name> max_abs_diff=<d> label_agree=<k>/<n>
 *   BENCHMARK model=<name> latency_us=<us> arena_used=<bytes> input_dim=<d> run=<i>
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

bool set_input(TfLiteTensor* tensor, const float* x, int dim) {
  if (tensor->type == kTfLiteFloat32) {
    for (int i = 0; i < dim; ++i) tensor->data.f[i] = x[i];
    return true;
  }
  if (tensor->type == kTfLiteInt8) {
    const float scale = tensor->params.scale;
    const int zp = tensor->params.zero_point;
    for (int i = 0; i < dim; ++i) {
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

// inputs: kNumTestVectors x dim, row-major; threshold: the model's deployment decision threshold
void benchmark_model(const char* name, const unsigned char* model_data, unsigned int model_len,
                     const float* inputs, int dim, const float* expected, float threshold) {
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
  if (input->dims->data[input->dims->size - 1] != dim) {
    Serial.printf("ERROR model=%s input dim %d, test vectors have %d\n", name,
                  input->dims->data[input->dims->size - 1], dim);
    return;
  }
  Serial.printf("MODEL name=%s bytes=%u input_type=%s output_type=%s\n", name,
                model_len, type_name(input->type), type_name(output->type));

  // Parity: device output vs. host TFLite interpreter output on the same inputs.
  float max_diff = 0.0f;
  int label_agree = 0;
  for (int v = 0; v < kNumTestVectors; ++v) {
    if (!set_input(input, inputs + v * dim, dim) || interpreter.Invoke() != kTfLiteOk) {
      Serial.printf("ERROR model=%s Invoke failed on vec %d\n", name, v);
      return;
    }
    const float y = get_output(output);
    const float diff = fabsf(y - expected[v]);
    if (diff > max_diff) max_diff = diff;
    if ((y >= threshold) == (expected[v] >= threshold)) ++label_agree;
    Serial.printf("PARITY model=%s vec=%d device=%.6f host=%.6f abs_diff=%.6f\n", name,
                  v, y, expected[v], diff);
  }
  Serial.printf("PARITY_SUMMARY model=%s max_abs_diff=%.6f label_agree=%d/%d\n", name,
                max_diff, label_agree, kNumTestVectors);

  for (int run = 0; run < kWarmupRuns; ++run) {
    set_input(input, inputs + (run % kNumTestVectors) * dim, dim);
    interpreter.Invoke();
  }

  for (int run = 0; run < kBenchmarkRuns; ++run) {
    set_input(input, inputs + (run % kNumTestVectors) * dim, dim);
    const unsigned long t0 = micros();
    const TfLiteStatus status = interpreter.Invoke();
    const unsigned long t1 = micros();
    if (status != kTfLiteOk) {
      Serial.printf("ERROR model=%s Invoke failed on run %d\n", name, run + 1);
      return;
    }
    Serial.printf("BENCHMARK model=%s latency_us=%lu arena_used=%u input_dim=%d run=%d\n",
                  name, t1 - t0, static_cast<unsigned>(interpreter.arena_used_bytes()),
                  dim, run + 1);
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

  benchmark_model("cic_deploy", g_model_cic_deploy, g_model_cic_deploy_len, &kCicInputs[0][0],
                  kCicInputDim, kExpected_cic_deploy, kThreshold_cic_deploy);
  benchmark_model("cic_fp32", g_model_cic_fp32, g_model_cic_fp32_len, &kCicInputs[0][0],
                  kCicInputDim, kExpected_cic_fp32, kThreshold_cic_fp32);
  benchmark_model("ton_deploy", g_model_ton_deploy, g_model_ton_deploy_len, &kTonInputs[0][0],
                  kTonInputDim, kExpected_ton_deploy, kThreshold_ton_deploy);
  benchmark_model("ton_fp32", g_model_ton_fp32, g_model_ton_fp32_len, &kTonInputs[0][0],
                  kTonInputDim, kExpected_ton_fp32, kThreshold_ton_fp32);

  Serial.println("BENCHMARK_DONE");
}

void loop() {
  delay(10000);
}
