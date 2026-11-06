/*
 * E7 runtime comparison: esp32_agreement's serial protocol on the current Espressif TFLM port
 * (espressif/esp-tflite-micro), so identical model files and inputs can be compared with the
 * Arduino TensorFlowLite_ESP32 1.0.0 runtime. Host side: scripts/device_agreement.py run.
 *
 * Host -> device (binary): 'S' <u8 model index> | 'X' <dim x float32 LE> | 'I'
 * Device -> host (text lines):
 *   DEVICE chip=<s> cores=<n> cpu_mhz=<n> flash_bytes=<n> sdk=<idf version>
 *   RUNTIME name=esp-tflite-micro kernels=<esp-nn|ansi-c>
 *   MEMORY free_heap=<b> min_free_heap=<b> arena_static=<b>
 *   MODEL idx=<i> name=<s> bytes=<n>
 *   READY models=<n> arena=<bytes>
 *   SELECTED idx=<i> name=<s> dim=<d> arena_used=<b> in_type=<t> in_scale=<f> in_zp=<z> out_type=<t> free_heap=<b>
 *   Y <output float32 bits, hex> <input_copy_us> <invoke_us> <output_read_us>
 *   ERROR <message>
 */
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <new>

#include "driver/uart.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_idf_version.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#if __has_include("tensorflow/lite/micro/schema/schema_generated.h")
#include "tensorflow/lite/micro/schema/schema_generated.h"  // esp-tflite-micro >= 1.3 (TFLM moved the schema)
#else
#include "tensorflow/lite/schema/schema_generated.h"
#endif
#include "model_data.h"

namespace {
constexpr uart_port_t kUart = UART_NUM_0;
constexpr int kArenaSize = 48 * 1024;
constexpr int kMaxDim = 256;
alignas(16) uint8_t tensor_arena[kArenaSize];
alignas(tflite::MicroInterpreter) uint8_t interpreter_buf[sizeof(tflite::MicroInterpreter)];
tflite::MicroInterpreter* interpreter = nullptr;
tflite::MicroMutableOpResolver<6> resolver;
float xbuf[kMaxDim];
int dim = 0;

void out(const char* fmt, ...) {
  char buf[320];
  va_list ap;
  va_start(ap, fmt);
  const int n = vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  if (n > 0) uart_write_bytes(kUart, buf, n < static_cast<int>(sizeof(buf)) ? n : sizeof(buf) - 1);
}

const char* type_name(TfLiteType t) {
  switch (t) {
    case kTfLiteFloat32: return "float32";
    case kTfLiteInt8: return "int8";
    case kTfLiteInt16: return "int16";
    case kTfLiteUInt8: return "uint8";
    default: return "other";
  }
}

const char* chip_name(esp_chip_model_t m) {
  switch (m) {
    case CHIP_ESP32: return "ESP32";
    case CHIP_ESP32S3: return "ESP32-S3";
    default: return "other";
  }
}

void info() {
  esp_chip_info_t chip;
  esp_chip_info(&chip);
  uint32_t flash = 0;
  esp_flash_get_size(nullptr, &flash);
  out("DEVICE chip=%s cores=%d cpu_mhz=%d flash_bytes=%u sdk=%s\n", chip_name(chip.model), chip.cores,
      CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, static_cast<unsigned>(flash), esp_get_idf_version());
#if defined(CONFIG_NN_OPTIMIZED)
  out("RUNTIME name=esp-tflite-micro kernels=esp-nn\n");
#else
  out("RUNTIME name=esp-tflite-micro kernels=ansi-c\n");
#endif
  out("MEMORY free_heap=%u min_free_heap=%u arena_static=%d\n",
      static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_8BIT)),
      static_cast<unsigned>(esp_get_minimum_free_heap_size()), kArenaSize);
  for (int i = 0; i < kNumModels; ++i) out("MODEL idx=%d name=%s bytes=%u\n", i, kModels[i].name, kModels[i].len);
  out("READY models=%d arena=%d\n", kNumModels, kArenaSize);
}

void select_model(int idx) {
  if (interpreter) {
    interpreter->~MicroInterpreter();
    interpreter = nullptr;
  }
  dim = 0;
  if (idx < 0 || idx >= kNumModels) {
    out("ERROR bad model index %d\n", idx);
    return;
  }
  const tflite::Model* model = tflite::GetModel(kModels[idx].data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    out("ERROR model=%s schema %d\n", kModels[idx].name, static_cast<int>(model->version()));
    return;
  }
  interpreter = new (interpreter_buf) tflite::MicroInterpreter(model, resolver, tensor_arena, kArenaSize);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    out("ERROR model=%s AllocateTensors failed (arena %d B or unsupported op)\n", kModels[idx].name, kArenaSize);
    interpreter->~MicroInterpreter();
    interpreter = nullptr;
    return;
  }
  TfLiteTensor* in = interpreter->input(0);
  TfLiteTensor* o = interpreter->output(0);
  dim = in->dims->data[in->dims->size - 1];
  if (dim > kMaxDim) {
    out("ERROR model=%s input dim %d > %d\n", kModels[idx].name, dim, kMaxDim);
    dim = 0;
    return;
  }
  out("SELECTED idx=%d name=%s dim=%d arena_used=%u in_type=%s in_scale=%.9g in_zp=%d out_type=%s free_heap=%u\n",
      idx, kModels[idx].name, dim, static_cast<unsigned>(interpreter->arena_used_bytes()), type_name(in->type),
      in->params.scale, static_cast<int>(in->params.zero_point), type_name(o->type),
      static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_8BIT)));
}

void run_vector() {
  TfLiteTensor* in = interpreter->input(0);
  TfLiteTensor* o = interpreter->output(0);
  const int64_t t0 = esp_timer_get_time();
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
  const int64_t t1 = esp_timer_get_time();
  if (interpreter->Invoke() != kTfLiteOk) {
    out("ERROR invoke failed\n");
    return;
  }
  const int64_t t2 = esp_timer_get_time();
  float y = NAN;
  if (o->type == kTfLiteFloat32) y = o->data.f[0];
  else if (o->type == kTfLiteInt8) y = (o->data.int8[0] - o->params.zero_point) * o->params.scale;
  const int64_t t3 = esp_timer_get_time();
  uint32_t bits;
  memcpy(&bits, &y, sizeof(bits));
  out("Y %08lx %ld %ld %ld\n", static_cast<unsigned long>(bits), static_cast<long>(t1 - t0),
      static_cast<long>(t2 - t1), static_cast<long>(t3 - t2));
}

bool read_exact(uint8_t* buf, int n) {
  int got = 0;
  while (got < n) {
    const int r = uart_read_bytes(kUart, buf + got, n - got, pdMS_TO_TICKS(2000));
    if (r <= 0) return false;
    got += r;
  }
  return true;
}
}  // namespace

extern "C" void app_main(void) {
  uart_config_t cfg = {};
  cfg.baud_rate = 921600;
  cfg.data_bits = UART_DATA_8_BITS;
  cfg.parity = UART_PARITY_DISABLE;
  cfg.stop_bits = UART_STOP_BITS_1;
  cfg.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;
  cfg.source_clk = UART_SCLK_DEFAULT;
  uart_driver_install(kUart, 4096, 4096, 0, nullptr, 0);
  uart_param_config(kUart, &cfg);
  uart_set_pin(kUart, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);

  resolver.AddFullyConnected();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddLogistic();
  resolver.AddRelu();

  vTaskDelay(pdMS_TO_TICKS(1500));
  info();
  while (true) {
    uint8_t cmd;
    if (uart_read_bytes(kUart, &cmd, 1, pdMS_TO_TICKS(100)) != 1) continue;
    if (cmd == 'S') {
      uint8_t idx;
      if (read_exact(&idx, 1)) select_model(idx);
    } else if (cmd == 'X') {
      if (!interpreter || dim == 0) {
        out("ERROR no model selected\n");
        continue;
      }
      if (!read_exact(reinterpret_cast<uint8_t*>(xbuf), dim * static_cast<int>(sizeof(float)))) {
        out("ERROR short frame\n");
        continue;
      }
      run_vector();
    } else if (cmd == 'I') {
      info();
    }
  }
}
