Subject: Completion table for your 2026-11-05 list, and runtime comparison results

Dear Dr. Saha,

Thank you for the list. Everything on it is now done. Nothing was retrained and every threshold stayed frozen. The full record is in `docs/SAHA_COMPLETION.md` on the branch. Here is the completion table:

| # | Item | Evidence | Unresolved | Extra execution |
|---|---|---|---|---|
| 1 | E1/E5 operating points | Recall and achieved test FAR per run for every arm, at both budgets, on the random split and on unseen hosts. Run-paired differences, plus record-level McNemar tests computed from the saved scores at the frozen thresholds. | None | Yes: existing predictions only (z9) |
| 2 | Exact deployed models | SHA-256, size, weight-scale type and conversion of every export (`docs/model_hashes.csv`). Per-tensor replacements for all 27 F8/P/S runs, with recall, FAR, size and hash. Tables 2-3 now report the per-tensor models; mean recall changes by at most 0.2 points (-0.98 for P on unseen hosts). | None | Yes (z9) |
| 3 | E7 disagreements, resets, completeness | Every disagreement is classified. Per-tensor INT8 models add no missed attacks, apart from one CIC record for S80 at 1%. Their other disagreements are false alarms on inputs we deliberately chose within two output steps of the threshold. All S3 resets are logged; the stream resumes after each one, and the final streams are complete with no missing or duplicated records. | None | No |
| 4 | Split provenance, freeze history | Split manifests with per-partition hashes and the freeze timeline. No new test record was in the old test split; all were in the old training pool, which only the exploratory models saw. | No validity problem found, so no re-split | Yes: documentation check (z9) |
| 5 | Resources, reproducibility | Static RAM/flash from the linker, arena, run-time heap for both runtimes, and the definition of latency (invoke only, including in-graph quantize/dequantize). Map of configs, seeds, checkpoints, thresholds, scores and scripts. | None | Yes: heap logged (z10) |
| - | Simulation vs export | All E3/E4 cells are exported TFLite models run in the interpreter. Only the "QAT float" stage is fake-quant simulation. | None | No |
| new | Runtime comparison | Section 7 below. | 16-bit failure not traced into the TFLM source | Yes (z10, z10d, z10f) |

**Runtime comparison.** We used the same classic ESP32 (240 MHz), the same three model files and the same 4,462 TON_IoT inputs. The thresholds were frozen and the files were converted with TensorFlow 2.21.0. Three runtime builds:
- **Old:** TFLM 1.0.0 (Arduino TensorFlowLite_ESP32), reference kernels.
- **Current:** esp-tflite-micro 1.4.2 with esp-nn 1.4.2 on ESP-IDF 5.5.0, built once with esp-nn kernels and once with reference kernels.

Component hashes, `dependencies.lock`, `sdkconfig` and the op versions are stored with the results. All three files declare QUANTIZE v1, FULLY_CONNECTED v4, LOGISTIC v2 and DEQUANTIZE v2.

Decisions agreeing with the host at the 1% FAR threshold (out of 4,462):

| Model | Old runtime | Current runtime |
|---|---|---|
| S80, per-channel weights (the failing model) | 2,209 (2,069 extra misses) | **4,459** (3 extra detections, nothing else) |
| S80, per-tensor re-export | 4,462 | 4,462 (outputs bit-identical to the old runtime) |
| 16-bit activations, per-channel | 3,379 | 1,894: one constant output for every input |

What this shows:
1. **Per-channel failure.** It comes from the old runtime, not from the export. The current Espressif port runs the per-channel file correctly.
2. **Per-tensor models.** They are runtime-independent here, so the per-tensor E7 results also hold on the current runtime.
3. **Kernel configuration.** It makes no difference on this chip. esp-nn and reference kernels give bit-identical outputs, the same image size, and latency within 0.3%. We did not test esp-nn on the ESP32-S3, where its optimizations apply.
4. **16-bit model.** It runs correctly on neither runtime.
   - On the current runtime its output equals the sigmoid of the output-layer bias for every input, so every hidden activation arrives at the last layer as zero.
   - We did not trace this further. The paper now presents 16-bit activations as a host-only result.
5. **Latency and memory.** The current runtime is 22% faster (13.3 ms to 10.4 ms per flow). Its arena use is 1.9 KB (per-tensor) and 5.5 KB (per-channel). Neither build allocates heap after start-up.

**Paper changes.**
- Section 7 has a new Table 9 and a short "current runtime" paragraph.
- The abstract, RQ3, recommendation 5, threats to validity and the conclusion now state the runtime scope: per-tensor models agree on both runtimes, per-channel models only on the current one, and the 16-bit model on neither.
- The paper is 8 pages (a title-macro bug had left page 1 blank; fixed).

**One question.** One follow-up could pin down the 16-bit failure. We would re-export the same 16-bit model with per-tensor weight scales and run it on the current runtime.
- If it then agrees with the host, the per-channel int16 path of the runtime is the cause.
- If it still fails, int16 support itself is the cause.

This needs one export and one board run (about an hour). It changes no other result. Without it, the paper keeps 16-bit activations as a host-only result, as it does now. Would you like us to run it, or should we leave the scope as it is?

Best regards,
Younsoo Park and Seokhyeon Bae
