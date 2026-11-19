Dear Dr. Saha,

Thank you for your patience. Following your email of November 11, we revised the manuscript using existing results only. There are no new experiments, no retraining and no new statistical tests, the thresholds remain frozen, and the 16-bit recovery is reported as host-only.

- **Model comparison.** A new Table 3 compares the principal deployable models: recall, achieved test FAR, size, latency and training cost, with each result linked to its exported file in the supplement.
- **Supporting results.** A supplement (S1–S8, PDF and CSV files) contains:
  - per-run results at both budgets
  - paired differences
  - device disagreement classes (additional misses, additional detections, false-alarm changes)
  - resource breakdown
  - stream completeness
  - the runtime comparison
  - split provenance with the freeze history

  The paper cites the supplement where each record applies.
- **Narrowed claims.** We limited the conclusions to what the records support, including the test-set history, the runtime scope, the E3 execution path and the int16 observation, and we now separate the validation FAR budget from the achieved test FAR.

| Finding | Evidence | Practical implication |
|---|---|---|
| A compact model trained at the target size matches the complete recovery recipe on CIC-IDS2017 and unseen hosts. The recipe is 0.5–1.6 points better on random TON_IoT, at a slightly higher test FAR. | Tables 2–4; S2, S8 | If the memory budget is known before training, start with the compact model. |
| The input quantization range causes the PTQ/QAT collapses, and bounding it prevents them. | Table 6, Figure 2; Table 2 | Clip or exclude extreme inputs before PTQ and QAT, and weigh the saturation that clipping causes. |
| Device agreement depends on the exact exported file and runtime. | Tables 9–10; S3, S5, S6 | Verify the exact file on the target runtime with near-threshold and out-of-range inputs. |
| A validation FAR budget does not fix the achieved test FAR. | S1 | Report the achieved test FAR next to the budget. |

Because of the new table, the tables you referred to as Tables 3, 7, 8 and 9 are now Tables 4, 8, 9 and 10. While reviewing the records, we also corrected two minor statements: the count of E7 exceptions and two unnecessary † marks in Table 2. Neither changes a principal conclusion.

We'd appreciate your feedback on the revised manuscript and supplement, especially on how we frame the main contribution.

Attached: the revised manuscript, the supplement (PDF), and the supplement CSV files (ZIP).

Best regards,
Younsoo Park and Seokhyeon Bae
