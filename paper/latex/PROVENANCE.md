# Provenance of every number in main.tex

All numbers come from our own runs on the user's PC (WSL2, RTX 5070 host, CPU simulation).
Each run directory under `data/processed/revision/<run>/` contains `job.yaml`, `git_commit.txt`, the configs, step logs
(`*.log`) and result files. External facts are cited in the paper (references.bib).

| Paper item | Run ID | Result file(s) |
|---|---|---|
| Table 1 (CIC rows) | B, I | `data/processed/revision/2026-10-03_b_v3_float_cic/baseline/baseline_ablation.json`, `.../non_iid/non_iid_ablation.json`, `data/processed/revision/2026-10-04_i_fixed_lr_cic/fixed_lr/non_iid_ablation.json` |
| Table 1 (TON rows) | S, U | `data/processed/revision/2026-10-05_s_ton_sweep/ton_sweep/ton_sweep.json` (federated near-IID = `best_text_alpha05_test`), `data/processed/revision/2026-10-05_u_toniot_v2/{baseline,non_iid,fixed_lr}/*.json` |
| Table 2 (local-only) | K, V | `data/processed/revision/2026-10-04_k_local_only/local_only/local_only.json`, `data/processed/revision/2026-10-05_v_toniot_v2_downstream/local_only/local_only.json` |
| Table 3, client-choice paragraph | E, V | `data/processed/revision/2026-10-04_e_float_compression_ablation/compression_ablation/compression_ablation.json`, `data/processed/revision/2026-10-05_v_toniot_v2_downstream/compression_ablation/compression_ablation.json` |
| Figure 1 | G, W | `data/processed/revision/2026-10-04_g_prune_sweep/compression_ablation/compression_ablation.json`, `data/processed/revision/2026-10-05_w_toniot_v2_sweep_robustness/compression_ablation/compression_ablation.json` (plot: `scripts/plot_prune_sweep.py`) |
| Table 4 (quantization methods) | J, V | `data/processed/revision/2026-10-04_j_quant_distill/quant_distill/quant_distill.json`, `data/processed/revision/2026-10-05_v_toniot_v2_downstream/quant_distill/quant_distill.json` |
| Table 5 (calibration), QAT draws | L, M, V | `data/processed/revision/2026-10-04_l_ptq_calibration/ptq_calibration/ptq_calibration.json`, `data/processed/revision/2026-10-04_m_qat_stability/qat_stability/qat_stability.json`, `data/processed/revision/2026-10-05_v_toniot_v2_downstream/{ptq_calibration,qat_stability}/*.json` |
| Table 6 (range estimators, data-free calibration) | X1, X3 | `data/processed/revision/2026-10-05_x1_calib_methods/quant_lit/quant_lit_summary.csv`, `data/processed/revision/2026-10-05_x3_data_free_calib/quant_lit/quant_lit_summary.csv` |
| Table 7 (QAT ranges, learned clipping), Section 4.4 range values | X2, X4 | `data/processed/revision/2026-10-05_x2_ptq_init_qat/quant_lit/{quant_lit_summary.csv,quant_lit.csv (ranges column)}`, `data/processed/revision/2026-10-05_x4_learned_clip_qat/quant_lit/quant_lit_summary.csv` |
| Section 4.5 (training-time QAT) | A, U | `data/processed/revision/2026-10-03_v2_cic_full/` (logs + baseline_ablation.json), `data/processed/revision/2026-10-05_u_toniot_v2/qat_fl/non_iid_ablation.json` |
| Table 8 (distillation) | J, V | same files as Table 4 (rows with part = distill) |
| Table 9 (pipelines) | N, V | `data/processed/revision/2026-10-05_n_compression_combos/compression_combos/compression_combos.json`, `data/processed/revision/2026-10-05_v_toniot_v2_downstream/compression_combos/compression_combos.json` |
| Table 10 (recall priority) | O, P, T | `data/processed/revision/2026-10-05_o_recall_priority/recall_priority/recall_priority.json`, `data/processed/revision/2026-10-05_p_recall_priority2/recall_priority/recall_priority.json`, `data/processed/revision/2026-10-05_t_recall_priority_ton2/recall_priority/recall_priority.json` |
| Section 4.8, fixed QAT ranges / CLE on the deployed recipes | X5, X6 | `data/processed/revision/2026-10-05_x5_recall_fixed_cic/recall_priority/recall_priority_summary.csv`, `data/processed/revision/2026-10-05_x6_recall_fixed_ton/recall_priority/recall_priority_summary.csv` |
| Table 11 (per attack type) | P, Q | `missed_by_type` fields in run P's json; test counts `data/processed/revision/2026-10-05_q_attack_counts/attack_counts/attack_type_counts.json` |
| Table 12 (robustness) | F, W | `data/processed/revision/2026-10-04_f_robustness/robustness/robustness.md`, `data/processed/revision/2026-10-05_w_toniot_v2_sweep_robustness/robustness/robustness.md` |
| Table 13 (failure modes) | — | `docs/REVISION_RESULTS.md` ("Bugs found"), fixes in `src/federated/client.py`, `compression.py`, `src/tinyml/export_tflite.py` |
| Section 4.10 / Appendix C (ESP32) | — | host build of `esp32_tflite_project/` against TensorFlowLite_ESP32 1.0.0; models staged by `scripts/prepare_esp32_benchmark.py` |
| Appendix B (TON recipe) | S | `data/processed/revision/2026-10-05_s_ton_sweep/ton_sweep/ton_sweep.json` |
| Dataset statistics (Section 3.1) | B, U | loader logs (`baseline_ablation.log`) of runs B and U |

Deployed models: CIC `data/processed/revision/2026-10-05_p_recall_priority2/recall_priority/cic_near_iid/kdclip_p30_d0.tflite` (threshold 0.1914),
TON `data/processed/revision/2026-10-05_t_recall_priority_ton2/recall_priority/ton_near_iid_v2/kdclip_fed_d0.tflite` (threshold 0.2070).
