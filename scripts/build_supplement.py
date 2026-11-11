#!/usr/bin/env python3
"""
Supplementary results for the paper, rebuilt only from records that already exist in the repository
(no training, no new statistical tests). Writes CSV files and a LaTeX supplement to paper/supplement/.

  S1  operating points per run: every arm, both FAR budgets, random splits and unseen hosts; recall,
      achieved test FAR, validation FAR, threshold, export file, SHA-256, weight-scale type
  S2  run-paired differences (recall and test FAR) between deployable models, plus the record-level
      McNemar tests of job z9 with the export they were computed on
  S3  E7 device disagreements, classified per board, model, FAR budget and input category into
      additional misses / additional detections / added false alarms / avoided false alarms /
      unlabelled synthetic probes
  S4  resources: firmware static RAM and flash (linker), arena used per model, heap, latency stages
  S5  stream completeness per board, job and model (outputs present, resets, skipped inputs,
      bit-identity between boards)
  S6  runtime comparison (z10 / z10d / z10f), from docs/runtime_compare
  S7  split provenance (overlap with the earlier split, manifest hashes) and method-freeze history
  S8  principal deployable models: recall, test FAR, size, latency, training cost, linked exports

  python scripts/build_supplement.py            # writes paper/supplement/*.csv and supplement.tex
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REV = ROOT / "data/processed/revision"
OUT = ROOT / "paper/supplement"

SPLITS = {  # split -> (main compare job, QC/SQC job, one-client job, training job)
    "TON": ("2026-10-29_z1d_compare_ton_fed", "2026-10-29_z1g_compare_ton_qc", "2026-10-29_z1e_compare_ton_client",
            "2026-10-29_z1_train_ton"),
    "CIC": ("2026-10-29_z2b_compare_cic_fed", "2026-10-29_z2e_compare_cic_qc", "2026-10-29_z2c_compare_cic_client",
            "2026-10-29_z2_train_cic"),
    "TON-hosts": ("2026-10-29_z3b_compare_ton_group", None, None, "2026-10-29_z3_train_ton_group"),
}
Z9 = REV / "2026-11-06_z9_saha_checks/saha_checks/saha_checks.json"
Z9_KEY = {"TON": "ton", "CIC": "cic", "TON-hosts": "ton_hosts"}
BUNDLE = REV / "2026-11-05_z7_device_bundle/device_bundle"
BOARDS = {  # board -> list of (job folder, sub folder); later entries replace earlier ones per model
    "ESP32 (TFLM 1.0.0)": [("2026-11-05_z8b_device_e7", "esp32")],
    "ESP32-S3 (TFLM 1.0.0)": [("2026-11-05_z8_device_e7", "esp32s3"), ("2026-11-05_z8c_device_e7_s3_fill", "esp32s3")],
}
FINAL_STAGE = {"fp32", "int8"}
ARM_FILE = {"F8": "F8.tflite", "P": "P_{b}.tflite", "Q": "Q_{b}.tflite", "R": "R_{b}.tflite", "S": "S_{b}.tflite",
            "SQ": "SQ_{b}.tflite", "QC": "QC_{b}.tflite", "SQC": "SQC_{b}.tflite"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""


def weight_scales(p: Path) -> str:
    try:
        import tflite
    except ImportError:
        return ""
    if not p.exists():
        return ""
    m = tflite.Model.GetRootAsModel(p.read_bytes(), 0)
    g = m.Subgraphs(0)
    n = sum(1 for i in range(g.TensorsLength())
            if g.Tensors(i).Quantization() and g.Tensors(i).Quantization().ScaleLength() > 1)
    return "per-channel" if n else "per-tensor"


def read_rows(job: str) -> list[dict]:
    with open(REV / job / "compare_arms/compare_arms_rows.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def tex_escape(s) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def pct(x, nd=2) -> str:
    return "" if x in ("", None) else f"{100 * float(x):.{nd}f}"


# ------------------------------------------------------------------------------------------- S1
def build_s1() -> list[dict]:
    z9 = json.loads(Z9.read_text())["exports"]
    out = []
    for split, (main, qc, one, _) in SPLITS.items():
        for job, ft in ((main, "FedAvg 4 clients"), (qc, "FedAvg 4 clients"), (one, "one client")):
            if not job:
                continue
            for r in read_rows(job):
                if r["stage"] not in FINAL_STAGE or r["arm"] in ("PQ",):
                    continue
                if job == qc and r["arm"] not in ("QC", "SQC"):
                    continue  # F / S fp32 rows of the QC job duplicate the main job
                if job == one and r["arm"] not in ("P", "Q", "R"):
                    continue  # the one-client job re-evaluates the same F / F8 / S models
                arm, b, seed = r["arm"], r["budget_kb"], int(r["seed"])
                if r["arm"] == "R" and r["stage"] == "fp32":
                    continue
                if r["arm"] == "S" and r["stage"] == "fp32":
                    continue
                f = ""
                if arm in ARM_FILE:
                    f = str((REV / job / "compare_arms" / f"s{seed}" / ARM_FILE[arm].format(b=b)).relative_to(ROOT))
                row = {"split": split, "fine_tune": ft if arm in ("P", "Q", "QC", "R") else "-",
                       "arm": arm + (b if arm == "S" else ""), "budget_kb": b, "seed": seed,
                       "far_budget": r["far_budget"], "stage": r["stage"], "size_kb": r["size_kb"],
                       "threshold": r["threshold"], "val_far": r["val_far"], "test_recall": r["test_recall"],
                       "test_far": r["test_far"], "test_fn": r["test_fn"], "test_fp": r["test_fp"],
                       "export": f, "sha256": sha(ROOT / f) if f else "",
                       "weight_scales": weight_scales(ROOT / f) if f else "float32",
                       "selection": "as selected", "source": f"{job}/compare_arms/compare_arms_rows.csv"}
                if arm in ("F", "S") and r["stage"] == "fp32":
                    row["weight_scales"] = "float32"
                out.append(row)
        # per-tensor strong-PTQ replacements (z9) supersede the first-selected F8 / P80 / S80 exports
        if job := Z9_KEY.get(split):
            for key, v in z9[job].items():
                if key.startswith("bundle:"):
                    continue
                arm, seed = key.split("_s")
                arm_name = {"F8": "F8", "P80": "P", "S80": "S80"}[arm]
                for fb in ("0.01", "0.05"):
                    pt = v["per_tensor_strong_ptq"]
                    for r in out:
                        if (r["split"] == split and r["arm"] == arm_name and int(r["seed"]) == int(seed)
                                and r["far_budget"] == fb and r["fine_tune"] != "one client"
                                and r["budget_kb"] in ("80", "-") and r["selection"] == "as selected"):
                            r["selection"] = ("superseded by per-tensor replacement"
                                              if pt["sha256"] != v["table2_export"]["sha256"] else "per-tensor (unchanged)")
                            r["note"] = ("first selection was per-tensor; the z9 re-export gives the same test decisions"
                                         if r["weight_scales"] == "per-tensor"
                                         and abs(pt[fb]["test_recall"] - float(r["test_recall"])) < 1e-12
                                         and abs(pt[fb]["test_far"] - float(r["test_far"])) < 1e-12 else "")
                    if pt["sha256"] == v["table2_export"]["sha256"]:
                        continue
                    out.append({"split": split, "fine_tune": "FedAvg 4 clients" if arm_name == "P" else "-",
                                "arm": arm_name, "budget_kb": "80" if arm_name != "F8" else "-", "seed": int(seed),
                                "far_budget": fb, "stage": "int8", "size_kb": f"{pt['bytes'] / 1024:.2f}",
                                "threshold": pt[fb]["threshold"], "val_far": pt[fb]["val_far"],
                                "test_recall": pt[fb]["test_recall"], "test_far": pt[fb]["test_far"],
                                "test_fn": pt[fb]["test_fn"], "test_fp": pt[fb]["test_fp"],
                                "export": pt["file"], "sha256": pt["sha256"], "weight_scales": "per-tensor",
                                "selection": f"per-tensor replacement ({v['per_tensor_selected']})",
                                "source": "2026-11-06_z9_saha_checks/saha_checks/saha_checks.json"})
    for r in out:  # the deployable (reported) row of every run
        r["reported"] = int(r["selection"] != "superseded by per-tensor replacement"
                            and r["weight_scales"] != "per-channel" or r["arm"] in ("F",))
    write_csv("S1_operating_points.csv", out)
    return out


def summarize(s1: list[dict]) -> dict:
    """(split, arm, budget_kb, far_budget, fine_tune) -> per-seed recall / FAR of the reported rows."""
    agg = defaultdict(dict)
    for r in s1:
        if not r["reported"]:
            continue
        k = (r["split"], r["arm"], r["budget_kb"], r["far_budget"], r["fine_tune"])
        agg[k][int(r["seed"])] = (float(r["test_recall"]), float(r["test_far"]), float(r["val_far"]))
    return agg


# ------------------------------------------------------------------------------------------- S2
PAIRS = [("R", "F"), ("S80", "F"), ("P", "F"), ("QC", "F"), ("R", "P"), ("S80", "P"), ("QC", "P"),
         ("R", "S80"), ("R", "QC"), ("SQC", "S80")]


def build_s2(agg: dict) -> list[dict]:
    z9 = json.loads(Z9.read_text())
    out = []
    for split in SPLITS:
        for fb in ("0.01", "0.05"):
            for a, b in PAIRS:
                ka = next((k for k in agg if k[0] == split and k[1] == a and k[3] == fb and k[2] in ("80", "-")
                           and k[4] != "one client"), None)
                kb = next((k for k in agg if k[0] == split and k[1] == b and k[3] == fb and k[2] in ("80", "-")
                           and k[4] != "one client"), None)
                if not ka or not kb:
                    continue
                for s in (1, 2, 3):
                    if s in agg[ka] and s in agg[kb]:
                        out.append({"split": split, "far_budget": fb, "pair": f"{a} - {b}", "seed": s,
                                    "recall_diff_pts": round(100 * (agg[ka][s][0] - agg[kb][s][0]), 3),
                                    "test_far_diff_pts": round(100 * (agg[ka][s][1] - agg[kb][s][1]), 3),
                                    "kind": "run-paired (reported exports)"})
    # record-level McNemar (z9): computed on the saved test scores of the first-selected exports
    exports = z9["exports"]
    for split, key in Z9_KEY.items():
        changed = defaultdict(set)  # (arm, seed) -> budgets where the reported export decides differently on test
        for k, v in exports[key].items():
            if k.startswith("bundle:"):
                continue
            arm = k.split("_s")[0].replace("80", "")
            for fb in ("0.01", "0.05"):
                a_, b_ = v["table2_export"][fb], v["per_tensor_strong_ptq"][fb]
                if a_["test_fn"] != b_["test_fn"] or a_["test_fp"] != b_["test_fp"]:
                    changed[(arm, int(k.split("_s")[1]))].add(fb)
        for p in z9["paired"][key]["pairs"]:
            arms = [x.strip() for x in p["pair"].split("vs")]
            stale = [x for x in arms if p["far_budget"] in changed.get((x.replace("80", ""), int(p["seed"])), set())]
            out.append({"split": split, "far_budget": p["far_budget"], "pair": p["pair"].replace(" vs ", " - "),
                        "seed": p["seed"], "recall_diff_pts": round(p["recall_diff_pts"], 3),
                        "test_far_diff_pts": round(p["far_diff_pts"], 3),
                        "attacks_only_first": p["attacks_only_A_detects"], "attacks_only_second": p["attacks_only_B_detects"],
                        "mcnemar_p_attacks": p["mcnemar_p_attacks"],
                        "false_alarms_only_first": p["false_alarms_only_A"], "false_alarms_only_second": p["false_alarms_only_B"],
                        "mcnemar_p_benign": p["mcnemar_p_benign"],
                        "kind": "record-level McNemar (z9, saved scores)",
                        "note": ("computed on the first-selected export of " + ", ".join(stale) +
                                 "; the reported per-tensor export has different test errors") if stale
                        else "first-selected and reported exports have the same test error counts"})
    write_csv("S2_paired_differences.csv", out)
    return out


# ------------------------------------------------------------------------------------------- S3 / S5
def device_outputs(board: str) -> tuple[dict, dict]:
    outs, meta = {}, {}
    for job, sub in BOARDS[board]:
        d = REV / job / "device_e7" / sub
        j = json.loads((d / "device_agreement.json").read_text())
        for m, v in j["models"].items():
            f = d / f"device_{m}.npy"
            meta.setdefault(m, []).append({"job": job, "n_reported": v.get("n"), "error": v.get("error"),
                                           "device_resets": v.get("device_resets"),
                                           "crashed_at_vector": v.get("crashed_at_vector"),
                                           "skipped": len(v.get("inputs_skipped_after_two_resets") or []),
                                           "file": f if f.exists() else None, "json": v})
            if f.exists():
                y = np.load(f).astype(np.float32)
                if m not in outs or np.isfinite(y).sum() >= np.isfinite(outs[m]).sum():
                    outs[m] = y
    return outs, meta


def build_s3_s5() -> tuple[list[dict], list[dict], list[dict]]:
    bundle = json.loads((BUNDLE / "bundle.json").read_text())
    s3, s5, s4 = [], [], []
    per_board = {b: device_outputs(b) for b in BOARDS}
    for m in bundle["models"]:
        name, ds = m["name"], m["dataset"]
        host = np.load(BUNDLE / f"host_{name}.npy").astype(np.float32)
        y = np.load(BUNDLE / f"labels_{ds}.npy")
        cat = np.array([str(c).split(":")[0] for c in np.load(BUNDLE / f"categories_{ds}.npy", allow_pickle=True)])
        scales = weight_scales(BUNDLE / m["file"])
        for board, (outs, meta) in per_board.items():
            dev = outs.get(name)
            if dev is None:
                continue
            for fb, t in m["thresholds"].items():
                hd, dd = host >= t, dev >= t
                for c in ["all"] + sorted(set(cat)):
                    sel = np.ones_like(hd) if c == "all" else (cat == c)
                    s3.append({"board": board, "model": name, "dataset": ds, "weight_scales": scales,
                               "activations": "int16" if "int16" in name else "int8", "far_budget": fb,
                               "input_category": c, "n": int(sel.sum()),
                               "agree": int((sel & (hd == dd)).sum()),
                               "additional_misses": int((sel & hd & ~dd & (y == 1)).sum()),
                               "additional_detections": int((sel & ~hd & dd & (y == 1)).sum()),
                               "added_false_alarms": int((sel & ~hd & dd & (y == 0)).sum()),
                               "avoided_false_alarms": int((sel & hd & ~dd & (y == 0)).sum()),
                               "unlabelled_probe_changes": int((sel & (hd != dd) & (y == -1)).sum())})
            other = [b for b in per_board if b != board and name in per_board[b][0]]
            final = meta[name][-1]
            s5.append({"board": board, "model": name, "inputs_in_bundle": len(host),
                       "final_job": final["job"], "outputs_present": int(np.isfinite(dev).sum()),
                       "missing_outputs": int((~np.isfinite(dev)).sum()),
                       "device_resets_final_run": ("not logged (z8 host loop)" if final["device_resets"] is None
                                                   else final["device_resets"]),
                       "inputs_skipped_final_run": final["skipped"],
                       "earlier_runs": "; ".join(f"{e['job']}: n={e['n_reported']}, crash at {e['crashed_at_vector']}, "
                                                 f"error={e['error']}" for e in meta[name][:-1]),
                       "duplicates_possible": "no (lockstep protocol: one reply per input index, written once)",
                       "bit_identical_to": "; ".join(f"{b}: {int((per_board[b][0][name] == dev).sum())}/{len(dev)}"
                                                     for b in other)})
            js = final["json"]
            lat = js.get("latency_us", {})
            s4.append({"board": board, "runtime": "TFLM 1.0.0 (TensorFlowLite_ESP32), reference kernels",
                       "model": name, "model_bytes": js.get("flash_bytes"),
                       "arena_used_bytes": js.get("selected", {}).get("arena_used"),
                       "arena_static_bytes": 49152,
                       **{f"{st}_us_{q}": lat.get(st, {}).get(q) for st in ("quantize_input", "invoke", "dequantize_output")
                          for q in ("median", "p95", "max")}})
    write_csv("S3_device_disagreements.csv", s3)
    write_csv("S5_stream_completeness.csv", s5)
    return s3, s5, s4


# ------------------------------------------------------------------------------------------- S4
def linker(log: Path) -> dict:
    t = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    ram = re.search(r"RAM:.*used (\d+) bytes from (\d+)", t)
    fl = re.search(r"Flash:.*used (\d+) bytes from (\d+)", t)
    return {"static_ram_bytes": ram and int(ram.group(1)), "ram_total_bytes": ram and int(ram.group(2)),
            "flash_bytes": fl and int(fl.group(1)), "flash_partition_bytes": fl and int(fl.group(2))}


def build_s4(model_rows: list[dict]) -> list[dict]:
    fw = []
    builds = [
        ("ESP32-S3", "TFLM 1.0.0 (Arduino TensorFlowLite_ESP32)", 14,
         REV / "2026-11-05_z8_device_e7/device_e7/esp32s3_platformio.log", REV / "2026-11-05_z8_device_e7/device_e7/esp32s3"),
        ("ESP32", "TFLM 1.0.0 (Arduino TensorFlowLite_ESP32)", 14,
         REV / "2026-11-05_z8b_device_e7/device_e7/esp32_platformio.log", None),
        ("ESP32", "TFLM 1.0.0, rebuilt for z10 (heap logged)", 14,
         REV / "2026-11-06_z10_runtime_compare/device_e7/esp32_runtime_arduino_tflm_1_0_0_platformio.log",
         REV / "2026-11-06_z10_runtime_compare/device_e7/esp32_runtime_arduino_tflm_1_0_0"),
        ("ESP32", "esp-tflite-micro 1.4.2 / ESP-IDF 5.5, esp-nn", 3,
         REV / "2026-11-06_z10d_runtime_compare_idf55/device_e7/esp32_runtime_esp_tflite_micro_platformio.log",
         REV / "2026-11-06_z10d_runtime_compare_idf55/device_e7/esp32_runtime_esp_tflite_micro"),
        ("ESP32", "esp-tflite-micro 1.4.2 / ESP-IDF 5.5, reference kernels", 3,
         REV / "2026-11-06_z10f_runtime_kernels_ansi_c/device_e7/esp32_runtime_esp_tflite_micro_ansi_c_platformio.log",
         REV / "2026-11-06_z10f_runtime_kernels_ansi_c/device_e7/esp32_runtime_esp_tflite_micro_ansi_c"),
    ]
    bundle = json.loads((BUNDLE / "bundle.json").read_text())
    for chip, rt, nmodels, log, d in builds:
        dev = json.loads((d / "device_agreement.json").read_text())["device"] if d and (d / "device_agreement.json").exists() else {}
        mb = (sum(m["bytes"] for m in bundle["models"]) if nmodels == 14 else
              sum(m["bytes"] for m in bundle["models"] if m["name"] in ("ton_S80", "ton_S80_pt", "ton_PQ80_int16")))
        fw.append({"chip": chip, "runtime": rt, "models_embedded": nmodels, **linker(log), "models_bytes": mb,
                   "arena_static_bytes": 49152, "heap_free_bytes": dev.get("memory_free_heap", ""),
                   "heap_min_free_bytes": dev.get("memory_min_free_heap", ""), "build_log": str(log.relative_to(ROOT))})
    write_csv("S4_firmware_memory.csv", fw)
    # per-model rows of the current runtime from the runtime comparison
    rc = ROOT / "docs/runtime_compare/runtime_compare.csv"
    if rc.exists():
        with open(rc, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["far"] == "0.01" and "esp-tflite" in r["runtime"]:
                    model_rows.append({"board": "ESP32", "runtime": r["runtime"], "model": r["model"],
                                       "arena_used_bytes": r["arena_used"], "arena_static_bytes": 49152,
                                       "invoke_us_median": r["invoke_us_median"]})
    write_csv("S4_model_memory_latency.csv", model_rows)
    return fw


# ------------------------------------------------------------------------------------------- S7
def build_s7() -> list[dict]:
    prov = json.loads(Z9.read_text())["provenance"]
    rows = []
    for split, v in prov.items():
        for part in ("new_test", "new_client_train", "new_client_val"):
            if part in v:
                rows.append({"split": split, "partition": part, **v[part],
                             "manifest_test_hash": v.get("manifest_test_hash", "")})
    write_csv("S7_split_provenance.csv", rows)
    return rows


FREEZE = [
    ("2026-10-28 02:00--02:14", "Pipeline v13, training recipe, arms F/F8/P/Q/R/S/SQ, budgets (80/128 KB; FAR 1\\%/5\\%), threshold rule and strong-PTQ candidates committed, before any v13 result."),
    ("2026-10-29 05:48", "v13.1 rules committed (drop impossible negative values; the new test split excludes every record of the earlier test split), before any v13 test-split result was used."),
    ("2026-10-29 07:24", "Float32 threshold fix (validation-side assertion); the two affected jobs were re-run (z1d, z1e) with unchanged models."),
    ("2026-10-29 07:29", "E3/E4 design committed before any E3/E4 result."),
    ("2026-10-29 15:58", "\\emph{Post hoc}, after E1--E4 test results: QC/SQC arms, E4 clipped-FP32 and 16-bit rows, E6 host split."),
    ("2026-11-05 00:57", "\\emph{Post hoc}, after device results: per-tensor re-exports of the TON S80/P80 device models."),
    ("2026-11-06", "\\emph{Post hoc}: per-tensor strong-PTQ replacements for every F8/P/S80 run (z9, same calibration draw and selection rule); runtime comparison (z10, z10d, z10f). No model retrained, no threshold changed."),
]


# ------------------------------------------------------------------------------------------- S8
def train_minutes(job: str) -> dict:
    t = (REV / job / "train_fl/train_fl.md").read_text()
    mins = defaultdict(list)
    for name, m in re.findall(r"\| (\w+)_s\d \| \d \| ([\d.]+) \|", t):
        mins[name].append(float(m))
    return {k: statistics.mean(v) for k, v in mins.items()}


COST = {  # federated training cost in rounds, beyond the shared data pipeline
    "F8": "teacher (60 rounds) + PTQ",
    "P": "teacher (60 rounds) + 3 fine-tuning rounds + PTQ",
    "QC": "teacher (60) + 3 fine-tuning + 2 QAT rounds",
    "R": "teacher (60) + 3 distilled fine-tuning (teacher pass) + 2 QAT rounds",
    "S80": "compact model (60 rounds) + PTQ",
    "SQC": "compact model (60) + 2 QAT rounds",
}


def build_s8(agg: dict, s4_models: list[dict]) -> list[dict]:
    lat = defaultdict(dict)
    dev_name = {"R": "R80", "QC": "QC80", "S80": "S80", "P": "P80"}
    for r in s4_models:
        if "TFLM 1.0.0" in r.get("runtime", "") and r.get("invoke_us_median"):
            lat[r["model"]][r["board"]] = float(r["invoke_us_median"]) / 1000
    out = []
    for split in SPLITS:
        tm = train_minutes(SPLITS[split][3])
        for arm in ("F", "F8", "P", "QC", "R", "S80", "SQC"):
            row = {"split": split, "arm": arm}
            for fb in ("0.01", "0.05"):
                k = next((k for k in agg if k[0] == split and k[1] == arm and k[3] == fb
                          and k[2] in ("80", "-") and k[4] != "one client"), None)
                if not k:
                    continue
                v = agg[k]
                rec = [100 * v[s][0] for s in sorted(v)]
                far = [100 * v[s][1] for s in sorted(v)]
                row[f"recall_{fb}_mean"] = round(statistics.mean(rec), 2)
                row[f"recall_{fb}_sd"] = round(statistics.stdev(rec), 2) if len(rec) > 1 else ""
                row[f"test_far_{fb}_mean"] = round(statistics.mean(far), 2)
                row[f"test_far_{fb}_range"] = f"{min(far):.2f}-{max(far):.2f}"
            row["training_cost"] = COST.get(arm, "teacher (60 rounds)")
            row["teacher_minutes"] = round(tm.get("F", float("nan")), 1)
            row["compact_minutes"] = round(tm.get("S80", float("nan")), 1) if arm in ("S80", "SQC") else ""
            ds = "ton" if split.startswith("TON") else "cic"
            dn = f"{ds}_{dev_name.get(arm, '')}" + ("_pt" if ds == "ton" and arm in ("S80", "P") else "")
            if split != "TON-hosts" and dn in lat:
                row["device_model"] = dn
                row["latency_ms_esp32s3"] = round(lat[dn].get("ESP32-S3 (TFLM 1.0.0)", float("nan")), 2)
                row["latency_ms_esp32"] = round(lat[dn].get("ESP32 (TFLM 1.0.0)", float("nan")), 2)
            row["exports"] = "S1 rows with reported=1 (file, SHA-256)"
            out.append(row)
    write_csv("S8_deployable_models.csv", out)
    return out


# ------------------------------------------------------------------------------------------- LaTeX
def tex_table(caption: str, header: list[str], rows: list[list], spec: str, size: str = "\\scriptsize") -> str:
    """A longtable (breaks across pages, stays in order)."""
    head = " & ".join(header) + r" \\"
    lines = ["{" + size, f"\\begin{{longtable}}{{{spec}}}", f"\\caption{{{caption}}}\\\\", r"\toprule", head, r"\midrule",
             r"\endfirsthead", r"\toprule", head, r"\midrule", r"\endhead", r"\bottomrule", r"\endlastfoot"]
    lines += [" & ".join(tex_escape(c) for c in r) + r" \\" for r in rows]
    lines += [r"\end{longtable}}", ""]
    return "\n".join(lines)


def build_tex(s1, s2, s3, s5, fw, s7, s8) -> None:
    parts = [r"""\documentclass[10pt]{article}
\usepackage[margin=1.8cm,landscape]{geometry}
\usepackage{booktabs,longtable,float}
\usepackage[T1]{fontenc}
\setlength{\tabcolsep}{3pt}
\title{Supplementary Results: When Does INT8 Compression Preserve Federated Intrusion Detection on Microcontrollers?}
\date{}
\begin{document}\maketitle
\noindent Every table is generated by \texttt{scripts/build\_supplement.py} from records that already exist in the
repository (no retraining, no new statistical tests); the complete rows, with file names and SHA-256 of every
export, are in the CSV files named in each caption (\texttt{paper/supplement/}). Recall and FAR in \%; ``test FAR''
is the false-alarm rate \emph{achieved} on the test split at the threshold chosen for the stated
\emph{validation} FAR budget. Reported rows use the per-tensor exports; first-selected per-channel exports are
kept in S1 and marked as superseded.
"""]
    # S8
    rows = []
    for r in s8:
        rows.append([r["split"], r["arm"],
                     f"{r.get('recall_0.01_mean', '')} $\\pm$ {r.get('recall_0.01_sd', '')}".replace("_", ""),
                     r.get("test_far_0.01_mean", ""), r.get("test_far_0.01_range", ""),
                     r.get("recall_0.05_mean", ""), r.get("test_far_0.05_mean", ""),
                     r.get("latency_ms_esp32s3", ""), r.get("latency_ms_esp32", ""), r["training_cost"],
                     r["teacher_minutes"], r["compact_minutes"]])
    parts.append(tex_table("S8 (\\texttt{S8\\_deployable\\_models.csv}): principal models, 80\\,KB arms; mean over three runs. "
                           "Latency: median invoke on the device model of the same arm (seed 1). Minutes: federated training "
                           "wall-clock per run on one PC.",
                           ["split", "arm", "recall@1\\%", "test FAR", "FAR range", "recall@5\\%", "test FAR", "S3 ms",
                            "ESP32 ms", "training", "teacher min", "compact min"],
                           [[c.replace("$\\pm$", "PM") if isinstance(c, str) else c for c in r] for r in rows],
                           "llrrrrrrrlrr").replace("PM", "$\\pm$"))
    # S1 per run
    rep = [r for r in s1 if r["reported"] and r["budget_kb"] in ("80", "-") and r["fine_tune"] != "one client"]
    by = defaultdict(dict)
    for r in rep:
        by[(r["split"], r["arm"])][(int(r["seed"]), r["far_budget"])] = r
    rows = []
    for (split, arm), v in sorted(by.items(), key=lambda x: (list(SPLITS).index(x[0][0]), x[0][1])):
        cells = []
        for fb in ("0.01", "0.05"):
            for s in (1, 2, 3):
                r = v.get((s, fb))
                cells.append(f"{pct(r['test_recall'])} / {pct(r['test_far'])}" if r else "")
        rows.append([split, arm] + cells)
    parts.append(tex_table("S1 (\\texttt{S1\\_operating\\_points.csv}): recall / achieved test FAR per run (seed 1, 2, 3) at "
                           "validation FAR budgets of 1\\% and 5\\%, 80\\,KB arms and the teacher. 128\\,KB arms and one-client "
                           "fine-tuning are in the CSV.",
                           ["split", "arm", "1\\%: s1", "s2", "s3", "5\\%: s1", "s2", "s3"], rows, "llrrrrrr"))
    # S2 run-paired at 1%
    rp = [r for r in s2 if r["kind"].startswith("run-paired")]
    rows = []
    for split in SPLITS:
        for fb in ("0.01", "0.05"):
            for a, b in PAIRS:
                xs = [r for r in rp if r["split"] == split and r["far_budget"] == fb and r["pair"] == f"{a} - {b}"]
                if xs:
                    rows.append([split, fb.replace("0.0", "") + "%", f"{a} - {b}",
                                 " ".join(f"{r['recall_diff_pts']:+.2f}" for r in xs),
                                 " ".join(f"{r['test_far_diff_pts']:+.2f}" for r in xs)])
    parts.append(tex_table("S2 (\\texttt{S2\\_paired\\_differences.csv}): run-paired differences in recall and in achieved test "
                           "FAR (points; seeds 1 2 3) between reported exports. The CSV also holds the record-level McNemar "
                           "tests of job z9 with, for each, whether it was computed on the reported export or on a first-selected "
                           "export that the per-tensor replacement superseded.",
                           ["split", "budget", "pair", "recall diff", "test FAR diff"],
                           rows, "lllll"))
    # S3 classification, all inputs
    rows = []
    for r in s3:
        if r["input_category"] != "all":
            continue
        rows.append([r["board"], r["model"], r["weight_scales"], r["activations"], r["far_budget"], f"{r['agree']}/{r['n']}",
                     r["additional_misses"], r["additional_detections"], r["added_false_alarms"],
                     r["avoided_false_alarms"], r["unlabelled_probe_changes"]])
    parts.append(tex_table("S3 (\\texttt{S3\\_device\\_disagreements.csv}): device decisions vs.\\ the host TFLite interpreter on the "
                           "same file, at the frozen thresholds; the CSV splits every count by input category (random test "
                           "records, near-threshold records, records beyond 5\\,SD, synthetic extremes, zero vectors).",
                           ["board", "model", "weights", "act.", "FAR", "agree", "add.\\ misses", "add.\\ detections",
                            "added FA", "avoided FA", "probes"], rows, "lllllrrrrrr", "\\tiny"))
    # S5
    rows = [[r["board"], r["model"], r["final_job"].split("_")[1], f"{r['outputs_present']}/{r['inputs_in_bundle']}",
             r["device_resets_final_run"], r["inputs_skipped_final_run"], r["bit_identical_to"],
             r["earlier_runs"][:70]] for r in s5]
    parts.append(tex_table("S5 (\\texttt{S5\\_stream\\_completeness.csv}): stream completeness. Final job, outputs present, device "
                           "resets and skipped inputs in the final run, bit-identity with the other board, earlier incomplete runs "
                           "(superseded). Duplicates are impossible: the protocol is lockstep, one reply per input index.",
                           ["board", "model", "job", "outputs", "resets", "skipped", "bit-identical", "earlier runs"],
                           rows, "lllrrrll", "\\tiny"))
    # S4
    rows = [[r["chip"], r["runtime"], r["models_embedded"], r["static_ram_bytes"], r["arena_static_bytes"],
             r["flash_bytes"], r["models_bytes"], r["heap_free_bytes"], r["heap_min_free_bytes"]] for r in fw]
    parts.append(tex_table("S4 (\\texttt{S4\\_firmware\\_memory.csv}, \\texttt{S4\\_model\\_memory\\_latency.csv}): firmware memory "
                           "from the linker (bytes). Static RAM includes the 49,152-byte tensor arena; each model uses 1.4--1.8\\,KB "
                           "of it on TFLM 1.0.0 and 1.9--5.9\\,KB on the current runtime (per-model file). Heap was logged only in "
                           "the z10 builds; it does not change during the stream. Latency is measured around \\texttt{Invoke()} "
                           "only (the graph's QUANTIZE, four FULLY\\_CONNECTED, LOGISTIC, DEQUANTIZE); copying the features into "
                           "the input tensor is reported separately and serial transfer is excluded.",
                           ["chip", "runtime", "models", "static RAM", "arena", "flash", "models", "heap free", "min free"],
                           rows, "llrrrrrrr"))
    # S6
    rc = ROOT / "docs/runtime_compare/runtime_compare.md"
    if rc.exists():
        parts.append("\\section*{S6: runtime comparison}\nSee \\texttt{docs/runtime\\_compare/runtime\\_compare.\\{md,csv\\}} "
                     "(same three files and 4,462 TON\\_IoT inputs on the same classic ESP32; runtime-comparison table of the paper). Build "
                     "provenance (component hashes, \\texttt{dependencies.lock}, \\texttt{sdkconfig}) is in each "
                     "\\texttt{device\\_e7/<board>/build\\_meta/} of jobs z10, z10d, z10f.\n")
    # S7
    rows = [[r["split"], r["partition"], r["records"], r["in_old_test"], r["in_old_training_pool"]] for r in s7]
    parts.append(tex_table("S7a (\\texttt{S7\\_split\\_provenance.csv}): overlap of the revised splits with the earlier (exploratory) "
                           "split. Test records were all in the earlier training pool, which only exploratory models of the earlier "
                           "pipeline used; none was in the earlier test split. No reported model, threshold or calibration draw used "
                           "a test record.",
                           ["split", "partition", "records", "in earlier test", "in earlier training pool"], rows, "llrrr"))
    parts.append("{\\scriptsize\\begin{longtable}{lp{20cm}}\\caption{S7b: method-freeze history (UTC, from the repository "
                 "history).}\\\\ \\toprule time & event \\\\ \\midrule\n" +
                 "\n".join(f"{t} & {e} \\\\" for t, e in FREEZE) + "\n\\bottomrule\\end{longtable}}\n")
    parts.append(r"\end{document}")
    (OUT / "supplement.tex").write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    s1 = build_s1()
    agg = summarize(s1)
    s2 = build_s2(agg)
    s3, s5, s4_models = build_s3_s5()
    fw = build_s4(s4_models)
    s7 = build_s7()
    s8 = build_s8(agg, s4_models)
    build_tex(s1, s2, s3, s5, fw, s7, s8)
    print("wrote", ", ".join(sorted(p.name for p in OUT.iterdir())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
