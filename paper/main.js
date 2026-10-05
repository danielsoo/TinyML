const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, PageBreak,
  P, RP, H1, H2, H3, caption, makeTable, img, UP, convertInchesToTwip
} = require("./build.js");
const fs = require("fs");
const path = require("path");

// Results that are still being produced are marked in red so they cannot be missed.
// Sources for every number: docs/REVISION_RESULTS.md and data/processed/revision/<run>/.
const PENDING_COLOR = "C00000";
function PEND(text) { return { text: `[PENDING: ${text}]`, bold: true, color: PENDING_COLOR }; }
function PP(parts, opts) { return RP(parts.map(p => (typeof p === "string" ? { text: p } : p)), opts); }

const FIG = path.join(__dirname, "figures") + "/";
const children = [];

// ---------------- Title block ----------------
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 40 },
  children: [new TextRun({ text: "Reliable Federated TinyML Deployment for IoT Security", bold: true, size: 32 })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 200 },
  children: [new TextRun({ text: "Extended version — revised in response to LCTES '26 Work-in-Progress reviews; all results recomputed after pipeline corrections", italics: true, size: 22 })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 40 },
  children: [new TextRun({ text: "Younsoo Park, Seokhyeon Bae", size: 22 })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 300 },
  children: [new TextRun({ text: "Pennsylvania State University", italics: true, size: 20 })],
}));

// ---------------- Reviewer response note ----------------
children.push(H2("Note to reviewers: how this revision addresses the WIP feedback"));
children.push(P(
  "We thank the reviewers for detailed, actionable feedback. While re-running every experiment the reviewers asked for, we found four implementation errors in our training and compression code (Section 6.2). Two of them affected the numbers in the WIP submission: the compressed model reported there did not inherit the federated model's weights, and the federated QAT runs exchanged incorrectly scaled weights. We fixed all four, re-ran every experiment from scratch, and every number in this version comes from the corrected pipeline. Claims that depended on the erroneous runs (the training-time-QAT/compression interaction, pruning as a regularizer, and the earlier robustness tables) have been removed or replaced.",
  { spacingAfter: 120 }
));
const respItems = [
  ["No comparison to other baselines (81A); fix the baseline (81B-1)", "Section 5.1 adds a centralized baseline trained with the identical recipe, on both datasets. Comparison with published IDS systems on CIC-IDS2017 is discussed in Section 7 with the comparability caveats that apply."],
  ["Non-IID clients / client count (81B-2, 81D)", "Section 5.2 evaluates a Dirichlet(α = 0.3) label partition on both datasets. Client-count scaling remains open (Section 7)."],
  ["On-device measurement (81B-3)", "The ESP32 firmware and collection tooling are complete and verified against the same TensorFlow Lite Micro library on a host build (Section 6.1, Appendix B)."],
  ["Missing FGSM results (81B-4, 81D-3)", "Section 5.7 (Table 5) reports FGSM and PGD on the corrected federated, centralized, and compressed models for both datasets."],
  ["Lead with QAT finding (81B-5)", "With the corrected pipeline, the earlier QAT finding does not hold. Section 5.6 reports what we now observe: training-time QAT inside FL collapses on CIC-IDS2017 because of heavy-tailed features, while it works on TON_IoT."],
  ["Decompose the compression ratio (81B-6)", "Section 5.3 separates INT8 quantization (3.5×) from structured pruning (a further 3.5×), Section 5.5 sweeps the pruning ratio, Section 5.9 compares five post-training quantization methods, and Section 5.10 evaluates client-local knowledge distillation as an alternative to pruning, and Section 5.12 compares 16 combinations of pruning, PTQ, QAT and distillation."],
  ["Dataset generalization (81B-7, 81C)", "Every experiment is repeated on TON_IoT (network), with identifier and timestamp columns removed (Section 3.1)."],
  ["Minor items: focal-loss motivation, duplicate table rows, BatchNorm derivation (81B)", "Section 3.2 gives the focal-loss setting. The duplicate-row table came from the superseded sweep and is removed. The BatchNorm folding derivation is in Appendix A, corrected for our post-activation BatchNorm placement."],
  ["Practical meaning of Attack Recall and false positives (81C)", "Section 5.8 reports missed-attack and false-alarm rates for every headline model."],
  ["Device heterogeneity (81C)", "Discussed in Section 6.1."],
  ["Related work depth and citation formatting (81C)", "Section 2 is reorganized by topic (24 references)."],
  ["Incremental novelty (meta-review)", "Section 6.3 states which results we consider non-obvious, including the negative QAT result and four silent failure modes in a common FL/TinyML toolchain."],
];
for (const [k, v] of respItems) {
  children.push(RP([{ text: "• " + k + ": ", bold: true }, { text: v }], { spacingAfter: 100 }));
}
children.push(new Paragraph({ children: [new PageBreak()] }));

// ---------------- Abstract ----------------
children.push(H1("Abstract"));
children.push(P(
  "Federated Learning (FL) combined with TinyML is an attractive basis for privacy-preserving intrusion detection on microcontroller-class IoT devices, but the claim that a federated model can be compressed for such devices without losing detection quality is rarely tested end to end. We train a multilayer-perceptron intrusion detector with Flower (FedAvgM, cosine learning-rate decay, focal loss) on CIC-IDS2017 and TON_IoT, and compress it for TensorFlow Lite Micro using only data a federated deployment would actually hold. Three results stand out. First, federated training comes close to centralized training on the same recipe (F1 85.70 vs. 87.35 on CIC-IDS2017; 99.40 vs. 99.48 on TON_IoT), and a strongly non-IID Dirichlet(0.3) partition costs at most about one F1 point. Second, structured pruning followed by fine-tuning and QAT fine-tuning on a single participating client's local data yields a 65 KB INT8 model (12.3× smaller than FP32) with no loss relative to the federated model (F1 85.59, Attack Recall 99.66% on CIC-IDS2017; F1 98.74 on TON_IoT). How well this works depends on how closely that client's class mix matches the global one. When missed attacks must be minimized, distillation from the federated model and a threshold chosen on the client's own held-out data together with fine-tuning on all clients' local data, let the 65 KB model miss 27 of 85,173 attacks, about half as many as the FP32 model at its default threshold (52), at an 11.3% false-alarm rate; on TON_IoT, after a validation-selected improvement of the federated model (text-derived features, focal α = 0.5), the 55 KB model misses 12 of 16,215 attacks at 6.6%. For on-device deployment we use a 112 KB CIC-IDS2017 variant (23 of 85,173 missed at 9.7%, mean over fine-tuning draws) and this 55 KB TON_IoT model. Third, training-time quantization-aware training inside FL collapses on CIC-IDS2017 (34.7% accuracy): heavy-tailed standardized features drive the learned INT8 input range to [−1231, 405]. The same method works on TON_IoT, and full-integer post-training quantization of the float model avoids that collapse but is unstable across calibration draws (F1 between 12 and 99 for the same TON_IoT model) until the calibration inputs are clipped, after which it is stable (F1 99.32–99.35). A local-only baseline shows where federation pays off: clients with few local attacks detect 45–93% of attacks alone and 98–99.8% with the federated model. We also document four silent failure modes in a common TensorFlow, tfmot, and Flower toolchain that had invalidated the compressed-model results of our own earlier report, together with the checks that catch them.",
  { spacingAfter: 200 }
));

// =================== 1 Introduction ===================
children.push(H1("1. Introduction"));
children.push(P(
  "The rapid proliferation of Internet of Things (IoT) devices has transformed sensing, monitoring, and control systems across smart homes, healthcare, industrial automation, and critical infrastructure [1]. These systems increasingly rely on machine learning models trained on sensitive data, raising concerns about privacy, ownership, and regulatory compliance. Traditional centralized learning requires raw data to be transmitted to cloud servers, introducing privacy risk, communication overhead, and exposure to data breaches."
));
children.push(P(
  "Federated Learning (FL) addresses this by letting clients train locally and share only model updates with a server [7, 8]. Deploying the result on real IoT hardware is still difficult. Devices are constrained in memory, compute, energy, and bandwidth, while standard FL pipelines assume comparatively large full-precision models. TinyML closes part of this gap by compressing models for microcontrollers [2, 19]. Combining the two raises a question that is easy to state and, as we found, easy to get wrong in practice: can a model trained without pooling data be compressed to microcontroller size without either losing what federated training learned or quietly relying on pooled data during compression?"
));
children.push(P(
  "Intrusion detection is a demanding test case. A missed attack is far more costly than a false alarm, so Attack Recall matters more than aggregate accuracy, and benchmark datasets are imbalanced and heavy-tailed. This revision answers the reviewers' requests for a centralized baseline, non-IID clients, a second dataset, and robustness results. While doing so, it also corrects our own earlier pipeline, whose compressed models turned out not to inherit the federated weights (Section 6.2)."
));
children.push(P(
  "Contributions. (1) An end-to-end evaluation of a federated TinyML intrusion detector on two datasets against a centralized baseline trained with the identical recipe, under both near-IID and Dirichlet(0.3) client partitions (Sections 5.1–5.2). (2) An FL-faithful compression procedure: exact BatchNorm folding, structured pruning, then fine-tuning and QAT fine-tuning on one client's local data. It produces a 65 KB INT8 model (12.3×) with no loss relative to the federated model, and we quantify how the outcome depends on which client performs the fine-tuning (Sections 5.3–5.5). We compare it with five post-training quantization methods and with client-local knowledge distillation, we test against clients training alone to show where federation helps, we compare 16 complete pipelines that combine structured and unstructured pruning, PTQ, QAT and distillation, and we show how to deploy the 65 KB model with fewer missed attacks than the FP32 model using only client-local data (Sections 5.9–5.13). (3) A negative result: training-time QAT inside FL fails on CIC-IDS2017 because its moving-average quantizers absorb the heavy-tailed feature outliers. It works on TON_IoT, and post-training quantization of a float model avoids the collapse (Section 5.6), although full-integer INT8 PTQ varies widely across calibration draws, more calibration data makes it worse, and clipping the calibration inputs removes the instability (Section 5.9). (4) Four silent failure modes in a widely used TensorFlow / TensorFlow Model Optimization / Flower toolchain, each of which yields plausible-looking but wrong compressed-model results, with the checks we now run to catch them (Section 6.2). (5) FGSM/PGD robustness of the corrected models (Section 5.7) and a verified ESP32 benchmark harness (Section 6.1, Appendix B)."
));

// =================== 2 Related Work ===================
children.push(H1("2. Related Work"));

children.push(H2("2.1 IoT Security and Intrusion Detection"));
children.push(P(
  "IoT deployments expose billions of resource-constrained devices to denial-of-service attacks, botnet recruitment, and other network intrusions [1]. Machine-learning-based IDS detect such traffic well [13], but most are designed for centralized training with resources beyond typical IoT hardware. CIC-IDS2017 [13] remains the most widely used benchmark. Its known labelling and flow-construction issues [24] mean that absolute numbers are hard to compare across papers. TON_IoT [15] and Bot-IoT [14] target IoT-specific traffic. We use CIC-IDS2017 and the TON_IoT network dataset."
));

children.push(H2("2.2 Federated Learning"));
children.push(P(
  "FedAvg [8] trains a shared model by iteratively aggregating client updates, and Konečný et al. study how to reduce its communication cost [7]. Client data is rarely IID. FedAvgM, which we use, adds server momentum to stabilize training under non-identical label distributions [10]. Zhao et al. characterize the degradation as client data becomes skewed [20], and FedProx adds a proximal term for the same reason [11]. Following this line of work, we evaluate a Dirichlet(α = 0.3) label partition alongside a near-IID one."
));

children.push(H2("2.3 TinyML and Model Compression"));
children.push(P(
  "TinyML enables inference directly on microcontrollers [2], and TensorFlow Lite Micro is the runtime we target [19]. Structured pruning removes redundant parameters [4], knowledge distillation transfers knowledge to a smaller student [5], and quantization reduces numerical precision [6]. Quantization-aware training (QAT) and post-training quantization (PTQ) are the two dominant routes to integer models. Gholami et al. survey them [18], Nagel et al. give practical guidance on when 8-bit PTQ suffices [17], and they also document oscillation-driven instability in QAT [16]. Section 5.6 reports a different, data-driven failure of QAT that arises when quantizer ranges are learned on heavy-tailed inputs."
));

children.push(H2("2.4 Adversarial Robustness of Compressed Models"));
children.push(P(
  "FGSM [3] and PGD [9] are the standard gradient-based attacks used to evaluate adversarial robustness. Gorsline et al. study how quantization level affects robustness [21], Song et al. recover robustness lost to weight quantization [22], and Ayaz et al. study deeply quantized TinyML-scale networks [23]. We evaluate whether the federated model's robustness carries over to its compressed INT8 derivatives in a federated IDS setting."
));

children.push(H2("2.5 Positioning"));
children.push(P(
  "Few studies evaluate federated training, microcontroller-scale compression, and intrusion detection together under the constraint that compression must not use pooled data. This paper is an empirical characterization of that combination rather than a new algorithm. It reports what works (federated training close to centralized, compression without loss using one client's data), what does not (training-time QAT on heavy-tailed IDS features), and which toolchain pitfalls make the latter easy to miss."
));

// =================== 3 Approach ===================
children.push(H1("3. Approach"));

children.push(H2("3.1 Datasets and Preprocessing"));
children.push(P(
  "CIC-IDS2017 [13]: the eight MachineLearningCSV files (2,830,743 flows, 78 numeric features). Labels are binarized (BENIGN → 0, all attacks → 1), ±∞ are replaced, and non-numeric values are coerced and zero-filled. We remove duplicate rows before any split (331,200 removed; 2,499,543 unique flows, of which 425,863 are attacks). The pool is shuffled with a fixed seed and split 80/20 with stratification (test set: 499,909 flows at the natural class ratio). On the training split only, the majority class is undersampled to at most 4:1 (balance_ratio = 4.0), a StandardScaler is fitted, and SMOTE balances the classes."
));
children.push(P(
  "TON_IoT [15]: the network train/test file (211,043 records). To avoid shortcut features, we drop host identifiers and ephemeral ports (src_ip, dst_ip, src_port; the file has no timestamp column) and keep dst_port, which corresponds to CIC-IDS2017's Destination Port feature. Text columns with at most 50 distinct values (protocol, service, connection state, DNS/SSL/HTTP flags) are label-encoded, and high-cardinality free text (DNS query, URI) is dropped, leaving 37 features. Duplicates are removed (105,176 unique records: 24,106 normal, 81,070 attack) and the same split, scaling, and SMOTE steps are applied (test set: 21,036 records). Section 5.14 additionally evaluates an improved TON_IoT federated model, selected on validation data, that adds six row-local features derived from the DNS query and the URI (length, digit count, and non-alphanumeric count of each field; 43 features) and uses focal α = 0.5; unless stated otherwise, TON_IoT results use the original 37-feature model."
));

children.push(H2("3.2 Federated Learning Setup"));
children.push(P(
  "The model is an MLP (512→256→128, ReLU, each hidden layer followed by BatchNormalization and Dropout, sigmoid output). We train it with Flower using four clients, FedAvgM (server momentum 0.5, server learning rate 0.1) [10], 60 rounds × 3 local epochs, batch size 128, and Adam. The server broadcasts a cosine learning-rate schedule (10⁻³ → 10⁻⁴) so that all clients decay in lock-step. Class imbalance is further addressed with focal loss [12] (α = 0.35, carried over from preliminary experiments in which larger α values were unstable) and class weights. Client data is partitioned either near-IID (each class split evenly across clients) or with a Dirichlet(α = 0.3) label partition (Section 5.2). The centralized baseline trains the identical model and loss for the same total epoch budget (180 epochs) with the same per-round cosine schedule. Training-time QAT is off in the main pipeline; Section 5.6 explains why."
));

children.push(H2("3.3 FL-Faithful Compression"));
children.push(P(
  "After federated training, the server holds only the global float model. Compression proceeds in four steps. (1) BatchNorm is folded exactly. Because BatchNorm follows the ReLU in our architecture, each BatchNorm is an affine map h ↦ s·h + t that we fold into the next Dense layer rather than the preceding one (Appendix A; maximum output deviation ≤ 2·10⁻⁵ on our models). (2) Neuron-level structured pruning removes 50% of each hidden layer by weight magnitude. (3) One participating client fine-tunes the pruned model for 3 epochs on 10,000 samples of its own local data. (4) The same client performs 2 epochs of QAT fine-tuning, and the model is exported to INT8 TensorFlow Lite. For comparison we also report INT8 PTQ without pruning, pruning without fine-tuning, and fine-tuning on 10,000 pooled training samples. The pooled variant requires server-side data a real deployment would not have, so we report it only as an upper bound."
));

children.push(H2("3.4 Evaluation"));
children.push(P(
  "All metrics are computed on the held-out test split at a fixed decision threshold of 0.3 (prob ≥ 0.3 → attack), chosen before these experiments. We report accuracy, precision, Attack Recall, F1, and the false-alarm rate (FAR, the fraction of benign flows flagged). Robustness is measured with FGSM and PGD (ε = 0.1; PGD with 10 steps). Adversarial examples are generated white-box on the float federated model and transferred to the centralized model and to every compressed TFLite variant, so all models face the same perturbed inputs."
));

// =================== 4 Experimental Setup ===================
children.push(H1("4. Experimental Setup"));
children.push(P(
  "All experiments ran on a single desktop CPU (16 cores, WSL2; federated simulation runs clients on CPU by design). A full CIC-IDS2017 run (centralized baseline, federated training, compression) takes about 5.4 hours, the Dirichlet run about 2.7 hours, and TON_IoT runs about 25 minutes. Each configuration was run once with a fixed seed. Seed variance is not yet reported (Section 7). The code, configurations (config/paper_v12_float, config/paper_v12_toniot), per-run logs, and result files are in the project repository."
));

// =================== 5 Results ===================
children.push(H1("5. Results"));

children.push(H2("5.1 Federated vs. Centralized Training"));
const t1w = [2600, 950, 950, 1000, 900, 850, 1900];
const t1h = ["Model (test split, threshold 0.3)", "Accuracy", "Precision", "Attack Recall", "F1", "FAR", "Missed attacks / all"];
children.push(makeTable(t1h, [
  ["CIC-IDS2017 — centralized", "95.07%", "77.60%", "99.90%", "87.35%", "5.92%", "88 / 85,173 (0.10%)"],
  ["CIC-IDS2017 — federated (near-IID)", "94.32%", "75.01%", "99.94%", "85.70%", "6.84%", "52 / 85,173 (0.061%)"],
  ["CIC-IDS2017 — federated, fixed LR", "95.01%", "77.40%", "99.84%", "87.20%", "5.99%", "136 / 85,173 (0.16%)"],
  ["TON_IoT — centralized", "99.20%", "99.32%", "99.64%", "99.48%", "2.28%", "59 / 16,215 (0.36%)"],
  ["TON_IoT — federated (near-IID)", "99.08%", "99.18%", "99.62%", "99.40%", "2.76%", "61 / 16,215 (0.38%)"],
  ["TON_IoT — federated, fixed LR", "99.02%", "99.00%", "99.73%", "99.36%", "3.38%", "44 / 16,215 (0.27%)"],
], t1w));
children.push(caption("Table 1. Federated vs. centralized training with the identical model, loss, and epoch budget (float models, before compression)."));
children.push(PP([
  "Federated training comes within 1.65 F1 points of centralized training on CIC-IDS2017 and within 0.08 on TON_IoT. On CIC-IDS2017 the gap lies almost entirely in precision (FAR 6.84% vs. 5.92%); both models detect more than 99.9% of attacks. ",
  "A fixed learning rate (10⁻³ throughout) does as well as the cosine schedule on both datasets: F1 87.20 vs. 85.70 on CIC-IDS2017 and 99.36 vs. 99.40 on TON_IoT. The WIP version attributed a jump in Attack Recall from 46.7% to 93.85% to the cosine schedule. With the weight-exchange error of Section 6.2 corrected, a fixed-LR federated model already reaches 99.84% Attack Recall, so that jump was an artifact of the erroneous pipeline and we withdraw the claim. We keep the cosine schedule only because the remaining experiments were run with it; it is not a contribution of this paper.",
]));

children.push(H2("5.2 Non-IID Clients"));
children.push(makeTable(["Model", "Client sizes (attack share)", "Accuracy", "F1", "Attack Recall", "FAR", "Missed attacks / all"], [
  ["CIC-IDS2017 — near-IID", "4 × equal (≈ class ratio)", "94.32%", "85.70%", "99.94%", "6.84%", "52 / 85,173 (0.061%)"],
  ["CIC-IDS2017 — Dirichlet(0.3)", "9.4k (0.1%), 1.98M (59.5%), 6.7k (2.6%), 728k (25.1%)", "94.27%", "85.59%", "99.83%", "6.87%", "144 / 85,173 (0.17%)"],
  ["TON_IoT — near-IID", "4 × equal (≈ class ratio)", "99.08%", "99.40%", "99.62%", "2.76%", "61 / 16,215 (0.38%)"],
  ["TON_IoT — Dirichlet(0.3)", "49.9k (0.4%), 62.9k (92.8%), 8.6k (65.5%), 8.4k (8.5%)", "97.63%", "98.45%", "97.99%", "3.59%", "326 / 16,215 (2.01%)"],
], [2100, 2500, 900, 800, 1000, 800, 1800]));
children.push(caption("Table 2. Effect of a Dirichlet(α = 0.3) label partition over four clients (float federated models). Client sizes are training samples after balancing and SMOTE."));
children.push(P(
  "The Dirichlet partition is extreme: on CIC-IDS2017 one client holds 73% of the training data and two clients see almost no attacks. Still, FedAvgM with the shared cosine schedule loses only 0.11 F1 on CIC-IDS2017 and 0.95 F1 on TON_IoT, where Attack Recall drops by 1.6 points and FAR rises from 2.76% to 3.59%. In the WIP-era runs, the non-IID partition appeared to collapse training (F1 63.5). That collapse was an artifact of the weight-exchange error described in Section 6.2."
));

children.push(H2("5.3 Compression Without Pooled Data"));
children.push(makeTable(["Variant (from the near-IID federated model)", "CIC size", "CIC F1 / FAR", "CIC missed of 85,173", "TON size", "TON F1 / FAR", "TON missed of 16,215"], [
  ["FP32 federated model (TFLite)", "802.5 KB", "85.70 / 6.84%", "52 (0.061%)", "720.6 KB", "99.40 / 2.76%", "61 (0.38%)"],
  ["INT8 PTQ only (no pruning)", "226.9 KB", "84.75 / 7.37%", "74 (0.087%)", "206.4 KB", "98.38 / 2.70%", "392 (2.42%)"],
  ["Prune 50%, no fine-tuning → PTQ", "75.1 KB", "42.28 / 56.04%", "46 (0.054%)", "64.9 KB", "96.78 / 18.15%", "192 (1.18%)"],
  ["Prune 50% → client FT → PTQ", "75.1 KB", "86.90 / 6.08%", "346 (0.41%)", "64.9 KB", "90.50 / 14.00%", "2,255 (13.91%)"],
  ["Prune 50% → client FT → QAT FT → INT8 (base recipe)", "65.4 KB", "85.59 / 6.82%", "291 (0.34%)", "55.2 KB", "98.74 / 2.88%", "268 (1.65%)"],
  ["Same, but fine-tuned on pooled data (upper bound)", "65.4 KB", "90.72 / 2.98%", "4,216 (4.95%)", "55.2 KB", "99.17 / 3.28%", "111 (0.68%)"],
], [3000, 900, 1300, 1200, 900, 1300, 1200]));
children.push(caption("Table 3. Compression of the federated model. \"Client FT\" uses 10,000 samples from one participating client's own partition (client 0); \"pooled\" uses 10,000 pooled training samples and is not available in a real federated deployment. The two rows without fine-tuning calibrate INT8 on 500 pooled samples (one draw; Section 5.9 shows how much full-integer PTQ varies across calibration draws)."));
children.push(P(
  "INT8 PTQ alone gives 3.5× at a cost of about one F1 point on both datasets for the calibration draw used here, but that cost varies widely from draw to draw (Section 5.9). Pruning half of each hidden layer gives a further 3.5×, but only if the pruned model is fine-tuned. Without fine-tuning, the pruned model breaks down (F1 42.28 on CIC-IDS2017, FAR 18% on TON_IoT). With fine-tuning and QAT fine-tuning on one client's local data, the resulting INT8 model is 65.4 KB on CIC-IDS2017 (802.5 / 65.4 = 12.27×) and 55.2 KB on TON_IoT (13.05×), and it matches the float federated model (F1 85.59 vs. 85.70 and 98.74 vs. 99.40). This CIC-IDS2017 model has 94.28% accuracy, 75.00% precision, and 99.66% Attack Recall. The QAT fine-tuning step matters on TON_IoT, where client-fine-tuned PTQ reaches only F1 90.50."
));
children.push(P(
  "Pooled fine-tuning scores higher on CIC-IDS2017 (F1 90.72), but only by shifting the operating point toward precision (FAR 2.98%): it misses 4,216 of 85,173 attacks (4.95%), against 291 for the client-fine-tuned model. The pooled sample's 20% attack share is closer to the test distribution than client 0's 50%. We treat this as an upper bound that requires server-held data, not as a federated result."
));

children.push(H2("5.4 Which Client Fine-Tunes Matters"));
children.push(makeTable(["Federated model", "Fine-tuning client (attack share)", "INT8 F1 (base recipe)", "FAR", "Missed attacks / all"], [
  ["CIC-IDS2017 near-IID", "client 0 (50.4%)", "85.59", "6.82%", "291 / 85,173 (0.34%)"],
  ["CIC-IDS2017 Dirichlet(0.3)", "client 3 (24.9%)", "91.08", "2.75%", "4,415 / 85,173 (5.18%)"],
  ["CIC-IDS2017 centralized model (reference)", "client 0 (50.4%)", "81.84", "8.84%", "771 / 85,173 (0.91%)"],
  ["TON_IoT near-IID", "client 0 (50.1%)", "98.74", "2.88%", "268 / 16,215 (1.65%)"],
  ["TON_IoT Dirichlet(0.3)", "client 2 (65.5%)", "98.72", "3.11%", "264 / 16,215 (1.63%)"],
], [2900, 2300, 1200, 900, 2000]));
children.push(caption("Table 4. Compressed-model quality as a function of the client that performs fine-tuning (prune 50% → client FT → QAT FT → INT8)."));
children.push(P(
  "The short fine-tuning phase recalibrates the decision boundary toward the fine-tuning client's class mix. A client whose attack share is close to the deployment distribution (client 3 of the CIC-IDS2017 Dirichlet run, 24.9% attacks) yields the highest F1 we observed (91.08, FAR 2.75%), but it misses 4,415 of 85,173 attacks (5.18%), fifteen times as many as the near-IID model fine-tuned on a 50%-attack client (291). A 50%-attack client yields a recall-heavy operating point. The choice of fine-tuning client is therefore a deployment decision that should consider class mix, and it should be reported. We did not tune it on the test set: the clients in Table 4 were chosen a priori by attack share."
));

children.push(H2("5.5 Compression Strength"));
children.push(P(
  "We sweep the structured-pruning ratio from 30% to 90% on the near-IID federated models. Each ratio is tested with no fine-tuning, client fine-tuning + PTQ, and client fine-tuning + QAT fine-tuning (Figure 1). This replaces the 48-configuration sweep of the WIP version, which was produced by the erroneous pipeline. Without fine-tuning, pruning breaks the CIC-IDS2017 model from 50% onward (from 70% it predicts every flow as an attack) and steadily degrades the TON_IoT model. With client fine-tuning, CIC-IDS2017 degrades gracefully: 30% pruning (112–126 KB) gives F1 87.6–89.7, 70% pruning (31 KB, 26×) gives 84.4–85.0, 85% (14 KB, 57×) gives 81.3–82.9, and 90% (9.9 KB, 81×) gives 78.0. PTQ and QAT fine-tuning track each other closely. On TON_IoT, client fine-tuning + QAT holds F1 between 98.6 and 98.9 all the way to 90% pruning (7.9 KB, 91× smaller than FP32), whereas client fine-tuning + PTQ fluctuates between 90.5 and 98.4. QAT fine-tuning is therefore the choice that is reliable on both datasets. On CIC-IDS2017 the useful range ends around 70% pruning (31 KB) if F1 within about 1.3 points of the federated model is required."
));
children.push(img(FIG + "prune_sweep.png", 600, 249));
children.push(caption("Figure 1. F1 vs. INT8 model size for structured-pruning ratios 30–90% (labels), with fine-tuning on one client's local data followed by QAT fine-tuning or PTQ, and without fine-tuning. Dashed line: the uncompressed federated model."));

children.push(H2("5.6 Training-Time QAT Inside FL: a Negative Result"));
children.push(P(
  "The WIP version reported that training-time QAT (fake quantization from the first federated round) helps at aggressive compression and hurts at moderate compression. That observation came from runs affected by the weight-exchange and QAT-stripping errors of Section 6.2. With both corrected, training-time QAT inside FL fails on CIC-IDS2017. The federated QAT model reaches 34.74% accuracy (F1 34.30, FAR 78.7%), and its float weights reach only F1 46.8. The reason is visible in the learned quantizer state. The input quantizer's moving-average range is [−1231, 405], and activation ranges reach 2342, because standardized CIC-IDS2017 features are heavy-tailed (rare flows lie hundreds to thousands of standard deviations from the mean). One INT8 step then spans about 6.4 standard deviations, so ordinary inputs collapse to the zero point. On TON_IoT, whose features are less extreme, the same procedure works (federated QAT model: accuracy 98.22%, F1 98.84). PTQ of a float federated model does not collapse on either dataset (Table 3), although Section 5.9 shows that its accuracy varies widely across calibration draws, and neither does QAT fine-tuning of a pruned float model, because its calibration happens on a small fine-tuning set after training. We therefore use post-training QAT fine-tuning, not training-time QAT, in the compression pipeline. Robust input scaling (clipping or a log transform before standardization) is a plausible fix for training-time QAT that we have not evaluated."
));

children.push(H2("5.7 Adversarial Robustness"));
children.push(P(
  "Table 5 reports accuracy on 20,000 test samples before and after FGSM and PGD (ε = 0.1 in standardized feature space; PGD with 10 steps). The perturbations are computed white-box on the float federated model and applied unchanged to every other model. The FGSM/FGM/GA/PGD table and the PGD adversarial-training study of the WIP version were computed on the erroneous compressed models and are withdrawn."
));
children.push(makeTable(["Model", "CIC clean", "CIC FGSM", "CIC PGD", "TON clean", "TON FGSM", "TON PGD"], [
  ["Federated FP32 (attack source)", "94.3%", "68.6%", "55.6%", "99.1%", "23.0%", "23.0%"],
  ["Centralized FP32 (transfer)", "95.1%", "30.8%", "29.0%", "99.2%", "23.0%", "23.0%"],
  ["INT8 PTQ only", "93.9%", "70.4%", "60.4%", "97.5%", "23.0%", "23.0%"],
  ["Prune 50% → client FT → PTQ", "94.8%", "32.5%", "23.9%", "86.1%", "23.0%", "23.0%"],
  ["Prune 50% → client FT → QAT FT (base recipe)", "94.2%", "44.3%", "45.6%", "98.1%", "23.0%", "23.0%"],
  ["Prune 50% → pooled FT → QAT FT (upper bound)", "96.3%", "44.7%", "46.2%", "98.7%", "56.2%", "55.6%"],
], [3200, 950, 950, 950, 950, 950, 950]));
children.push(caption("Table 5. Accuracy under FGSM and PGD (ε = 0.1, standardized features), perturbations computed on the federated FP32 model and transferred to the other models. On TON_IoT, 23.0% equals the benign share of the test set: those models label every perturbed record benign."));
children.push(P(
  "On CIC-IDS2017, INT8 PTQ preserves the federated model's robustness (PGD 55.6% → 60.4%). Pruning followed by fine-tuning reduces it. Of the pruned models, the QAT-fine-tuned deployment model (PGD 45.6%) is clearly more robust than the PTQ one (23.9%), which echoes the quantization-robustness interaction reported in [21–23]. The centralized model is markedly more fragile (PGD 29.0%) even though it is attacked only by transfer. On TON_IoT, ε = 0.1 is strong enough that every model except the pooled-fine-tuned one predicts all perturbed records as benign. Two caveats apply. First, these are unconstrained L∞ perturbations in standardized feature space that ignore feature semantics (integer counts, encoded categories), so they overstate what a network attacker can realize. Second, the per-model differences come from a single seed, and the PTQ rows from a single calibration draw (Section 5.9). We report them as relative robustness under a common perturbation, not as absolute security guarantees."
));

children.push(H2("5.8 Practical Significance"));
children.push(P(
  "We deploy one model per dataset, chosen in Section 5.13 for recall priority: for CIC-IDS2017 the 112 KB INT8 model (30% pruning, distillation and QAT fine-tuning on clipped inputs, threshold 0.191 chosen on client data), and for TON_IoT the 55 KB INT8 model of the same recipe with federated fine-tuning, built from the improved federated model of Section 5.14 (50% pruning, threshold 0.207). On the test split the CIC-IDS2017 model misses 27 of 85,173 attacks (0.032%) and flags 10.0% of benign flows, about one in ten; the TON_IoT model misses 12 of 16,215 attacks (0.074%) and flags 6.5% of benign records, about one in fifteen. For comparison, the base recipe of Table 3 at the fixed 0.3 threshold misses 291 of 85,173 CIC-IDS2017 attacks (FAR 6.82%) and 268 of 16,215 TON_IoT attacks (FAR 2.88%). The trade-off is therefore explicit: roughly ten to twenty times fewer missed attacks for 1.5–2.3 times as many false alarms. A deployment with a fixed alert budget would choose the threshold on client data for that budget instead; Section 5.4 shows that the fine-tuning client's class mix shifts the trade-off as well."
));

children.push(H2("5.9 Quantization Methods"));
children.push(P(
  "INT8 is not the only post-training option. Table 6 applies five TensorFlow Lite conversions to the federated model and to the pruned (50%), client-fine-tuned model, with every calibration set drawn from client 0's local data so that the comparison stays FL-faithful. Dynamic-range quantization stores INT8 weights but computes in float; float16 halves the weights; full-integer INT8 quantizes weights and activations to 8 bits; int16x8 keeps INT8 weights and uses 16-bit activations."
));
children.push(makeTable(["Method", "Runs on our TFLM build", "CIC federated", "CIC pruned + client FT", "TON federated", "TON pruned + client FT"], [
  ["FP32", "yes", "802 KB · 85.70", "243 KB · 89.39", "721 KB · 99.40", "202 KB · 99.35"],
  ["Dynamic range (INT8 weights, float compute)", "no", "216 KB · 85.58", "69 KB · 90.13", "196 KB · 99.40", "59 KB · 99.35"],
  ["Float16 weights", "no", "404 KB · 85.70", "124 KB · 89.35", "363 KB · 99.40", "103 KB · 99.35"],
  ["Full-integer INT8 PTQ", "yes", "227 KB · 85.02", "75 KB · 86.90", "206 KB · 91.24", "65 KB · 90.50"],
  ["int16x8 (INT8 weights, 16-bit activations)", "untested", "238 KB · 85.48", "81 KB · 89.61", "217 KB · 99.40", "70 KB · 99.35"],
  ["INT8 via QAT fine-tuning (base recipe, Table 3)", "yes", "—", "65 KB · 85.59", "—", "55 KB · 98.74"],
], [2700, 1100, 1450, 1450, 1450, 1450]));
children.push(caption("Table 6. Size and F1 for post-training quantization methods (test split, threshold 0.3). Calibration for INT8 and int16x8 uses one draw of 500 samples of client 0's data (Table 7 shows the spread over draws). \"Pruned + client FT\" is prune 50% → 3 epochs of fine-tuning on 10,000 client-0 samples. FAR for the full-integer INT8 rows on TON_IoT is 14.3% and 14.0%; for all float-activation and int16x8 rows it is within 0.5 points of FP32."));
children.push(P(
  "Every method that keeps activations in float or 16 bits loses at most 0.25 F1 on both datasets (dynamic range even gains 0.7 on the pruned CIC-IDS2017 model), so weight precision is not the bottleneck. Full-integer INT8 is the only post-training method that loses accuracy: 0.7 F1 on the CIC-IDS2017 federated model, 2.5 F1 on the pruned CIC-IDS2017 model, and 8.2–8.9 F1 on TON_IoT, where FAR rises from 2.8% to 14%. Because int16x8 uses the same INT8 weights and recovers fully, the loss comes from 8-bit activations."
));
children.push(P(
  "Full-integer PTQ is also unstable. Table 3 (500 pooled samples) and Table 6 (500 client-0 samples) each rest on a single calibration draw and disagree on TON_IoT (F1 98.38 vs. 91.24). We therefore converted each federated model 45 more times, varying the calibration source (pooled data or any one of the four clients), the number of samples (100, 500, 2,000), and the random draw (three per setting). The source does not matter: for every source the median F1 is 98.97–99.30 on TON_IoT and 83.1–83.8 on CIC-IDS2017. The draw does (Table 7). On TON_IoT the same model ranges from F1 12.28 to 99.38, and on CIC-IDS2017 from 29.16 to 85.02. More calibration data makes it worse, not better: with 100 samples, 12 of 15 TON_IoT conversions are within one F1 point of FP32, with 2,000 samples only 3 of 15, and 10 of those 15 fall below F1 95. Allowing float fallback in the converter changes nothing."
));
children.push(makeTable(["Calibration samples", "TON_IoT median F1", "TON_IoT min F1", "TON_IoT within 1 F1 of FP32", "CIC median F1", "CIC min F1", "CIC within 1 F1 of FP32"], [
  ["100", "99.31", "90.99", "12 / 15", "84.60", "81.47", "5 / 15"],
  ["500", "99.30", "26.19", "11 / 15", "83.11", "80.91", "2 / 15"],
  ["2,000", "91.23", "12.28", "3 / 15", "82.60", "29.16", "1 / 15"],
  ["2,000, inputs clipped to ±5", "99.34", "99.32", "15 / 15", "84.61", "83.02", "4 / 15"],
  ["FP32 reference", "99.40", "", "", "85.70", "", ""],
], [1700, 1300, 1200, 1500, 1200, 1100, 1500]));
children.push(caption("Table 7. Full-integer INT8 PTQ of the near-IID federated models over 15 calibration sets per size (5 sources: pooled or one of four clients, × 3 random draws). The clipped row uses the same 15 sets of 2,000 samples with every calibration input clipped to ±5 standard deviations."));
children.push(P(
  "The activation ranges point to the heavy-tailed inputs of Section 5.6. The TFLite converter sets each activation range from the minimum and maximum it sees during calibration, and a larger calibration set is more likely to contain a rare extreme record. The worst conversions coincide with the widest ranges: the three TON_IoT conversions with F1 12–22 all have a hidden-layer maximum near 500, and the worst CIC-IDS2017 conversion (F1 29.16) one of 9,609, against 52–230 for most of the others. The relation is not strictly monotonic (two TON_IoT conversions with small ranges land at F1 91.2), so outliers are the main but not the only cause. The difference between Tables 3 and 6 is therefore a draw effect, not an effect of federated data, and any single full-integer PTQ number, including Table 3's, is one sample from a wide distribution."
));
children.push(P(
  "A one-line change removes the instability: clip the calibration inputs to ±5 standard deviations before conversion (inputs beyond the calibrated range then saturate at inference). Over the same 15 calibration sets of 2,000 samples, clipped calibration gives F1 99.32–99.35 on TON_IoT (FP32 99.40) and 83.02–84.85 on CIC-IDS2017 (FP32 85.70), against 12.28–99.35 and 29.16–84.99 without clipping (Table 7). Clipping at ±10 works equally well, while ±3 is too tight for TON_IoT (F1 96.7–96.9). The remaining loss of about one F1 point on CIC-IDS2017 is the cost of 8-bit activations seen in Table 6. This is the same remedy that Section 5.6 suggests for training-time QAT, applied only at calibration time."
));
children.push(P(
  "QAT fine-tuning learns its activation ranges as moving averages during fine-tuning. It is more stable than PTQ but not independent of the fine-tuning draw. Repeating the base recipe with five different 10,000-sample fine-tuning draws from client 0 gives F1 98.74–99.38 on TON_IoT (PTQ of the same pruned models: 97.44–99.21) and 86.77–89.28 on CIC-IDS2017 (PTQ: 79.37–89.05). The Table 3 deployment (F1 85.59) lies below all five repeated draws, so it is a conservative rather than a selected result. A spread of about 2.5 F1 from the fine-tuning draw alone should be kept in mind when reading Tables 3–4 and Figure 1."
));
children.push(P(
  "This determines the deployment recipe. The ESP32 build of TensorFlow Lite Micro that we use runs FP32 and full-integer INT8 kernels; dynamic-range (hybrid) and float16 models need kernels it does not provide, and int16x8 kernels exist only for some operators in newer TFLM releases, which we have not tested on the device. Dynamic-range quantization would otherwise be the best size–accuracy point in Table 6 (69 KB, F1 90.13 on CIC-IDS2017), so it is a reasonable choice for Linux-class gateways but not for the microcontroller. On the microcontroller, full-integer INT8 is required. QAT fine-tuning and PTQ with clipped calibration both reach it reliably; plain PTQ does not. A side observation: fine-tuning the pruned model on client 0 raises float F1 on CIC-IDS2017 from 85.70 to 89.39, by shifting the operating point toward precision (FAR 4.78%, Attack Recall 99.62%). INT8 quantization then gives part of that back."
));

children.push(H2("5.10 Client-Local Knowledge Distillation"));
children.push(P(
  "Knowledge distillation is a second way to obtain a small model without pooled data: a client trains a narrow student on its own data, using the federated model's predictions as soft targets. We train students with the same three-layer MLP at 1/2, 1/4, and 1/8 of the width of 512-256-128 (no BatchNorm or Dropout) on client 0's data for 10 epochs, either distilled from the federated teacher (temperature 2, targets = 0.5·label + 0.5·soft teacher output) or from scratch on hard labels. On CIC-IDS2017 we use 50,000 of client 0's 681,380 samples; on TON_IoT all 32,428. Each student is exported as FP32, INT8 PTQ (client-calibrated), and INT8 after two epochs of QAT fine-tuning."
));
children.push(makeTable(["Student width", "INT8 size (CIC / TON)", "CIC KD", "CIC scratch", "TON KD", "TON scratch"], [
  ["1/2 (256-128-64)", "75.1 / 64.9 KB", "87.90 / 87.64 / 87.36", "89.75 / 89.36 / 87.21", "99.35 / 99.31 / 98.90", "99.35 / 97.06 / 98.93"],
  ["1/4 (128-64-32)", "29.2 / 24.1 KB", "87.31 / 87.37 / 86.42", "88.64 / 88.74 / 86.46", "99.35 / 99.24 / 99.01", "99.36 / 98.75 / 98.98"],
  ["1/8 (64-32-16)", "13.7 / 11.2 KB", "86.30 / 86.07 / 86.55", "88.29 / 88.51 / 91.10", "99.16 / 99.16 / 98.82", "99.14 / 98.70 / 98.81"],
  ["Reference: federated teacher (FP32)", "802.5 / 720.6 KB", "85.70", "—", "99.40", "—"],
], [2200, 1600, 1600, 1600, 1600, 1600]));
children.push(caption("Table 8. F1 of client-local students, given as FP32 / INT8 PTQ / INT8 QAT fine-tuned. KD: distilled from the federated model; scratch: trained on client 0's hard labels only. QAT-fine-tuned students are about 15% smaller than the PTQ sizes shown (e.g. 11.9 / 9.4 KB at 1/8 width)."));
children.push(P(
  "On TON_IoT, distillation gives the most compact models in this paper: the 1/8-width distilled student keeps F1 99.16 at 11.2 KB with plain INT8 PTQ (64× smaller than the FP32 federated model), and distillation makes the student robust to INT8 PTQ (at most 0.11 F1 lost, against up to 2.3 F1 for scratch students). On CIC-IDS2017, distillation does not help. The distilled students inherit the teacher's recall-heavy operating point (Attack Recall 99.5–99.7%, FAR 5.6–6.5%), while scratch students in FP32 and PTQ form trade some recall for precision (Attack Recall 98.9–99.5%, FAR 4.5–5.3%), which raises their F1 at the fixed 0.3 threshold. The highest F1 in Table 8, 91.10 for the 1/8-width scratch student after QAT fine-tuning, is an outlier against the same student's FP32 and PTQ results (88.29, 88.51) and reaches its F1 by missing 3.68% of attacks (3,137 of 85,173, against 291 for the base-recipe model of Table 3). We therefore do not select it, and differences of one or two F1 points in this table should be read as operating-point shifts under a single seed rather than as better detectors."
));
children.push(P(
  "The comparison also raises a question the reviewers did not ask but that we consider essential: a scratch student trained on one client's data alone already matches the federated model on near-IID CIC-IDS2017 (F1 89.75 vs. 85.70). Section 5.11 asks when federation helps at all."
));

children.push(H2("5.11 Local-Only Training: When Does Federation Help?"));
children.push(P(
  "Each client trains the same MLP with the same loss alone on its own partition (10 epochs, at most 200,000 of its samples) and is evaluated on the shared test split. Table 9 compares this with the federated model trained on the same partition."
));
children.push(makeTable(["Partition / client", "Local data (attack share)", "Local-only F1 / Recall / FAR", "Missed attacks / all", "Federated F1 / Recall / FAR", "Missed attacks / all"], [
  ["CIC near-IID, clients 0–3", "681k each (50%)", "88.3–90.0 / 98.85–99.78% / 4.5–5.2%", "185–982 / 85,173 (0.22–1.15%)", "85.70 / 99.94% / 6.84%", "52 / 85,173 (0.061%)"],
  ["CIC Dirichlet, client 0", "9.4k (0.1%)", "62.04 / 44.97% / 0.00%", "46,872 / 85,173 (55.03%)", "85.59 / 99.83% / 6.87%", "144 / 85,173 (0.17%)"],
  ["CIC Dirichlet, client 1", "1.98M (59.5%)", "90.70 / 99.75% / 4.15%", "214 / 85,173 (0.25%)", "", ""],
  ["CIC Dirichlet, client 2", "6.7k (2.6%)", "88.74 / 91.11% / 2.92%", "7,572 / 85,173 (8.89%)", "", ""],
  ["CIC Dirichlet, client 3", "728k (25.1%)", "89.80 / 99.60% / 4.56%", "339 / 85,173 (0.40%)", "", ""],
  ["TON near-IID, clients 0–3", "32k each (50%)", "99.16–99.31 / 99.65–99.79% / 3.5–4.8%", "34–57 / 16,215 (0.21–0.35%)", "99.40 / 99.62% / 2.76%", "61 / 16,215 (0.38%)"],
  ["TON Dirichlet, client 0", "49.9k (0.4%)", "90.16 / 82.13% / 0.23%", "2,897 / 16,215 (17.87%)", "98.45 / 97.99% / 3.59%", "326 / 16,215 (2.01%)"],
  ["TON Dirichlet, client 1", "62.9k (92.8%)", "96.90 / 99.92% / 21.20%", "13 / 16,215 (0.080%)", "", ""],
  ["TON Dirichlet, client 2", "8.6k (65.5%)", "99.02 / 99.80% / 5.97%", "33 / 16,215 (0.20%)", "", ""],
  ["TON Dirichlet, client 3", "8.4k (8.5%)", "96.34 / 93.22% / 1.00%", "1,099 / 16,215 (6.78%)", "", ""],
], [1900, 1400, 2200, 1500, 1700, 1500]));
children.push(caption("Table 9. Each client trained alone vs. the federated model of the same partition (float models, test split, threshold 0.3). Test attacks: 85,173 (CIC-IDS2017), 16,215 (TON_IoT). One federated model serves all clients of a partition."));
children.push(P(
  "Federation pays off where the reviewers' non-IID concern points: clients whose local data contains few attacks. Alone, CIC-IDS2017 Dirichlet client 0 (0.1% attacks) detects 44.97% of attacks and client 2 (2.6%) 91.11%; TON_IoT client 0 (0.4%) detects 82.13% and client 3 (8.5%) 93.22%. On the same partitions the federated model detects 99.83% (CIC-IDS2017) and 97.99% (TON_IoT). The attack-heavy TON_IoT client 1 (92.8% attacks) has the opposite problem: alone, it flags 21.2% of benign records, against 3.59% for the federated model. Clients that already hold large, representative data gain little or nothing. On near-IID CIC-IDS2017 each client alone reaches F1 88.3–90.0 against 85.70 for the federated model, at a precision-heavier operating point that misses 3.6–19 times as many attacks (185–982 vs. 52). The near-IID partition, which gives every client several hundred thousand representative flows, is therefore the setting in which federation is least needed, and the Dirichlet partition is the one in which it is needed. Three caveats apply: local and federated training budgets are not matched (10 local epochs vs. 60 rounds × 3 epochs), each configuration ran once, and at the fixed 0.3 threshold the F1 differences among data-rich clients mainly reflect operating points."
));

children.push(H2("5.12 Combining Pruning, PTQ, QAT, and Distillation"));
children.push(P(
  "Sections 5.3–5.10 vary one step of the compression pipeline at a time. Table 10 compares 16 complete pipelines that combine quantization (PTQ, PTQ with clipped calibration, QAT fine-tuning, QAT fine-tuning on clipped inputs), pruning (structured 50%, unstructured magnitude pruning at 50% and 80%, and both together), distillation fine-tuning, and the order of pruning and quantization. Every pipeline starts from the near-IID federated model, uses only client 0's data, and is repeated over three random 10,000-sample fine-tuning draws. We report the mean and the range over draws. Magnitude pruning ramps sparsity up during three fine-tuning epochs; sparsity-preserving QAT (PQAT) is tfmot's prune-preserving quantization scheme."
));
children.push(makeTable(["Pipeline", "Flash KB (TON / CIC)", "gzip KB (TON / CIC)", "TON F1 mean (range)", "CIC F1 mean (range)", "CIC missed of 85,173"], [
  ["FP32 federated model (reference)", "720 / 803", "672 / 755", "99.40", "85.70", "52 (0.061%)"],
  ["INT8 PTQ", "206 / 227", "156 / 150", "98.87 (98.37–99.26)", "82.36 (81.04–83.21)", "163 (0.19%)"],
  ["INT8 PTQ, clipped calibration", "206 / 227", "156 / 150", "99.34 (99.34–99.34)", "83.97 (83.27–84.35)", "57 (0.067%)"],
  ["QAT fine-tuning", "186 / 207", "125 / 99", "99.10 (98.98–99.33)", "90.51 (89.27–91.99)", "580 (0.68%)"],
  ["Structured 50% → FT → PTQ", "65 / 75", "47 / 47", "98.50 (97.44–99.21)", "84.39 (79.37–87.27)", "500 (0.59%)"],
  ["Structured 50% → FT → PTQ, clipped calibration", "65 / 75", "47 / 47", "99.37 (99.34–99.39)", "87.67 (86.86–88.09)", "830 (0.97%)"],
  ["Structured 50% → FT → QAT (base recipe)", "55 / 65", "39 / 34", "99.08 (98.88–99.38)", "88.37 (87.64–89.28)", "498 (0.58%)"],
  ["Structured 50% → FT → QAT on clipped inputs", "55 / 65", "39 / 35", "99.40 (99.34–99.44)", "90.36 (88.54–92.44)", "1,163 (1.37%)"],
  ["Structured 50% → KD fine-tuning → QAT", "55 / 65", "39 / 34", "99.13 (98.97–99.43)", "86.04 (85.61–86.56)", "274 (0.32%)"],
  ["Structured 50% → KD fine-tuning → PTQ, clipped", "65 / 75", "47 / 47", "99.37 (99.36–99.38)", "84.85 (84.23–85.16)", "237 (0.28%)"],
  ["QAT first → structured 50% → FT → QAT", "55 / 65", "39 / 34", "99.12 (98.98–99.36)", "88.64 (86.65–90.58)", "687 (0.81%)"],
  ["Magnitude 50% → PTQ, clipped", "206 / 227", "124 / 128", "99.40 (99.39–99.41)", "89.72 (89.48–89.95)", "491 (0.58%)"],
  ["Magnitude 50% → PQAT", "186 / 207", "100 / 90", "99.12 (99.00–99.35)", "90.70 (89.95–91.71)", "487 (0.57%)"],
  ["Magnitude 80% → PTQ, clipped", "206 / 227", "71 / 76", "99.34 (99.31–99.37)", "87.98 (87.57–88.18)", "923 (1.08%)"],
  ["Magnitude 80% → PQAT", "186 / 207", "56 / 57", "98.85 (98.51–99.36)", "89.77 (88.99–91.10)", "609 (0.72%)"],
  ["Magnitude 80% → standard QAT", "186 / 207", "76 / 62", "99.07 (98.89–99.33)", "89.77 (89.31–90.52)", "571 (0.67%)"],
  ["Structured 50% → FT → magnitude 50% → PQAT", "55 / 65", "31 / 31", "98.95 (98.70–99.37)", "88.66 (88.05–89.86)", "699 (0.82%)"],
], [3300, 1250, 1250, 1500, 1500, 1000]));
children.push(caption("Table 10. Complete compression pipelines from the near-IID federated model, using only client 0's data; mean and range over three fine-tuning draws (test split, threshold 0.3). Flash is the TFLite file that TensorFlow Lite Micro stores; gzip shows what a compressed weight format or an over-the-air update would transfer. Clipping is at ±5 standard deviations. CIC-IDS2017 missed attacks are means over draws (85,173 test attacks). Table 3's single draw of the base recipe (F1 85.59) lies below this table's range for the same pipeline."));
children.push(P(
  "Only structured pruning shrinks what the microcontroller stores. TensorFlow Lite Micro keeps weights dense, so magnitude pruning leaves the flash size at that of the unpruned INT8 model (186–227 KB) even at 80% sparsity; it only reduces the gzip size (to 56–76 KB at 80%), which matters for transferring model updates but not for flash. Adding magnitude pruning on top of structured pruning likewise leaves flash at 55–65 KB and reduces gzip to 31 KB. PQAT keeps the target sparsity (50% and 80%; slightly more on CIC-IDS2017, where INT8 rounding adds zeros), whereas standard QAT after 80% magnitude pruning lets part of the pruned weights grow back (to 62% zeros on TON_IoT and 75% on CIC-IDS2017, where INT8 rounding alone already zeroes 33–40% of the QAT-fine-tuned weights)."
));
children.push(P(
  "Clipping the inputs at ±5 standard deviations helps every pipeline on TON_IoT. PTQ becomes stable (structured + PTQ: range 97.44–99.21 → 99.34–99.39), and QAT fine-tuning on clipped inputs matches the FP32 model at 55 KB (F1 99.40, range 99.34–99.44, 73 missed attacks against 158 for the base recipe). On CIC-IDS2017, clipped calibration also removes the PTQ collapses (structured + PTQ: 79.37–87.27 → 86.86–88.09) and, for the unpruned model, lowers both FAR and missed attacks. After pruning and fine-tuning, however, clipping raises F1 by moving the operating point toward precision: QAT on clipped inputs reaches the highest structured-pruning F1 (90.36) but misses 1,163 attacks on average (one draw 2,747), against 498 for the base recipe. Distillation fine-tuning goes the other way: it lowers CIC-IDS2017 F1 to 86.04 but misses the fewest attacks of any 65 KB pipeline (274), because the student inherits the teacher's recall-heavy operating point (Section 5.10). Quantizing before pruning changes nothing beyond the draw-to-draw spread (99.12 vs. 99.08 and 88.64 vs. 88.37)."
));
children.push(P(
  "Three recommendations follow. First, always clip calibration inputs for PTQ; it costs nothing and removes the failures of Section 5.9. Second, for a TensorFlow Lite Micro target, use structured pruning; unstructured sparsity helps only if the runtime or the update channel compresses weights. Third, on CIC-IDS2017 no pipeline dominates: at 65 KB the choice between QAT on clipped inputs (higher F1, more missed attacks), the base recipe, and distillation fine-tuning (fewest missed attacks) is a choice of operating point, which should be made with a validation-set threshold rather than by picking a pipeline (Section 5.13). On TON_IoT, structured pruning followed by QAT fine-tuning on clipped inputs is the best 55 KB model we found and loses nothing relative to the federated FP32 model."
));

children.push(H2("5.13 Recall-Priority Deployment"));
children.push(P(
  "For intrusion detection a missed attack usually costs more than a false alarm. The FP32 federated model misses 52 of 85,173 CIC-IDS2017 test attacks at the 0.3 threshold, whereas the 65 KB base-recipe model misses 463 on average over three fine-tuning draws (Table 10 setting). We therefore ask how close the 65 KB model can get to the FP32 model's missed-attack count without pooled data. Two levers are combined: fine-tuning targets that preserve the teacher's recall (distillation from the federated model with α = 0.5 or α = 0, or a 4× loss weight on attacks), and a decision threshold chosen on client 0's own held-out data (20,000 samples disjoint from every fine-tuning draw) as the largest threshold that reaches a target Attack Recall there. The test split is never used for selection."
));
children.push(makeTable(["Model (CIC-IDS2017)", "Threshold selection", "Threshold", "Missed attacks / all (mean)", "Missed share", "Worst of 3 draws", "FAR"], [
  ["FP32 federated, 803 KB", "fixed", "0.30", "52 / 85,173", "0.061%", "", "6.84%"],
  ["FP32 federated, 803 KB", "client val., recall ≥ 99.99%", "0.269", "37 / 85,173", "0.043%", "", "7.56%"],
  ["65 KB, base recipe", "fixed", "0.30", "463 / 85,173", "0.54%", "483", "5.97%"],
  ["65 KB, base recipe", "client val., recall ≥ 99.95%", "0.005", "44 / 85,173", "0.052%", "58", "23.39%"],
  ["65 KB, attack weight 4×", "client val., recall ≥ 99.95%", "0.027", "63 / 85,173", "0.074%", "81", "20.31%"],
  ["65 KB, distillation α = 0.5", "client val., recall ≥ 99.95%", "0.164", "37 / 85,173", "0.044%", "52", "19.92%"],
  ["65 KB, distillation + clipped inputs", "fixed", "0.30", "231 / 85,173", "0.27%", "292", "6.04%"],
  ["65 KB, distillation + clipped inputs", "client val., recall ≥ 99.9%", "0.213", "77 / 85,173", "0.090%", "133", "9.54%"],
  ["65 KB, distillation + clipped inputs", "client val., recall ≥ 99.95%", "0.198", "44 / 85,173", "0.052%", "55", "10.43%"],
  ["65 KB, distillation + clipped inputs", "client val., recall ≥ 99.99%", "0.171", "24 / 85,173", "0.028%", "25", "14.33%"],
  ["65 KB, same + longer fine-tuning (6 + 4 ep)", "client val., recall ≥ 99.99%", "0.184", "31 / 85,173", "0.036%", "45", "10.94%"],
  ["65 KB, same + federated fine-tuning (4 clients)", "client val., recall ≥ 99.95%", "0.204", "51 / 85,173", "0.060%", "79", "9.31%"],
  ["65 KB, same + federated fine-tuning (4 clients)", "client val., recall ≥ 99.99%", "0.180", "27 / 85,173", "0.032%", "33", "11.28%"],
  ["112 KB (prune 30%), distillation + clipped inputs", "client val., recall ≥ 99.95%", "0.221", "54 / 85,173", "0.063%", "73", "7.63%"],
  ["112 KB (prune 30%), distillation + clipped inputs", "client val., recall ≥ 99.99%", "0.189", "23 / 85,173", "0.027%", "27", "9.65%"],
  ["207 KB (no pruning), distillation + clipped inputs", "client val., recall ≥ 99.95%", "0.229", "51 / 85,173", "0.060%", "52", "6.71%"],
  ["207 KB (no pruning), distillation + clipped inputs", "client val., recall ≥ 99.99%", "0.182", "23 / 85,173", "0.027%", "28", "10.39%"],
], [2700, 1900, 800, 1500, 900, 900, 800]));
children.push(caption("Table 11. Recall-priority deployment on CIC-IDS2017 (85,173 test attacks). All models: structured pruning (50% unless stated) → fine-tuning → QAT fine-tuning → INT8, on client 0's data unless \"federated fine-tuning\" (FedAvg over all four clients' local data after every epoch); means over three fine-tuning draws. \"Client val.\": threshold chosen on 20,000 held-out samples of client 0 to reach the stated recall there."));
children.push(P(
  "Lowering the threshold of the base-recipe model alone does reach the FP32 model's missed-attack count (44), but only at a 23.4% FAR. The hard-label QAT model pushes most attack and benign scores into the lowest output levels, and the selected thresholds (0.004–0.012) are one to three steps of the INT8 output's 1/256 resolution, so the threshold cannot be tuned finely. Distillation keeps the student's scores in the middle of the range, where the INT8 output has resolution (selected thresholds 0.15–0.26), and clipping the inputs during fine-tuning (Section 5.12) separates the classes further. Together they let the 65 KB model miss 44 attacks, fewer than the FP32 model at 0.3, at a 10.4% FAR, or 24 attacks at a 14.3% FAR. A 4× attack weight, pure teacher targets (α = 0; 66 missed at 18.5% FAR), and five times more fine-tuning data (50,000 samples; 40 missed at 17.9% FAR) do not do better. Measured at equal recall, compression still costs FAR: on the test sweep the FP32 model misses 22 attacks at an 11.0% FAR, against 24 at 14.3% for the best 65 KB model."
));
children.push(P(
  "The client's own held-out data is a reliable guide. Its FAR predicts the test FAR to within 0.35 points in every draw, and the 99.9% and 99.95% recall targets are met on the test split on average (99.91% and 99.95%). The 99.99% target is not (99.97%): with about 10,000 attacks in the client validation set it allows a single miss, and the client data are SMOTE-augmented, so very high recall targets need a larger validation set."
));
children.push(P(
  "Three further levers improve the trade-off (lower rows of Table 11). Federated fine-tuning, in which all four clients fine-tune on their own data and the server averages the weights after every fine-tuning and QAT epoch, gives the best 65 KB operating points: 27 missed attacks at an 11.3% FAR, against 24 at 14.3% when client 0 fine-tunes alone, or 51 at 9.3%. It also removes the dependence on a single fine-tuning client (Section 5.4) without moving any data. Longer fine-tuning helps to a similar degree (31 missed at 10.9%), whereas a higher distillation temperature (T = 4) is worse (39 missed at 13.0%). Pruning less trades flash for false alarms: at 112 KB (30% pruning) the model misses 23 attacks at a 9.7% FAR, and at 207 KB (no pruning) 51 at 6.7%. The larger models thus reach the FP32 model's missed-attack counts at a lower FAR than the FP32 model itself (52 missed at 6.84%, 22 at 11.0% on the test sweep), because distillation on clipped inputs moves the whole recall–FAR curve and not only the operating point."
));
children.push(P(
  "The remaining misses show where the ceiling is (Table 12). At the 99.99% target the best models miss 22–31 of 85,173 attacks. The FP32 federated model itself misses 7 of the 11 Infiltration test flows and 1 of the 4 SQL-injection flows at every threshold for which we recorded attack types, and 11–32 of 368 Bot flows depending on the threshold; Infiltration has only 25 training flows. The compressed models share these misses, and compression adds one of its own: the 65 KB models miss on average 3.3 of the 5 Heartbleed test flows, which the FP32 model detects, whereas the 112 KB model detects all five. Heartbleed has only 6 training flows, and halving the hidden layers evidently removes the capacity that represented them. All other attack types lose at most a few flows in tens of thousands. Engelen et al. [24] also report labelling errors in CIC-IDS2017, including attack-labelled flows without malicious content, which we did not check for these flows. Fewer misses would therefore require a better federated model (for example, class-aware objectives for rare attack types or corrected labels), not a better compression pipeline."
));
children.push(makeTable(["Attack type (CIC-IDS2017)", "Test flows", "FP32, threshold 0.3", "FP32, client val. ≥ 99.99%", "65 KB, federated FT, ≥ 99.99%", "112 KB, ≥ 99.99%"], [
  ["Infiltration", "11", "7 (64%)", "7 (64%)", "8.3 (76%)", "6.3 (58%)"],
  ["Heartbleed", "5", "0", "0", "3.3 (67%)", "0"],
  ["Web Attack – SQL injection", "4", "1 (25%)", "1 (25%)", "0", "0.3 (8%)"],
  ["Bot", "368", "20 (5.4%)", "11 (3.0%)", "6.0 (1.6%)", "6.7 (1.8%)"],
  ["Web Attack – brute force", "301", "3 (1.0%)", "2 (0.7%)", "1.3 (0.4%)", "1.0 (0.3%)"],
  ["DoS slowloris", "1,063", "2", "2", "2.0", "1.7"],
  ["DoS Slowhttptest", "1,071", "2", "0", "1.0", "1.0"],
  ["FTP-Patator", "1,236", "3", "3", "0", "1.3"],
  ["SSH-Patator", "650", "1", "0", "0", "0"],
  ["PortScan", "18,282", "4", "3", "3.7", "3.3"],
  ["DDoS", "25,569", "9", "8", "1.3", "1.7"],
  ["All attacks", "85,173", "52 (0.061%)", "37 (0.043%)", "27 (0.032%)", "23 (0.027%)"],
], [2600, 1000, 1400, 1600, 1700, 1300]));
children.push(caption("Table 12. Missed test flows per attack type (means over three fine-tuning draws for the compressed models; share of that type's test flows in parentheses where it exceeds 0.1%). DoS Hulk (34,487 test flows), DoS GoldenEye (2,006) and Web Attack – XSS (120) are not missed by any of these four models."));
children.push(P(
  "The same recipe also works on TON_IoT (Table 13), where the role of distillation is even clearer. The FP32 federated model and the base-recipe hard-label 55 KB model are so confident that a few client-validation attacks score near zero; reaching a 99.9% recall target on client data then pushes the threshold to almost zero, and FAR to 36–97%. Clipped inputs alone help up to a 99.9% target (21 missed at 7.0%) but collapse to a zero threshold at 99.99%. Distilled students keep their scores in the middle of the range, and their thresholds (0.16–0.22) carry over to the test split: the 55 KB distilled model with clipped inputs misses 19 of 16,215 attacks at a 5.9% FAR, the same trade-off as the FP32 model at threshold 0.2 on the test sweep (21 missed at 5.85%), and 7–10 attacks at an 11–13% FAR. Federated fine-tuning and lighter pruning bring no further gain on TON_IoT, where the 55 KB model already matches the FP32 curve. Here the selected thresholds slightly undershoot the recall targets on the test split (99.88% for a 99.9% target, 99.94–99.96% for 99.99%), because client 0's 8,000-sample validation set holds only about 4,000 attacks."
));
children.push(makeTable(["Model (TON_IoT)", "Threshold selection", "Threshold", "Missed attacks / all (mean)", "Missed share", "Worst of 3 draws", "FAR"], [
  ["FP32 federated, 720 KB", "fixed", "0.30", "61 / 16,215", "0.38%", "", "2.76%"],
  ["FP32 federated, 720 KB", "client val., recall ≥ 99.9%", "0.021", "2 / 16,215", "0.012%", "", "96.76%"],
  ["55 KB, base recipe", "fixed", "0.30", "182 / 16,215", "1.12%", "233", "3.01%"],
  ["55 KB, base recipe", "client val., recall ≥ 99.9%", "0.040", "16 / 16,215", "0.099%", "29", "36.44%"],
  ["55 KB, QAT on clipped inputs", "fixed", "0.30", "80 / 16,215", "0.49%", "89", "2.33%"],
  ["55 KB, QAT on clipped inputs", "client val., recall ≥ 99.9%", "0.038", "21 / 16,215", "0.13%", "25", "7.02%"],
  ["55 KB, distillation + clipped inputs", "client val., recall ≥ 99.9%", "0.220", "19 / 16,215", "0.12%", "20", "5.94%"],
  ["55 KB, distillation + clipped inputs", "client val., recall ≥ 99.99%", "0.171", "10 / 16,215", "0.062%", "10", "11.17%"],
  ["55 KB, same + federated fine-tuning (4 clients)", "client val., recall ≥ 99.99%", "0.159", "7 / 16,215", "0.045%", "10", "12.60%"],
  ["98 KB (prune 30%), distillation + clipped inputs", "client val., recall ≥ 99.9%", "0.210", "21 / 16,215", "0.13%", "25", "5.79%"],
], [2700, 1900, 800, 1500, 900, 900, 800]));
children.push(caption("Table 13. Recall-priority deployment on TON_IoT (16,215 test attacks), same procedure as Table 11; the client validation set is 8,000 held-out samples of client 0."));
children.push(P(
  "For a recall-priority deployment we therefore recommend one recipe for both datasets: distillation from the federated model, QAT fine-tuning on clipped inputs, and a threshold chosen on client-held data for the required recall. Federated fine-tuning across clients helps on CIC-IDS2017; on TON_IoT it is neutral for the original federated model and helps for the improved one (Section 5.14). On a 65 KB budget for CIC-IDS2017 the price is about one benign flow in nine raising an alert; where 112 KB of flash is available, 30% pruning lowers that to about one in ten at the same recall and keeps the rare Heartbleed attacks. On TON_IoT the 55 KB model reaches the FP32 model's trade-off at about one false alarm in seventeen benign records; Section 5.14 improves the TON_IoT federated model itself, and with it this result. The CIC-IDS2017 112 KB model (first fine-tuning draw) and the TON_IoT model of Section 5.14 are the ones we deploy (Section 5.8) and benchmark on the ESP32 (Section 6.1)."
));

children.push(H2("5.14 Improving the TON_IoT Federated Model"));
children.push(P(
  "After compression reached the FP32 model's trade-off on TON_IoT (Table 13), the remaining room lay in the federated model itself. Three properties of our TON_IoT setup were inherited from CIC-IDS2017 without being tuned: focal α = 0.35 down-weights the attack class, which on TON_IoT is the majority; SMOTE therefore synthesizes benign rather than attack records; and the DNS query and URI fields are dropped entirely. We trained eight variants of the federated model with 10% of the training split held out and selected among them only on that validation set, by the false-alarm rate at the threshold that reaches 99.9% validation recall (Table 14). The test split was not used for selection."
));
children.push(makeTable(["Variant (TON_IoT, federated, 60 rounds)", "FAR at 99.9% val. recall", "F1 at 0.3", "Missed val. attacks at 0.3 (of 6,485)", "PR-AUC"], [
  ["Original model (37 features, α = 0.35, SMOTE)", "6.84%", "99.36", "18", "0.99932"],
  ["Focal α = 0.5", "3.78%", "99.35", "12", "0.99941"],
  ["Focal α = 0.65", "7.72%", "99.06", "8", "0.99927"],
  ["No SMOTE", "4.30%", "99.28", "12", "0.99957"],
  ["Text-derived features (43 features)", "3.21%", "99.45", "6", "0.99942"],
  ["100 rounds", "5.18%", "99.36", "14", "0.99806"],
  ["Text-derived features + focal α = 0.5 (selected)", "3.01%", "99.39", "5", "0.99887"],
  ["Text-derived features + no SMOTE", "3.06%", "99.41", "3", "0.99960"],
], [3600, 1500, 1000, 1900, 1000]));
children.push(caption("Table 14. Federated-training variants for TON_IoT, evaluated on a validation split held out from the training data (8,414 records, 6,485 attacks). The selected variant was retrained on the full training split and evaluated once on the test split (Table 15)."));
children.push(P(
  "The text-derived features and the higher focal α each roughly halve the validation false-alarm rate at 99.9% recall, and together they are best (3.01% vs. 6.84%); α = 0.65 and longer training do not help. Retrained on the full training split, the selected model misses 19 instead of 61 of the 16,215 test attacks at the fixed 0.3 threshold (FAR 3.61% vs. 2.76%), and at an equal 99.9% test recall its FAR falls from 7.16% to 4.52% (ROC-AUC 0.99895 vs. 0.99860). Missed DoS, MITM, and DDoS records fall from 16, 14, and 10 to 3, 5, and 1."
));
children.push(makeTable(["Model (TON_IoT, improved federated model)", "Threshold selection", "Threshold", "Missed attacks / all (mean)", "Missed share", "Worst of 3 draws", "FAR"], [
  ["Original FP32 federated model, 720 KB (reference)", "fixed", "0.30", "61 / 16,215", "0.38%", "", "2.76%"],
  ["Improved FP32 federated model, 732 KB", "fixed", "0.30", "19 / 16,215", "0.12%", "", "3.61%"],
  ["Improved FP32 federated model, 732 KB", "client val., recall ≥ 99.95%", "0.259", "9 / 16,215", "0.056%", "", "4.71%"],
  ["55 KB, base recipe", "fixed", "0.30", "168 / 16,215", "1.04%", "274", "6.45%"],
  ["55 KB, distillation + clipped inputs", "client val., recall ≥ 99.99%", "0.165", "4 / 16,215", "0.023%", "8", "12.30%"],
  ["55 KB, same + federated fine-tuning (4 clients)", "client val., recall ≥ 99.9%", "0.244", "30 / 16,215", "0.19%", "32", "3.68%"],
  ["55 KB, same + federated fine-tuning (4 clients)", "client val., recall ≥ 99.95%", "0.215", "18 / 16,215", "0.11%", "20", "5.61%"],
  ["55 KB, same + federated fine-tuning (4 clients)", "client val., recall ≥ 99.99%", "0.203", "12 / 16,215", "0.074%", "14", "6.61%"],
  ["100 KB (prune 30%), distillation + clipped inputs", "client val., recall ≥ 99.99%", "0.186", "5 / 16,215", "0.031%", "8", "9.07%"],
  ["Original model's 55 KB deployment (Table 13)", "client val., recall ≥ 99.9%", "0.220", "19 / 16,215", "0.12%", "20", "5.94%"],
], [3000, 1900, 800, 1500, 900, 900, 800]));
children.push(caption("Table 15. Recall-priority compression of the improved TON_IoT federated model (same procedure as Tables 11 and 13; client 0's data, three fine-tuning draws)."));
children.push(P(
  "Compressing the improved model with the recipe of Section 5.13 gives better 55 KB models than before, and here federated fine-tuning makes them stable across draws. At a 99.99% client-validation target, the 55 KB federated-fine-tuned model misses 12 of 16,215 attacks (at most 14 over draws) at a 6.6% FAR (at most 6.8%), against 19 at 5.9% for the original model's 55 KB deployment; at a 99.9% target it misses 30 at a 3.7% FAR. Client-only fine-tuning misses fewer attacks on average (4) but its FAR varies between 7.2% and 17.3% across draws. XSS accounts for about half of the remaining misses. We deploy the first draw of the 55 KB federated-fine-tuned model (12 missed, FAR 6.49%, threshold 0.207). Two caveats apply. The variants of Table 14 were designed after we saw that the original TON_IoT model left room for improvement, although the choice among them used validation data only. And only this section, Section 5.8, and Section 6.1 use the improved model; the earlier TON_IoT tables use the original 37-feature model and were not re-run."
));

// =================== 6 Discussion ===================
children.push(H1("6. Discussion"));

children.push(H2("6.1 Device Heterogeneity and On-Device Measurement"));
children.push(PP([
  "IoT deployments vary in MCU architecture, memory, and clock, and interpreter latency on a workstation does not establish on-device latency. We provide ESP32 firmware that embeds both deployment models (CIC-IDS2017, 112 KB INT8; TON_IoT, 55 KB INT8) and the FP32 federated models they were compressed from (803 KB and 732 KB). It checks on-device outputs against the host TFLite interpreter on fixed inputs, including decision agreement at each model's deployment threshold, and times 100 inferences per model. A host build against the same TensorFlow Lite Micro library reproduces the interpreter outputs exactly (FP32) and within one INT8 output step (at most 0.004), with all decisions in agreement. Tensor-arena use is 2.2–4.6 KB, well within ESP32 SRAM. ",
  PEND("ESP32 latency, arena use and parity measured on hardware (Appendix B)"),
]));

children.push(H2("6.2 Four Silent Failure Modes"));
children.push(P(
  "Re-running the reviewers' requested experiments exposed four errors in our own pipeline. None raised an exception, and each produced plausible numbers. We describe them because the same components (Flower, TensorFlow Model Optimization, Keras BatchNorm) are widely used, and because the earlier version of this paper reported results built on the first two."
));
const failureModes = [
  ["Weight exchange without scales", "Clients sent int8-rounded weights without their per-tensor scales, and the server averaged the integer codes as if they were weights. Every tensor was therefore rescaled to max |w| = 127 each round, which was visible afterwards as global weights and QAT ranges at exactly ±127. Check: assert that dequantize(quantize(w)) ≈ w on the receiving side."],
  ["QAT wrapper stripping that copies nothing", "Converting a QAT model back to float copied weights positionally from the wrapped layer, which does not own its kernel. Every assignment failed silently and the 'stripped' model kept its random initialization. Short fine-tuning on pooled data then produced a deployable-looking model that had learned nothing from FL. Check: copy weights by name, refuse to proceed if nothing was copied, and compare outputs before and after stripping."],
  ["BatchNorm folded on the wrong side of the activation", "With Dense → ReLU → BatchNorm, folding BatchNorm into the preceding Dense is not equivalent (max output error 0.86, 11.6% of decisions flipped on our model). Check: fold into the next layer and assert numerical equivalence on random inputs."],
  ["Pruning that resets BatchNorm statistics", "Re-creating BatchNorm layers from their configuration during pruning discarded the learned statistics (accuracy 95.4% → 32.7% before fine-tuning). Check: fold BatchNorm before pruning."],
];
for (const [k, v] of failureModes) {
  children.push(RP([{ text: k + ". ", bold: true }, { text: v }], { spacingAfter: 120 }));
}
children.push(P(
  "What exposed them was an ablation that evaluates the federated model itself, unchanged, at every stage of compression (the FP32 rows in Table 3). That row should reproduce the float model's metrics exactly, and in the erroneous pipeline it was at chance level. We recommend this invariant check for any FL-plus-compression study."
));

children.push(H2("6.3 What Goes Beyond Pipeline Composition"));
children.push(P(
  "The meta-review asked what this work contributes beyond composing known techniques. We consider four results non-obvious from the individual literatures: (1) a federated IDS that is within about 1.7 F1 of centralized training and nearly insensitive to an extreme Dirichlet(0.3) partition on two datasets; (2) compression to 65 KB with no loss using only one client's data, together with a quantified dependence on that client's class mix; (3) the dataset-dependent failure of training-time QAT under FL, traced to heavy-tailed features rather than to optimization; and (4) the four silent failure modes and the invariant that detects them; and (5) the instability of full-integer PTQ across calibration draws and a one-line fix (clipped calibration), together with a local-only baseline that locates the benefit of federation in clients with few local attacks. The paper is an evidence-backed characterization, not a new algorithm, and we state that explicitly."
));

// =================== 7 Limitations ===================
children.push(H1("7. Limitations"));
const limItems = [
  ["Single seed", "Every configuration was run once. Differences below about one F1 point (e.g. near-IID vs. Dirichlet on CIC-IDS2017) should not be over-interpreted until seed variance is measured."],
  ["Client count", "All experiments use four clients. Scaling to 20–50 clients is supported by our scripts but not yet run."],
  ["Choice of fine-tuning client", "Deployed-model quality depends on the fine-tuning client's class mix (Table 4). We chose clients a priori by attack share. Federated fine-tuning over all clients removes this dependence in the recall-priority setting (Section 5.13), but we have not repeated Table 4 with it."],
  ["Comparison to published systems", "We compare only against our own centralized baseline. Published CIC-IDS2017 results mostly use multi-class weighted metrics, different splits, and no deduplication, and the dataset has known labelling issues [24], so cross-paper numbers are not directly comparable. We have compiled candidate comparisons but report none until each number is verified against its source."],
  ["Single architecture", "All results use one MLP; the QAT failure mode (Section 5.6) may differ for architectures with input normalization layers."],
  ["Two TON_IoT federated models", "Only Sections 5.8, 5.14, and 6.1 use the improved TON_IoT federated model; the other TON_IoT results use the original 37-feature model and were not re-run with the improved one."],
  ["Threshold selection data", "Recall-target thresholds (Section 5.13) are chosen on the fine-tuning client's SMOTE-augmented training data; targets above 99.95% need a larger, non-augmented validation set."],
  ["Transfer attacks only", "Robustness of TFLite models is measured with adversarial examples transferred from the float federated model, not with attacks computed through the integer model."],
  ["On-device measurement", "See Section 6.1."],
];
for (const [k, v] of limItems) {
  children.push(RP([{ text: k + ". ", bold: true }, { text: v }], { spacingAfter: 140 }));
}

// =================== 8 Future Work ===================
children.push(H1("8. Future Work"));
children.push(P("Beyond the items in Section 7, we plan to evaluate robust feature scaling as a fix for training-time QAT on heavy-tailed IDS data, federated fine-tuning for the other compression settings (Section 5.13 applies it to the recall-priority case only), threshold selection against an explicit alert budget (Section 5.13 selects for a recall target only), and deployment on heterogeneous microcontrollers beyond the ESP32."));

// =================== 9 Conclusion ===================
children.push(H1("9. Conclusion"));
children.push(P(
  "A federated MLP intrusion detector trained with FedAvgM, a shared cosine schedule, and focal loss comes close to centralized training on CIC-IDS2017 and TON_IoT and is robust to a strongly non-IID partition. It can be compressed to a 65 KB INT8 TensorFlow Lite model (12.3×) without loss using only one participating client's data, provided that client's class mix is representative, and with distillation, federated fine-tuning and a client-chosen threshold the 65 KB model misses about half as many attacks as the FP32 model at its default threshold (27 vs. 52 of 85,173) at an 11.3% false-alarm rate; the remaining misses are concentrated in the rarest attack classes, which the FP32 model also misses. Training-time QAT inside FL, by contrast, fails on heavy-tailed CIC-IDS2017 features while working on TON_IoT, and full-integer post-training quantization is unstable across calibration draws unless its calibration inputs are clipped. Federation matters most for clients with few local attacks, which alone miss up to 55% of attacks. Finally, four silent toolchain failures had invalidated the compressed-model results of our earlier report. An invariant check that re-evaluates the uncompressed federated model at every stage would have caught all of them."
));

// =================== References ===================
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H1("References"));
const refs = [
  "[1] Luigi Atzori, Antonio Iera, and Giacomo Morabito. 2010. The Internet of Things: A Survey. Computer Networks 54, 15 (2010), 2787–2805.",
  "[2] Colby Banbury, Chia-Yu Zhou, and Igor Fedorov. 2020. TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers. arXiv preprint arXiv:2003.05403 (2020).",
  "[3] Ian Goodfellow, Jonathon Shlens, and Christian Szegedy. 2015. Explaining and Harnessing Adversarial Examples. In International Conference on Learning Representations (ICLR).",
  "[4] Song Han, Huizi Mao, and William J. Dally. 2016. Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding. In International Conference on Learning Representations (ICLR).",
  "[5] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the Knowledge in a Neural Network. arXiv preprint arXiv:1503.02531 (2015).",
  "[6] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. 2018. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).",
  "[7] Jakub Konečný, H. Brendan McMahan, Felix Yu, and Peter Richtárik. 2016. Federated Learning: Strategies for Improving Communication Efficiency. arXiv preprint arXiv:1610.05492 (2016).",
  "[8] H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y Arcas. 2017. Communication-Efficient Learning of Deep Networks from Decentralized Data. In Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS).",
  "[9] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. 2018. Towards Deep Learning Models Resistant to Adversarial Attacks. In International Conference on Learning Representations (ICLR).",
  "[10] Tzu-Ming Harry Hsu, Hang Qi, and Matthew Brown. 2019. Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification. arXiv preprint arXiv:1909.06335 (2019).",
  "[11] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. 2020. Federated Optimization in Heterogeneous Networks. In Proceedings of Machine Learning and Systems (MLSys).",
  "[12] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. 2017. Focal Loss for Dense Object Detection. In Proceedings of the IEEE International Conference on Computer Vision (ICCV).",
  "[13] Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani. 2018. Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. In Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP).",
  "[14] Nickolaos Koroniotis, Nour Moustafa, Elena Sitnikova, and Benjamin Turnbull. 2019. Towards the Development of Realistic Botnet Dataset in the Internet of Things for Network Forensic Analytics: Bot-IoT Dataset. Future Generation Computer Systems 100 (2019), 779–796.",
  "[15] Nour Moustafa. 2021. A New Distributed Architecture for Evaluating AI-Based Security Systems at the Edge: Network TON_IoT Datasets. Sustainable Cities and Society 72 (2021), 102994.",
  "[16] Markus Nagel, Marios Fournarakis, Yelysei Bondarenko, and Tijmen Blankevoort. 2022. Overcoming Oscillations in Quantization-Aware Training. In Proceedings of the 39th International Conference on Machine Learning (ICML), PMLR 162, 16318–16330.",
  "[17] Markus Nagel, Marios Fournarakis, Rana Ali Amjad, Yelysei Bondarenko, Mart van Baalen, and Tijmen Blankevoort. 2021. A White Paper on Neural Network Quantization. arXiv preprint arXiv:2106.08295 (2021).",
  "[18] Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W. Mahoney, and Kurt Keutzer. 2021. A Survey of Quantization Methods for Efficient Neural Network Inference. arXiv preprint arXiv:2103.13630 (2021).",
  "[19] Robert David, Jared Duke, Advait Jain, Vijay Janapa Reddi, Nat Jeffries, Jian Li, Nick Kreeger, Ian Nappier, Meghna Natraj, Tiezhen Wang, Pete Warden, and Rocky Rhodes. 2021. TensorFlow Lite Micro: Embedded Machine Learning for TinyML Systems. In Proceedings of Machine Learning and Systems (MLSys).",
  "[20] Yue Zhao, Meng Li, Liangzhen Lai, Naveen Suda, Damon Civin, and Vikas Chandra. 2018. Federated Learning with Non-IID Data. arXiv preprint arXiv:1806.00582 (2018).",
  "[21] Micah Gorsline, James Smith, and Cory Merkel. 2021. On the Adversarial Robustness of Quantized Neural Networks. arXiv preprint arXiv:2105.00227 (2021).",
  "[22] Chang Song, Elias Fallon, and Hai Li. 2021. Improving Adversarial Robustness in Weight-quantized Neural Networks. arXiv preprint arXiv:2012.14965 (2021).",
  "[23] Ferheen Ayaz, Idris Zakariyya, José Cano, Sye Loong Keoh, Jeremy Singer, Danilo Pau, and Mounia Kharbouche-Harrari. 2023. Improving Robustness Against Adversarial Attacks with Deeply Quantized Neural Networks. arXiv preprint arXiv:2304.12829 (2023).",
  "[24] Gints Engelen, Vera Rimmer, and Wouter Joosen. 2021. Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study. In 2021 IEEE Security and Privacy Workshops (SPW), 7–12. doi:10.1109/SPW53761.2021.00009.",
];
for (const r of refs) children.push(P(r, { spacingAfter: 100, size: 20 }));

// =================== Appendix A ===================
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H1("Appendix A: Exact BatchNorm Folding for Post-Activation BatchNorm"));
children.push(P("In our MLP each hidden layer is Dense → ReLU → BatchNorm → Dropout. At inference, BatchNorm with scale γ, shift β, running mean μ, running variance σ² and epsilon ε is the affine map", { spacingAfter: 120 }));
children.push(P("BN(h) = s ⊙ h + t,   s = γ / √(σ² + ε),   t = β − s ⊙ μ", { spacingAfter: 120, alignment: AlignmentType.CENTER }));
children.push(P("Because it is applied after the ReLU, it cannot be folded into the preceding Dense layer. Dropout is the identity at inference, so BN can be folded into the next Dense layer (weights W, bias b):", { spacingAfter: 120 }));
children.push(P("W′ = diag(s) · W,   b′ = b + tᵀ W", { spacingAfter: 160, alignment: AlignmentType.CENTER }));
children.push(P("This is exact. On our trained models the folded network matches the original to within 2·10⁻⁵. The commonly used pre-activation formula W′ = W·s, b′ = s·(b − μ) + β is valid only for Dense → BatchNorm → activation. Applied to our post-activation layout, it changed 11.6% of decisions (Section 6.2). Folding is applied before pruning, before QAT fine-tuning, and before every TFLite export."));

// =================== Appendix B ===================
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H1("Appendix B: ESP32 On-Device Benchmark Protocol"));
children.push(P("The harness is complete and verified on a host build. Running it requires only a connected ESP32-class board.", { spacingAfter: 160 }));
const steps = [
  "Models: the firmware embeds the two deployment models and their FP32 federated models (models/*.tflite, 1.7 MB in total), converted to 16-byte-aligned C arrays at build time (esp32_tflite_project/gen_model_data.py). scripts/prepare_esp32_benchmark.py stages the models, records each model's deployment threshold, and regenerates the parity vectors.",
  "Build and flash (PlatformIO, TensorFlowLite_ESP32 1.0.0, huge_app partition table so the FP32 models fit): pio run -e <esp32dev | esp32-s3-devkitc-1 | esp32-c3-devkitm-1> -t upload.",
  "Parity check: for 8 fixed standardized input vectors per dataset (78 features for CIC-IDS2017, 43 for the improved TON_IoT model), the firmware compares each model's on-device output with the host TFLite output (include/test_vectors.h) and reports the maximum absolute difference and decision agreement at the model's deployment threshold. A host build against the same TensorFlow Lite Micro library matched exactly (FP32) and within one quantization step (INT8, 1/256), with 8/8 decisions in agreement for every model.",
  "Timing: after 5 warm-up inferences, 100 timed Invoke() calls per model are measured with micros(). The firmware reports chip model, core count, CPU clock and ESP-IDF version. Tensor-arena use is 2.2–4.6 KB (16 KB allocated).",
  "Collect: python scripts/collect_esp32_benchmark.py --port <serial port> writes per-model mean / median / std / p95 latency, arena use and parity to data/processed/ablation/esp32_benchmark.json.",
];
steps.forEach((s, i) => children.push(RP([{ text: `${i+1}. `, bold: true }, { text: s }], { spacingAfter: 120 })));
children.push(PP([PEND("run on hardware and report latency")], { spacingAfter: 160 }));

// ---------------- Build document ----------------
const doc = new Document({
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 }, // US Letter
        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
      },
    },
    children,
  }],
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 22 } },
    },
  },
});

Packer.toBuffer(doc).then(buf => {
  const out = path.join(__dirname, "TinyML_Federated_IDS_Revised.docx");
  fs.writeFileSync(out, buf);
  console.log("done");
});
