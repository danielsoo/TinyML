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
  ["Decompose the compression ratio (81B-6)", "Section 5.3 separates INT8 quantization (3.5×) from structured pruning (a further 3.5×), and Section 5.5 sweeps the pruning ratio."],
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
  "Federated Learning (FL) combined with TinyML is an attractive basis for privacy-preserving intrusion detection on microcontroller-class IoT devices, but the claim that a federated model can be compressed for such devices without losing detection quality is rarely tested end to end. We train a multilayer-perceptron intrusion detector with Flower (FedAvgM, cosine learning-rate decay, focal loss) on CIC-IDS2017 and TON_IoT, and compress it for TensorFlow Lite Micro using only data a federated deployment would actually hold. Three results stand out. First, federated training comes close to centralized training on the same recipe (F1 85.70 vs. 87.35 on CIC-IDS2017; 99.40 vs. 99.48 on TON_IoT), and a strongly non-IID Dirichlet(0.3) partition costs at most about one F1 point. Second, structured pruning followed by fine-tuning and QAT fine-tuning on a single participating client's local data yields a 65 KB INT8 model (12.3× smaller than FP32) with no loss relative to the federated model (F1 85.59, Attack Recall 99.66% on CIC-IDS2017; F1 98.74 on TON_IoT). How well this works depends on how closely that client's class mix matches the global one. Third, training-time quantization-aware training inside FL collapses on CIC-IDS2017 (34.7% accuracy): heavy-tailed standardized features drive the learned INT8 input range to [−1231, 405]. The same method works on TON_IoT, and post-training quantization of the float model costs only about one F1 point. We also document four silent failure modes in a common TensorFlow, tfmot, and Flower toolchain that had invalidated the compressed-model results of our own earlier report, together with the checks that catch them.",
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
  "Contributions. (1) An end-to-end evaluation of a federated TinyML intrusion detector on two datasets against a centralized baseline trained with the identical recipe, under both near-IID and Dirichlet(0.3) client partitions (Sections 5.1–5.2). (2) An FL-faithful compression procedure: exact BatchNorm folding, structured pruning, then fine-tuning and QAT fine-tuning on one client's local data. It produces a 65 KB INT8 model (12.3×) with no loss relative to the federated model, and we quantify how the outcome depends on which client performs the fine-tuning (Sections 5.3–5.5). (3) A negative result: training-time QAT inside FL fails on CIC-IDS2017 because its moving-average quantizers absorb the heavy-tailed feature outliers. It works on TON_IoT, and post-training quantization of a float model is unaffected (Section 5.6). (4) Four silent failure modes in a widely used TensorFlow / TensorFlow Model Optimization / Flower toolchain, each of which yields plausible-looking but wrong compressed-model results, with the checks we now run to catch them (Section 6.2). (5) FGSM/PGD robustness of the corrected models (Section 5.7) and a verified ESP32 benchmark harness (Section 6.1, Appendix B)."
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
  "TON_IoT [15]: the network train/test file (211,043 records). To avoid shortcut features, we drop host identifiers and ephemeral ports (src_ip, dst_ip, src_port; the file has no timestamp column) and keep dst_port, which corresponds to CIC-IDS2017's Destination Port feature. Text columns with at most 50 distinct values (protocol, service, connection state, DNS/SSL/HTTP flags) are label-encoded, and high-cardinality free text (DNS query, URI) is dropped, leaving 37 features. Duplicates are removed (105,176 unique records: 24,106 normal, 81,070 attack) and the same split, scaling, and SMOTE steps are applied (test set: 21,036 records)."
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
const t1w = [3000, 1150, 1150, 1150, 1150, 1100];
const t1h = ["Model (test split, threshold 0.3)", "Accuracy", "Precision", "Attack Recall", "F1", "FAR"];
children.push(makeTable(t1h, [
  ["CIC-IDS2017 — centralized", "95.07%", "77.60%", "99.90%", "87.35%", "5.92%"],
  ["CIC-IDS2017 — federated (near-IID)", "94.32%", "75.01%", "99.94%", "85.70%", "6.84%"],
  ["CIC-IDS2017 — federated, fixed LR", "pending", "", "", "", ""],
  ["TON_IoT — centralized", "99.20%", "99.32%", "99.64%", "99.48%", "2.28%"],
  ["TON_IoT — federated (near-IID)", "99.08%", "99.18%", "99.62%", "99.40%", "2.76%"],
  ["TON_IoT — federated, fixed LR", "99.02%", "99.00%", "99.73%", "99.36%", "3.38%"],
], t1w));
children.push(caption("Table 1. Federated vs. centralized training with the identical model, loss, and epoch budget (float models, before compression)."));
children.push(PP([
  "Federated training comes within 1.65 F1 points of centralized training on CIC-IDS2017 and within 0.08 on TON_IoT. On CIC-IDS2017 the gap lies almost entirely in precision (FAR 6.84% vs. 5.92%); both models detect more than 99.9% of attacks. ",
  "On TON_IoT, a fixed learning rate performs as well as the cosine schedule (F1 99.36 vs. 99.40), so the large effect of the cosine schedule reported in the WIP version is not visible on this dataset. ",
  PEND("CIC-IDS2017 fixed-learning-rate row from job 2026-10-04_i"),
]));

children.push(H2("5.2 Non-IID Clients"));
children.push(makeTable(["Model", "Client sizes (attack share)", "Accuracy", "F1", "Attack Recall", "FAR"], [
  ["CIC-IDS2017 — near-IID", "4 × equal (≈ class ratio)", "94.32%", "85.70%", "99.94%", "6.84%"],
  ["CIC-IDS2017 — Dirichlet(0.3)", "9.4k (0.1%), 1.98M (59.5%), 6.7k (2.6%), 728k (25.1%)", "94.27%", "85.59%", "99.83%", "6.87%"],
  ["TON_IoT — near-IID", "4 × equal (≈ class ratio)", "99.08%", "99.40%", "99.62%", "2.76%"],
  ["TON_IoT — Dirichlet(0.3)", "49.9k (0.4%), 62.9k (92.8%), 8.6k (65.5%), 8.4k (8.5%)", "97.63%", "98.45%", "97.99%", "3.59%"],
], [2600, 3000, 1000, 900, 1200, 900]));
children.push(caption("Table 2. Effect of a Dirichlet(α = 0.3) label partition over four clients (float federated models). Client sizes are training samples after balancing and SMOTE."));
children.push(P(
  "The Dirichlet partition is extreme: on CIC-IDS2017 one client holds 73% of the training data and two clients see almost no attacks. Still, FedAvgM with the shared cosine schedule loses only 0.11 F1 on CIC-IDS2017 and 0.95 F1 on TON_IoT, where Attack Recall drops by 1.6 points and FAR rises from 2.76% to 3.59%. In the WIP-era runs, the non-IID partition appeared to collapse training (F1 63.5). That collapse was an artifact of the weight-exchange error described in Section 6.2."
));

children.push(H2("5.3 Compression Without Pooled Data"));
children.push(makeTable(["Variant (from the near-IID federated model)", "CIC size", "CIC F1 / FAR", "TON size", "TON F1 / FAR"], [
  ["FP32 federated model (TFLite)", "802.5 KB", "85.70 / 6.84%", "720.6 KB", "99.40 / 2.76%"],
  ["INT8 PTQ only (no pruning)", "226.9 KB", "84.75 / 7.37%", "206.4 KB", "98.38 / 2.70%"],
  ["Prune 50%, no fine-tuning → PTQ", "75.1 KB", "42.28 / 56.04%", "64.9 KB", "96.78 / 18.15%"],
  ["Prune 50% → client FT → PTQ", "75.1 KB", "86.90 / 6.08%", "64.9 KB", "90.50 / 14.00%"],
  ["Prune 50% → client FT → QAT FT → INT8 (deployed)", "65.4 KB", "85.59 / 6.82%", "55.2 KB", "98.74 / 2.88%"],
  ["Same, but fine-tuned on pooled data (upper bound)", "65.4 KB", "90.72 / 2.98%", "55.2 KB", "99.17 / 3.28%"],
], [3700, 1000, 1500, 1000, 1500]));
children.push(caption("Table 3. Compression of the federated model. \"Client FT\" uses 10,000 samples from one participating client's own partition (client 0); \"pooled\" uses 10,000 pooled training samples and is not available in a real federated deployment."));
children.push(P(
  "INT8 PTQ alone gives 3.5× at a cost of about one F1 point on both datasets. Pruning half of each hidden layer gives a further 3.5×, but only if the pruned model is fine-tuned. Without fine-tuning, the pruned model breaks down (F1 42.28 on CIC-IDS2017, FAR 18% on TON_IoT). With fine-tuning and QAT fine-tuning on one client's local data, the deployed INT8 model is 65.4 KB on CIC-IDS2017 (802.5 / 65.4 = 12.27×) and 55.2 KB on TON_IoT (13.05×), and it matches the float federated model (F1 85.59 vs. 85.70 and 98.74 vs. 99.40). The deployed CIC-IDS2017 model has 94.28% accuracy, 75.00% precision, and 99.66% Attack Recall. The QAT fine-tuning step matters on TON_IoT, where client-fine-tuned PTQ reaches only F1 90.50."
));
children.push(P(
  "Pooled fine-tuning scores higher on CIC-IDS2017 (F1 90.72), mostly by shifting the operating point toward precision (FAR 2.98%). The pooled sample's 20% attack share is closer to the test distribution than client 0's 50%. We treat this as an upper bound that requires server-held data, not as a federated result."
));

children.push(H2("5.4 Which Client Fine-Tunes Matters"));
children.push(makeTable(["Federated model", "Fine-tuning client (attack share)", "Deployed INT8 F1", "FAR"], [
  ["CIC-IDS2017 near-IID", "client 0 (50.4%)", "85.59", "6.82%"],
  ["CIC-IDS2017 Dirichlet(0.3)", "client 3 (24.9%)", "91.08", "2.75%"],
  ["CIC-IDS2017 centralized model (reference)", "client 0 (50.4%)", "81.84", "8.84%"],
  ["TON_IoT near-IID", "client 0 (50.1%)", "98.74", "2.88%"],
  ["TON_IoT Dirichlet(0.3)", "client 2 (65.5%)", "98.72", "3.11%"],
], [3300, 2800, 1600, 1100]));
children.push(caption("Table 4. Deployed-model quality as a function of the client that performs fine-tuning (prune 50% → client FT → QAT FT → INT8)."));
children.push(P(
  "The short fine-tuning phase recalibrates the decision boundary toward the fine-tuning client's class mix. A client whose attack share is close to the deployment distribution (client 3 of the CIC-IDS2017 Dirichlet run, 24.9% attacks) yields the best model we observed (F1 91.08, FAR 2.75%). A 50%-attack client yields a recall-heavy operating point. The choice of fine-tuning client is therefore a deployment decision that should consider class mix, and it should be reported. We did not tune it on the test set: the clients in Table 4 were chosen a priori by attack share."
));

children.push(H2("5.5 Compression Strength"));
children.push(P(
  "We sweep the structured-pruning ratio from 30% to 90% on the near-IID federated models. Each ratio is tested with no fine-tuning, client fine-tuning + PTQ, and client fine-tuning + QAT fine-tuning (Figure 1). This replaces the 48-configuration sweep of the WIP version, which was produced by the erroneous pipeline. Without fine-tuning, pruning breaks the CIC-IDS2017 model from 50% onward (from 70% it predicts every flow as an attack) and steadily degrades the TON_IoT model. With client fine-tuning, CIC-IDS2017 degrades gracefully: 30% pruning (112–126 KB) gives F1 87.6–89.7, 70% pruning (31 KB, 26×) gives 84.4–85.0, 85% (14 KB, 57×) gives 81.3–82.9, and 90% (9.9 KB, 81×) gives 78.0. PTQ and QAT fine-tuning track each other closely. On TON_IoT, client fine-tuning + QAT holds F1 between 98.6 and 98.9 all the way to 90% pruning (7.9 KB, 91× smaller than FP32), whereas client fine-tuning + PTQ fluctuates between 90.5 and 98.4. QAT fine-tuning is therefore the choice that is reliable on both datasets. On CIC-IDS2017 the useful range ends around 70% pruning (31 KB) if F1 within about 1.3 points of the federated model is required."
));
children.push(img(FIG + "prune_sweep.png", 600, 249));
children.push(caption("Figure 1. F1 vs. deployed INT8 model size for structured-pruning ratios 30–90% (labels), with fine-tuning on one client's local data followed by QAT fine-tuning or PTQ, and without fine-tuning. Dashed line: the uncompressed federated model."));

children.push(H2("5.6 Training-Time QAT Inside FL: a Negative Result"));
children.push(P(
  "The WIP version reported that training-time QAT (fake quantization from the first federated round) helps at aggressive compression and hurts at moderate compression. That observation came from runs affected by the weight-exchange and QAT-stripping errors of Section 6.2. With both corrected, training-time QAT inside FL fails on CIC-IDS2017. The federated QAT model reaches 34.74% accuracy (F1 34.30, FAR 78.7%), and its float weights reach only F1 46.8. The reason is visible in the learned quantizer state. The input quantizer's moving-average range is [−1231, 405], and activation ranges reach 2342, because standardized CIC-IDS2017 features are heavy-tailed (rare flows lie hundreds to thousands of standard deviations from the mean). One INT8 step then spans about 6.4 standard deviations, so ordinary inputs collapse to the zero point. On TON_IoT, whose features are less extreme, the same procedure works (federated QAT model: accuracy 98.22%, F1 98.84). PTQ of a float federated model is unaffected on both datasets (Table 3), and so is QAT fine-tuning of a pruned float model, because its calibration happens on a small fine-tuning set after training. We therefore use post-training QAT fine-tuning, not training-time QAT, in the deployed pipeline. Robust input scaling (clipping or a log transform before standardization) is a plausible fix for training-time QAT that we have not evaluated."
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
  ["Prune 50% → client FT → QAT FT (deployed)", "94.2%", "44.3%", "45.6%", "98.1%", "23.0%", "23.0%"],
  ["Prune 50% → pooled FT → QAT FT (upper bound)", "96.3%", "44.7%", "46.2%", "98.7%", "56.2%", "55.6%"],
], [3200, 950, 950, 950, 950, 950, 950]));
children.push(caption("Table 5. Accuracy under FGSM and PGD (ε = 0.1, standardized features), perturbations computed on the federated FP32 model and transferred to the other models. On TON_IoT, 23.0% equals the benign share of the test set: those models label every perturbed record benign."));
children.push(P(
  "On CIC-IDS2017, INT8 PTQ preserves the federated model's robustness (PGD 55.6% → 60.4%). Pruning followed by fine-tuning reduces it. Of the pruned models, the QAT-fine-tuned deployment model (PGD 45.6%) is clearly more robust than the PTQ one (23.9%), which echoes the quantization-robustness interaction reported in [21–23]. The centralized model is markedly more fragile (PGD 29.0%) even though it is attacked only by transfer. On TON_IoT, ε = 0.1 is strong enough that every model except the pooled-fine-tuned one predicts all perturbed records as benign. Two caveats apply. First, these are unconstrained L∞ perturbations in standardized feature space that ignore feature semantics (integer counts, encoded categories), so they overstate what a network attacker can realize. Second, the per-model differences come from a single seed. We report them as relative robustness under a common perturbation, not as absolute security guarantees."
));

children.push(H2("5.8 Practical Significance"));
children.push(P(
  "For the deployed CIC-IDS2017 model (65.4 KB INT8), 99.66% Attack Recall means 0.34% of attack flows are missed (291 of 85,173 test attacks). The 6.82% FAR means about one in fifteen benign flows raises an alert. For the deployed TON_IoT model, 1.65% of attacks are missed (268 of 16,214) and 2.88% of benign records are flagged. The fixed 0.3 threshold deliberately favours recall. A deployment with a known alert budget should instead pick the threshold from a precision-recall curve on validation data, which we leave as future work (Section 8). Section 5.4 shows that the fine-tuning client's class mix shifts this trade-off as well."
));

// =================== 6 Discussion ===================
children.push(H1("6. Discussion"));

children.push(H2("6.1 Device Heterogeneity and On-Device Measurement"));
children.push(PP([
  "IoT deployments vary in MCU architecture, memory, and clock, and interpreter latency on a workstation does not establish on-device latency. We provide ESP32 firmware that embeds an FP32 model and an INT8 deployment model of the architecture used here. It checks on-device outputs against the host TFLite interpreter on fixed inputs and times 100 inferences per model. A host build against the same TensorFlow Lite Micro library reproduces the interpreter outputs exactly (FP32) and within one quantization step (INT8). Tensor-arena use is 2–4 KB, well within ESP32 SRAM. ",
  PEND("ESP32 latency, arena use and parity for the final deployment models (Appendix B)"),
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
  "The meta-review asked what this work contributes beyond composing known techniques. We consider four results non-obvious from the individual literatures: (1) a federated IDS that is within about 1.7 F1 of centralized training and nearly insensitive to an extreme Dirichlet(0.3) partition on two datasets; (2) compression to 65 KB with no loss using only one client's data, together with a quantified dependence on that client's class mix; (3) the dataset-dependent failure of training-time QAT under FL, traced to heavy-tailed features rather than to optimization; and (4) the four silent failure modes and the invariant that detects them. The paper is an evidence-backed characterization, not a new algorithm, and we state that explicitly."
));

// =================== 7 Limitations ===================
children.push(H1("7. Limitations"));
const limItems = [
  ["Single seed", "Every configuration was run once. Differences below about one F1 point (e.g. near-IID vs. Dirichlet on CIC-IDS2017) should not be over-interpreted until seed variance is measured."],
  ["Client count", "All experiments use four clients. Scaling to 20–50 clients is supported by our scripts but not yet run."],
  ["Choice of fine-tuning client", "Deployed-model quality depends on the fine-tuning client's class mix (Table 4). We chose clients a priori by attack share; a principled selection or a federated fine-tuning round is future work."],
  ["Comparison to published systems", "We compare only against our own centralized baseline. Published CIC-IDS2017 results mostly use multi-class weighted metrics, different splits, and no deduplication, and the dataset has known labelling issues [24], so cross-paper numbers are not directly comparable. We have compiled candidate comparisons but report none until each number is verified against its source."],
  ["Single architecture", "All results use one MLP; the QAT failure mode (Section 5.6) may differ for architectures with input normalization layers."],
  ["Transfer attacks only", "Robustness of TFLite models is measured with adversarial examples transferred from the float federated model, not with attacks computed through the integer model."],
  ["On-device measurement", "See Section 6.1."],
];
for (const [k, v] of limItems) {
  children.push(RP([{ text: k + ". ", bold: true }, { text: v }], { spacingAfter: 140 }));
}

// =================== 8 Future Work ===================
children.push(H1("8. Future Work"));
children.push(P("Beyond the items in Section 7, we plan to evaluate robust feature scaling as a fix for training-time QAT on heavy-tailed IDS data, federated (multi-client) fine-tuning during compression, threshold selection from validation precision-recall curves against an alert budget, and deployment on heterogeneous microcontrollers beyond the ESP32."));

// =================== 9 Conclusion ===================
children.push(H1("9. Conclusion"));
children.push(P(
  "A federated MLP intrusion detector trained with FedAvgM, a shared cosine schedule, and focal loss comes close to centralized training on CIC-IDS2017 and TON_IoT and is robust to a strongly non-IID partition. It can be compressed to a 65 KB INT8 TensorFlow Lite model (12.3×) without loss using only one participating client's data, provided that client's class mix is representative. Training-time QAT inside FL, by contrast, fails on heavy-tailed CIC-IDS2017 features while working on TON_IoT. Finally, four silent toolchain failures had invalidated the compressed-model results of our earlier report. An invariant check that re-evaluates the uncompressed federated model at every stage would have caught all of them."
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
  "Models: the firmware embeds an FP32 model and an INT8 deployment model (models/*.tflite), converted to 16-byte-aligned C arrays at build time (esp32_tflite_project/gen_model_data.py). scripts/prepare_esp32_benchmark.py swaps in any model pair and regenerates the parity vectors.",
  "Build and flash (PlatformIO, TensorFlowLite_ESP32 1.0.0, huge_app partition table so the FP32 model fits): pio run -e <esp32dev | esp32-s3-devkitc-1 | esp32-c3-devkitm-1> -t upload.",
  "Parity check: for 8 fixed standardized input vectors, the firmware compares each model's on-device output with the host TFLite output (include/test_vectors.h) and reports the maximum absolute difference and decision agreement. A host build against the same TensorFlow Lite Micro library matched exactly (FP32) and within one quantization step (INT8, 1/256).",
  "Timing: after 5 warm-up inferences, 100 timed Invoke() calls per model are measured with micros(). The firmware reports chip model, core count, CPU clock and ESP-IDF version. Tensor-arena use is 2–4 KB (16 KB allocated).",
  "Collect: python scripts/collect_esp32_benchmark.py --port <serial port> writes per-model mean / median / std / p95 latency, arena use and parity to data/processed/ablation/esp32_benchmark.json.",
];
steps.forEach((s, i) => children.push(RP([{ text: `${i+1}. `, bold: true }, { text: s }], { spacingAfter: 120 })));
children.push(PP([PEND("swap in the final deployment models (Table 3) and run on hardware")], { spacingAfter: 160 }));

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
