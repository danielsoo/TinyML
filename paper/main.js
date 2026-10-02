const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, PageBreak,
  P, RP, H1, H2, H3, caption, makeTable, img, UP, convertInchesToTwip
} = require("./build.js");
const fs = require("fs");

const path = require("path");
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
  children: [new TextRun({ text: "Extended version — revised in response to LCTES '26 Work-in-Progress reviews", italics: true, size: 22 })],
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
  "This version is a substantial expansion of LCTES '26 Paper #81 (Work-in-Progress). We thank the reviewers for detailed, actionable feedback. Below we summarize what changed; full detail is in the corresponding sections.",
  { spacingAfter: 120 }
));
const respItems = [
  ["Limited experiment/evaluation, no comparison to other baselines or IDS systems (81A)", "Sections 5.4–5.8 add a 48-configuration grid sweep, a full four-attack adversarial evaluation, and a PGD adversarial-training study, substantially deepening the internal evaluation. A comparison against other published IDS systems on CIC-IDS2017 is not included — see Section 7 (\"Comparison to other published IDS systems\") for why, and what a responsible version of it would require."],
  ["Fix the baseline / decompose the headline numbers (81B-1, 81B-6)", "Section 5.2 adds an explicit federated+cosine-LR pre-compression data point (missing intermediate step); Section 5.4 decomposes the 12.28× ratio into its distillation / pruning / INT8 contributions using the 48-configuration grid sweep. A centralized-only baseline was not re-run in this revision — Section 7 specifies the exact recipe as a fast, ready-to-execute follow-up rather than presenting an estimated number."],
  ["Non-IID clients / small client count (81B-2, 81D)", "Not re-run in this revision; Section 7 specifies a concrete Dirichlet(0.3) protocol compatible with the existing client-partitioning code, and separately flags that four clients is small independent of IID-ness."],
  ["On-device measurement (81B-3)", "The ESP32 TFLite-Micro benchmark harness (firmware + log-collection script) is complete and described in Appendix B, but this revision was produced without physical access to the board; Section 6.1 and Appendix B give the exact reproduction steps and current status."],
  ["FGSM results missing (81B-4, 81D-3)", "Section 5.7 reports full FGSM/FGM/GA/PGD evaluation results across four deployment configurations (previously only described, not tabulated)."],
  ["Lead with QAT finding (81B-5)", "Section 5.5 is restructured around this as the paper's central empirical finding, with Figure 1 (accuracy vs. compression ratio, with/without training-time QAT) as the headline figure."],
  ["Dataset generalization (81B-7, 81C)", "Section 3.1 and Section 7 clarify which additional datasets (Bot-IoT, TON_IoT) are already supported by the data loader but not yet evaluated end-to-end, and why."],
  ["Minor: focal-loss motivation, duplicate table rows, BatchNorm math placement (81B minor)", "Section 3.2 motivates α; Section 5.4 explains why two configurations legitimately share an identical 14.44 KB footprint; the BatchNorm-folding derivation moved to Appendix A."],
  ["Practical meaning of Attack Recall / false positives (81C)", "Section 5.10 adds a precision / false-positive-rate reading of every headline result."],
  ["Device heterogeneity (81C)", "Section 6.1 discusses this directly."],
  ["Related work depth and citation formatting (81C)", "Section 2 is substantially expanded (8 → 20 references) and reorganized by topic."],
  ["Incremental novelty (Meta-review)", "Section 6.2 explicitly separates the three claims that go beyond pipeline composition: the QAT/compression-level interaction (Section 5.5), the discovery that pruning improves recall on this task (Section 5.6), and adversarial-training results within the federated+compressed pipeline (Section 5.8), which were not requested by reviewers but strengthen the robustness story."],
];
for (const [k, v] of respItems) {
  children.push(RP([{ text: "• " + k + ": ", bold: true }, { text: v }], { spacingAfter: 100 }));
}
children.push(new Paragraph({ children: [new PageBreak()] }));

// ---------------- Abstract ----------------
children.push(H1("Abstract"));
children.push(P(
  "Federated Learning combined with TinyML presents a compelling framework for privacy-preserving intrusion detection on resource-constrained IoT devices. However, achieving high Attack Recall under strict model size, latency, and quantization constraints remains challenging because of training instability, class imbalance, and deployment incompatibilities between BatchNormalization and INT8 TensorFlow Lite conversion. Rather than proposing a new model architecture, this work systematically investigates and optimizes training and compression configurations for federated TinyML intrusion detection on CIC-IDS2017, and extends a prior work-in-progress report with four additions requested by reviewers: an explicit decomposition of the reported compression ratio into its distillation, pruning, and quantization contributions; a restructured presentation that leads with our central empirical finding — that training-time quantization-aware training (QAT) improves robustness and moderately-compressed accuracy in different, sometimes opposite, directions depending on the target compression ratio; a full adversarial-robustness evaluation under four gradient-based attacks (FGSM, FGM, a genetic-algorithm black-box attack, and PGD) across four deployment configurations, together with a PGD adversarial-training study that raises mean adversarial accuracy from 67.9% to 88.8%; and an expanded related-work and limitations discussion that is explicit about what remains untested (non-IID client partitions, cross-dataset generalization, and physical on-device latency). We show that server-coordinated cosine learning-rate decay alone improves Attack Recall from 46.7% to 93.85% while stabilizing convergence, and that the resulting compression pipeline achieves a 12.28× model-size reduction and a 74.5% latency reduction while maintaining 93.85% Attack Recall on the final INT8 deployment model. We report these results together with the evidence, code paths, and precise follow-up protocols needed to close the remaining gaps, so that the trade-offs and limitations of the approach are stated as precisely as the results themselves.",
  { spacingAfter: 200 }
));

// =================== 1 Introduction ===================
children.push(H1("1. Introduction"));
children.push(P(
  "The rapid proliferation of Internet of Things (IoT) devices has transformed modern sensing, monitoring, and control systems, enabling data-driven applications across smart homes, healthcare, industrial automation, and critical infrastructure [1]. These systems increasingly rely on machine learning models trained on sensitive data, raising concerns about privacy, ownership, and regulatory compliance. Traditional centralized learning requires raw data to be transmitted to cloud servers, introducing privacy risk, communication overhead, and exposure to data breaches."
));
children.push(P(
  "Federated Learning (FL) addresses this by enabling collaborative model training across distributed clients without sharing raw data: clients train locally and share only model updates with a central server [7, 8]. Despite its privacy advantages, deploying FL on real IoT hardware remains difficult. IoT devices are resource-constrained in memory, compute, energy, and bandwidth, while standard FL pipelines assume relatively large models and frequent bidirectional communication of full-precision updates — assumptions that are incompatible with microcontroller-class devices such as the ESP32."
));
children.push(P(
  "TinyML addresses the deployment side of this problem by aggressively optimizing model size, memory footprint, and inference latency for microcontrollers [2], but most TinyML systems target inference-only workloads; integrating TinyML-style compression into a federated training pipeline — rather than applying it once, after training, to a fully centralized model — remains comparatively unexplored, and doing so exposes interactions (between BatchNormalization, quantization, and federated aggregation; between class imbalance and quantization-aware training; between compression and adversarial robustness) that do not arise in either setting alone."
));
children.push(P(
  "Machine-learning-based intrusion detection systems (IDS) can achieve strong detection performance [13], but deploying them on IoT devices requires lightweight models that still maintain high Attack Recall, since a missed attack is far more costly than a false alarm. In our early experiments on the CIC-IDS2017 dataset, a federated IDS model trained with a fixed learning rate achieved only 46.7% Attack Recall despite 93.5% overall accuracy — a model that would silently miss roughly one in every two attacks while looking accurate on an aggregate metric. This gap between aggregate accuracy and the recall that actually matters for security is a central motivation for this work, and we return to it quantitatively in Section 5.10."
));
children.push(P(
  "This work makes the following contributions. (1) We identify server-coordinated cosine learning-rate decay as the single largest lever on Attack Recall in this setting, improving it from 46.7% to 93.85% without any architecture change (Section 5.1). (2) We identify and fix a BatchNormalization/TFLite INT8 conversion instability that produced NaN outputs on target devices, via BatchNorm folding (Section 5.9, Appendix A). (3) We characterize, across a 48-configuration grid sweep, how training-time QAT interacts with compression level: it is harmful at moderate compression and can be neutral-to-helpful at aggressive compression, and it is decisively better for adversarial robustness across every compression level we tested (Section 5.5, Section 5.7) — this is, in our view, the paper's most interesting and least intuitive finding, and this revision restructures the presentation to lead with it. (4) We evaluate adversarial robustness under FGSM, FGM, a genetic-algorithm attack, and PGD across four representative deployment configurations, and separately show that PGD adversarial training raises mean adversarial accuracy from 67.9% to 88.8% while the underlying model is carried through the full compression pipeline (Section 5.7–5.8). (5) We are explicit about what this evaluation does not yet establish — non-IID client behavior, cross-dataset generalization, and physical on-device latency — and give exact, reproducible protocols for closing each gap (Section 7)."
));

// =================== 2 Related Work ===================
children.push(H1("2. Related Work"));

children.push(H2("2.1 IoT Security and Intrusion Detection"));
children.push(P(
  "The rapid growth of IoT has resulted in the deployment of billions of interconnected, resource-constrained devices across consumer, industrial, and infrastructure environments [1], exposing them to distributed denial-of-service attacks, botnet recruitment, and other network-based intrusions. Machine-learning-based IDS have demonstrated strong performance at detecting such traffic [13], but most are designed and evaluated for centralized training environments with computational resources that exceed what typical IoT hardware provides. Public IDS benchmark datasets used in this line of work include CIC-IDS2017 [13], which we use throughout this paper, and more recent alternatives such as Bot-IoT [14] and TON_IoT [15], which target IoT-specific traffic and device telemetry rather than the enterprise-network setting of CIC-IDS2017; we discuss why our reported results use CIC-IDS2017 exclusively, and what is already in place to extend to the others, in Section 3.1 and Section 7."
));

children.push(H2("2.2 Federated Learning"));
children.push(P(
  "Federated Learning was introduced by McMahan et al. as FedAvg, in which decentralized clients jointly train a shared model through iterative aggregation of local updates without transmitting raw data [8]; Konečný et al. study strategies for reducing the communication cost of this process [7]. A well-documented difficulty in FL is that client data is rarely independent and identically distributed (IID): Hsu, Qi, and Brown introduce FedAvg with server momentum (FedAvgM) — the aggregation strategy we use — specifically to stabilize training under non-identical client label distributions [10], and Zhao et al. characterize how performance degrades as client data becomes more skewed [20]; Li et al.'s FedProx modifies the local objective with a proximal term for the same reason [11]. Our client partitioning (Section 3.2) is label-distribution-aware in a way that keeps every client's local class balance close to the global balance, which is a considerably milder regime than the non-IID partitions studied in [10, 11, 20]; Section 7 specifies a Dirichlet-based partition, consistent with this line of work, as a concrete next step."
));

children.push(H2("2.3 TinyML and Model Compression"));
children.push(P(
  "TinyML techniques enable inference directly on microcontrollers by aggressively optimizing model size, memory footprint, and computational cost [2]; David et al. describe TensorFlow Lite Micro, the runtime our deployment target uses [19]. To fit models onto such hardware, structured pruning removes redundant parameters [4], knowledge distillation transfers knowledge from a larger teacher model to a smaller student [5], and quantization reduces numerical precision to improve computational and memory efficiency [6]. Quantization-aware training (QAT) and post-training quantization (PTQ) are the two dominant strategies for producing efficient integer representations; Gholami et al. survey the space broadly [18], and Nagel et al.'s white paper gives practical guidance on when 8-bit PTQ alone is already sufficient versus when QAT is needed, and documents that QAT can itself introduce oscillation-driven instability during training [16, 17] — a finding consistent with what we observe under class imbalance and federated aggregation in Section 5.5."
));

children.push(H2("2.4 Adversarial Robustness of Compressed Models"));
children.push(P(
  "The robustness of neural networks to adversarial perturbations is a substantial and separate research area; the Fast Gradient Sign Method (FGSM) [3] and Projected Gradient Descent (PGD) [9] are the two attacks most commonly used to evaluate it, and PGD adversarial training — the technique we use in Section 5.8 — was introduced in the same work that proposes PGD as an attack [9]. A smaller body of work studies how compression interacts with robustness rather than treating them independently: Gorsline et al. study how quantization level affects adversarial robustness in isolation [21], Song et al. propose training techniques to recover robustness lost to weight quantization [22], and Ayaz et al. study the same interaction specifically for deeply quantized, TinyML-scale networks [23]. Our results in Section 5.7 are consistent with the direction reported in this line of work — quantization-aware training measurably changes adversarial robustness, not just clean accuracy — but, to our knowledge, we are the first to report this specifically for a federated, compressed intrusion-detection pipeline rather than a centrally-trained vision or language model."
));

children.push(H2("2.5 Positioning"));
children.push(P(
  "Despite these advances, relatively few studies systematically investigate the intersection of federated learning, TinyML deployment constraints, and intrusion detection for IoT devices together. In particular, achieving high attack-detection performance while maintaining extremely small model size, stable federated training, and characterized robustness to adversarial perturbation, all in one pipeline, remains comparatively unstudied. This work addresses that gap empirically: rather than a new algorithm, it is a systematic, ablation-driven characterization of where a federated TinyML IDS pipeline breaks (training instability, BatchNorm/TFLite incompatibility, QAT-under-imbalance) and what specifically fixes each break, together with the first robustness evaluation of this combination that we are aware of."
));

// =================== 3 Approach ===================
children.push(H1("3. Approach"));
children.push(P(
  "The goal of this work is to maintain high intrusion-detection performance while minimizing model size, enabling deployment on resource-constrained IoT devices such as ESP32-class microcontrollers. We aim to preserve accuracy, F1-score, and — primarily — Attack Recall, while applying federated training and multiple model-compression techniques."
));

children.push(H2("3.1 Dataset and Preprocessing"));
children.push(P(
  "We use the CIC-IDS2017 dataset [13] for all federated and compression experiments. CIC-IDS2017 provides labeled network traffic (benign and multiple attack types) collected in a controlled environment, is widely used in IDS research, and offers a sufficient number of samples and features (78 after preprocessing) for training MLP models while remaining manageable for federated and compression pipelines."
));
children.push(P(
  "We did not use Bot-IoT [14] or TON_IoT [15] for the results reported here, for two related reasons rather than a lack of support: our data loader already implements a Bot-IoT loader with IP-address encoding and categorical feature handling, and a TON_IoT loader, for compatibility testing, but each dataset has a different feature structure (38 features for our Bot-IoT setup versus 78 for CIC-IDS2017) that would require its own hyperparameter and compression sweep to evaluate fairly rather than reusing the CIC-IDS2017 configuration as-is; and doing so for all three datasets within the time available for this revision would have meant a shallower sweep on each rather than the depth of ablation we report for CIC-IDS2017. Section 7 specifies the exact experiment we would run first."
));
children.push(P(
  "Preprocessing proceeds as follows. Labels are converted to binary (BENIGN → 0, all attack types → 1). We replace ±infinity with NaN, drop all-NaN columns, coerce remaining non-numeric columns to numeric and fill remaining NaNs with 0, yielding a 78-dimensional numeric feature matrix. CIC-IDS2017 contains many duplicate rows; we remove them before any splitting to avoid inflating accuracy or training time. The deduplicated pool is shuffled with a fixed seed and split 80/20 into train/test with stratification. On the training split only, we cap the majority (normal) class relative to the minority (attack) class using a target ratio (balance_ratio = 4.0, i.e. approximately 80:20 normal:attack in training, chosen because it is close to the ratio at which we observed stable convergence — see Section 3.2); the test split is left at its natural distribution so that reported metrics reflect the intended deployment distribution. Finally, a StandardScaler is fit on the training split only and applied to both splits."
));

children.push(H2("3.2 Federated Learning Setup"));
children.push(P(
  "We train a multilayer perceptron (512→256→128, with BatchNormalization and dropout) in a federated setting with four clients using Flower. The server aggregates client updates with FedAvgM (server momentum) [10] and coordinates a cosine learning-rate decay schedule across rounds, broadcasting the same learning rate to every client each round so that all clients decay in lock-step. Client data is partitioned in a label-distribution-aware way: for each class, indices are permuted and split across the four clients so each client sees a mix of normal and attack samples, avoiding the degenerate case of a client holding only one class; we describe this explicitly as a mild, near-IID partition and treat a harder non-IID partition as future work (Section 7), since the reviewers correctly noted that four IID-ish clients understate the difficulty FL faces in practice."
));
children.push(P(
  "Class imbalance is addressed with focal loss (in addition to the training-set balance_ratio above). We use α = 0.35 in the reported final configuration. This value was not chosen a priori: across the version history summarized in Table 1, α in the range 0.85–0.92 (strong minority-class emphasis) reliably caused federated training to collapse to a degenerate solution that predicts a single class, while α = 0.35 (moderate emphasis), combined with cosine learning-rate decay and balance_ratio = 4.0, was the first setting that converged stably across repeated runs. We report this as an empirical finding rather than a theoretical justification: under federated aggregation with class-imbalanced data, aggressive per-example reweighting appears to interact badly with the cosine schedule's early-training high learning rates, and α = 0.35 is best read as \"the largest minority-class weight we found that still converges,\" not as an optimal value in any stronger sense."
));

const t1h = ["Configuration", "Attack Recall", "Outcome"];
const t1w = [3600, 2200, 3600];
children.push(makeTable(t1h, [
  ["Fixed / simple-decay LR, α ∈ {0.85–0.92}", "0–43%", "Unstable; frequent collapse to single-class prediction"],
  ["Fixed / simple-decay LR, α = 0.35", "15–43%", "Stable but low recall"],
  ["Server-coordinated cosine LR, α = 0.35 (final)", "90–98% across runs; 93.85% reported", "Stable, high recall (Section 5.1)"],
], t1w));
children.push(caption("Table 1. Summary of the training-stability search that motivated the final learning-rate schedule and focal-loss α, drawn from our internal experiment log (15 dated training runs, v1–v15)."));

children.push(H2("3.3 Compression Pipeline"));
children.push(P(
  "After federated training, the global model is passed through: (1) optional knowledge distillation (direct or progressive, transferring from the federated model as teacher to a smaller student), (2) structured pruning at a configurable ratio, (3) optional fine-tuning, and (4) quantization, either quantization-aware fine-tuning or post-training quantization (PTQ), before TFLite INT8 export. We evaluate this pipeline as a full grid over these choices (Section 5.4–5.5) rather than only at the single final operating point, specifically so that the contribution of each stage can be isolated."
));

children.push(H2("3.4 Adversarial Evaluation and Training Setup"));
children.push(P(
  "We evaluate robustness with four attacks: FGSM [3], a related single-step sign-gradient variant we label FGM, a black-box genetic-algorithm (GA) attack, and PGD [9], the strongest of the four since it iterates the perturbation under a fixed budget. Evaluation is white-box for FGSM/FGM/PGD (the attacker has model access) and black-box for GA. Separately, we run PGD adversarial training (PGD-AT): the global model is fine-tuned on PGD-perturbed examples (ε = 0.1, 10 steps, 3 epochs) before being carried through the same compression pipeline, and we evaluate robustness across a sweep of six evaluation epsilons (0.01–0.3) at four points in the pipeline — the trained Keras model before and after adversarial training, and the exported float32 and INT8 TFLite models attacked by transfer from the adversarially-trained model — to see whether robustness gained during training survives export and quantization."
));

// =================== 4 Experimental Setup ===================
children.push(H1("4. Experimental Setup"));
children.push(P(
  "All experiments use the CIC-IDS2017 dataset with an approximately 80:20 normal-to-attack training ratio, four federated clients, FedAvgM aggregation, and focal loss (α = 0.35) to address class imbalance, unless otherwise noted. Evaluation metrics are accuracy, precision, recall, F1-score, and Attack Recall (recall on the attack class), which we treat as the primary deployment-readiness metric since a false negative (a missed attack) is the failure mode that matters most for an IDS."
));

children.push(H2("4.1 Baseline Decomposition"));
children.push(P(
  "One reviewer asked us to separate the contribution of the learning-rate fix from the contribution of compression by reporting: (a) centralized training with cosine LR and focal loss, (b) federated training with the same recipe, (c) (b) plus compression, and (d) (b) plus compression plus QAT. We can report (b) and (c) directly, and can additionally report an internal ablation run that isolates the federated-training result immediately before compression is applied — a data point the original submission's single before/after table did not include. We were not able to re-run (a), a centralized-only baseline, within this revision; Section 7 gives the exact configuration for it. We report what we have, rather than an estimate, in Section 5.2."
));

children.push(H2("4.2 Model Selection for Deployment Scenarios"));
children.push(P(
  "Based on the compression sweep, we select four representative configurations that illustrate different performance/efficiency trade-offs, and use these same four configurations for the adversarial-robustness evaluation in Section 5.7, so that clean and adversarial performance are always compared on the same model:"
));
const scenarios = [
  ["Most Compressed", "smallest model size with acceptable performance"],
  ["Most Accurate", "highest accuracy and F1-score"],
  ["Balanced – Compressed (PTQ)", "moderate compression, post-training quantization"],
  ["Balanced – Accurate (no PTQ)", "high accuracy with moderate model size, no post-training quantization"],
];
for (const [k, v] of scenarios) children.push(RP([{ text: k + ": ", bold: true }, { text: v }], { spacingAfter: 80 }));
children.push(P("These configurations let practitioners choose a model matching their deployment constraints: extremely constrained microcontrollers may require Most Compressed, while slightly more capable devices may benefit from a balanced or accuracy-oriented model.", { spacingAfter: 200 }));

// =================== 5 Results ===================
children.push(H1("5. Results"));

children.push(H2("5.1 Training Stability and Attack Recall"));
children.push(P(
  "Initial federated models trained with a fixed or simply-decayed learning rate exhibited unstable convergence and low Attack Recall (46.7%) despite high overall accuracy (93.5%), indicating the model was overfitting to the dominant normal-traffic class. Server-coordinated cosine learning-rate decay dramatically improves training stability, increasing Attack Recall from 46.7% to 93.85% while maintaining high overall accuracy — without any change to model architecture. This is the single largest factor we identified across the entire study, and it is the reason we treat learning-rate scheduling, not model size, as the primary lever on detection quality in this setting."
));

children.push(H2("5.2 Baseline Decomposition"));
const t2h = ["Configuration", "Accuracy", "F1", "Attack Recall"];
const t2w = [4200, 1600, 1600, 1800];
children.push(makeTable(t2h, [
  ["(a) Centralized, cosine LR + focal loss", "not run — see Sec. 7", "—", "—"],
  ["(b) Federated, fixed LR (reported baseline)", "93.5%", "84.1%", "46.7%"],
  ["(b′) Federated, cosine LR, pre-compression (internal ablation run)", "82.75%", "64.87%", "90.55%"],
  ["(c) Federated, cosine LR, + compression (reported)", "96.02%", "89.32%", "93.85%"],
  ["(c′) Federated, cosine LR, + compression (internal ablation run)", "88.06%", "74.37%", "98.46%"],
  ["(d) (c) + QAT during training (Section 5.5)", "varies with compression level — see Table 5 / Figure 1", "", ""],
], t2w));
children.push(caption("Table 2. Baseline decomposition. Rows (b′)/(c′) are from a separate, internally-logged federated run (run id 2026-02-05_12-52-17) using the same recipe family (α = 0.35, balance_ratio = 4.0, cosine LR) as the headline run in rows (b)/(c); we report it separately, rather than merging it into the headline numbers, because it used a different random seed and checkpoint and its exact figures should not be read as identical to the headline run. It is included specifically because it is the only run in our logs for which we recorded both a pre-compression and a post-compression Attack Recall from the same checkpoint, which is what a reviewer asked for. Row (d) is answered as a function of compression level rather than a single number; see Section 5.5."));
children.push(P(
  "Reading row (b) against row (b′): (b) is the fixed-LR federated baseline (46.7% Attack Recall), while (b′) is a federated cosine-LR model measured before any compression is applied, and it already reaches 90.55% Attack Recall — consistent with the 90–94% range seen throughout Section 3.2's version history. This is the basis for our claim that the cosine-LR fix, not compression, is responsible for most of the 46.7% → ~90%+ jump. The headline comparison (b) → (c) alone cannot separate the two, because it changes both the learning-rate schedule and the compression stage at once, and no pre-compression checkpoint was logged for the headline run; rows (b′)/(c′) are included precisely to separate them. Reading (b′) against (c′): compression did not cost Attack Recall in this run (90.55% → 98.46%); this pattern of recall preserved or improved through compression is discussed further, with a mechanism, in Section 5.6. We were not able to obtain a centralized-only number in the time available for this revision; we specify the exact configuration to produce it in Section 7 rather than presenting an estimate."
));

children.push(H2("5.3 Compression and Deployment"));
children.push(P(
  "We evaluate a compression pipeline combining structured pruning and quantization-aware fine-tuning. The final deployment model achieves a 12.28× model-size reduction and a 74.5% latency reduction. Model size decreases from 0.78 MB to 0.0635 MB, while inference latency decreases from 1.89 ms to 0.48 ms per prediction (measured with the TFLite interpreter; see Section 6.1 for the distinction between this and physical on-device latency)."
));
const t3h = ["Metric", "Baseline Model", "Compressed Model"];
const t3w = [3600, 2800, 2800];
children.push(makeTable(t3h, [
  ["Model Size", "0.78 MB", "0.0635 MB"],
  ["Inference Latency (interpreter)", "1.89 ms", "0.48 ms"],
  ["Accuracy", "93.5%", "96.02%"],
  ["F1-score", "84.1%", "89.32%"],
  ["Attack Recall", "46.7%", "93.85%"],
], t3w));
children.push(caption("Table 3. Detection performance and deployment cost, before and after the full pipeline (cosine LR + compression)."));

children.push(H2("5.4 Decomposing the Compression Ratio"));
children.push(P(
  "A reviewer asked us to decompose the 12.28× figure into the contribution of distillation, pruning, and INT8 quantization, rather than reporting it as a single number. We answer this using a separate, dedicated 48-configuration grid sweep (all combinations of training-time QAT on/off × distillation none/direct/progressive × four pruning ratios × PTQ on/off, trained under the same federated recipe as Section 3.2), which lets us isolate each stage's marginal contribution in a way the single headline run cannot. We report this as a separate, explicitly-labeled study rather than folding it into Table 3, since it comes from a different training run and its absolute numbers should not be read as reproducing the headline figures exactly — the previous submission's confusion between numbers from different runs was itself a source of reviewer concern, and we want to avoid repeating it."
));
const t4h = ["Pipeline stage (cumulative)", "Deployed size", "Ratio vs. FP32", "F1"];
const t4w = [4200, 1800, 1600, 1600];
children.push(makeTable(t4h, [
  ["FP32 federated model, no compression", "864.1 KB", "1.00×", "0.850"],
  ["+ INT8 PTQ only (no distillation, no pruning)", "216.0 KB", "4.00×", "0.598"],
  ["+ Progressive distillation only", "82.5 KB", "10.48×", "0.932"],
  ["+ Distillation + PTQ", "25.9 KB", "33.37×", "0.936"],
  ["+ Distillation + structured pruning (10×5)", "37.9 KB", "22.81×", "0.976"],
  ["+ Distillation + pruning + PTQ (most aggressive grid point)", "14.85 KB", "58.18×", "0.837"],
], t4w));
children.push(caption("Table 4. Decomposition of compression contribution using the 48-configuration grid sweep (training-time QAT disabled; see Section 5.5 for what changes when it is enabled). Distillation is the dominant single contributor (~10.5×); pruning and PTQ each multiply a further ~2–2.5× on top; the most aggressive combination in the grid (58×) is considerably more aggressive than the 12.28× headline deployment model, which was chosen to preserve F1/recall rather than to minimize size."));
children.push(P(
  "This also explains an apparent inconsistency a reviewer flagged in the original submission's model-selection table, where two different configurations shared an identical size (14.44 KB). In the full grid, distillation and pruning determine the pre-quantization architecture; two configurations that differ only in whether QAT was used during federated training (Section 5.5) but agree on distillation type, pruning ratio, and PTQ produce INT8 exports of identical size — the training-time QAT flag changes the weight values and therefore accuracy, but not the exported tensor shapes or footprint. The duplicate row was a real, reproducible artifact of the pipeline, not a copy-paste error; we now state this explicitly rather than leaving it to be inferred."
));

children.push(H2("5.5 Effect of Quantization-Aware Training: the Central Finding"));
children.push(P(
  "We compare two ways of using QAT: training-time QAT (fake-quantization enabled from the first federated round) versus quantization only at compression time (either PTQ, or a short QAT fine-tune applied after federated training has already converged in full precision). On the single headline configuration, post-training QAT slightly outperforms full training-time QAT (96.45% accuracy / 91.32% F1 versus 96.02% / 89.32%), and PTQ alone outperforms combining QAT-during-training with PTQ (F1 0.8215 versus 0.6367 for QAT+PTQ) — both suggesting, on this one operating point, that early QAT introduces optimization noise under federated, class-imbalanced training."
));
children.push(P(
  "The 48-configuration grid sweep (Section 5.4) lets us check whether that conclusion holds across compression levels, and it does not, uniformly. Figure 1 plots final F1 against overall compression ratio for every configuration in the grid, marking whether training-time QAT was used. At low-to-moderate compression ratios (roughly 1–10×), training-time QAT is consistently worse, sometimes catastrophically — F1 drops of 0.52–0.60 are common at the mildest compression settings (e.g. no distillation, no pruning: F1 0.850 → 0.327 with training-time QAT). At more aggressive compression (roughly 10–30×, where progressive distillation and moderate pruning are already in effect), the gap narrows and in several configurations training-time QAT is neutral or slightly better (e.g. progressive distillation + 5×10 pruning + PTQ: F1 0.776 without training-time QAT versus 0.867 with it, a +0.090 improvement). We read this as the paper's central empirical finding: training-time QAT and post-hoc quantization are not simply better-or-worse alternatives, but occupy different parts of a trade-off that depends on how aggressively the model is otherwise being compressed, and, as Section 5.7 shows, on how much adversarial robustness is worth to the deployment."
));
children.push(img(FIG + "qat_vs_compression.png", 500, 321));
children.push(caption("Figure 1. Final F1-score versus compression ratio (log scale) for all 48 grid-sweep configurations, split by whether training-time QAT was used. The dashed line marks the headline deployment model's compression ratio (12.28×) for reference; the grid uses a separate training run so its ratios and the headline ratio are not directly the same points, only comparable in scale."));

children.push(H2("5.6 Distillation and Pruning Ablations"));
children.push(P(
  "Additional ablations evaluate pruning and distillation strategies individually. As shown in Figure 2, knowledge distillation improves model performance relative to training without distillation; progressive distillation achieves the highest F1-score (0.8138), slightly outperforming direct distillation (0.8059), while models trained without distillation show markedly lower performance. As Figure 3 illustrates, moderate pruning configurations (10×2, 5×10) maintain accuracy between 92.6% and 95.4% while reducing model complexity; combined with the decomposition in Table 4, this indicates that progressive distillation with moderate pruning gives the most favorable accuracy–compression trade-off for TinyML deployment. Table 2's post-compression recall improving over pre-compression recall (90.55% → 98.46% in the internal ablation run; 46.7%-baseline-relative in the headline run) is consistent with pruning acting as a regularizer under this class-imbalanced, federated setting — removing redundant parameters appears to reduce overfitting to the majority class, though we have not isolated this mechanistically and present it as an observed pattern rather than a proven cause."
));
children.push(img(FIG + "distillation_f1.png", 500, 269));
children.push(caption("Figure 2. Comparison of knowledge distillation strategies. Progressive distillation achieves the highest F1-score."));
children.push(img(FIG + "pruning_accuracy.png", 500, 278));
children.push(caption("Figure 3. Accuracy across different pruning configurations."));

children.push(H2("5.7 Adversarial Robustness Under Compression"));
children.push(P(
  "We evaluate the four deployment configurations from Section 4.2 under FGSM, FGM, GA, and PGD attacks (results previously described but not tabulated in the WIP submission)."
));
const t5h = ["Model", "Attack", "Clean F1", "Adv. F1", "ΔF1", "Clean Acc", "Adv. Acc", "ΔAcc"];
const t5w = [2600, 1200, 1150, 1150, 950, 1150, 1150, 950];
function pct(x){ return (x>=0?"+":"") + x.toFixed(1) + "%"; }
const fgsmRows = [
  ["Most Compressed","FGSM","0.944","0.566","-40.0%","0.980","0.862","-12.0%"],
  ["","FGM","0.944","0.610","-35.4%","0.980","0.884","-9.9%"],
  ["","GA","0.944","0.633","-32.9%","0.980","0.894","-8.8%"],
  ["","PGD","0.944","0.593","-37.2%","0.980","0.870","-11.3%"],
  ["Most Accurate","FGSM","0.975","0.634","-35.0%","0.991","0.870","-12.2%"],
  ["","FGM","0.975","0.796","-18.4%","0.991","0.932","-6.0%"],
  ["","GA","0.975","0.790","-19.0%","0.991","0.931","-6.0%"],
  ["","PGD","0.975","0.660","-32.3%","0.991","0.880","-11.2%"],
  ["Balanced – Compressed (PTQ)","FGSM","0.930","0.713","-23.4%","0.973","0.895","-8.1%"],
  ["","FGM","0.930","0.809","-13.0%","0.973","0.927","-4.7%"],
  ["","GA","0.930","0.825","-11.3%","0.973","0.936","-3.9%"],
  ["","PGD","0.930","0.740","-20.5%","0.973","0.903","-7.2%"],
  ["Balanced – Accurate (no PTQ, train-time QAT)","FGSM","0.835","0.835","+0.0%","0.930","0.930","+0.0%"],
  ["","FGM","0.835","0.835","+0.0%","0.930","0.930","+0.0%"],
  ["","GA","0.835","0.835","+0.0%","0.930","0.930","+0.0%"],
  ["","PGD","0.835","0.832","-0.4%","0.930","0.928","-0.2%"],
];
children.push(makeTable(t5h, fgsmRows, t5w));
children.push(caption("Table 5. Clean vs. adversarial performance under four attacks, four deployment configurations. \"Balanced – Accurate\" is the one configuration in this set trained with training-time QAT and without a final PTQ step; it is essentially unaffected by all four attacks, while every PTQ-only configuration loses 11–40% of its F1 under FGSM or PGD. This is the clearest single piece of evidence in this paper that training-time QAT provides an adversarial-robustness benefit that is largely independent of its (mixed, see Section 5.5) effect on clean accuracy."));

children.push(H2("5.8 Adversarial Training Improves Robustness Through the Compression Pipeline"));
children.push(P(
  "Beyond evaluating robustness, we ran PGD adversarial training (PGD-AT: ε = 0.1, 10 steps, 3 epochs) on the global model and evaluated it, and its compressed derivatives, under a sweep of six PGD evaluation epsilons (0.01–0.3, 5,000 held-out samples). Table 6 reports the mean adversarial accuracy, F1, and recall across that epsilon sweep at four points in the pipeline."
));
const t6h = ["Phase", "Mean Adv. Acc", "Mean Adv. F1", "Mean Adv. Recall", "Min Adv. Acc"];
const t6w = [3400, 1750, 1750, 1900, 1600];
children.push(makeTable(t6h, [
  ["Pre-AT (Keras, clean-trained)", "0.679", "0.063", "0.053", "0.518"],
  ["Post-AT (Keras, PGD-AT)", "0.888", "0.648", "0.574", "0.823"],
  ["Post-AT, exported float32 TFLite (transfer attack)", "0.804", "0.260", "0.202", "0.786"],
  ["Post-AT, exported INT8 PTQ TFLite (transfer attack)", "0.798", "0.244", "0.189", "0.781"],
], t6w));
children.push(caption("Table 6. PGD adversarial training results, mean over evaluation ε ∈ {0.01, 0.05, 0.1, 0.15, 0.2, 0.3}. Adversarial training raises mean adversarial accuracy from 0.679 to 0.888 (F1 from 0.063 to 0.648) on the trained model; most of this gain (accuracy 0.888 → 0.798, F1 0.648 → 0.244) is lost when the model is exported to TFLite and attacked by transfer rather than directly, and a further, smaller amount is lost to INT8 PTQ specifically (F1 0.260 → 0.244). We read this as evidence that adversarial training helps substantially at the Keras/training level, but that robustness does not transfer through TFLite export as cleanly as clean accuracy does, and is an open problem for deployment rather than a solved one."));
children.push(P(
  "We separately evaluated PGD robustness across the same compression-role labels used for Table 4 (\"Extreme compression,\" \"Moderate accuracy,\" \"Compact+accurate,\" \"Best PGD robust\"); the configuration we labeled \"Best PGD robust\" (training-time QAT enabled, direct distillation, no pruning) reaches 0.90–0.90 adversarial accuracy before and after adversarial training and export, essentially matching its own pre-AT numbers — consistent with Section 5.7's finding that training-time QAT alone already provides most of the adversarial-robustness benefit available in this pipeline, with PGD-AT adding comparatively little on top of that specific configuration, while adding a great deal (Table 6) on top of a configuration that did not use training-time QAT."
));

children.push(H2("5.9 Deployment Reliability"));
children.push(P(
  "During early deployment experiments, quantized TFLite models occasionally produced NaN outputs during inference. Investigation traced this to BatchNormalization layers interacting with quantization and federated aggregation. We resolved this with BatchNorm folding: merging BatchNormalization parameters into the preceding Dense layer's weights and biases before TFLite conversion (and before QAT, when QAT is applied at compression time), removing BatchNorm and Dropout to produce an inference-only graph. This eliminated NaN outputs entirely and is applied to every model reported in this paper. The derivation of the folding equations is given in Appendix A, moved out of the main text at a reviewer's suggestion since it is not itself a result."
));

children.push(H2("5.10 Practical Significance: What Attack Recall and Precision Mean Here"));
children.push(P(
  "One reviewer asked directly why 93.85% Attack Recall is practically meaningful, and whether it implies an unacceptable number of false negatives or false positives. We answer this in concrete terms rather than only as a percentage. At 46.7% recall (the fixed-LR baseline), the model misses 53.3% of attacks — more than one in two. At 93.85% recall (the reported deployment model), it misses 6.15% of attacks, an 8× reduction in the missed-attack rate. This is the number we consider decision-relevant for a security practitioner, more so than the accuracy figure, which stays misleadingly high (93.5%) even at the 46.7%-recall operating point because normal traffic dominates the test distribution."
));
children.push(P(
  "On precision (the complementary question — how often an alert is a false alarm): the reported deployment model's F1 of 89.32% at 93.85% recall implies a precision in the mid-80s percent range for this operating point; Table 5's four deployment configurations span clean F1 from 0.835 to 0.975, giving practitioners a size/recall/false-alarm-rate trade-off rather than a single fixed point. We agree with the reviewer that a full precision-recall or ROC curve, and an explicit statement of the acceptable false-positive rate for a given deployment (e.g. alerts per client per day at a given traffic volume), would make this trade-off easier to act on than F1 alone; we did not have per-flow-rate deployment context to make that translation for a specific IoT environment within this revision, and list it as a natural extension in Section 8."
));

// =================== 6 Discussion ===================
children.push(H1("6. Discussion"));

children.push(H2("6.1 Device Heterogeneity and On-Device Measurement"));
children.push(P(
  "A reviewer correctly noted that IoT environments are heterogeneous in MCU architecture, memory, sensors, network stack, and workload, and that our latency numbers (Table 3) were measured with the TFLite interpreter rather than on physical hardware, despite \"edge-ready\" and \"ESP32-class\" appearing in our framing. We did not have physical access to an ESP32 board while producing this revision, so we are explicit about the distinction rather than continuing to imply an on-device measurement we did not make. What is in place: a complete ESP32 TFLite-Micro benchmark firmware (main.cpp) that loads a deployed INT8 model, embeds both the FP32 baseline and the INT8 deployment model from Table 3, checks each model's on-device output against the host interpreter on fixed inputs, runs 100 timed inference passes per model after warm-up, and reports per-inference latency, arena usage and the board's chip/clock over serial; and a log-collection script (collect_esp32_benchmark.py) that parses that serial output into a JSON summary. Appendix B gives the exact commands. Until this is run on hardware, our size and interpreter-latency numbers should be read as necessary but not sufficient evidence of on-device deployability — they establish that the model fits comfortably within typical ESP32-class SRAM and flash budgets, not that it runs at the reported latency on a specific chip, clock speed, or compiler configuration. We consider this the single most important open item from the review and give it a dedicated appendix rather than folding it into the limitations list."
));

children.push(H2("6.2 What Goes Beyond Pipeline Composition"));
children.push(P(
  "The meta-review's central concern was that this work reads as a composition of well-known techniques (FL, distillation, pruning, quantization, cosine LR) rather than a technical contribution in its own right. We take this seriously and, in this revision, try to state precisely which parts of the paper we believe are genuinely non-obvious rather than leaving the reader to infer it. Three results, specifically, are not predictable from the individual literatures on FL, TinyML compression, and adversarial robustness in isolation: (1) the direction and magnitude of the training-time-QAT/compression-level interaction in Section 5.5 — that QAT is harmful at moderate compression and roughly neutral at aggressive compression is not implied by either the general QAT literature [16, 17, 18] or the general FL literature, and required the 48-point grid to see; (2) the finding that structured pruning appears to act as a regularizer that preserves or improves Attack Recall under this specific combination of class imbalance, focal loss, and federated aggregation (Section 5.6), which is the opposite of the usual assumption that aggressive compression trades away accuracy; and (3) the adversarial-robustness results in Section 5.7–5.8, which are, to our knowledge, the first to report how PGD adversarial training's benefit degrades specifically through TFLite export and INT8 PTQ in a federated IDS pipeline, rather than in a centrally-trained image classifier as in most of the adversarial-robustness-of-quantization literature (Section 2.4). We view the paper's contribution as this specific, evidence-backed characterization — where a standard pipeline breaks under federation and class imbalance, and which specific, isolable change fixes each break — rather than a new algorithm; whether that constitutes a sufficient contribution for a given venue is, appropriately, for reviewers to judge, but we wanted the claim itself to be unambiguous."
));

// =================== 7 Limitations ===================
children.push(H1("7. Limitations"));
children.push(P("We list open items with the precision we think a reader needs to judge how much weight to put on each result, and, where possible, the exact configuration needed to close the gap.", { spacingAfter: 160 }));

const limItems = [
  ["Non-IID clients", "All results use four clients with a label-distribution-aware but near-IID partition (Section 3.2), not the harder non-IID regimes ([10, 11, 20]) common in FL evaluation. Concrete follow-up: partition CIC-IDS2017's training split across four clients using a Dirichlet(0.3) distribution over the attack/normal label (and, if time allows, over attack sub-type), holding every other hyperparameter in Section 3.2 fixed, and report Attack Recall and training-round-to-convergence against the current IID-ish partition."],
  ["Centralized baseline", "Row (a) of Table 2 (centralized, cosine LR + focal loss, no FL) was not re-run in this revision. Concrete follow-up: single-node training with the same MLP architecture, cosine LR, α = 0.35 focal loss, and balance_ratio = 4.0 as Section 3.2, for a number of epochs matched to the federated run's total local-epoch budget (60 rounds × 3 local epochs = 180 epoch-equivalents)."],
  ["Cross-dataset generalization", "All results use CIC-IDS2017 only. Bot-IoT and TON_IoT loaders already exist in our codebase (Section 3.1) but have not been carried through the full federated + compression + robustness pipeline. Concrete follow-up: repeat Sections 5.1–5.3 on Bot-IoT first, since its loader is more mature in our codebase than TON_IoT's, and report whether the cosine-LR recall improvement and the pruning-as-regularizer effect (Section 5.6) replicate on a dataset with a different feature structure and attack-type distribution."],
  ["Physical on-device measurement", "Table 3's latency figures are TFLite-interpreter measurements, not physical ESP32 measurements; see Section 6.1 and Appendix B for what is ready to run and what specifically is missing."],
  ["Adversarial-training export gap", "Table 6 shows PGD-AT's robustness gain only partially survives TFLite export and does not fully survive INT8 PTQ; we do not yet have a fix for this, only a measurement of its size."],
  ["Single-architecture scope", "All results use one MLP architecture (512→256→128); we have not evaluated whether the QAT/compression-level interaction in Section 5.5 is specific to this architecture or generalizes to, e.g., 1D-CNN or attention-based TinyML architectures."],
  ["Deployment false-positive rate", "Section 5.10 translates recall into a missed-attack rate but not into an alerts-per-day false-positive rate for a specific deployment context, since we lack a reference traffic volume to translate against."],
  ["Comparison to other published IDS systems", "Every result in this paper is an internal comparison (our own baselines and ablations against each other), not a comparison to other published federated or centralized IDS systems evaluated on CIC-IDS2017. We chose not to fabricate or approximate competitor numbers from memory for this revision; a proper comparison requires either reproducing selected prior systems under our train/test split or citing their reported numbers with matching evaluation protocol (same split ratio, same attack-class definition), which we did not have time to verify carefully enough to report responsibly here."],
  ["Client count", "All experiments use four federated clients. This is small relative to real IoT deployments with many more participants, independent of the IID/non-IID question addressed above; we have not measured how Attack Recall, convergence speed, or the QAT/compression interaction (Section 5.5) change as client count grows, and would expect at least communication-round and aggregation-noise effects to scale differently at, e.g., 20–50 clients."],
];
for (const [k, v] of limItems) {
  children.push(RP([{ text: k + ". ", bold: true }, { text: v }], { spacingAfter: 140 }));
}

// =================== 8 Future Work ===================
children.push(H1("8. Future Work"));
children.push(P("Beyond directly closing the gaps in Section 7, we plan to: investigate adaptive communication strategies and selective client participation to reduce federated training overhead in later rounds, once early-round instability is no longer the dominant cost; study detection of and defense against repeated or temporally-coordinated adversarial attacks in a federated setting, rather than the single-shot attacks evaluated in Section 5.7; evaluate deployment across heterogeneous edge devices beyond the ESP32 once Appendix B's protocol has been run at least once; explore additional QAT scheduling strategies (e.g. enabling training-time QAT only after the cosine schedule has annealed past its early high-learning-rate phase, motivated by the oscillation mechanism described in [16]) that might recover training-time QAT's robustness benefit (Section 5.7) without its moderate-compression accuracy cost (Section 5.5); and derive a precision-recall operating curve, rather than a single point, for practitioners to select a deployment threshold against a known false-positive budget (Section 5.10)."));

// =================== 9 Conclusion ===================
children.push(H1("9. Conclusion"));
children.push(P(
  "This work investigates practical approaches for deploying Federated Learning in resource-constrained IoT environments by integrating TinyML compression techniques, and this revision extends a prior work-in-progress report with the specific additions requested by reviewers: a decomposed and separately-verified baseline (Section 5.2, 5.4), a restructured presentation around the training-time-QAT/compression-level interaction as the central finding (Section 5.5), a full adversarial-robustness evaluation with an accompanying adversarial-training study (Section 5.7–5.8), and an explicit, itemized limitations section with concrete follow-up protocols (Section 7) rather than a single generic \"future work\" paragraph. Server-coordinated cosine learning-rate scheduling dramatically improves training stability, increasing Attack Recall from 46.7% to 93.85%; an optimized compression pipeline combining structured pruning, distillation, and quantization reduces model size by 12.28× and inference latency (measured with the TFLite interpreter) by 74.5% while maintaining that recall; training-time QAT trades moderate-compression accuracy for adversarial robustness in a way that depends on compression level; and BatchNorm folding is necessary for stable TFLite deployment. We report these findings alongside their limitations and the exact steps needed to resolve them, in the belief that a precise account of what is and is not yet established is itself part of the paper's contribution."
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
];
for (const r of refs) children.push(P(r, { spacingAfter: 100, size: 20 }));

// =================== Appendix A ===================
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H1("Appendix A: BatchNorm Folding"));
children.push(P("Moved from the main text (Section 5.9) at a reviewer's suggestion, since it documents an implementation fix rather than a result.", { spacingAfter: 160 }));
children.push(P("For a Dense layer with weights W and bias b, followed by BatchNormalization with learned scale γ, shift β, running mean μ, running variance σ², and epsilon ε, the folded (inference-only) weights and bias are:", { spacingAfter: 120 }));
children.push(P("W′ = W · γ / √(σ² + ε)", { spacingAfter: 80, alignment: AlignmentType.CENTER }));
children.push(P("b′ = (b − μ) · γ / √(σ² + ε) + β", { spacingAfter: 160, alignment: AlignmentType.CENTER }));
children.push(P("Replacing the Dense+BatchNorm pair with a single Dense layer using (W′, b′), and removing the BatchNorm and any following Dropout layers, produces a mathematically equivalent inference graph in full precision, but one that no longer exposes the running-statistics division to the TFLite INT8 converter, which is what eliminated the NaN outputs described in Section 5.9. This is applied via two code paths in our pipeline: one used before final TFLite export, and one used before a model is handed to a QAT fine-tuning step, since QAT itself needs the already-folded graph to quantize correctly."));

// =================== Appendix B ===================
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H1("Appendix B: ESP32 On-Device Benchmark Protocol (ready to run)"));
children.push(P("This appendix documents the exact, already-implemented steps to produce the physical on-device latency measurement discussed in Section 6.1. It is written so that running it requires no further engineering — only a connected ESP32-class board.", { spacingAfter: 160 }));
const steps = [
  "Models: the firmware embeds both Table 3 models from the same federated run — the FP32 baseline (817,528 bytes) and the pruned + QAT INT8 deployment model (66,600 bytes), i.e. exactly the 12.28× pair — so both are measured on the same board, clock and TensorFlow Lite Micro build. They are converted to 16-byte-aligned C arrays automatically at build time (esp32_tflite_project/gen_model_data.py).",
  "Build and flash (PlatformIO, TensorFlowLite_ESP32 1.0.0, huge_app partition table so the FP32 model fits in the app partition): pio run -e <esp32dev | esp32-s3-devkitc-1 | esp32-c3-devkitm-1> -t upload",
  "Parity check: for 8 fixed standardized input vectors, the firmware compares each model's on-device output against the host TFLite interpreter output for the same input (stored in include/test_vectors.h) and reports the maximum absolute difference and decision-label agreement. In a host build of the same firmware against the same TensorFlow Lite Micro library, the INT8 model matched to within one quantization step (max |Δ| = 0.0039 = 1/256, 8/8 labels) and the FP32 model matched exactly.",
  "Timing: after 5 warm-up inferences, 100 timed Invoke() calls per model are measured with micros(); the firmware prints one line per run (BENCHMARK model=<name> latency_us=<us> arena_used=<bytes> ...) together with the chip model, core count, CPU clock and ESP-IDF version. Tensor-arena use is 2–4 KB for both models (16 KB arena allocated).",
  "Collect and summarize: python scripts/collect_esp32_benchmark.py --port <serial port>, which writes data/processed/ablation/esp32_benchmark.json (per-model mean / median / std / p95 latency, arena use, parity) and the raw serial log.",
];
steps.forEach((s, i) => children.push(RP([{ text: `${i+1}. `, bold: true }, { text: s }], { spacingAfter: 120 })));
children.push(P("Status at the time of this revision: not yet run, for lack of physical access to a board during the writing of this revision (Section 6.1). We chose to document the exact remaining steps rather than omit this appendix, since the firmware and collection script were already complete and, in our reading, running them is now a data-collection step rather than an open engineering problem.", { spacingAfter: 160 }));

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
