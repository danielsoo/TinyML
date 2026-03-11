# 논문용: 전체 과정 나열 (데이터·전처리·학습·수정·변경)

아래 내용을 논문 해당 섹션에 복사·붙여넣기 후, 문맥에 맞게 문장만 다듬어 사용하세요. 숫자(rounds, alpha, ratio 등)는 실제 실험 설정과 일치하는지 확인해 수정하세요.

---

## 1. 데이터셋 선택 및 사용 이유 (Introduction 또는 5.1 앞단에 넣을 문단)

**Dataset selection.** We use the CIC-IDS2017 dataset [citation] for all federated and compression experiments. CIC-IDS2017 provides labeled network traffic (BENIGN and multiple attack types) collected in a controlled environment, is widely used in intrusion detection research, and offers a sufficient number of samples and features (78 after preprocessing) for training MLP models while remaining manageable for federated and compression pipelines.

**Why not other datasets.** We did not use Bot-IoT, TON_IoT, or MNIST for the main results for the following reasons. (1) **Bot-IoT**: We implemented and tested a Bot-IoT loader (with IP encoding and categorical feature handling) for compatibility and sanity checks; however, our primary focus was on CIC-IDS2017 because of its broader adoption in IDS benchmarks and its different feature structure (e.g., 38 features in our Bot-IoT setup vs 78 for CIC-IDS2017), which would require separate hyperparameter and compression sweeps. (2) **TON_IoT**: The TON_IoT dataset (UNSW Canberra) was integrated in the data loader for potential future work on heterogeneous IoT traffic; we did not run full federated or compression experiments on it to keep the experimental scope consistent and comparable. (3) **MNIST**: MNIST is supported only for quick pipeline tests (e.g., input shape and FL connectivity); it is not an intrusion detection dataset and was not used for any reported accuracy or compression results.

**Summary.** All reported results in this paper (training stability, Attack Recall, compression ratios, and deployment reliability) use CIC-IDS2017 with the preprocessing and training configuration described below.

---

## 2. 데이터 전처리 (전체 과정 나열)

**Loading and label handling.** Raw data is loaded from CIC-IDS2017 CSV files (`.pcap_ISCX.csv`) under a single data directory. The label column is the last column (nominally " Label"); we strip leading/trailing spaces from column names. Labels are converted to binary: BENIGN → 0, all other attack types → 1, unless multi-class mode is explicitly enabled (we use binary for all reported results).

**Missing and invalid values.** We replace positive and negative infinity with NaN, drop columns that are entirely NaN, and fill remaining NaN values with 0. Non-numeric columns are converted to numeric with `pd.to_numeric(..., errors='coerce')` and then filled with 0; columns that cannot be converted are dropped. This yields a numeric feature matrix of 78 dimensions for CIC-IDS2017.

**Duplicate removal.** CIC-IDS2017 contains many duplicate rows (same feature vector and label). We remove duplicates by constructing a temporary DataFrame of (features, label) and calling `drop_duplicates()`, then report the number of rows removed (e.g., "Removed N duplicates (M unique)"). All subsequent steps (shuffling, splitting, balancing) operate on this deduplicated set to avoid inflating accuracy and to reduce training time.

**Shuffling and sampling.** The full deduplicated pool is shuffled with a fixed random seed (e.g., 42) so that train/test splits and mini-batches are not biased by file order. If a global sample cap (`max_samples`) is set, we either (a) sample to a target normal-to-attack ratio (e.g., 80:20 via `balance_ratio`) or (b) perform a stratified train split to the cap when no ratio is specified.

**Train/test split.** We use an 80/20 train/test split with stratification when both classes have at least two samples (`stratify=y`), and a fixed random state for reproducibility.

**Class imbalance (training set only).** For binary classification we optionally cap the majority class size relative to the minority using `balance_ratio` (e.g., 4.0 or 10.0), meaning the training set has at most `balance_ratio` times as many normal samples as attack samples (e.g., 80:20 normal-to-attack). This undersampling is applied only to the training set; the test set is left unchanged to reflect the intended evaluation distribution. Optionally, we also support SMOTE (oversampling of the minority class on the training set only); when used, we report it in the configuration.

**Feature scaling.** We fit a StandardScaler on the training set and apply it to both training and test features (no information from the test set is used in scaling). This improves training stability and convergence.

**Summary of preprocessing order.** (1) Load CSVs → (2) extract labels, clean column names → (3) handle inf/NaN and non-numeric columns → (4) remove duplicates → (5) shuffle → (6) optional cap with ratio or stratified sample → (7) train/test split → (8) optional balance_ratio undersampling on train → (9) optional SMOTE on train → (10) StandardScaler fit on train, transform train and test.

---

## 3. 연방 학습 설정 및 변경 이력

**Baseline FL setup.** We use Flower (flwr) with a server-coordinated strategy. The aggregation algorithm is FedAvg with server-side momentum (FedAvgM): server momentum 0.5 (or 0.9 in some configs), server learning rate 1.0. There are four clients; each client receives a partition of the (preprocessed) training data. Data is partitioned in a label-distribution-aware way (non-IID by label): for each class, indices are permuted and split across clients so that each client sees a mix of normal and attack samples, reducing the risk of one client holding only one class.

**Training hyperparameters (representative).** Local training: batch size 128, local epochs 2 per round (or 15 in quick-test configs), initial learning rate 0.0005 (or 0.001). Number of federated rounds: 80 for full runs (or 3 for quick tests). Class imbalance is addressed with (1) class weights (`use_class_weights: true`) and (2) focal loss with α = 0.7 (or 0.75 in some configs). The model is an MLP (architecture and layer sizes as in the codebase/config).

**Learning rate schedule (key change).** Initially, we used a fixed learning rate (or simple step/exponential decay), which led to unstable convergence and very low Attack Recall (46.7%). We then introduced server-coordinated Cosine learning rate decay: the learning rate follows a cosine schedule from the initial value down to a minimum (e.g., 0.0001) over the federated rounds. This single change dramatically improved Attack Recall to 93.85% and stabilized training. All reported results use this Cosine LR schedule unless stated otherwise.

**QAT vs non-QAT training.** We compare two training modes: (1) **QAT-trained**: quantization-aware training is enabled from the first federated round (`use_qat: true`); clients train with fake quantization. (2) **Traditional (non-QAT)**: federated training is performed in full precision (`use_qat: false`); quantization is applied only later during the compression pipeline (QAT fine-tuning or PTQ). The server strategy is QAT-aware when `use_qat` is true (e.g., handling float32 aggregation and optional re-quantization for sending).

**Config changes over time (abridged).** (a) `num_rounds`: increased from a small value (e.g., 3) to 80 for full experiments. (b) `local_epochs`: reduced to 2 per round in main configs to avoid client drift and NaN. (c) `max_samples`: capped (e.g., 2M) to avoid memory issues and NaN when using the full dataset with SMOTE. (d) `balance_ratio`: set to 4.0 or 10.0 to achieve approximately 80:20 normal-to-attack in training. (e) `focal_loss_alpha`: tuned (e.g., 0.7) for better Attack Recall. (f) `lr_decay_type`: set to `cosine` with `lr_min`; other decay types (exponential, step) were tried before settling on cosine.

---

## 4. 압축 파이프라인 및 배포 수정

**Compression pipeline order.** After federated training, the global model is passed through: (1) optional knowledge distillation (progressive or direct, or none), (2) structured pruning (e.g., 50% or configurable ratios such as 10×5, 5×10), (3) optional fine-tuning, (4) quantization (QAT fine-tuning or PTQ), (5) TFLite INT8 export.

**BatchNorm and TFLite NaN (critical fix).** During early deployment, the quantized TFLite model sometimes produced NaN outputs on the target device. We identified that BatchNormalization layers, when combined with quantization and federated aggregation, were causing numerical instability during TFLite conversion. We implemented BatchNorm folding: before TFLite conversion (and before QAT when applying QAT at compression), we merge each BatchNorm layer into the preceding Dense layer (absorbing scale and shift into the Dense weights and biases) and remove BatchNorm and Dropout layers to produce an inference-only graph. This is done in two code paths: `_strip_bn_dropout_for_tflite` (for final export) and `_strip_bn_dropout_for_qat` (for models that will be passed to QAT). After this change, NaN outputs were eliminated and inference became stable.

**QAT at compression.** When the federated model was trained without QAT (traditional), we optionally apply QAT only during the compression phase: we strip BatchNorm/Dropout, wrap the model with `tfmot.quantization.keras.quantize_model`, fine-tune for a short number of epochs, then export to TFLite INT8. We compare this "post-training QAT" with "full QAT training" (QAT from round one); our results show that post-training QAT slightly outperforms full QAT in accuracy and F1 (e.g., 96.45% vs 96.02%, 91.32% vs 89.32% F1), suggesting that QAT during federated training can introduce optimization noise under imbalanced settings.

**Distillation and pruning variants.** We tested distillation modes (none, direct, progressive) and pruning configurations (e.g., prune_none, prune_10x5, prune_5x10, prune_10x2) in combination with QAT/PTQ. Results are reported in the tables and figures; progressive distillation with moderate pruning (e.g., 10×5) gave a good accuracy–size trade-off.

**2×2 experimental design.** For deployment model selection, we compare: (1) Traditional (non-QAT) FL model vs QAT-trained FL model, and (2) PTQ vs QAT fine-tuning at compression. This yields four representative TFLite models (e.g., saved_model_original, saved_model_no_qat_ptq, saved_model_traditional_qat, saved_model_qat_ptq, saved_model_pruned_qat) for evaluation and ratio sweeps.

---

## 5. 실험 설정 요약 (5.1에 넣을 확장 문단)

**Dataset.** CIC-IDS2017, binary labels (BENIGN=0, ATTACK=1), 80:20 train/test split after deduplication and optional balance_ratio (e.g., 80:20 normal-to-attack in training). Features: 78 dimensions; StandardScaler fit on train. Duplicates removed; inf/NaN handled as above.

**Federated.** Four clients, FedAvgM (server momentum 0.5, server_lr 1.0), 80 rounds, 2 local epochs, batch size 128, initial LR 0.0005, Cosine LR decay to minimum 0.0001. Focal loss α = 0.7, class weights enabled. Data partitioned by label for non-IID simulation.

**Compression.** Structured pruning (e.g., 50% or sweep over 10×5, 5×10, 10×2), optional progressive/direct distillation, QAT fine-tuning or PTQ, BatchNorm folding before conversion, TFLite INT8 export.

**Evaluation.** Metrics: accuracy, precision, recall, F1, Attack Recall (recall on attack class). Deployment: model size (MB), inference latency (ms), compression ratio and latency reduction. Ablation: QAT during training vs at compression; distillation and pruning variants.

---

## 6. 체크리스트 (논문에 넣기 전 확인)

- [ ] 데이터셋 선택 이유 문단에서 [citation]을 CIC-IDS2017 논문으로 교체
- [ ] 숫자 통일: num_rounds (80), focal_loss_alpha (0.7), balance_ratio (4.0 → 80:20), local_epochs (2), learning_rate (0.0005)
- [ ] 표/그림 참조 추가 (예: "Table X summarizes preprocessing steps", "Figure Y shows the effect of Cosine LR")
- [ ] Related Work에 Bot-IoT/TON_IoT/MNIST를 사용하지 않은 이유를 짧게 넣을지 결정
- [ ] "3.1 Dataset Selection and Preprocessing" 또는 "5.1 Experimental Setup" 확장으로 위 문단 배치 후 문장 연결 다듬기
