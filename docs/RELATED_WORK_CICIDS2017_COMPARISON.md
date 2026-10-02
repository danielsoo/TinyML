# Related Work Comparison on CIC-IDS2017 (for reviewer response)

Prepared 2026-10-02. Our system, for reference: binary (BENIGN vs ATTACK), 4 FL clients, FedAvgM, INT8 TFLite MLP 512-256-128.
Results: accuracy 96.02 %, F1 89.32 %, attack recall 93.85 %, model 0.0635 MB (12.28x compression).

## 0. How the numbers below were checked (read this first)

The research environment's network policy blocked every publisher and preprint host (arxiv.org, ieeexplore.ieee.org,
mdpi.com, dl.acm.org, link.springer.com, sciencedirect, semanticscholar, crossref, doi.org, the KU Leuven / imec
repositories, distrinet-research.be). The only reachable source was `raw.githubusercontent.com`. Every number in this
file therefore comes from one of three kinds of document fetched from GitHub. Each entry says which kind it is:

| Tag | Meaning | Strength |
|---|---|---|
| **[FT]** | Full text of the published article: a PDF-to-text conversion of the publisher PDF, stored in a third-party GitHub repo. The page headers in the text show the journal, DOI and licence. | Strong. Spot-check against the publisher PDF before camera-ready. |
| **[ABS]** | Publisher abstract, mirrored in a BibTeX/metadata file on GitHub (DBLP-derived crawler, Zotero export, or Semantic-Scholar export). | Medium. The number is the authors' own wording, but you cannot see the setting/split behind it. |
| **[SEC]** | Number reported by a *survey/thesis table* about the paper, not by the paper itself. | Weak. Use only with the survey as the citation, or verify against the primary paper. |

I did not use any number from memory. Search-engine snippets were not used as evidence. A paper whose numbers I could
not see in a fetched file is listed in Section 5 without numbers.

---

## 1. Summary comparison table (verified numbers only: [FT] and [ABS])

Metric columns are copied exactly as the source reports them. "n/s" = not stated in the fetched text.
"Bin" = binary, "MC" = multi-class. Precision/recall/F1 for multi-class papers are usually *weighted* averages over classes,
so they cannot be compared directly with our attack-class F1 (see Section 4).

| # | Paper (venue, year) | Tag | Setting | Classes | Acc. | Prec. | Recall | F1 | Model size / params | On-device |
|---|---|---|---|---|---|---|---|---|---|---|
| **Ours** | TinyML FL IDS | – | FL, 4 clients, FedAvgM | Bin | 96.02 % | n/a | 93.85 % (attack) | 89.32 % | 0.0635 MB INT8 TFLite | (ours) |
| 1 | Huang et al., *FED-IoV* (Computers & Security, 2024) | ABS | FL | n/s | 97.74 % | n/s | n/s | n/s | "MobileNet-Tiny" (size n/s) | Raspberry Pi 4: "under 10 milliseconds per sample" |
| 2 | Yang et al., Dependable FL vs. poisoning (Computers & Security, 2023) | ABS | FL with label-flipping attackers | n/s | 84.3 % → 97.1 % (without → with their defence) | n/s | n/s | n/s | n/s | none stated |
| 3 | Misrak & Melaku, DNN-BiLSTMQ (Discover Internet of Things, 2025) | FT | Centralized, QAT + dynamic quantization | MC (4: BENIGN, Bot, DDoS, PortScan; **Friday files only**) | 99.73 % | 99.57 % | 99.73 % | 99.64 % | **1000 params, 25.60 KB** | none (no MCU measurement found) |
| 4 | Wisanwanichthan & Thammawichai, KD student (Computers, 2025) | FT | Centralized, knowledge distillation | MC (grouped classes) | 97.88 % | 97.95 % | n/s | 97.85 % | **9,527 params** (teacher 191,063) | inference time only, platform n/s |
| 5 | Sharafaldin et al., dataset paper (ICISSP, 2018) | FT | Centralized ML baselines | MC, weighted avg. | n/s | MLP 0.77; RF 0.98 | MLP 0.83; RF 0.97 | MLP 0.76; RF 0.97 | n/s | none |
| 6 | Cao et al., CNN-GRU (Applied Sciences, 2022) | FT | Centralized DL | MC (9) | 99.65 % | 99.65 % | 99.63 % | 99.64 % | n/s | none |
| 7 | Sun et al., DL-IDS CNN-LSTM (Hindawi, 2020) | ABS | Centralized DL | MC | 98.67 % | n/s | n/s | n/s | n/s | none |
| 8 | Wu et al., RTIDS Transformer (IEEE Access, 2022) | ABS | Centralized DL | n/s | n/s | n/s | n/s | 99.17 % | n/s | none |
| 9 | Wang, Xu & Liu, Res-TranBiLSTM (Computer Networks, 2023) | ABS | Centralized DL, SMOTE-ENN | n/s | 99.15 % | n/s | n/s | n/s | n/s | none |

Counts: **9 papers with verified numbers** (4 from full text, 5 from the abstract only), plus **2 FL papers with
survey-reported numbers only** (Section 2), plus **7 dataset-quality papers** cited for the caveats (Section 4).

### 1a. Survey-reported FL results on CIC-IDS2017 ([SEC], not verified against the primary paper)

Source: Léo Lavaur's PhD thesis (IMT Atlantique, defended 7 Oct 2024). Its SOTA performance table
(`src/chapters/30_sota/figures/table-perf.tex`) extends the table in Lavaur et al., "The Evolution of Federated
Learning-based Intrusion Detection and Mitigation: A Survey", IEEE TNSM 2022. Table footnotes, verbatim:
"\(\ast\) Value is an average of those provided by the authors." / "\(\ddagger\) Value is read from a graph in the article" /
"\(K\) is the highest number of client considered in the experiments."

| Paper | Local model / aggregation | Acc. | Prec. | Recall | FPR | F1 | K (max clients) | Datasets averaged |
|---|---|---|---|---|---|---|---|---|
| Qin et al., IFIP Networking 2020 | **Binarized NN** / SignSGD | *0.9640 | *0.9555 | *0.8645 | – | *0.9055 | 8 | CICIDS2017 + ISCX Botnet 2014 |
| Chen et al., FedAGRU, IEEE Access 2020 | GRU-SVM / FedAGRU | *0.9905 | – | – | *0.0108 | *0.9762 | 20 | CICIDS2017 + KDD99 + WSN-DS |

Because of the `*` (averaged) marks, these figures may mix CIC-IDS2017 with other datasets. Cite the survey as the source of these figures, or get the
primary PDFs before quoting them as CIC-IDS2017 numbers. Qin et al. is still the closest *conceptual* match to our work: FL plus an extremely
compressed (1-bit) model.

---

## 2. Per-paper details

### 2.1 Federated-learning IDS on CIC-IDS2017

#### [1] Huang, Xian, Xian, Wang, Ni — FED-IoV  ([ABS])
- **Citation:** K. Huang, R. Xian, M. Xian, H. Wang, L. Ni, "A comprehensive intrusion detection method for the internet of vehicles based on federated learning architecture," *Computers & Security*, vol. 147, 104067, 2024. DOI 10.1016/j.cose.2024.104067.
- **Source fetched:** https://raw.githubusercontent.com/Lraxer/paper_metadata/main/journal/compsec/compsec147.bib (DBLP record plus publisher abstract)
- **Quotes:** "Vehicular communication traffic data is transformed into images, and a bespoke, efficient model, MobileNet-Tiny, is employed for feature extraction" / "Through evaluation against the authoritative datasets CAN-Intrusion and CICIDS2017, exceptional accuracy rates of 98.51 % and 97.74 %, respectively, were demonstrated by FED-IoV within a federated learning context" / "a prediction latency of under 10 milliseconds per sample was maintained on devices with limited computational power, such as the Raspberry Pi 4 8GB".
- **Setting:** FL; number of clients, IID/non-IID, binary vs multi-class, split, deduplication and model size: not in the abstract.
- **Relevance:** the closest peer-reviewed match (FL + lightweight model + on-device latency). Its on-device target is a Raspberry Pi 4 (Cortex-A72, 8 GB), not a microcontroller.

#### [2] Yang, He, Wang, Qu, Zhang — dependable FL against poisoning  ([ABS])
- **Citation:** R. Yang, H. He, Y. Wang, Y. Qu, W. Zhang, "Dependable federated learning for IoT intrusion detection against poisoning attacks," *Computers & Security*, vol. 132, 103381, 2023. DOI 10.1016/j.cose.2023.103381.
- **Sources fetched:** https://raw.githubusercontent.com/Lraxer/paper_metadata/main/journal/compsec/compsec132.bib ; the same abstract also appears in https://raw.githubusercontent.com/leolavaur/thesis/main/src/biblio/references.bib
- **Quote:** "On the CIC-IDS-2017 dataset, our method can improve the accuracy of the intrusion detection model trained based on FL from 84.3% to 97.1%, while enhancing the protection of IoT network security."
- **Setting:** FL under label-flipping poisoning. The 84.3 % figure is FL *under attack without* their defence, not clean FL. Clients, split and class setting are not in the abstract.

#### [SEC] Qin, Poularakis, Leung, Tassiulas — BNN + FL at line speed
- **Citation (from thesis bib):** Q. Qin, K. Poularakis, K. K. Leung, L. Tassiulas, "Line-Speed and Scalable Intrusion Detection at the Network Edge via Federated Learning," *2020 IFIP Networking Conference*, pp. 352–360, 2020.
- **Sources fetched:** https://raw.githubusercontent.com/leolavaur/thesis/main/src/chapters/30_sota/figures/table-perf.tex (numbers), https://raw.githubusercontent.com/leolavaur/thesis/main/src/biblio/references.bib (citation and abstract), https://raw.githubusercontent.com/vxxx03/IFIPNetworking20/master/README.md (code repo; no numbers in it).
- **Abstract quote:** "We show that Binarized Neural Networks (BNNs) can be implemented as switch functions at the network edge classifying incoming packets at the line speed of the switches. To train BNNs in a scalable manner, we adopt a federated learning approach".
- **Survey table row (verbatim):** `BNN & SignSGD & \(\ast\) 0.9640 & \(\ast\) 0.9555 & \(\ast\) 0.8645 & -- & \(\ast\) 0.9055 & 8 & CICIDS2017 ... ISCX Botnet 2014`.

#### [SEC] Chen, Lv, Liu, Fang, Chen, Pan — FedAGRU
- **Citation (from thesis bib):** Z. Chen, N. Lv, P. Liu, Y. Fang, K. Chen, W. Pan, "Intrusion Detection for Wireless Edge Networks Based on Federated Learning," *IEEE Access*, vol. 8, pp. 217463–217472, 2020. DOI 10.1109/ACCESS.2020.3041793.
- **Sources fetched:** the same Lavaur table and bib as above.
- **Abstract quote:** "FedAGRU improves detection accuracy by approximately 8\%. In addition, FedAGRU's communication cost is 70\% less than other federated learning algorithms".
- **Survey table row:** `GRU-SVM & FedAGRU & \(\ast\) 0.9905 & -- & -- & \(\ast\) 0.0108 & \(\ast\) 0.9762 & 20 & CICIDS2017 ... KDD 99 ... WSN-DS`.

### 2.2 Lightweight / quantized / compressed IDS on CIC-IDS2017

#### [3] Misrak & Melaku — DNN-BiLSTMQ with QAT and dynamic quantization  ([FT])
- **Citation:** S. F. Misrak, H. M. Melaku, "Lightweight intrusion detection system for IoT with improved feature engineering and advanced dynamic quantization," *Discover Internet of Things*, 5:97, 2025. DOI 10.1007/s43926-025-00203-8.
- **Source fetched:** https://raw.githubusercontent.com/nishantharkut/TinyRF-KD/main/docs/literature/papers/_extract/misrak2025quantization.full.txt
- **Quotes:**
  - Abstract: "Using the CIC-IDS2017 dataset, a detection accuracy of 99.73% is achieved with a model size of just 25.6 KB".
  - Results: "For accuracy, the DNN-BiLSTMQ model achieves 99.73%" / "For precision, the DNN-BiLSTMQ model obtains 99.57%" / "For recall, the DNN-BiLSTMQ model achieves 99.73%" / "For the F1-score, the DNN-BiLSTMQ model obtains 99.64%".
  - Table 6(a), CIC-IDS2017 (Params / FLOPs / Size KB): "BiLSTM 18,724 456,768 47.56; 2D-CNN 1488 558,080 40.00; DNN 33,244 12,888,960 46.46; DNN-BiLSTMQ 1000 2784 25.60".
- **Setting / split:** "we specifically focus on the subset of data collected and generated on a Friday". Table 2 classes: BENIGN, Bot, DDoS, PortScan (train 450,076 / test 140,649 / validation 112,520). "data is randomly divided into into 80 % training, 20% testing, and then further divide the training set to training and validation 80% and 20%" (stratified). Feature reduction: "from 78 to 26 features". The text does not mention duplicate removal (the word "duplicate" does not occur). No microcontroller or on-device measurement was found. Quantization is PyTorch QAT + post-training dynamic quantization.
- **Comparability:** this is a 4-class problem on the easiest day of CIC-IDS2017 (volumetric DDoS/PortScan plus Bot), not all five days, so it is **not comparable** to our full-dataset binary task. 25.60 KB vs our 0.0635 MB (about 65 KB): their model is about 2.5x smaller on a much easier sub-task.

#### [4] Wisanwanichthan & Thammawichai — KD shallow student  ([FT])
- **Citation:** T. Wisanwanichthan, M. Thammawichai, "A Lightweight Intrusion Detection System for IoT and UAV Using Deep Neural Networks with Knowledge Distillation," *Computers*, 14(7), 291, 2025. DOI 10.3390/computers14070291.
- **Source fetched:** https://raw.githubusercontent.com/nishantharkut/TinyRF-KD/main/docs/literature/papers/_extract/wisanwanichthan2025kd.full.txt
- **Quotes:** Table 10, "Comparison of teacher and student models on the CIC-IDS2017 dataset": Teacher DNN "191,063 / 98.22 / 98.30 / 98.20"; Student without KD "9527 / 97.81 / 97.97 / 97.79"; Student with KD "9527 ... 97.88 ... 97.95 ... 97.85" (columns: No. of Parameters, Accuracy (%), Precision (%), F1 Score (%)).
- **Setting / split:** multi-class. "all DoS records were grouped into a single DoS category, FTP and SSH Patator records were combined into brute force, and all web attack records were consolidated. Infiltration and Heartbleed attacks were excluded". "After merging the files in MachineLearningCSV.zip, the dataset comprises 2,830,743 records ... an 80:20 stratified train/test split". The excerpt says nothing about deduplication. Inference time is reported as about 2.4 x 10^-5 s; the platform is not identified in the excerpt.
- **Note:** the running text gives "97.88% accuracy and 97.85% F1" for the student *without* KD, but the table gives 97.81 / 97.79. Quote the table values.

#### [1] FED-IoV (above) also belongs in this category (MobileNet-Tiny, Raspberry Pi 4).

### 2.3 Centralized baselines on CIC-IDS2017

#### [5] Sharafaldin, Habibi Lashkari, Ghorbani — the CIC-IDS2017 dataset paper  ([FT])
- **Citation:** I. Sharafaldin, A. Habibi Lashkari, A. A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization," *Proc. ICISSP 2018*, pp. 108–116. DOI 10.5220/0006639801080116.
- **Source fetched:** https://raw.githubusercontent.com/grejc/A.L.E.P.H./main/UndergraduateThesis/bibliography/pdf_txt/Toward_Generating_a_New_Intrusion_Detection_Dataset_and_Intrusion_Traffic_Characterization.txt
- **Quote (Table 4, "The Performance Examination Results"):** Pr: KNN 0.96, RF 0.98, ID3 0.98, Adaboost 0.77, MLP 0.77, Naive-Bayes 0.88, QDA 0.97. Rc: 0.96, 0.97, 0.98, 0.84, 0.83, 0.04, 0.88. F1: 0.96, 0.97, 0.98, 0.77, 0.76, 0.04, 0.92. Text: "Table 4 shows the performance examination results in terms of the weighted average of our evaluation metrics".
- **Setting:** multi-class, weighted averages. Split and deduplication are not described in the excerpt. The **MLP row (F1 0.76)** is the closest architectural comparison to our MLP.

#### [6] Cao, Li, Song, Qin, Chen — CNN-GRU  ([FT])
- **Citation:** B. Cao, C. Li, Y. Song, Y. Qin, C. Chen, "Network Intrusion Detection Model Based on CNN and GRU," *Applied Sciences*, 12(9), 4184, 2022. DOI 10.3390/app12094184.
- **Source fetched:** https://raw.githubusercontent.com/grejc/A.L.E.P.H./main/UndergraduateThesis/bibliography/pdf_txt/applsci-12-04184-v2.txt
- **Quote:** "the detection accuracy, recall, precision, and F1 score of dataset CIC-IDS2017 reached 99.65%, 99.63%, 99.65%, and 99.64%, respectively."
- **Setting / split:** multi-class: "the final dataset contains nine types of attacks: Benign, Dos, Portscan, Ddos, Patator, Bot, Web attack, Infiltration, and Heartbleed". Table 5 total: "2,046,465 877,057 2,923,522" (train / test / total, about 70/30). Uses ADASYN+RENN resampling and RF + Pearson feature selection (52 features remain). The excerpt does not mention deduplication; the total exceeds the raw row count.

#### [7] Sun et al. — DL-IDS (CNN-LSTM)  ([ABS])
- **Citation:** P. Sun, P. Liu, Q. Li, C. Liu, X. Lu, R. Hao, J. Chen, "DL-IDS: Extracting Features Using CNN-LSTM Hybrid Network for Intrusion Detection System," 2020. DOI 10.1155/2020/8890306 (a Hindawi journal; the fetched record does not name the journal, so check it before citing).
- **Source fetched:** https://raw.githubusercontent.com/LCorti/acm_articles/main/download_scripts/semscholar_bib/q40.bib
- **Quote:** "In the multiclassification test, DL-IDS reached 98.67% in overall accuracy, and the accuracy of each attack type was above 99.50%."

#### [8] Wu, Zhang, Wang, Sun — RTIDS  ([ABS])
- **Citation:** Z. Wu, H. Zhang, P. Wang, Z. Sun, "RTIDS: A Robust Transformer-Based Approach for Intrusion Detection System," *IEEE Access*, vol. 10, pp. 64375–64387, 2022. DOI 10.1109/ACCESS.2022.3182333.
- **Source fetched:** https://raw.githubusercontent.com/leolavaur/thesis/main/src/biblio/references.bib
- **Quote:** "Extensive experiments reveal the effectiveness of the proposed RTIDS on two publicly available real traffic intrusion detection datasets named CICIDS2017 and CIC-DDoS2019 with F1-Score of 99.17\% and 98.48\% respectively."

#### [9] Wang, Xu, Liu — Res-TranBiLSTM  ([ABS])
- **Citation:** S. Wang, W. Xu, Y. Liu, "Res-TranBiLSTM: An intelligent approach for intrusion detection in the Internet of Things," *Computer Networks*, vol. 235, 109982, 2023. DOI 10.1016/j.comnet.2023.109982.
- **Source fetched:** https://raw.githubusercontent.com/Lraxer/paper_metadata/main/journal/cn/cn235.bib
- **Quote:** "with accuracy reaching 90.99%, 99.15% and 99.56%, on NSL-KDD dataset, CIC-IDS2017 dataset and MQTTset dataset, respectively." Also uses "Synthetic Minor Overriding Technique (SMOTE) – Edited Nearest Neighbor (ENN)".

---

## 3. Suggested wording for the paper (based only on the verified items)

- Centralized deep models on CIC-IDS2017 commonly report 98.7–99.7 % accuracy or F1 [6–9]. Most of these use multi-class
  weighted metrics, resampling (ADASYN/SMOTE) and random splits without stated deduplication. We therefore treat them as
  upper-bound reference points, not as directly comparable baselines.
- The dataset authors' own MLP baseline reached a weighted F1 of 0.76 (RF 0.97) [5].
- FL-based IDS on CIC-IDS2017 report 97.1–97.74 % accuracy [1, 2]. Only [1] reports an on-device latency, and on a
  Raspberry Pi 4 rather than a microcontroller-class device.
- Among compressed models, [3] reports 25.6 KB / 1000 parameters, but only on the Friday subset with 4 classes. [4]
  reports a 9,527-parameter distilled student at 97.88 % accuracy on grouped multi-class CIC-IDS2017. Neither combines
  compression with federated training or reports a microcontroller deployment.

---

## 4. Comparability caveats

1. **Known labelling and flow-construction errors in CIC-IDS2017.**
   - Engelen, Rimmer, Joosen, "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study," *2021 IEEE Security and Privacy Workshops (SPW)*, pp. 7–12, DOI 10.1109/SPW53761.2021.00009. Abstract (fetched from https://raw.githubusercontent.com/leolavaur/thesis/main/src/biblio/references.bib): "we uncover a series of problems with traffic generation, flow construction, feature extraction and labelling that severely affect the aforementioned properties ... As a result, more than 20 percent of original traffic traces are reconstructed or relabelled. Machine learning benchmarks on the final dataset demonstrate significant improvements." Official code README (https://raw.githubusercontent.com/GintsEngelen/WTMC2021-Code/main/README.md) confirms the BibTeX and adds: "in our experiments, we chose to relabel all "Attempted" flows as BENIGN."
   - Liu, Engelen, Lynar, Essam, Joosen, "Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018," *IEEE CNS 2022*, pp. 254–262, DOI 10.1109/CNS56114.2022.9947235. Abstract (same bib): "We report a large number of previously undocumented errors throughout the dataset creation lifecycle, including in attack orchestration, feature generation, documentation, and labeling. The errors destabilize the results and challenge the findings of numerous publications that have relied on it as a benchmark." (Repo: https://raw.githubusercontent.com/GintsEngelen/CNS2022_Code/main/README.md)
   - Lanvin, Gimenez, Han, Majorczyk, Mé, Totel, "Errors in the CICIDS2017 Dataset and the Significant Differences in Detection Performances It Makes," *CRiSIS 2022*. Abstract (same bib): "we present several flaws we identified in the labelling of the CICIDS2017 dataset and in the traffic capture, such as packet misorder, packet duplication and attack that were performed but not correctly labelled."
   - Mondragon et al., "Advanced IDS: a comparative study of datasets and machine learning algorithms for network flow-based intrusion detection systems," *Applied Intelligence* 55:608, 2025, DOI 10.1007/s10489-025-06422-4 ([FT], https://raw.githubusercontent.com/grejc/A.L.E.P.H./main/UndergraduateThesis/bibliography/pdf_txt/Advanced_IDS_a_comparative_study_of_datasets_and_m.txt): "the original version had 2 828 164 flows, while this newer has 2 097 863" and "the results of the methods' characterizations are not entirely conclusive as it is not clear if the tests were done on the erroneous CIC IDS 2017 [10] and CSE CIC 2018 [39], or on the corrected ones."
   - Counterpoint: Pekár & Jozsa, "Evaluating ML-based anomaly detection across datasets of varied integrity: A case study," *Computer Networks* 251, 110617, 2024, DOI 10.1016/j.comnet.2024.110617 ([ABS], https://raw.githubusercontent.com/Lraxer/paper_metadata/main/journal/cn/cn251.bib): "the RF model exhibits exceptional robustness, achieving consistent high-performance metrics irrespective of the underlying dataset quality". So the size of the error's effect depends on the model. Cite this alongside Engelen to keep the claim balanced.
   - Borgioli et al., *J. Systems Architecture* 156:103283, 2024 ([FT]) did not use CIC-IDS2017 at all for that reason: "this dataset also does not provide per-packet labeling, making it unsuitable for our purposes. Recent work also discovered several flaws affecting the CIC-IDS2017 dataset ... including errors such as duplicate packets and incorrect labeling".
2. **Dataset version and subset.** Papers differ in which files they use: all 5 days vs. the Friday subset only [3]; the original CSVs vs. the Engelen/Lanvin corrected versions; and whether Infiltration and Heartbleed are dropped [4]. Row counts differ accordingly (2,830,743 merged CSV rows in [4]; 2,923,522 after processing in [6]).
3. **Binary vs. multi-class, and how F1 is computed.** Our F1 (89.32 %) is the F1 of the ATTACK class on an imbalanced binary task. Most baselines report multi-class *weighted* averages; [5] says so explicitly. Benign traffic dominates CIC-IDS2017, so weighted F1 is pulled towards the benign-class F1 and is systematically higher than an attack-class F1. Accuracy has the same bias.
4. **Duplicates and leakage.** Our loader (`src/data/loader.py`, the line "Remove duplicates (CIC-IDS2017 has many duplicate rows)") deduplicates *before* the stratified 80/20 split. None of the verified baselines state that they deduplicate. Several use random splits (80/20 in [3, 4]; about 70/30 in [6]), and some resample with SMOTE/ADASYN [6, 9]. Identical rows can then land in both train and test, which inflates test scores. A Bot or PortScan flow with duplicates is easy to "memorise".
5. **FL setting.** The number of clients, the IID vs. non-IID partition and the aggregator differ (8 clients with SignSGD in Qin; up to 20 clients in FedAGRU; FedAvg/FedAvgM in Lazzarini). [2] reports accuracy *under poisoning*. Our 4-client FedAvgM result should be compared with the authors' own centralized baseline (as in our paper), not with other papers' FL numbers.
6. **Model-size reporting.** Sizes are given in different units: float vs. INT8, parameters vs. KB vs. FLOPs, PyTorch dynamic-quantized vs. TFLite flatbuffer. [3] gives KB for a PyTorch dynamic-quantized model. Ours is an INT8 TFLite file. Neither [3] nor [4] measures on a microcontroller. [1] measures on a Raspberry Pi 4.
7. **Survey-derived numbers ([SEC])** are averages across the datasets a paper used, and some are read from graphs (Lavaur table footnotes). They should not be presented as CIC-IDS2017-specific.

---

## 5. Unverified / could not access (no numbers reported)

| Paper | Why listed | What was verified | Why no numbers |
|---|---|---|---|
| Lazzarini, Tianfield, Charissis, "Federated Learning for IoT Intrusion Detection," *AI* 4(3):509–530, 2023, DOI 10.3390/ai4030028 | **Most similar setting** (FL, shallow ANN, CICIDS2017 binary + multi-class, FedAvg vs **FedAvgM**/FedAdam/FedAdagrad via Flower) | Abstract (Lavaur bib; also https://raw.githubusercontent.com/hetiantian10/TWAPPA-FRL/main/article/bibliography.bib): "The experiments are completed on the ToN\_IoT and CICIDS2017 datasets in binary and multiclass classification" / "FedAvg and FedAvgM tend to perform better compared to the two adaptive algorithms" | mdpi.com blocked; the abstract has no numbers. **Strongly recommend fetching this paper manually**: it is the most direct FedAvgM + CICIDS2017 binary comparison. |
| Zang, Zheng, Koziak, Zilberman, Dittmann, "Federated In-Network Machine Learning for Privacy-Preserving IoT Traffic Analysis" (FLIP4), *ACM Trans. Internet Technol.*, 2024, DOI 10.1145/3696354 | FL on CIC-IDS2017, lightweight in-switch models, Raspberry Pi (P4Pi) | Repo README (https://raw.githubusercontent.com/In-Network-Machine-Learning/FLIP4/main/README.md): "use case with dataset $CIC-IDS2017$ as an example", FedAvg aggregator | dl.acm.org blocked; the README has no metrics. |
| Idrissi et al., "Fed-ANIDS," *Expert Systems with Applications* 234:121000, 2023, DOI 10.1016/j.eswa.2023.121000 | FL autoencoder IDS on CIC-IDS2017 (FedAvg vs FedProx) | Abstract (Lavaur bib) | The abstract has no numbers; the repo README (https://raw.githubusercontent.com/meryemJanatiIdrissi/Fed-ANIDS/main/README.md) says "The code will be available upon publication". |
| Tang, Hu, Xu, "A Federated Learning Method for Network Intrusion Detection," *Concurrency Computat. Pract. Exper.* 34(10):e6812, 2022, DOI 10.1002/cpe.6812 | FL on CICIDS2017 | Abstract (Lavaur bib): "The accuracy and other performance of federated learning are almost equal to those of centralized deep learning models." | No numbers in the abstract. |
| Ayed & Talhi, "Federated Learning for Anomaly-Based Intrusion Detection," *ISNCC 2021* | FL CNN on CICIDS2017 | Abstract (Lavaur bib) | No numbers in the abstract. |
| Zhao et al., "Multi-Task Network Anomaly Detection using Federated Learning," *SoICT 2019*, DOI 10.1145/3368926.3369705; Fan et al., "IoTDefender," *IEEE BigDataSE 2020*, DOI 10.1109/BigDataSE50710.2020.00020 | FL with CICIDS2017 among other datasets | Survey rows exist in the Lavaur table, but the values are averaged across datasets | Primary papers not reachable; the survey values are multi-dataset averages, so they are omitted. |
| Doriguzzi-Corin et al., "LUCID: A Practical, Lightweight Deep Learning Solution for DDoS Attack Detection," *IEEE TNSM* 17(2):876–889, 2020, DOI 10.1109/TNSM.2020.2971776 | Lightweight CNN, CIC-IDS2017 DDoS | Citation and CIC-IDS2017 support confirmed in the repo README (https://raw.githubusercontent.com/doriguzzi/lucid-ddos/master/README.md) | The README shows only a CIC-DDoS2019 example; the paper itself was not reachable. |
| "SecFedIDS" (Edraoui), README at https://raw.githubusercontent.com/YEdraoui/SecFedIDS/main/README.md | FL, CICIDS2017, 10 clients, Dirichlet 0.3 | README states it was "Submitted to: ICONs-IoT 2026" | Not peer-reviewed at fetch time, so not used. |

---

## 6. Source index (all fetched via raw.githubusercontent.com)

- Lavaur thesis table: https://raw.githubusercontent.com/leolavaur/thesis/main/src/chapters/30_sota/figures/table-perf.tex
- Lavaur thesis bibliography (abstracts): https://raw.githubusercontent.com/leolavaur/thesis/main/src/biblio/references.bib
- DBLP + abstract metadata: https://raw.githubusercontent.com/Lraxer/paper_metadata/main/journal/compsec/compsec147.bib , .../compsec/compsec132.bib , .../cn/cn235.bib , .../cn/cn251.bib
- Semantic-Scholar export (DL-IDS): https://raw.githubusercontent.com/LCorti/acm_articles/main/download_scripts/semscholar_bib/q40.bib
- Full texts: https://raw.githubusercontent.com/grejc/A.L.E.P.H./main/UndergraduateThesis/bibliography/pdf_txt/ (Sharafaldin 2018; Cao 2022 `applsci-12-04184-v2.txt`; Mondragon 2025; Borgioli 2024 `borgioli-jsa24.txt`) and https://raw.githubusercontent.com/nishantharkut/TinyRF-KD/main/docs/literature/papers/_extract/ (`misrak2025quantization.full.txt`, `wisanwanichthan2025kd.full.txt`)
- Official repos: GintsEngelen/WTMC2021-Code, GintsEngelen/CNS2022_Code, In-Network-Machine-Learning/FLIP4, doriguzzi/lucid-ddos, vxxx03/IFIPNetworking20, meryemJanatiIdrissi/Fed-ANIDS

**Before submission:** open the publisher PDFs for [1]–[9] and confirm each quoted number, especially the [ABS] entries [1, 2, 7, 8, 9]. For those, also extract the class setting, split and client count from the full text.
