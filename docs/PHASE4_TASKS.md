# Phase 4: Adversarial Hardening & Deployment Tasks

## Task 1: Complete FGSM Attack Implementation

## Task Description
Enhance the basic FGSM perturbation utilities to include full attack implementation with gradient computation, epsilon parameter tuning, and adversarial example generation for Bot-IoT dataset.

## Related Files
- `src/adversarial/fgsm_hook.py` (currently has basic perturbation function)
- `src/models/nets.py` (model architecture)
- `src/data/loader.py` (dataset loading)

## Completed Tasks
- [x] Basic FGSM perturbation function (`fgsm_perturb`)
- [ ] Full FGSM attack implementation with gradient computation
- [ ] Epsilon parameter tuning and validation
- [ ] Adversarial example generation for Bot-IoT dataset
- [ ] Attack success rate evaluation

## Status
⏳ **IN PROGRESS** - Basic utilities scaffolded

## Completion Date
Week 12

---

## Task 2: FGSM Integration into FL Training Loop

## Task Description
Integrate adversarial example generation into the federated learning client training loop. Modify the KerasClient class to generate adversarial examples during local training epochs.

## Related Files
- `src/federated/client.py` (KerasClient class)
- `src/adversarial/fgsm_hook.py` (FGSM attack functions)
- `config/federated_local.yaml` (training configuration)
- `config/federated_colab.yaml` (Colab configuration)

## Completed Tasks
- [ ] FGSM integration into KerasClient.fit() method
- [ ] Adversarial example generation during training batches
- [ ] Configuration parameters for adversarial training (epsilon, ratio of adversarial examples)
- [ ] Testing adversarial example generation in FL simulation

## Status
❌ **NOT STARTED**

## Completion Date
Week 12-13

---

## Task 3: Adversarial Training in FL

## Task Description
Run the complete federated learning process with adversarial training enabled. Train the global model using a mix of clean and adversarial examples across all clients.

## Related Files
- `src/federated/client.py`
- `src/federated/server.py`
- `scripts/run_fl_sim.sh`
- `config/federated_local.yaml`
- `config/federated_colab.yaml`

## Completed Tasks
- [ ] FL simulation with adversarial training enabled
- [ ] Hyperparameter tuning for adversarial training
- [ ] Robust global model generation
- [ ] Comparison of robust model vs. standard model performance

## Status
❌ **NOT STARTED**

## Completion Date
Week 13

---

## Task 4: Re-compression of Robust Model

## Task Description
Apply the TinyML compression pipeline (quantization, pruning, knowledge distillation) to the adversarially trained robust model. Analyze size vs. robustness trade-offs.

## Related Files
- `src/tinyml/export_tflite.py`
- `scripts/analyze_compression.py`
- `scripts/visualize_results.py`
- `src/models/global_model.h5` (robust model)

## Completed Tasks
- [ ] TFLite export of robust model
- [ ] 8-bit quantization (INT8) of robust model
- [ ] Structured pruning of robust model
- [ ] Knowledge distillation from robust teacher model
- [ ] Compression analysis and trade-off evaluation

## Status
❌ **NOT STARTED**

## Completion Date
Week 13-14

---

## Task 5: Microcontroller Deployment

## Task Description
Deploy the final compressed robust model to actual microcontroller hardware (ESP32 or Raspberry Pi Pico). Measure inference latency, memory usage, and real-world performance.

## Related Files
- `data/processed/` (compressed TFLite models)
- Deployment scripts (to be created)
- Hardware test code (to be created)

## Completed Tasks
- [ ] Hardware platform selection (ESP32/Raspberry Pi Pico)
- [ ] TFLite model deployment to microcontroller
- [ ] Inference latency measurement
- [ ] Memory footprint analysis
- [ ] Real-world performance evaluation
- [ ] Deployment documentation

## Status
❌ **NOT STARTED**

## Completion Date
Week 14

