# FGSM (Fast Gradient Sign Method) — Quick Guide for Varun

## What is FGSM?

**FGSM** (Goodfellow et al.) is a one-step adversarial attack: we take the gradient of the **loss with respect to the input**, take its **sign**, and add a small perturbation `ε` (epsilon) in that direction:  
`x_adv = x + ε * sign(∇_x Loss(model(x), y))`.  
The goal is to flip the model’s prediction (e.g., make an attack sample classified as benign) with minimal change to the input. For IDS, we use it to test whether our classifier is robust to such evasion.

---

## What we use to generate attacks

We **do not** use ART (Adversarial Robustness Toolbox) for generating FGSM. We use a **custom implementation** with **TensorFlow/Keras**:

- **Gradient computation:** `tf.GradientTape` — we feed inputs as `tf.Variable`, run a forward pass, compute the loss with the true labels, then `tape.gradient(loss, x)` to get ∇_x loss.
- **Perturbation:** `x_adv = x + eps * sign(gradients)`, then clip to `[0, 1]` (assuming normalized features).

**Code:** `src/adversarial/fgsm_hook.py`  
- `compute_gradients(model, x, y)` — gradients w.r.t. input  
- `generate_fgsm_attack(model, x, y, eps=0.05, ...)` — returns adversarial examples  
- `evaluate_attack_success(...)` — original vs adversarial accuracy, attack success rate, perturbation size  
- `tune_epsilon(...)` — sweep epsilons to hit a target attack success rate  
- `generate_adversarial_dataset(...)` — batch generation of adversarial dataset  

**Test script:** `scripts/test_fgsm_attack.py`  
- Loads **Bot-IoT** (configurable path), loads or trains a model, runs FGSM with several epsilons (e.g. 0.01, 0.05, 0.1, 0.15, 0.2), runs epsilon tuning, and generates an adversarial dataset.  
- Run: `python scripts/test_fgsm_attack.py` (optional: `--model path/to/model.h5`).

**Dependency:** `requirements.txt` includes `adversarial-robustness-toolbox>=1,<2` for possible future use; the **current FGSM pipeline uses only TensorFlow/Keras** (no ART calls in `fgsm_hook.py`).

---

## Is there info on FGSM in the slides?

The repo has slides (e.g. `tinyml_meeting_deck.pptx`, `PSU_Capstone.pptx`) for project/meeting context; they may not have a dedicated FGSM section. You can use this doc for a short slide: “What is FGSM?” + “We use TensorFlow GradientTape and custom code in `fgsm_hook.py`; test via `scripts/test_fgsm_attack.py`.”

---

## What you can do next

1. **Run the attack and log results:**  
   `python scripts/test_fgsm_attack.py` (and optionally with `--model models/global_model.h5` or a specific run’s model). Log accuracy drop and attack success rate for different epsilons.

2. **Try different epsilons:**  
   In `test_fgsm_attack.py`, the list `epsilon_values = [0.01, 0.05, 0.1, 0.15, 0.2]` can be extended (e.g. 0.25, 0.3). `tune_epsilon()` in `fgsm_hook.py` uses a default range `[0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]` — you can add more values or make this configurable via CLI.

3. **CIC-IDS2017:**  
   The current test script is wired for **Bot-IoT**. To run FGSM on **CIC-IDS2017**, we’d need to plug in the CIC-IDS2017 loader (same preprocessing as in FL training) and pass that data into `generate_fgsm_attack` / `evaluate_attack_success`. The attack logic in `fgsm_hook.py` is dataset-agnostic (it only needs `model`, `x`, `y`).

4. **Summarize in report/slides:**  
   Use this doc and the script output (tables of epsilon vs. accuracy / attack success rate) for a short “Adversarial robustness (FGSM)” subsection and one or two slides.

If you want, next step can be: add a CLI flag to `test_fgsm_attack.py` for epsilon list and output path for logs, and/or a small script that runs FGSM on CIC-IDS2017 using the existing data loader.
