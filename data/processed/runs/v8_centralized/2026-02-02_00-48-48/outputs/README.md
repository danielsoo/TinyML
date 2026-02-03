# Outputs Directory

This directory stores outputs from pruning demonstrations and tests.

## Subdirectories

### `pruning_demo/`
Created by `demo_pruning.py`. Contains:
- `original_model.h5` / `.tflite` - Original trained model
- `pruned_model.h5` / `.tflite` - Pruned model
- `layer_comparison.png` - Visual comparison of layer sizes
- `accuracy_comparison.png` - Accuracy through the pruning pipeline

## Usage

Run the demo to generate outputs:
```bash
python demo_pruning.py
```

All visualization plots and models will be saved here.
