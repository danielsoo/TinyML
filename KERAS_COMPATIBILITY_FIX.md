# Keras 3 / tf_keras Compatibility Fix

## Problem
Models were being saved with Keras 3 format during FL training, but compression script tried to load them with tf_keras (TensorFlow's built-in), causing `InputLayer 'batch_shape'` errors and QAT compatibility issues.

## Solution Applied
Added `os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')` at the **very beginning** of all files that import TensorFlow, BEFORE any imports.

## Files Modified

### ✅ Local files updated (ready to sync):

1. **src/federated/client.py** (line 11)
   - Sets `TF_USE_LEGACY_KERAS=1` before any TF imports
   - Ensures models saved during FL training use tf_keras format

2. **src/federated/server.py** (line 6)
   - Sets `TF_USE_LEGACY_KERAS=1` before importing tfmot
   - Ensures server-side operations use tf_keras

3. **scripts/train.py** (line 18)
   - Sets `TF_USE_LEGACY_KERAS=1` before any TF imports
   - Main training entry point now uses tf_keras

4. **compression.py** (line 15)
   - Already had the fix
   - Ensures compression pipeline loads models correctly

5. **scripts/convert_keras_model.py** (NEW)
   - Utility to convert existing Keras 3 models to tf_keras format
   - Use if you have models already saved in Keras 3 format

## How to Sync to Server

### Option 1: Sync specific files (fast)
```bash
# From your Mac terminal, in the TinyML directory:
scp -P 42538 src/federated/client.py root@96.241.192.5:/workspace/TinyML/src/federated/
scp -P 42538 src/federated/server.py root@96.241.192.5:/workspace/TinyML/src/federated/
scp -P 42538 scripts/train.py root@96.241.192.5:/workspace/TinyML/scripts/
scp -P 42538 scripts/convert_keras_model.py root@96.241.192.5:/workspace/TinyML/scripts/
scp -P 42538 compression.py root@96.241.192.5:/workspace/TinyML/
```

### Option 2: Full sync (if you have a sync script)
```bash
./scripts/sync_to_psu.sh  # Or whatever your server sync script is
```

### Option 3: Git (if server has git access)
```bash
# On your Mac:
git add src/federated/client.py src/federated/server.py scripts/train.py scripts/convert_keras_model.py compression.py
git commit -m "Fix Keras 3 compatibility: Add TF_USE_LEGACY_KERAS=1 to all TF imports"
git push

# On server:
cd /workspace/TinyML
git pull
```

## Converting Existing Models (If Needed)

If you have models already saved in Keras 3 format on the server, convert them:

```bash
# On the server:
python scripts/convert_keras_model.py src/models/global_model.h5 models/global_model.h5
```

This will:
1. Load the Keras 3 model
2. Extract weights
3. Rebuild using `build_mlp()` with tf_keras
4. Save in tf_keras format

## Testing After Sync

On the server, run the full pipeline:

```bash
python compression.py --config config/federated.yaml --dataset cicids2017 --mode full --num-rounds 2
```

Expected result:
- ✅ FL training completes (saves model in tf_keras format)
- ✅ Model loads successfully in compression stage
- ✅ QAT works (tfmot accepts tensorflow.keras models)
- ✅ All compression stages complete
- ✅ TFLite models generated

## What This Fixes

1. **Model loading errors**: No more `batch_shape` or deserialization errors
2. **QAT compatibility**: tensorflow_model_optimization now accepts models
3. **Format consistency**: All saves/loads use same Keras version
4. **End-to-end pipeline**: Training → Compression → Export all work together

## Why This Works

- `TF_USE_LEGACY_KERAS=1` tells TensorFlow to use its built-in `tf_keras` instead of standalone Keras 3
- This env var **must** be set BEFORE importing TensorFlow (hence placed at top of files)
- All model saves/loads now use consistent format
- tensorflow_model_optimization (tfmot) requires `tensorflow.keras`, which is what we now use

## Verification

To verify the fix worked:
```python
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
from tensorflow import keras

# Build a simple model
model = keras.Sequential([keras.layers.Dense(10, input_shape=(5,))])
print(model.__class__.__module__)  # Should print: tf_keras... or tensorflow.keras...
```
