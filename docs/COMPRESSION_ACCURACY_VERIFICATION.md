# Compression Accuracy 검증 결과

## 문제 현상

v15 리포트에서:
- **Original** (saved_model_original.tflite): Accuracy 0.54, F1 0.41
- **Compressed (PTQ)** (saved_model_pruned_quantized.tflite): Accuracy 0.90, F1 0.78

압축 후 오히려 정확도가 크게 상승 → 정상적이지 않음.

---

## 코드 분석 결과

### 1. 파이프라인 (compression.py)

- **saved_model_original.tflite**: `global_model.h5` (full, 205K params) → `_strip_bn_dropout_for_tflite()` → float32 TFLite export
- **saved_model_pruned_quantized.tflite**: Full → 50% pruning → PTQ → TFLite (61K params)

즉, Original은 **full model**, Compressed는 **pruned + quantized** 모델이다.

### 2. 평가 파이프라인 (analyze_compression.py)

- 두 모델 모두 **같은 config**로 `load_dataset()` 호출
- 같은 `x_test`, `y_test` 사용
- TFLite: INT8/FLOAT32 입력에 따라 quantization/dequantization 처리
- binary: `y_pred_proba[:, 0] >= 0.5` → class 1 예측 (sigmoid 1출력 기준으로는 올바름)

### 3. 가능한 원인

#### A. TFLite float32 변환 이슈 (의심 높음)

`export_tflite.py`에서 Original 변환 시:
```python
model = _strip_bn_dropout_for_tflite(model)
```

- BatchNorm을 Dense에 folding
- `_strip_bn_dropout_for_tflite`가 Sequential로 재구성할 때
  - Dense 순서/연결이 잘못되거나
  - weight 복사가 누락되면  
  TFLite Original 출력이 Keras와 달라질 수 있음.

#### B. 모델/데이터 버전 불일치

- `models/tflite/`는 run마다 덮어쓰여서, 현재 디스크의 TFLite가 v15 학습 run과 다른 버전일 수 있음.
- 분석 config와 학습 config가 다르면 (예: 다른 데이터 경로) test set이 달라질 수 있음.

#### C. output 해석 버그 가능성

- binary 모델: `Dense(1, sigmoid)` → shape `(batch, 1)`, `[:, 0]` = P(class=1)
- 이 해석이 Original/Compressed에서 동일하다고 가정했는데, TFLite export 과정에서 output shape/의미가 바뀌면 잘못된 예측이 나올 수 있음.

---

## 검증 방법

### 1. 스크립트 실행

```bash
cd /Users/younsoopark/Documents/Privacy/Research/TinyML-decayinglr
python scripts/verify_compression_accuracy.py
```

동일 test set에 대해 다음 세 모델의 정확도를 비교:

- Keras `global_model.h5` 또는 `src/models/global_model.h5`
- TFLite Original
- TFLite Compressed

### 2. 기대 결과

- **정상**: Keras ≈ Original > Compressed (압축 시 소폭 하락)
- **비정상**: Original ≪ Compressed → Original TFLite export에 문제 있음

### 3. 추가 확인

1. **Keras 직접 평가**
   ```bash
   python -c "
   import numpy as np
   from src.data.loader import load_dataset
   from src.models import nets
   import yaml
   with open('config/federated.yaml') as f: cfg = yaml.safe_load(f)
   dc = cfg['data']
   dk = {k:v for k,v in dc.items() if k not in {'name','num_clients'}}
   dk.setdefault('data_path', dk.pop('path', None))
   _,_,xt,yt = load_dataset(dc['name'],**dk)
   m = tf.keras.models.load_model('models/global_model.h5', compile=False)
   yp = (m.predict(xt)[:,0] >= 0.5).astype(int)
   print('Keras Acc:', np.mean(yp==yt))
   "
   ```

2. **TFLite Original vs Keras 출력 비교**
   - 일부 샘플에 대해 Keras `model.predict(x)`와 TFLite `interpreter.get_tensor(output_idx)`를 비교
   - 값 차이가 크면 TFLite 변환/스트립 로직에 문제가 있는 것

---

## 결론

- 평가 로직은 두 모델에 동일하게 적용됨.
- Original TFLite가 Keras보다 정확도가 크게 낮고, Compressed보다도 낮다면, **TFLite Original export(특히 BN/Dropout stripping) 과정**에 이슈가 있을 가능성이 큼.
- `verify_compression_accuracy.py` 실행 결과와 Keras 직접 평가 결과를 함께 확인하면 원인을 더 정확히 좁힐 수 있음.
