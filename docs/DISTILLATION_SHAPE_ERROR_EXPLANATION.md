# Knowledge Distillation Shape Error 원인 분석

## 문제 발생 원인

### 1. TensorFlow Graph Mode의 Shape Inference 한계

TensorFlow는 두 가지 실행 모드가 있습니다:
- **Eager Mode**: 즉시 실행, shape을 런타임에 알 수 있음
- **Graph Mode**: 그래프를 먼저 생성 후 실행, shape 추론이 어려움

`tf.cond` 내부에서 shape을 추출하려 할 때:
```python
student_pred_shape = tf.shape(student_predictions)  # [batch_size, 1]
student_batch_size = student_pred_shape[0]  # ❌ 문제 발생 지점
```

**문제점**:
- Graph mode에서 `batch_size`가 unknown일 수 있음
- `student_pred_shape[0]`이 `[0]`이거나 unknown이면 오류 발생
- `tf.cond` 내부에서 shape 추론이 실패할 수 있음

### 2. Dynamic Batch Size와 Static Shape의 충돌

**시나리오**:
```python
# train_step에서
x, y = data  # batch_size는 런타임에 결정됨

# distillation_loss_fn 내부
student_pred_shape = tf.shape(student_predictions)  # [?, 1] (unknown batch)
student_batch_size = student_pred_shape[0]  # ? (unknown)

# reshape 시도
y_true_flat = tf.reshape(y_true_float, [student_batch_size])  # ❌ unknown으로 reshape 불가
```

**문제점**:
- 배치 크기가 런타임에 결정되는데, graph 생성 시점에는 알 수 없음
- Unknown shape으로 reshape를 시도하면 오류 발생

### 3. Binary Classification의 Shape 불일치

**Binary vs Multi-class**:
```python
# Binary classification
student_predictions: (batch, 1)  # sigmoid output
y_true: (batch,)  # labels

# Multi-class classification  
student_predictions: (batch, num_classes)  # softmax output
y_true: (batch,)  # labels
```

**문제점**:
- Binary: `(batch, 1)` → `(batch,)`로 변환 필요
- Multi-class: `(batch, num_classes)` → 그대로 사용
- `tf.cond`로 분기 처리 시 shape 추론이 복잡해짐

### 4. Knowledge Distillation의 Shape 차이

**Teacher와 Student의 출력**:
```python
teacher_predictions = self.teacher(x)  # (batch, 1) or (batch, num_classes)
student_predictions = self.student(x)  # (batch, 1) or (batch, num_classes)

# distillation_loss 계산
distillation_loss = categorical_crossentropy(...)  # (batch,)

# student_loss 계산
student_loss = binary_crossentropy(...)  # (batch,)
```

**문제점**:
- `distillation_loss`와 `student_loss`의 batch size가 다를 수 있음
- 두 loss를 더할 때 shape 불일치 오류 발생

## 해결 방법

### 1. `-1`을 사용한 자동 Batch Size 감지

```python
# ❌ 기존 방법 (문제 발생)
student_batch_size = student_pred_shape[0]  # unknown이면 오류
y_true_flat = tf.reshape(y_true_float, [student_batch_size])

# ✅ 개선된 방법 (안전)
y_true_flat = tf.reshape(y_true_float, [-1])  # 자동으로 batch size 감지
student_pred_flat = tf.reshape(student_predictions, [-1])
```

**장점**:
- Unknown shape에서도 안전하게 작동
- Graph mode에서도 shape 추론 가능
- Runtime에 batch size를 자동으로 감지

### 2. Reshape 후 Batch Size 확인

```python
# Reshape 후 실제 batch size 가져오기
y_true_flat = tf.reshape(y_true_float, [-1])
student_pred_flat = tf.reshape(student_predictions, [-1])

# 실제 batch size 확인
y_true_batch_size = tf.shape(y_true_flat)[0]
student_batch_size = tf.shape(student_pred_flat)[0]

# 최소 batch size로 통일
min_batch_size = tf.minimum(y_true_batch_size, student_batch_size)
y_true_flat = y_true_flat[:min_batch_size]
student_pred_flat = student_pred_flat[:min_batch_size]
```

### 3. Loss 계산 후 Batch Size 통일

```python
# distillation_loss와 student_loss 계산 후
distillation_batch_size = tf.shape(distillation_loss)[0]
student_batch_size = tf.shape(student_loss)[0]

# 최소 batch size로 slice
min_batch_size = tf.minimum(distillation_batch_size, student_batch_size)
distillation_loss = distillation_loss[:min_batch_size]
student_loss = student_loss[:min_batch_size]

# 안전하게 더하기
total_loss = alpha * distillation_loss + (1 - alpha) * student_loss
```

## 주요 교훈

1. **Graph Mode에서는 shape 추론이 제한적**: `tf.shape()[0]` 같은 접근은 위험할 수 있음
2. **`-1`을 활용한 reshape**: Unknown shape에서도 안전하게 작동
3. **Reshape 후 shape 확인**: Reshape 후 실제 shape을 확인하는 것이 안전
4. **Batch size 통일**: 여러 텐서를 더하기 전에 batch size를 통일해야 함

## 참고 자료

- [TensorFlow Shape Inference](https://www.tensorflow.org/guide/function#shape_inference)
- [tf.reshape with -1](https://www.tensorflow.org/api_docs/python/tf/reshape)
- [tf.cond limitations](https://www.tensorflow.org/api_docs/python/tf/cond)
