# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 8 models (same as compression_analysis) |
| **Config** | `./config/federated_local_sky.yaml` |
| **Generated** | 2026-03-14 11:52:07 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | None |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 70 |
| **Local epochs** | 3 |
| **Batch size** | 128 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |
| **Prediction threshold** | 0.3 |
| **Ratio sweep models** | 8 models |
| **PGD top-N** | 4 |
| **PGD metric** | f1_score |
| **Adversarial training enabled** | True |
| **AT attack** | pgd |
| **AT epsilon** | 0.05 |
| **Distillation first** | False |

전체 실험 설정: 이 run 디렉터리의 `run_config.yaml` 및 `experiment_record.md` 참조 (run pipeline으로 실행한 경우).

## Summary

Total models: 8, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0004 | 0.1003 | 0.2003 | 0.3002 | 0.4002 | 0.5002 | 0.6002 | 0.7001 | 0.8001 | 0.9001 | 1.0000 |
| noQAT+PTQ | 0.0547 | 0.1492 | 0.2437 | 0.3384 | 0.4328 | 0.5272 | 0.6215 | 0.7162 | 0.8107 | 0.9052 | 1.0000 |
| Traditional+QAT (no QAT in FL, QAT fine-tune) | 0.9717 | 0.9698 | 0.9675 | 0.9652 | 0.9632 | 0.9614 | 0.9592 | 0.9573 | 0.9551 | 0.9527 | 0.9508 |
| QAT+PTQ | 0.9457 | 0.9482 | 0.9493 | 0.9502 | 0.9517 | 0.9529 | 0.9539 | 0.9557 | 0.9565 | 0.9576 | 0.9592 |
| Compressed (QAT) | 0.9729 | 0.9678 | 0.9623 | 0.9568 | 0.9515 | 0.9464 | 0.9410 | 0.9358 | 0.9303 | 0.9249 | 0.9197 |
| saved_model_pruned_10x5_qat | 0.9729 | 0.9678 | 0.9623 | 0.9568 | 0.9515 | 0.9464 | 0.9410 | 0.9358 | 0.9303 | 0.9249 | 0.9197 |
| saved_model_pruned_10x2_qat | 0.9653 | 0.9639 | 0.9627 | 0.9616 | 0.9604 | 0.9595 | 0.9586 | 0.9573 | 0.9563 | 0.9552 | 0.9542 |
| saved_model_pruned_5x10_qat | 0.9705 | 0.9681 | 0.9653 | 0.9625 | 0.9599 | 0.9575 | 0.9550 | 0.9525 | 0.9497 | 0.9470 | 0.9445 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1819 | 0.3334 | 0.4616 | 0.5715 | 0.6667 | 0.7501 | 0.8236 | 0.8890 | 0.9474 | 1.0000 |
| noQAT+PTQ | 0.0000 | 0.1903 | 0.3459 | 0.4756 | 0.5851 | 0.6790 | 0.7602 | 0.8315 | 0.8942 | 0.9500 | 1.0000 |
| Traditional+QAT (no QAT in FL, QAT fine-tune) | 0.0000 | 0.8629 | 0.9212 | 0.9426 | 0.9538 | 0.9610 | 0.9655 | 0.9689 | 0.9713 | 0.9731 | 0.9748 |
| QAT+PTQ | 0.0000 | 0.7877 | 0.8833 | 0.9204 | 0.9408 | 0.9532 | 0.9615 | 0.9681 | 0.9725 | 0.9761 | 0.9792 |
| Compressed (QAT) | 0.0000 | 0.8514 | 0.9071 | 0.9273 | 0.9381 | 0.9449 | 0.9493 | 0.9525 | 0.9548 | 0.9566 | 0.9582 |
| saved_model_pruned_10x5_qat | 0.0000 | 0.8514 | 0.9071 | 0.9273 | 0.9381 | 0.9449 | 0.9493 | 0.9525 | 0.9548 | 0.9566 | 0.9582 |
| saved_model_pruned_10x2_qat | 0.0000 | 0.8412 | 0.9110 | 0.9371 | 0.9507 | 0.9593 | 0.9651 | 0.9691 | 0.9722 | 0.9746 | 0.9766 |
| saved_model_pruned_5x10_qat | 0.0000 | 0.8558 | 0.9159 | 0.9380 | 0.9496 | 0.9569 | 0.9618 | 0.9653 | 0.9678 | 0.9697 | 0.9715 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0004 | 0.0003 | 0.0003 | 0.0003 | 0.0004 | 0.0004 | 0.0004 | 0.0003 | 0.0007 | 0.0010 | 0.0000 |
| noQAT+PTQ | 0.0547 | 0.0547 | 0.0547 | 0.0549 | 0.0547 | 0.0544 | 0.0538 | 0.0542 | 0.0539 | 0.0528 | 0.0000 |
| Traditional+QAT (no QAT in FL, QAT fine-tune) | 0.9717 | 0.9718 | 0.9717 | 0.9714 | 0.9715 | 0.9720 | 0.9718 | 0.9724 | 0.9722 | 0.9697 | 0.0000 |
| QAT+PTQ | 0.9457 | 0.9469 | 0.9468 | 0.9464 | 0.9466 | 0.9466 | 0.9458 | 0.9477 | 0.9458 | 0.9435 | 0.0000 |
| Compressed (QAT) | 0.9729 | 0.9730 | 0.9730 | 0.9727 | 0.9727 | 0.9731 | 0.9730 | 0.9734 | 0.9729 | 0.9717 | 0.0000 |
| saved_model_pruned_10x5_qat | 0.9729 | 0.9730 | 0.9730 | 0.9727 | 0.9727 | 0.9731 | 0.9730 | 0.9734 | 0.9729 | 0.9717 | 0.0000 |
| saved_model_pruned_10x2_qat | 0.9653 | 0.9649 | 0.9648 | 0.9647 | 0.9646 | 0.9648 | 0.9653 | 0.9647 | 0.9647 | 0.9638 | 0.0000 |
| saved_model_pruned_5x10_qat | 0.9705 | 0.9706 | 0.9705 | 0.9703 | 0.9702 | 0.9705 | 0.9708 | 0.9711 | 0.9704 | 0.9688 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0004 | 0.0000 | 0.0000 | 0.0000 | 0.0004 | 1.0000 |
| 90 | 10 | 460,810 | 0.1003 | 0.1000 | 1.0000 | 0.1819 | 0.0003 | 1.0000 |
| 80 | 20 | 425,865 | 0.2003 | 0.2001 | 1.0000 | 0.3334 | 0.0003 | 1.0000 |
| 70 | 30 | 283,910 | 0.3002 | 0.3001 | 1.0000 | 0.4616 | 0.0003 | 1.0000 |
| 60 | 40 | 212,930 | 0.4002 | 0.4001 | 1.0000 | 0.5715 | 0.0004 | 1.0000 |
| 50 | 50 | 170,346 | 0.5002 | 0.5001 | 1.0000 | 0.6667 | 0.0004 | 1.0000 |
| 40 | 60 | 141,955 | 0.6002 | 0.6001 | 1.0000 | 0.7501 | 0.0004 | 1.0000 |
| 30 | 70 | 121,672 | 0.7001 | 0.7001 | 1.0000 | 0.8236 | 0.0003 | 1.0000 |
| 20 | 80 | 106,465 | 0.8001 | 0.8001 | 1.0000 | 0.8890 | 0.0007 | 1.0000 |
| 10 | 90 | 94,630 | 0.9001 | 0.9001 | 1.0000 | 0.9474 | 0.0010 | 1.0000 |
| 0 | 100 | 85,173 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### noQAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0547 | 0.0000 | 0.0000 | 0.0000 | 0.0547 | 1.0000 |
| 90 | 10 | 460,810 | 0.1492 | 0.1052 | 1.0000 | 0.1903 | 0.0547 | 1.0000 |
| 80 | 20 | 425,865 | 0.2437 | 0.2091 | 1.0000 | 0.3459 | 0.0547 | 0.9998 |
| 70 | 30 | 283,910 | 0.3384 | 0.3120 | 1.0000 | 0.4756 | 0.0549 | 0.9996 |
| 60 | 40 | 212,930 | 0.4328 | 0.4136 | 1.0000 | 0.5851 | 0.0547 | 0.9994 |
| 50 | 50 | 170,346 | 0.5272 | 0.5140 | 1.0000 | 0.6790 | 0.0544 | 0.9991 |
| 40 | 60 | 141,955 | 0.6215 | 0.6132 | 1.0000 | 0.7602 | 0.0538 | 0.9987 |
| 30 | 70 | 121,672 | 0.7162 | 0.7116 | 1.0000 | 0.8315 | 0.0542 | 0.9980 |
| 20 | 80 | 106,465 | 0.8107 | 0.8087 | 1.0000 | 0.8942 | 0.0539 | 0.9965 |
| 10 | 90 | 94,630 | 0.9052 | 0.9048 | 1.0000 | 0.9500 | 0.0528 | 0.9921 |
| 0 | 100 | 85,173 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### Traditional+QAT (no QAT in FL, QAT fine-tune)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9717 | 0.0000 | 0.0000 | 0.0000 | 0.9717 | 1.0000 |
| 90 | 10 | 460,810 | 0.9698 | 0.7892 | 0.9518 | 0.8629 | 0.9718 | 0.9945 |
| 80 | 20 | 425,865 | 0.9675 | 0.8935 | 0.9508 | 0.9212 | 0.9717 | 0.9875 |
| 70 | 30 | 283,910 | 0.9652 | 0.9345 | 0.9508 | 0.9426 | 0.9714 | 0.9788 |
| 60 | 40 | 212,930 | 0.9632 | 0.9569 | 0.9508 | 0.9538 | 0.9715 | 0.9673 |
| 50 | 50 | 170,346 | 0.9614 | 0.9714 | 0.9508 | 0.9610 | 0.9720 | 0.9518 |
| 40 | 60 | 141,955 | 0.9592 | 0.9806 | 0.9508 | 0.9655 | 0.9718 | 0.9294 |
| 30 | 70 | 121,672 | 0.9573 | 0.9877 | 0.9508 | 0.9689 | 0.9724 | 0.8944 |
| 20 | 80 | 106,465 | 0.9551 | 0.9927 | 0.9508 | 0.9713 | 0.9722 | 0.8316 |
| 10 | 90 | 94,630 | 0.9527 | 0.9965 | 0.9508 | 0.9731 | 0.9697 | 0.6865 |
| 0 | 100 | 85,173 | 0.9508 | 1.0000 | 0.9508 | 0.9748 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9457 | 0.0000 | 0.0000 | 0.0000 | 0.9457 | 1.0000 |
| 90 | 10 | 460,810 | 0.9482 | 0.6675 | 0.9605 | 0.7877 | 0.9469 | 0.9954 |
| 80 | 20 | 425,865 | 0.9493 | 0.8184 | 0.9592 | 0.8833 | 0.9468 | 0.9893 |
| 70 | 30 | 283,910 | 0.9502 | 0.8846 | 0.9592 | 0.9204 | 0.9464 | 0.9819 |
| 60 | 40 | 212,930 | 0.9517 | 0.9230 | 0.9592 | 0.9408 | 0.9466 | 0.9721 |
| 50 | 50 | 170,346 | 0.9529 | 0.9473 | 0.9592 | 0.9532 | 0.9466 | 0.9587 |
| 40 | 60 | 141,955 | 0.9539 | 0.9637 | 0.9592 | 0.9615 | 0.9458 | 0.9392 |
| 30 | 70 | 121,672 | 0.9557 | 0.9772 | 0.9592 | 0.9681 | 0.9477 | 0.9087 |
| 20 | 80 | 106,465 | 0.9565 | 0.9861 | 0.9592 | 0.9725 | 0.9458 | 0.8529 |
| 10 | 90 | 94,630 | 0.9576 | 0.9935 | 0.9592 | 0.9761 | 0.9435 | 0.7199 |
| 0 | 100 | 85,173 | 0.9592 | 1.0000 | 0.9592 | 0.9792 | 0.0000 | 0.0000 |

### Compressed (QAT)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9729 | 0.0000 | 0.0000 | 0.0000 | 0.9729 | 1.0000 |
| 90 | 10 | 460,810 | 0.9678 | 0.7914 | 0.9214 | 0.8514 | 0.9730 | 0.9911 |
| 80 | 20 | 425,865 | 0.9623 | 0.8948 | 0.9197 | 0.9071 | 0.9730 | 0.9798 |
| 70 | 30 | 283,910 | 0.9568 | 0.9351 | 0.9197 | 0.9273 | 0.9727 | 0.9658 |
| 60 | 40 | 212,930 | 0.9515 | 0.9573 | 0.9197 | 0.9381 | 0.9727 | 0.9478 |
| 50 | 50 | 170,346 | 0.9464 | 0.9716 | 0.9197 | 0.9449 | 0.9731 | 0.9237 |
| 40 | 60 | 141,955 | 0.9410 | 0.9808 | 0.9197 | 0.9493 | 0.9730 | 0.8898 |
| 30 | 70 | 121,672 | 0.9358 | 0.9878 | 0.9197 | 0.9525 | 0.9734 | 0.8385 |
| 20 | 80 | 106,465 | 0.9303 | 0.9927 | 0.9197 | 0.9548 | 0.9729 | 0.7517 |
| 10 | 90 | 94,630 | 0.9249 | 0.9966 | 0.9197 | 0.9566 | 0.9717 | 0.5734 |
| 0 | 100 | 85,173 | 0.9197 | 1.0000 | 0.9197 | 0.9582 | 0.0000 | 0.0000 |

### saved_model_pruned_10x5_qat

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9729 | 0.0000 | 0.0000 | 0.0000 | 0.9729 | 1.0000 |
| 90 | 10 | 460,810 | 0.9678 | 0.7914 | 0.9214 | 0.8514 | 0.9730 | 0.9911 |
| 80 | 20 | 425,865 | 0.9623 | 0.8948 | 0.9197 | 0.9071 | 0.9730 | 0.9798 |
| 70 | 30 | 283,910 | 0.9568 | 0.9351 | 0.9197 | 0.9273 | 0.9727 | 0.9658 |
| 60 | 40 | 212,930 | 0.9515 | 0.9573 | 0.9197 | 0.9381 | 0.9727 | 0.9478 |
| 50 | 50 | 170,346 | 0.9464 | 0.9716 | 0.9197 | 0.9449 | 0.9731 | 0.9237 |
| 40 | 60 | 141,955 | 0.9410 | 0.9808 | 0.9197 | 0.9493 | 0.9730 | 0.8898 |
| 30 | 70 | 121,672 | 0.9358 | 0.9878 | 0.9197 | 0.9525 | 0.9734 | 0.8385 |
| 20 | 80 | 106,465 | 0.9303 | 0.9927 | 0.9197 | 0.9548 | 0.9729 | 0.7517 |
| 10 | 90 | 94,630 | 0.9249 | 0.9966 | 0.9197 | 0.9566 | 0.9717 | 0.5734 |
| 0 | 100 | 85,173 | 0.9197 | 1.0000 | 0.9197 | 0.9582 | 0.0000 | 0.0000 |

### saved_model_pruned_10x2_qat

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9653 | 0.0000 | 0.0000 | 0.0000 | 0.9653 | 1.0000 |
| 90 | 10 | 460,810 | 0.9639 | 0.7515 | 0.9551 | 0.8412 | 0.9649 | 0.9949 |
| 80 | 20 | 425,865 | 0.9627 | 0.8715 | 0.9542 | 0.9110 | 0.9648 | 0.9883 |
| 70 | 30 | 283,910 | 0.9616 | 0.9206 | 0.9542 | 0.9371 | 0.9647 | 0.9801 |
| 60 | 40 | 212,930 | 0.9604 | 0.9473 | 0.9542 | 0.9507 | 0.9646 | 0.9693 |
| 50 | 50 | 170,346 | 0.9595 | 0.9644 | 0.9542 | 0.9593 | 0.9648 | 0.9547 |
| 40 | 60 | 141,955 | 0.9586 | 0.9763 | 0.9542 | 0.9651 | 0.9653 | 0.9336 |
| 30 | 70 | 121,672 | 0.9573 | 0.9844 | 0.9542 | 0.9691 | 0.9647 | 0.9003 |
| 20 | 80 | 106,465 | 0.9563 | 0.9908 | 0.9542 | 0.9722 | 0.9647 | 0.8404 |
| 10 | 90 | 94,630 | 0.9552 | 0.9958 | 0.9542 | 0.9746 | 0.9638 | 0.7005 |
| 0 | 100 | 85,173 | 0.9542 | 1.0000 | 0.9542 | 0.9766 | 0.0000 | 0.0000 |

### saved_model_pruned_5x10_qat

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9705 | 0.0000 | 0.0000 | 0.0000 | 0.9705 | 1.0000 |
| 90 | 10 | 460,810 | 0.9681 | 0.7813 | 0.9461 | 0.8558 | 0.9706 | 0.9939 |
| 80 | 20 | 425,865 | 0.9653 | 0.8890 | 0.9445 | 0.9159 | 0.9705 | 0.9859 |
| 70 | 30 | 283,910 | 0.9625 | 0.9316 | 0.9445 | 0.9380 | 0.9703 | 0.9761 |
| 60 | 40 | 212,930 | 0.9599 | 0.9548 | 0.9445 | 0.9496 | 0.9702 | 0.9633 |
| 50 | 50 | 170,346 | 0.9575 | 0.9697 | 0.9445 | 0.9569 | 0.9705 | 0.9459 |
| 40 | 60 | 141,955 | 0.9550 | 0.9798 | 0.9445 | 0.9618 | 0.9708 | 0.9210 |
| 30 | 70 | 121,672 | 0.9525 | 0.9871 | 0.9445 | 0.9653 | 0.9711 | 0.8824 |
| 20 | 80 | 106,465 | 0.9497 | 0.9922 | 0.9445 | 0.9678 | 0.9704 | 0.8139 |
| 10 | 90 | 94,630 | 0.9470 | 0.9963 | 0.9445 | 0.9697 | 0.9688 | 0.6599 |
| 0 | 100 | 85,173 | 0.9445 | 1.0000 | 0.9445 | 0.9715 | 0.0000 | 0.0000 |


## Threshold Tuning (Original)

Model: `models/tflite/saved_model_original.tflite`

### Normal 90% : Attack 10%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_original.tflite

Test set: Normal 90% : Attack 10%  (n=460,810, N=414,729, A=46,081)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.1000   0.1818   0.0000   1.0000   1.0000   0.1000  
0.20       0.1000   0.1818   0.0000   1.0000   1.0000   0.1000  
0.25       0.1000   0.1818   0.0000   1.0000   1.0000   0.1000  
0.30       0.1003   0.1819   0.0003   1.0000   1.0000   0.1000  
0.35       0.1016   0.1821   0.0018   0.9987   1.0000   0.1002  
0.40       0.1026   0.1822   0.0028   0.9983   1.0000   0.1003  
0.45       0.1022   0.1792   0.0047   0.6764   0.9799   0.0986  
0.50       0.2363   0.1988   0.1573   0.9642   0.9475   0.1111  
0.55       0.8340   0.2837   0.8902   0.9227   0.3286   0.2496   <--
0.60       0.8789   0.0747   0.9711   0.9019   0.0489   0.1584  
0.65       0.8923   0.0222   0.9901   0.9002   0.0122   0.1205  
0.70       0.8973   0.0119   0.9963   0.9002   0.0062   0.1571  
0.75       0.8990   0.0099   0.9983   0.9003   0.0050   0.2497  
0.80       0.8990   0.0030   0.9987   0.9000   0.0015   0.1160  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.8340, F1=0.2837, Normal Recall=0.8902, Normal Precision=0.9227, Attack Recall=0.3286, Attack Precision=0.2496

```

### Normal 80% : Attack 20%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_original.tflite

Test set: Normal 80% : Attack 20%  (n=425,865, N=340,692, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.2000   0.3333   0.0000   1.0000   1.0000   0.2000  
0.20       0.2000   0.3333   0.0000   1.0000   1.0000   0.2000  
0.25       0.2000   0.3333   0.0000   1.0000   1.0000   0.2000  
0.30       0.2003   0.3334   0.0003   1.0000   1.0000   0.2001  
0.35       0.2014   0.3337   0.0018   0.9983   1.0000   0.2003  
0.40       0.2023   0.3340   0.0028   0.9959   1.0000   0.2004  
0.45       0.1998   0.3289   0.0047   0.4898   0.9806   0.1976  
0.50       0.3153   0.3563   0.1573   0.9229   0.9474   0.2194  
0.55       0.7781   0.3724   0.8903   0.8415   0.3291   0.4286   <--
0.60       0.7867   0.0843   0.9712   0.8034   0.0491   0.2986  
0.65       0.7945   0.0227   0.9901   0.8003   0.0119   0.2318  
0.70       0.7983   0.0119   0.9963   0.8004   0.0061   0.2915  
0.75       0.7996   0.0095   0.9983   0.8005   0.0048   0.4181  
0.80       0.7993   0.0029   0.9987   0.8000   0.0015   0.2230  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.7781, F1=0.3724, Normal Recall=0.8903, Normal Precision=0.8415, Attack Recall=0.3291, Attack Precision=0.4286

```

### Normal 70% : Attack 30%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/MachineLearningCVE
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 286,467 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 529,918 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 288,602 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 445,909 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 692,703 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 2,830,743
[load_cicids2017] Removed 331,200 duplicates (2,499,543 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=2073680, ATTACK=425863
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1658944 -> 1362760 (ratio<=4.0), total=1,703,450
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 2,725,520 (BENIGN=1,362,760, ATTACK=1,362,760)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 2,725,520, Test samples: 499,909
  Test: 499,909 (Normal=414,736, Attack=85,173)
Loading model: models/tflite/saved_model_original.tflite

Test set: Normal 70% : Attack 30%  (n=283,910, N=198,737, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.3000   0.4615   0.0000   1.0000   1.0000   0.3000  
0.20       0.3000   0.4615   0.0000   1.0000   1.0000   0.3000  
0.25       0.3000   0.4615   0.0000   1.0000   1.0000   0.3000  
0.30       0.3002   0.4616   0.0003   1.0000   1.0000   0.3001  
0.35       0.3013   0.4620   0.0018   0.9972   1.0000   0.3004  
0.40       0.3020   0.4622   0.0028   0.9929   1.0000   0.3006  
0.45       0.2974   0.4557   0.0046   0.3557   0.9806   0.2969  
0.50       0.3942   0.4841   0.1571   0.8746   0.9474   0.3251   <--
0.55       0.7221   0.4155   0.8906   0.7560   0.3291   0.5632  
0.60       0.6947   0.0880   0.9714   0.7045   0.0491   0.4240  
0.65       0.6968   0.0231   0.9903   0.7005   0.0119   0.3449  
0.70       0.6993   0.0120   0.9964   0.7005   0.0061   0.4205  
0.75       0.7003   0.0096   0.9984   0.7007   0.0048   0.5607  
0.80       0.6996   0.0029   0.9987   0.7000   0.0015   0.3351  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.5
  At threshold 0.5: Accuracy=0.3942, F1=0.4841, Normal Recall=0.1571, Normal Precision=0.8746, Attack Recall=0.9474, Attack Precision=0.3251

```

