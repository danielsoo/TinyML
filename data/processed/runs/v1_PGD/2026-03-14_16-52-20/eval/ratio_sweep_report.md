# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 8 models (same as compression_analysis) |
| **Config** | `./config/federated_local_sky.yaml` |
| **Generated** | 2026-03-14 17:10:45 |

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
| Original (TFLite) | 0.0091 | 0.1069 | 0.2051 | 0.3029 | 0.4011 | 0.4991 | 0.5971 | 0.6955 | 0.7934 | 0.8915 | 0.9896 |
| noQAT+PTQ | 0.0502 | 0.1453 | 0.2402 | 0.3354 | 0.4302 | 0.5250 | 0.6199 | 0.7150 | 0.8099 | 0.9049 | 1.0000 |
| Traditional+QAT (no QAT in FL, QAT fine-tune) | 0.9757 | 0.9715 | 0.9669 | 0.9625 | 0.9581 | 0.9540 | 0.9492 | 0.9453 | 0.9406 | 0.9362 | 0.9318 |
| QAT+PTQ | 0.9439 | 0.9467 | 0.9482 | 0.9495 | 0.9514 | 0.9532 | 0.9546 | 0.9568 | 0.9583 | 0.9596 | 0.9616 |
| Compressed (QAT) | 0.9525 | 0.9531 | 0.9532 | 0.9533 | 0.9533 | 0.9540 | 0.9547 | 0.9551 | 0.9547 | 0.9546 | 0.9552 |
| saved_model_pruned_10x5_qat | 0.9525 | 0.9531 | 0.9532 | 0.9533 | 0.9533 | 0.9540 | 0.9547 | 0.9551 | 0.9547 | 0.9546 | 0.9552 |
| saved_model_pruned_10x2_qat | 0.9443 | 0.9471 | 0.9490 | 0.9507 | 0.9530 | 0.9551 | 0.9574 | 0.9594 | 0.9613 | 0.9631 | 0.9656 |
| saved_model_pruned_5x10_qat | 0.9667 | 0.9648 | 0.9625 | 0.9603 | 0.9585 | 0.9567 | 0.9546 | 0.9529 | 0.9509 | 0.9486 | 0.9468 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1814 | 0.3324 | 0.4600 | 0.5693 | 0.6639 | 0.7467 | 0.8198 | 0.8846 | 0.9426 | 0.9948 |
| noQAT+PTQ | 0.0000 | 0.1896 | 0.3449 | 0.4745 | 0.5840 | 0.6780 | 0.7594 | 0.8308 | 0.8938 | 0.9498 | 1.0000 |
| Traditional+QAT (no QAT in FL, QAT fine-tune) | 0.0000 | 0.8675 | 0.9185 | 0.9371 | 0.9468 | 0.9529 | 0.9566 | 0.9597 | 0.9617 | 0.9634 | 0.9647 |
| QAT+PTQ | 0.0000 | 0.7832 | 0.8812 | 0.9196 | 0.9406 | 0.9536 | 0.9621 | 0.9689 | 0.9736 | 0.9772 | 0.9804 |
| Compressed (QAT) | 0.0000 | 0.8031 | 0.8910 | 0.9247 | 0.9424 | 0.9541 | 0.9620 | 0.9675 | 0.9712 | 0.9743 | 0.9771 |
| saved_model_pruned_10x5_qat | 0.0000 | 0.8031 | 0.8910 | 0.9247 | 0.9424 | 0.9541 | 0.9620 | 0.9675 | 0.9712 | 0.9743 | 0.9771 |
| saved_model_pruned_10x2_qat | 0.0000 | 0.7851 | 0.8834 | 0.9216 | 0.9426 | 0.9555 | 0.9645 | 0.9708 | 0.9756 | 0.9792 | 0.9825 |
| saved_model_pruned_5x10_qat | 0.0000 | 0.8433 | 0.9100 | 0.9347 | 0.9480 | 0.9562 | 0.9616 | 0.9657 | 0.9686 | 0.9707 | 0.9727 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0091 | 0.0089 | 0.0089 | 0.0086 | 0.0087 | 0.0086 | 0.0083 | 0.0091 | 0.0085 | 0.0087 | 0.0000 |
| noQAT+PTQ | 0.0502 | 0.0503 | 0.0503 | 0.0506 | 0.0503 | 0.0501 | 0.0498 | 0.0500 | 0.0498 | 0.0495 | 0.0000 |
| Traditional+QAT (no QAT in FL, QAT fine-tune) | 0.9757 | 0.9758 | 0.9757 | 0.9756 | 0.9756 | 0.9761 | 0.9753 | 0.9766 | 0.9755 | 0.9755 | 0.0000 |
| QAT+PTQ | 0.9439 | 0.9449 | 0.9448 | 0.9444 | 0.9446 | 0.9447 | 0.9440 | 0.9456 | 0.9448 | 0.9417 | 0.0000 |
| Compressed (QAT) | 0.9525 | 0.9528 | 0.9527 | 0.9525 | 0.9520 | 0.9528 | 0.9539 | 0.9547 | 0.9526 | 0.9492 | 0.0000 |
| saved_model_pruned_10x5_qat | 0.9525 | 0.9528 | 0.9527 | 0.9525 | 0.9520 | 0.9528 | 0.9539 | 0.9547 | 0.9526 | 0.9492 | 0.0000 |
| saved_model_pruned_10x2_qat | 0.9443 | 0.9449 | 0.9449 | 0.9443 | 0.9446 | 0.9446 | 0.9450 | 0.9450 | 0.9441 | 0.9406 | 0.0000 |
| saved_model_pruned_5x10_qat | 0.9667 | 0.9666 | 0.9665 | 0.9661 | 0.9663 | 0.9665 | 0.9664 | 0.9672 | 0.9671 | 0.9648 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0091 | 0.0000 | 0.0000 | 0.0000 | 0.0091 | 1.0000 |
| 90 | 10 | 460,810 | 0.1069 | 0.0999 | 0.9895 | 0.1814 | 0.0089 | 0.8837 |
| 80 | 20 | 425,865 | 0.2051 | 0.1998 | 0.9896 | 0.3324 | 0.0089 | 0.7745 |
| 70 | 30 | 283,910 | 0.3029 | 0.2996 | 0.9896 | 0.4600 | 0.0086 | 0.6596 |
| 60 | 40 | 212,930 | 0.4011 | 0.3996 | 0.9896 | 0.5693 | 0.0087 | 0.5563 |
| 50 | 50 | 170,346 | 0.4991 | 0.4995 | 0.9896 | 0.6639 | 0.0086 | 0.4517 |
| 40 | 60 | 141,955 | 0.5971 | 0.5995 | 0.9896 | 0.7467 | 0.0083 | 0.3481 |
| 30 | 70 | 121,672 | 0.6955 | 0.6997 | 0.9896 | 0.8198 | 0.0091 | 0.2732 |
| 20 | 80 | 106,465 | 0.7934 | 0.7997 | 0.9896 | 0.8846 | 0.0085 | 0.1689 |
| 10 | 90 | 94,630 | 0.8915 | 0.8998 | 0.9896 | 0.9426 | 0.0087 | 0.0847 |
| 0 | 100 | 85,173 | 0.9896 | 1.0000 | 0.9896 | 0.9948 | 0.0000 | 0.0000 |

### noQAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0502 | 0.0000 | 0.0000 | 0.0000 | 0.0502 | 1.0000 |
| 90 | 10 | 460,810 | 0.1453 | 0.1047 | 1.0000 | 0.1896 | 0.0503 | 1.0000 |
| 80 | 20 | 425,865 | 0.2402 | 0.2084 | 1.0000 | 0.3449 | 0.0503 | 0.9998 |
| 70 | 30 | 283,910 | 0.3354 | 0.3110 | 1.0000 | 0.4745 | 0.0506 | 0.9996 |
| 60 | 40 | 212,930 | 0.4302 | 0.4124 | 1.0000 | 0.5840 | 0.0503 | 0.9994 |
| 50 | 50 | 170,346 | 0.5250 | 0.5128 | 1.0000 | 0.6780 | 0.0501 | 0.9991 |
| 40 | 60 | 141,955 | 0.6199 | 0.6122 | 1.0000 | 0.7594 | 0.0498 | 0.9986 |
| 30 | 70 | 121,672 | 0.7150 | 0.7107 | 1.0000 | 0.8308 | 0.0500 | 0.9978 |
| 20 | 80 | 106,465 | 0.8099 | 0.8080 | 1.0000 | 0.8938 | 0.0498 | 0.9962 |
| 10 | 90 | 94,630 | 0.9049 | 0.9045 | 1.0000 | 0.9498 | 0.0495 | 0.9915 |
| 0 | 100 | 85,173 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### Traditional+QAT (no QAT in FL, QAT fine-tune)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9757 | 0.0000 | 0.0000 | 0.0000 | 0.9757 | 1.0000 |
| 90 | 10 | 460,810 | 0.9715 | 0.8105 | 0.9330 | 0.8675 | 0.9758 | 0.9924 |
| 80 | 20 | 425,865 | 0.9669 | 0.9056 | 0.9318 | 0.9185 | 0.9757 | 0.9828 |
| 70 | 30 | 283,910 | 0.9625 | 0.9424 | 0.9318 | 0.9371 | 0.9756 | 0.9709 |
| 60 | 40 | 212,930 | 0.9581 | 0.9622 | 0.9318 | 0.9468 | 0.9756 | 0.9555 |
| 50 | 50 | 170,346 | 0.9540 | 0.9750 | 0.9318 | 0.9529 | 0.9761 | 0.9347 |
| 40 | 60 | 141,955 | 0.9492 | 0.9826 | 0.9318 | 0.9566 | 0.9753 | 0.9051 |
| 30 | 70 | 121,672 | 0.9453 | 0.9894 | 0.9318 | 0.9597 | 0.9766 | 0.8600 |
| 20 | 80 | 106,465 | 0.9406 | 0.9935 | 0.9318 | 0.9617 | 0.9755 | 0.7816 |
| 10 | 90 | 94,630 | 0.9362 | 0.9971 | 0.9318 | 0.9634 | 0.9755 | 0.6139 |
| 0 | 100 | 85,173 | 0.9318 | 1.0000 | 0.9318 | 0.9647 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9439 | 0.0000 | 0.0000 | 0.0000 | 0.9439 | 1.0000 |
| 90 | 10 | 460,810 | 0.9467 | 0.6600 | 0.9628 | 0.7832 | 0.9449 | 0.9957 |
| 80 | 20 | 425,865 | 0.9482 | 0.8133 | 0.9616 | 0.8812 | 0.9448 | 0.9899 |
| 70 | 30 | 283,910 | 0.9495 | 0.8811 | 0.9616 | 0.9196 | 0.9444 | 0.9829 |
| 60 | 40 | 212,930 | 0.9514 | 0.9205 | 0.9616 | 0.9406 | 0.9446 | 0.9736 |
| 50 | 50 | 170,346 | 0.9532 | 0.9457 | 0.9616 | 0.9536 | 0.9447 | 0.9610 |
| 40 | 60 | 141,955 | 0.9546 | 0.9626 | 0.9616 | 0.9621 | 0.9440 | 0.9425 |
| 30 | 70 | 121,672 | 0.9568 | 0.9763 | 0.9616 | 0.9689 | 0.9456 | 0.9135 |
| 20 | 80 | 106,465 | 0.9583 | 0.9859 | 0.9616 | 0.9736 | 0.9448 | 0.8603 |
| 10 | 90 | 94,630 | 0.9596 | 0.9933 | 0.9616 | 0.9772 | 0.9417 | 0.7317 |
| 0 | 100 | 85,173 | 0.9616 | 1.0000 | 0.9616 | 0.9804 | 0.0000 | 0.0000 |

### Compressed (QAT)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9525 | 0.0000 | 0.0000 | 0.0000 | 0.9525 | 1.0000 |
| 90 | 10 | 460,810 | 0.9531 | 0.6923 | 0.9563 | 0.8031 | 0.9528 | 0.9949 |
| 80 | 20 | 425,865 | 0.9532 | 0.8348 | 0.9552 | 0.8910 | 0.9527 | 0.9884 |
| 70 | 30 | 283,910 | 0.9533 | 0.8960 | 0.9552 | 0.9247 | 0.9525 | 0.9803 |
| 60 | 40 | 212,930 | 0.9533 | 0.9299 | 0.9552 | 0.9424 | 0.9520 | 0.9696 |
| 50 | 50 | 170,346 | 0.9540 | 0.9530 | 0.9552 | 0.9541 | 0.9528 | 0.9551 |
| 40 | 60 | 141,955 | 0.9547 | 0.9688 | 0.9552 | 0.9620 | 0.9539 | 0.9342 |
| 30 | 70 | 121,672 | 0.9551 | 0.9801 | 0.9552 | 0.9675 | 0.9547 | 0.9014 |
| 20 | 80 | 106,465 | 0.9547 | 0.9878 | 0.9552 | 0.9712 | 0.9526 | 0.8418 |
| 10 | 90 | 94,630 | 0.9546 | 0.9941 | 0.9552 | 0.9743 | 0.9492 | 0.7020 |
| 0 | 100 | 85,173 | 0.9552 | 1.0000 | 0.9552 | 0.9771 | 0.0000 | 0.0000 |

### saved_model_pruned_10x5_qat

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9525 | 0.0000 | 0.0000 | 0.0000 | 0.9525 | 1.0000 |
| 90 | 10 | 460,810 | 0.9531 | 0.6923 | 0.9563 | 0.8031 | 0.9528 | 0.9949 |
| 80 | 20 | 425,865 | 0.9532 | 0.8348 | 0.9552 | 0.8910 | 0.9527 | 0.9884 |
| 70 | 30 | 283,910 | 0.9533 | 0.8960 | 0.9552 | 0.9247 | 0.9525 | 0.9803 |
| 60 | 40 | 212,930 | 0.9533 | 0.9299 | 0.9552 | 0.9424 | 0.9520 | 0.9696 |
| 50 | 50 | 170,346 | 0.9540 | 0.9530 | 0.9552 | 0.9541 | 0.9528 | 0.9551 |
| 40 | 60 | 141,955 | 0.9547 | 0.9688 | 0.9552 | 0.9620 | 0.9539 | 0.9342 |
| 30 | 70 | 121,672 | 0.9551 | 0.9801 | 0.9552 | 0.9675 | 0.9547 | 0.9014 |
| 20 | 80 | 106,465 | 0.9547 | 0.9878 | 0.9552 | 0.9712 | 0.9526 | 0.8418 |
| 10 | 90 | 94,630 | 0.9546 | 0.9941 | 0.9552 | 0.9743 | 0.9492 | 0.7020 |
| 0 | 100 | 85,173 | 0.9552 | 1.0000 | 0.9552 | 0.9771 | 0.0000 | 0.0000 |

### saved_model_pruned_10x2_qat

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9443 | 0.0000 | 0.0000 | 0.0000 | 0.9443 | 1.0000 |
| 90 | 10 | 460,810 | 0.9471 | 0.6608 | 0.9670 | 0.7851 | 0.9449 | 0.9961 |
| 80 | 20 | 425,865 | 0.9490 | 0.8141 | 0.9656 | 0.8834 | 0.9449 | 0.9910 |
| 70 | 30 | 283,910 | 0.9507 | 0.8814 | 0.9656 | 0.9216 | 0.9443 | 0.9846 |
| 60 | 40 | 212,930 | 0.9530 | 0.9207 | 0.9656 | 0.9426 | 0.9446 | 0.9763 |
| 50 | 50 | 170,346 | 0.9551 | 0.9457 | 0.9656 | 0.9555 | 0.9446 | 0.9648 |
| 40 | 60 | 141,955 | 0.9574 | 0.9634 | 0.9656 | 0.9645 | 0.9450 | 0.9482 |
| 30 | 70 | 121,672 | 0.9594 | 0.9762 | 0.9656 | 0.9708 | 0.9450 | 0.9217 |
| 20 | 80 | 106,465 | 0.9613 | 0.9857 | 0.9656 | 0.9756 | 0.9441 | 0.8727 |
| 10 | 90 | 94,630 | 0.9631 | 0.9932 | 0.9656 | 0.9792 | 0.9406 | 0.7523 |
| 0 | 100 | 85,173 | 0.9656 | 1.0000 | 0.9656 | 0.9825 | 0.0000 | 0.0000 |

### saved_model_pruned_5x10_qat

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9667 | 0.0000 | 0.0000 | 0.0000 | 0.9667 | 1.0000 |
| 90 | 10 | 460,810 | 0.9648 | 0.7592 | 0.9484 | 0.8433 | 0.9666 | 0.9941 |
| 80 | 20 | 425,865 | 0.9625 | 0.8759 | 0.9468 | 0.9100 | 0.9665 | 0.9864 |
| 70 | 30 | 283,910 | 0.9603 | 0.9229 | 0.9468 | 0.9347 | 0.9661 | 0.9769 |
| 60 | 40 | 212,930 | 0.9585 | 0.9493 | 0.9468 | 0.9480 | 0.9663 | 0.9646 |
| 50 | 50 | 170,346 | 0.9567 | 0.9658 | 0.9468 | 0.9562 | 0.9665 | 0.9478 |
| 40 | 60 | 141,955 | 0.9546 | 0.9769 | 0.9468 | 0.9616 | 0.9664 | 0.9237 |
| 30 | 70 | 121,672 | 0.9529 | 0.9853 | 0.9468 | 0.9657 | 0.9672 | 0.8863 |
| 20 | 80 | 106,465 | 0.9509 | 0.9914 | 0.9468 | 0.9686 | 0.9671 | 0.8197 |
| 10 | 90 | 94,630 | 0.9486 | 0.9959 | 0.9468 | 0.9707 | 0.9648 | 0.6684 |
| 0 | 100 | 85,173 | 0.9468 | 1.0000 | 0.9468 | 0.9727 | 0.0000 | 0.0000 |


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
0.15       0.1010   0.1820   0.0011   0.9913   0.9999   0.1001  
0.20       0.1014   0.1820   0.0016   0.9807   0.9997   0.1001  
0.25       0.1044   0.1824   0.0050   0.9777   0.9990   0.1004   <--
0.30       0.1069   0.1814   0.0089   0.8835   0.9895   0.0998  
0.35       0.1135   0.1812   0.0172   0.8899   0.9809   0.0998  
0.40       0.1757   0.1676   0.1029   0.8450   0.8301   0.0932  
0.45       0.5036   0.0939   0.5310   0.8655   0.2571   0.0574  
0.50       0.8969   0.0019   0.9964   0.8998   0.0010   0.0293  
0.55       0.9000   0.0002   1.0000   0.9000   0.0001   0.5000  
0.60       0.9000   0.0001   1.0000   0.9000   0.0001   0.7500  
0.65       0.9000   0.0000   1.0000   0.9000   0.0000   1.0000  
0.70       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.75       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.80       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.25
  At threshold 0.25: Accuracy=0.1044, F1=0.1824, Normal Recall=0.0050, Normal Precision=0.9777, Attack Recall=0.9990, Attack Precision=0.1004

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
0.15       0.2009   0.3335   0.0011   0.9673   0.9998   0.2002  
0.20       0.2012   0.3336   0.0016   0.9550   0.9997   0.2002  
0.25       0.2038   0.3342   0.0050   0.9505   0.9990   0.2006   <--
0.30       0.2050   0.3324   0.0089   0.7731   0.9896   0.1998  
0.35       0.2099   0.3318   0.0171   0.7820   0.9809   0.1997  
0.40       0.2486   0.3066   0.1031   0.7090   0.8308   0.1880  
0.45       0.4770   0.1651   0.5316   0.7415   0.2586   0.1213  
0.50       0.7974   0.0019   0.9965   0.7996   0.0010   0.0658  
0.55       0.8000   0.0001   1.0000   0.8000   0.0001   0.6667  
0.60       0.8000   0.0001   1.0000   0.8000   0.0001   0.8333  
0.65       0.8000   0.0000   1.0000   0.8000   0.0000   1.0000  
0.70       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.75       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.80       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.25
  At threshold 0.25: Accuracy=0.2038, F1=0.3342, Normal Recall=0.0050, Normal Precision=0.9505, Attack Recall=0.9990, Attack Precision=0.2006

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
0.15       0.3007   0.4618   0.0011   0.9449   0.9998   0.3002  
0.20       0.3010   0.4618   0.0016   0.9238   0.9997   0.3003  
0.25       0.3031   0.4624   0.0049   0.9177   0.9990   0.3008   <--
0.30       0.3030   0.4600   0.0087   0.6608   0.9896   0.2996  
0.35       0.3061   0.4589   0.0168   0.6728   0.9809   0.2995  
0.40       0.3210   0.4233   0.1025   0.5856   0.8308   0.2840  
0.45       0.4504   0.2201   0.5326   0.6263   0.2586   0.1916  
0.50       0.6978   0.0020   0.9964   0.6995   0.0010   0.1061  
0.55       0.7000   0.0001   1.0000   0.7000   0.0001   0.7500  
0.60       0.7000   0.0001   1.0000   0.7000   0.0001   0.8333  
0.65       0.7000   0.0000   1.0000   0.7000   0.0000   1.0000  
0.70       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.75       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.80       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.25
  At threshold 0.25: Accuracy=0.3031, F1=0.4624, Normal Recall=0.0049, Normal Precision=0.9177, Attack Recall=0.9990, Attack Precision=0.3008

```

### Normal 60% : Attack 40%

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

Test set: Normal 60% : Attack 40%  (n=212,930, N=127,758, A=85,172)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.4006   0.5717   0.0012   0.9202   0.9998   0.4002  
0.20       0.4009   0.5717   0.0016   0.8884   0.9997   0.4003  
0.25       0.4025   0.5722   0.0049   0.8759   0.9990   0.4009   <--
0.30       0.4011   0.5693   0.0087   0.5577   0.9896   0.3996  
0.35       0.4024   0.5677   0.0167   0.5673   0.9809   0.3994  
0.40       0.3938   0.5230   0.1025   0.4759   0.8308   0.3816  
0.45       0.4234   0.2640   0.5333   0.5190   0.2586   0.2697  
0.50       0.5982   0.0020   0.9963   0.5994   0.0010   0.1514  
0.55       0.6000   0.0001   1.0000   0.6000   0.0001   0.7500  
0.60       0.6000   0.0001   1.0000   0.6000   0.0001   0.8333  
0.65       0.6000   0.0000   1.0000   0.6000   0.0000   1.0000  
0.70       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.75       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.80       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.25
  At threshold 0.25: Accuracy=0.4025, F1=0.5722, Normal Recall=0.0049, Normal Precision=0.8759, Attack Recall=0.9990, Attack Precision=0.4009

```

### Normal 50% : Attack 50%

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

Test set: Normal 50% : Attack 50%  (n=170,346, N=85,173, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.5006   0.6669   0.0013   0.8943   0.9998   0.5003  
0.20       0.5007   0.6669   0.0017   0.8480   0.9997   0.5003  
0.25       0.5019   0.6673   0.0049   0.8254   0.9990   0.5010   <--
0.30       0.4993   0.6640   0.0089   0.4621   0.9896   0.4996  
0.35       0.4989   0.6619   0.0168   0.4688   0.9809   0.4994  
0.40       0.4666   0.6090   0.1025   0.3773   0.8308   0.4807  
0.45       0.3965   0.3000   0.5345   0.4189   0.2586   0.3571  
0.50       0.4986   0.0020   0.9963   0.4993   0.0010   0.2090  
0.55       0.5000   0.0001   1.0000   0.5000   0.0001   0.7500  
0.60       0.5000   0.0001   1.0000   0.5000   0.0001   0.8333  
0.65       0.5000   0.0000   1.0000   0.5000   0.0000   1.0000  
0.70       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.75       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.80       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.25
  At threshold 0.25: Accuracy=0.5019, F1=0.6673, Normal Recall=0.0049, Normal Precision=0.8254, Attack Recall=0.9990, Attack Precision=0.5010

```


## Threshold Tuning (noQAT+PTQ)

Model: `models/tflite/saved_model_no_qat_ptq.tflite`

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
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 90% : Attack 10%  (n=460,810, N=414,729, A=46,081)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.1057   0.1828   0.0064   1.0000   1.0000   0.1006  
0.20       0.1091   0.1833   0.0102   1.0000   1.0000   0.1009  
0.25       0.1228   0.1857   0.0253   0.9999   1.0000   0.1023  
0.30       0.1453   0.1896   0.0503   0.9999   1.0000   0.1047  
0.35       0.1609   0.1925   0.0677   0.9999   1.0000   0.1065  
0.40       0.1727   0.1947   0.0808   0.9999   1.0000   0.1078  
0.45       0.1947   0.1989   0.1052   1.0000   1.0000   0.1105  
0.50       0.9267   0.6154   0.9645   0.9545   0.5864   0.6474  
0.55       0.9558   0.7249   0.9972   0.9556   0.5831   0.9580  
0.60       0.9565   0.7277   0.9983   0.9554   0.5810   0.9737  
0.65       0.9567   0.7283   0.9986   0.9554   0.5801   0.9780  
0.70       0.9570   0.7292   0.9991   0.9552   0.5786   0.9858   <--
0.75       0.9571   0.7291   0.9993   0.9551   0.5772   0.9894  
0.80       0.9572   0.7290   0.9997   0.9549   0.5754   0.9946  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.7
  At threshold 0.7: Accuracy=0.9570, F1=0.7292, Normal Recall=0.9991, Normal Precision=0.9552, Attack Recall=0.5786, Attack Precision=0.9858

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
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 80% : Attack 20%  (n=425,865, N=340,692, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.2051   0.3347   0.0063   0.9995   1.0000   0.2010  
0.20       0.2081   0.3356   0.0102   0.9997   1.0000   0.2016  
0.25       0.2202   0.3390   0.0253   0.9998   1.0000   0.2041  
0.30       0.2401   0.3449   0.0502   0.9998   1.0000   0.2084  
0.35       0.2540   0.3490   0.0675   0.9998   1.0000   0.2114  
0.40       0.2645   0.3522   0.0806   0.9999   1.0000   0.2138  
0.45       0.2840   0.3584   0.1050   0.9998   0.9999   0.2183  
0.50       0.8891   0.6794   0.9644   0.9034   0.5877   0.8050  
0.55       0.9145   0.7321   0.9971   0.9056   0.5840   0.9808  
0.60       0.9150   0.7324   0.9982   0.9052   0.5819   0.9881   <--
0.65       0.9150   0.7322   0.9985   0.9050   0.5809   0.9900  
0.70       0.9151   0.7318   0.9991   0.9047   0.5793   0.9935  
0.75       0.9150   0.7312   0.9993   0.9045   0.5779   0.9952  
0.80       0.9150   0.7304   0.9996   0.9042   0.5762   0.9976  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.6
  At threshold 0.6: Accuracy=0.9150, F1=0.7324, Normal Recall=0.9982, Normal Precision=0.9052, Attack Recall=0.5819, Attack Precision=0.9881

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
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 70% : Attack 30%  (n=283,910, N=198,737, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.3045   0.4631   0.0064   0.9992   1.0000   0.3013  
0.20       0.3071   0.4641   0.0102   0.9995   1.0000   0.3021  
0.25       0.3178   0.4679   0.0254   0.9996   1.0000   0.3054  
0.30       0.3352   0.4744   0.0503   0.9996   1.0000   0.3109  
0.35       0.3474   0.4790   0.0677   0.9997   1.0000   0.3149  
0.40       0.3564   0.4825   0.0806   0.9998   1.0000   0.3179  
0.45       0.3733   0.4891   0.1048   0.9997   0.9999   0.3237  
0.50       0.8515   0.7036   0.9645   0.8452   0.5877   0.8765  
0.55       0.8732   0.7343   0.9971   0.8483   0.5840   0.9886   <--
0.60       0.8733   0.7338   0.9982   0.8478   0.5819   0.9930  
0.65       0.8733   0.7333   0.9985   0.8476   0.5809   0.9942  
0.70       0.8731   0.7326   0.9991   0.8471   0.5793   0.9962  
0.75       0.8729   0.7318   0.9993   0.8467   0.5779   0.9972  
0.80       0.8726   0.7307   0.9996   0.8462   0.5762   0.9985  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.8732, F1=0.7343, Normal Recall=0.9971, Normal Precision=0.8483, Attack Recall=0.5840, Attack Precision=0.9886

```

### Normal 60% : Attack 40%

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
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 60% : Attack 40%  (n=212,930, N=127,758, A=85,172)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.4038   0.5730   0.0063   0.9988   1.0000   0.4015  
0.20       0.4061   0.5739   0.0102   0.9992   1.0000   0.4025  
0.25       0.4153   0.5777   0.0256   0.9994   1.0000   0.4062  
0.30       0.4302   0.5840   0.0504   0.9994   1.0000   0.4125  
0.35       0.4408   0.5886   0.0680   0.9995   1.0000   0.4170  
0.40       0.4487   0.5920   0.0812   0.9996   1.0000   0.4205  
0.45       0.4631   0.5984   0.1052   0.9996   0.9999   0.4269  
0.50       0.8140   0.7165   0.9649   0.7783   0.5877   0.9178  
0.55       0.8319   0.7354   0.9972   0.7824   0.5840   0.9927   <--
0.60       0.8317   0.7345   0.9983   0.7817   0.5819   0.9956  
0.65       0.8315   0.7339   0.9986   0.7814   0.5809   0.9963  
0.70       0.8311   0.7329   0.9990   0.7808   0.5793   0.9975  
0.75       0.8307   0.7320   0.9993   0.7803   0.5779   0.9981  
0.80       0.8302   0.7308   0.9996   0.7796   0.5762   0.9990  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.8319, F1=0.7354, Normal Recall=0.9972, Normal Precision=0.7824, Attack Recall=0.5840, Attack Precision=0.9927

```

### Normal 50% : Attack 50%

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
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 50% : Attack 50%  (n=170,346, N=85,173, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.5032   0.6681   0.0064   0.9982   1.0000   0.5016  
0.20       0.5051   0.6689   0.0102   0.9989   1.0000   0.5026  
0.25       0.5129   0.6724   0.0258   0.9991   1.0000   0.5065  
0.30       0.5252   0.6781   0.0505   0.9991   1.0000   0.5129  
0.35       0.5339   0.6821   0.0678   0.9993   1.0000   0.5175  
0.40       0.5406   0.6852   0.0812   0.9994   1.0000   0.5211  
0.45       0.5526   0.6909   0.1054   0.9993   0.9999   0.5278  
0.50       0.7761   0.7241   0.9645   0.7005   0.5877   0.9431  
0.55       0.7905   0.7360   0.9970   0.7056   0.5840   0.9950   <--
0.60       0.7901   0.7349   0.9982   0.7048   0.5819   0.9969  
0.65       0.7897   0.7342   0.9985   0.7044   0.5809   0.9974  
0.70       0.7891   0.7331   0.9989   0.7036   0.5793   0.9981  
0.75       0.7886   0.7322   0.9992   0.7030   0.5779   0.9987  
0.80       0.7879   0.7309   0.9996   0.7022   0.5762   0.9993  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.7905, F1=0.7360, Normal Recall=0.9970, Normal Precision=0.7056, Attack Recall=0.5840, Attack Precision=0.9950

```


## Threshold Tuning (saved_model_traditional_qat)

Model: `models/tflite/saved_model_traditional_qat.tflite`

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
Loading model: models/tflite/saved_model_traditional_qat.tflite

Test set: Normal 90% : Attack 10%  (n=460,810, N=414,729, A=46,081)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9492   0.7946   0.9457   0.9978   0.9815   0.6674  
0.20       0.9503   0.7940   0.9493   0.9952   0.9588   0.6776  
0.25       0.9682   0.8544   0.9719   0.9926   0.9344   0.7870  
0.30       0.9713   0.8664   0.9758   0.9922   0.9309   0.8102  
0.35       0.9717   0.8682   0.9763   0.9922   0.9306   0.8136  
0.40       0.9733   0.8717   0.9808   0.9895   0.9059   0.8400   <--
0.45       0.9730   0.8673   0.9832   0.9868   0.8818   0.8533  
0.50       0.9752   0.8671   0.9938   0.9790   0.8082   0.9352  
0.55       0.9712   0.8346   0.9982   0.9706   0.7275   0.9787  
0.60       0.9686   0.8161   0.9987   0.9674   0.6975   0.9833  
0.65       0.9684   0.8152   0.9987   0.9673   0.6960   0.9837  
0.70       0.9685   0.8154   0.9988   0.9673   0.6957   0.9849  
0.75       0.9684   0.8144   0.9989   0.9671   0.6939   0.9857  
0.80       0.9684   0.8142   0.9989   0.9670   0.6932   0.9862  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9733, F1=0.8717, Normal Recall=0.9808, Normal Precision=0.9895, Attack Recall=0.9059, Attack Precision=0.8400

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
Loading model: models/tflite/saved_model_traditional_qat.tflite

Test set: Normal 80% : Attack 20%  (n=425,865, N=340,692, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9528   0.8927   0.9455   0.9953   0.9820   0.8183  
0.20       0.9511   0.8870   0.9491   0.9893   0.9591   0.8250  
0.25       0.9644   0.9132   0.9718   0.9836   0.9352   0.8922  
0.30       0.9669   0.9184   0.9756   0.9828   0.9318   0.9053  
0.35       0.9673   0.9193   0.9762   0.9828   0.9316   0.9073   <--
0.40       0.9659   0.9141   0.9807   0.9768   0.9069   0.9215  
0.45       0.9627   0.9045   0.9830   0.9708   0.8816   0.9286  
0.50       0.9567   0.8819   0.9937   0.9541   0.8086   0.9698  
0.55       0.9441   0.8390   0.9982   0.9362   0.7278   0.9903  
0.60       0.9385   0.8196   0.9987   0.9297   0.6981   0.9923  
0.65       0.9383   0.8187   0.9987   0.9294   0.6966   0.9925  
0.70       0.9383   0.8187   0.9988   0.9294   0.6964   0.9931  
0.75       0.9380   0.8175   0.9989   0.9290   0.6946   0.9935  
0.80       0.9379   0.8172   0.9989   0.9288   0.6939   0.9937  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.9673, F1=0.9193, Normal Recall=0.9762, Normal Precision=0.9828, Attack Recall=0.9316, Attack Precision=0.9073

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
Loading model: models/tflite/saved_model_traditional_qat.tflite

Test set: Normal 70% : Attack 30%  (n=283,910, N=198,737, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9560   0.9305   0.9448   0.9919   0.9820   0.8841  
0.20       0.9517   0.9225   0.9485   0.9819   0.9591   0.8886  
0.25       0.9605   0.9342   0.9713   0.9722   0.9352   0.9332  
0.30       0.9623   0.9369   0.9754   0.9709   0.9318   0.9419  
0.35       0.9626   0.9373   0.9759   0.9708   0.9316   0.9432   <--
0.40       0.9583   0.9288   0.9804   0.9609   0.9069   0.9519  
0.45       0.9524   0.9174   0.9827   0.9509   0.8816   0.9562  
0.50       0.9381   0.8869   0.9936   0.9237   0.8086   0.9819  
0.55       0.9171   0.8405   0.9983   0.8954   0.7278   0.9945  
0.60       0.9085   0.8208   0.9987   0.8853   0.6981   0.9958  
0.65       0.9081   0.8198   0.9988   0.8848   0.6966   0.9959  
0.70       0.9081   0.8197   0.9989   0.8847   0.6964   0.9962  
0.75       0.9076   0.8185   0.9989   0.8841   0.6946   0.9964  
0.80       0.9074   0.8181   0.9990   0.8839   0.6939   0.9965  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.9626, F1=0.9373, Normal Recall=0.9759, Normal Precision=0.9708, Attack Recall=0.9316, Attack Precision=0.9432

```

### Normal 60% : Attack 40%

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
Loading model: models/tflite/saved_model_traditional_qat.tflite

Test set: Normal 60% : Attack 40%  (n=212,930, N=127,758, A=85,172)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9596   0.9511   0.9446   0.9875   0.9820   0.9220   <--
0.20       0.9526   0.9419   0.9483   0.9720   0.9591   0.9253  
0.25       0.9570   0.9456   0.9715   0.9574   0.9352   0.9563  
0.30       0.9581   0.9467   0.9755   0.9555   0.9318   0.9621  
0.35       0.9583   0.9470   0.9761   0.9553   0.9316   0.9629  
0.40       0.9509   0.9366   0.9803   0.9404   0.9069   0.9684  
0.45       0.9422   0.9242   0.9826   0.9256   0.8816   0.9713  
0.50       0.9196   0.8894   0.9936   0.8862   0.8086   0.9883  
0.55       0.8901   0.8412   0.9982   0.8462   0.7278   0.9964  
0.60       0.8785   0.8213   0.9987   0.8323   0.6981   0.9972  
0.65       0.8779   0.8203   0.9987   0.8316   0.6966   0.9973  
0.70       0.8778   0.8201   0.9988   0.8315   0.6964   0.9974  
0.75       0.8771   0.8189   0.9989   0.8307   0.6945   0.9976  
0.80       0.8769   0.8185   0.9989   0.8304   0.6939   0.9976  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9596, F1=0.9511, Normal Recall=0.9446, Normal Precision=0.9875, Attack Recall=0.9820, Attack Precision=0.9220

```

### Normal 50% : Attack 50%

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
Loading model: models/tflite/saved_model_traditional_qat.tflite

Test set: Normal 50% : Attack 50%  (n=170,346, N=85,173, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9629   0.9636   0.9438   0.9813   0.9820   0.9459   <--
0.20       0.9533   0.9536   0.9475   0.9586   0.9591   0.9481  
0.25       0.9531   0.9523   0.9711   0.9374   0.9352   0.9700  
0.30       0.9536   0.9526   0.9754   0.9347   0.9318   0.9742  
0.35       0.9537   0.9527   0.9758   0.9345   0.9316   0.9747  
0.40       0.9434   0.9413   0.9800   0.9132   0.9069   0.9785  
0.45       0.9321   0.9284   0.9826   0.8924   0.8816   0.9806  
0.50       0.9010   0.8909   0.9935   0.8384   0.8086   0.9920  
0.55       0.8630   0.8416   0.9982   0.7857   0.7278   0.9975  
0.60       0.8484   0.8216   0.9986   0.7679   0.6981   0.9980  
0.65       0.8477   0.8206   0.9987   0.7670   0.6966   0.9981  
0.70       0.8476   0.8204   0.9987   0.7669   0.6964   0.9982  
0.75       0.8467   0.8192   0.9988   0.7658   0.6946   0.9983  
0.80       0.8464   0.8187   0.9988   0.7654   0.6939   0.9983  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9629, F1=0.9636, Normal Recall=0.9438, Normal Precision=0.9813, Attack Recall=0.9820, Attack Precision=0.9459

```


## Threshold Tuning (QAT+PTQ)

Model: `models/tflite/saved_model_qat_ptq.tflite`

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
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 90% : Attack 10%  (n=460,810, N=414,729, A=46,081)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9379   0.7577   0.9342   0.9966   0.9712   0.6212  
0.20       0.9432   0.7734   0.9402   0.9965   0.9699   0.6432  
0.25       0.9455   0.7804   0.9428   0.9964   0.9689   0.6532  
0.30       0.9466   0.7826   0.9449   0.9955   0.9618   0.6597  
0.35       0.9542   0.8073   0.9536   0.9953   0.9594   0.6968  
0.40       0.9638   0.8386   0.9663   0.9933   0.9412   0.7561  
0.45       0.9687   0.8557   0.9733   0.9918   0.9275   0.7943  
0.50       0.9687   0.8557   0.9733   0.9918   0.9275   0.7943  
0.55       0.9715   0.8645   0.9785   0.9897   0.9087   0.8244   <--
0.60       0.9711   0.8618   0.9789   0.9889   0.9013   0.8257  
0.65       0.9693   0.8503   0.9800   0.9858   0.8725   0.8292  
0.70       0.9745   0.8623   0.9941   0.9779   0.7980   0.9378  
0.75       0.9749   0.8614   0.9966   0.9760   0.7795   0.9626  
0.80       0.9747   0.8596   0.9972   0.9753   0.7728   0.9683  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9715, F1=0.8645, Normal Recall=0.9785, Normal Precision=0.9897, Attack Recall=0.9087, Attack Precision=0.8244

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
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 80% : Attack 20%  (n=425,865, N=340,692, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9414   0.8689   0.9340   0.9923   0.9710   0.7862  
0.20       0.9460   0.8777   0.9400   0.9920   0.9697   0.8017  
0.25       0.9479   0.8815   0.9427   0.9918   0.9686   0.8087  
0.30       0.9481   0.8812   0.9448   0.9899   0.9616   0.8132  
0.35       0.9546   0.8943   0.9535   0.9894   0.9591   0.8377  
0.40       0.9612   0.9065   0.9662   0.9850   0.9411   0.8743  
0.45       0.9640   0.9115   0.9732   0.9816   0.9271   0.8965   <--
0.50       0.9640   0.9115   0.9732   0.9816   0.9271   0.8965  
0.55       0.9646   0.9112   0.9784   0.9773   0.9090   0.9134  
0.60       0.9634   0.9080   0.9788   0.9756   0.9020   0.9141  
0.65       0.9587   0.8942   0.9799   0.9688   0.8736   0.9158  
0.70       0.9552   0.8772   0.9942   0.9520   0.7994   0.9717  
0.75       0.9535   0.8705   0.9966   0.9479   0.7810   0.9831  
0.80       0.9525   0.8669   0.9972   0.9463   0.7737   0.9857  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.9640, F1=0.9115, Normal Recall=0.9732, Normal Precision=0.9816, Attack Recall=0.9271, Attack Precision=0.8965

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
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 70% : Attack 30%  (n=283,910, N=198,737, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9447   0.9134   0.9335   0.9868   0.9710   0.8622  
0.20       0.9485   0.9186   0.9394   0.9863   0.9697   0.8727  
0.25       0.9501   0.9209   0.9422   0.9859   0.9686   0.8777  
0.30       0.9495   0.9195   0.9442   0.9829   0.9616   0.8808  
0.35       0.9548   0.9272   0.9530   0.9819   0.9591   0.8973  
0.40       0.9582   0.9311   0.9656   0.9745   0.9411   0.9214  
0.45       0.9591   0.9315   0.9728   0.9689   0.9271   0.9359   <--
0.50       0.9591   0.9315   0.9728   0.9689   0.9271   0.9359  
0.55       0.9573   0.9274   0.9780   0.9616   0.9090   0.9466  
0.60       0.9555   0.9240   0.9784   0.9588   0.9020   0.9471  
0.65       0.9478   0.9094   0.9796   0.9476   0.8736   0.9482  
0.70       0.9357   0.8818   0.9941   0.9204   0.7994   0.9831  
0.75       0.9320   0.8732   0.9967   0.9139   0.7810   0.9902  
0.80       0.9302   0.8692   0.9972   0.9114   0.7737   0.9916  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.9591, F1=0.9315, Normal Recall=0.9728, Normal Precision=0.9689, Attack Recall=0.9271, Attack Precision=0.9359

```

### Normal 60% : Attack 40%

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
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 60% : Attack 40%  (n=212,930, N=127,758, A=85,172)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9482   0.9375   0.9330   0.9797   0.9710   0.9062  
0.20       0.9513   0.9409   0.9390   0.9789   0.9697   0.9138  
0.25       0.9525   0.9422   0.9417   0.9783   0.9686   0.9172  
0.30       0.9510   0.9401   0.9439   0.9736   0.9616   0.9195  
0.35       0.9551   0.9447   0.9524   0.9722   0.9591   0.9308   <--
0.40       0.9557   0.9444   0.9654   0.9609   0.9411   0.9478  
0.45       0.9544   0.9420   0.9726   0.9524   0.9271   0.9575  
0.50       0.9544   0.9420   0.9726   0.9524   0.9271   0.9575  
0.55       0.9503   0.9361   0.9779   0.9416   0.9090   0.9649  
0.60       0.9478   0.9326   0.9784   0.9374   0.9020   0.9653  
0.65       0.9371   0.9175   0.9794   0.9208   0.8736   0.9659  
0.70       0.9162   0.8841   0.9941   0.8814   0.7994   0.9890  
0.75       0.9103   0.8745   0.9965   0.8722   0.7810   0.9934  
0.80       0.9077   0.8702   0.9971   0.8686   0.7737   0.9943  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.9551, F1=0.9447, Normal Recall=0.9524, Normal Precision=0.9722, Attack Recall=0.9591, Attack Precision=0.9308

```

### Normal 50% : Attack 50%

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
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 50% : Attack 50%  (n=170,346, N=85,173, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9519   0.9528   0.9329   0.9698   0.9710   0.9354  
0.20       0.9543   0.9550   0.9389   0.9687   0.9697   0.9407  
0.25       0.9551   0.9557   0.9416   0.9678   0.9686   0.9431  
0.30       0.9526   0.9530   0.9436   0.9609   0.9616   0.9446  
0.35       0.9556   0.9558   0.9522   0.9588   0.9591   0.9525   <--
0.40       0.9532   0.9526   0.9652   0.9425   0.9411   0.9644  
0.45       0.9498   0.9486   0.9725   0.9302   0.9271   0.9712  
0.50       0.9498   0.9486   0.9725   0.9302   0.9271   0.9712  
0.55       0.9435   0.9415   0.9781   0.9148   0.9090   0.9765  
0.60       0.9402   0.9379   0.9785   0.9090   0.9020   0.9767  
0.65       0.9266   0.9225   0.9795   0.8857   0.8736   0.9771  
0.70       0.8967   0.8856   0.9940   0.8321   0.7994   0.9925  
0.75       0.8888   0.8753   0.9965   0.8198   0.7810   0.9956  
0.80       0.8854   0.8710   0.9971   0.8150   0.7737   0.9963  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.9556, F1=0.9558, Normal Recall=0.9522, Normal Precision=0.9588, Attack Recall=0.9591, Attack Precision=0.9525

```


## Threshold Tuning (saved_model_pruned_qat)

Model: `models/tflite/saved_model_pruned_qat.tflite`

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
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 90% : Attack 10%  (n=460,810, N=414,729, A=46,081)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9191   0.7095   0.9114   0.9986   0.9881   0.5534  
0.20       0.9243   0.7229   0.9173   0.9985   0.9875   0.5702  
0.25       0.9492   0.7901   0.9483   0.9950   0.9570   0.6728  
0.30       0.9530   0.8024   0.9528   0.9948   0.9548   0.6919  
0.35       0.9564   0.8132   0.9571   0.9942   0.9498   0.7109  
0.40       0.9583   0.8199   0.9593   0.9941   0.9492   0.7217  
0.45       0.9605   0.8276   0.9620   0.9939   0.9472   0.7348  
0.50       0.9638   0.8382   0.9668   0.9928   0.9372   0.7581  
0.55       0.9651   0.8431   0.9683   0.9928   0.9369   0.7664  
0.60       0.9663   0.8460   0.9707   0.9917   0.9267   0.7782  
0.65       0.9659   0.8420   0.9725   0.9895   0.9073   0.7854  
0.70       0.9684   0.8500   0.9764   0.9883   0.8964   0.8082  
0.75       0.9684   0.8458   0.9798   0.9851   0.8663   0.8262  
0.80       0.9734   0.8635   0.9882   0.9823   0.8402   0.8882   <--
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.8
  At threshold 0.8: Accuracy=0.9734, F1=0.8635, Normal Recall=0.9882, Normal Precision=0.9823, Attack Recall=0.8402, Attack Precision=0.8882

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
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 80% : Attack 20%  (n=425,865, N=340,692, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9267   0.8435   0.9113   0.9967   0.9881   0.7359  
0.20       0.9313   0.8518   0.9172   0.9966   0.9875   0.7488  
0.25       0.9502   0.8849   0.9483   0.9889   0.9575   0.8225  
0.30       0.9533   0.8910   0.9528   0.9884   0.9552   0.8349  
0.35       0.9557   0.8955   0.9570   0.9871   0.9501   0.8469  
0.40       0.9573   0.8989   0.9592   0.9870   0.9494   0.8535  
0.45       0.9591   0.9025   0.9620   0.9865   0.9475   0.8616  
0.50       0.9609   0.9056   0.9667   0.9842   0.9378   0.8756  
0.55       0.9621   0.9082   0.9682   0.9841   0.9375   0.8807   <--
0.60       0.9619   0.9069   0.9707   0.9815   0.9269   0.8876  
0.65       0.9594   0.8993   0.9725   0.9766   0.9068   0.8919  
0.70       0.9604   0.9004   0.9765   0.9741   0.8960   0.9049  
0.75       0.9572   0.8900   0.9799   0.9670   0.8664   0.9149  
0.80       0.9588   0.8908   0.9883   0.9612   0.8406   0.9474  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9621, F1=0.9082, Normal Recall=0.9682, Normal Precision=0.9841, Attack Recall=0.9375, Attack Precision=0.8807

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
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 70% : Attack 30%  (n=283,910, N=198,737, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9342   0.9001   0.9111   0.9944   0.9881   0.8265  
0.20       0.9381   0.9054   0.9169   0.9942   0.9875   0.8359  
0.25       0.9511   0.9215   0.9483   0.9812   0.9575   0.8881  
0.30       0.9535   0.9250   0.9528   0.9803   0.9552   0.8966  
0.35       0.9550   0.9268   0.9571   0.9781   0.9501   0.9046  
0.40       0.9563   0.9288   0.9593   0.9779   0.9494   0.9091  
0.45       0.9577   0.9308   0.9621   0.9772   0.9475   0.9146  
0.50       0.9580   0.9305   0.9667   0.9732   0.9378   0.9234  
0.55       0.9590   0.9321   0.9683   0.9731   0.9375   0.9268   <--
0.60       0.9576   0.9291   0.9707   0.9688   0.9269   0.9313  
0.65       0.9528   0.9201   0.9725   0.9605   0.9068   0.9339  
0.70       0.9523   0.9185   0.9764   0.9563   0.8960   0.9421  
0.75       0.9459   0.9057   0.9799   0.9448   0.8664   0.9487  
0.80       0.9441   0.9002   0.9884   0.9354   0.8406   0.9689  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9590, F1=0.9321, Normal Recall=0.9683, Normal Precision=0.9731, Attack Recall=0.9375, Attack Precision=0.9268

```

### Normal 60% : Attack 40%

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
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 60% : Attack 40%  (n=212,930, N=127,758, A=85,172)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9414   0.9310   0.9103   0.9914   0.9881   0.8801  
0.20       0.9447   0.9346   0.9162   0.9910   0.9875   0.8870  
0.25       0.9517   0.9407   0.9479   0.9710   0.9575   0.9245  
0.30       0.9536   0.9427   0.9524   0.9696   0.9552   0.9305  
0.35       0.9540   0.9430   0.9567   0.9664   0.9501   0.9360  
0.40       0.9551   0.9442   0.9590   0.9660   0.9494   0.9391  
0.45       0.9561   0.9453   0.9619   0.9649   0.9475   0.9431   <--
0.50       0.9549   0.9433   0.9664   0.9588   0.9378   0.9490  
0.55       0.9558   0.9443   0.9680   0.9587   0.9375   0.9513  
0.60       0.9530   0.9404   0.9704   0.9522   0.9269   0.9543  
0.65       0.9461   0.9308   0.9723   0.9399   0.9068   0.9561  
0.70       0.9441   0.9276   0.9762   0.9337   0.8960   0.9617  
0.75       0.9344   0.9135   0.9798   0.9167   0.8664   0.9661  
0.80       0.9291   0.9047   0.9881   0.9029   0.8406   0.9793  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.9561, F1=0.9453, Normal Recall=0.9619, Normal Precision=0.9649, Attack Recall=0.9475, Attack Precision=0.9431

```

### Normal 50% : Attack 50%

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
Loading model: models/tflite/saved_model_pruned_qat.tflite

Test set: Normal 50% : Attack 50%  (n=170,346, N=85,173, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9493   0.9512   0.9105   0.9871   0.9881   0.9169  
0.20       0.9519   0.9535   0.9163   0.9865   0.9875   0.9219  
0.25       0.9528   0.9530   0.9480   0.9571   0.9575   0.9485  
0.30       0.9537   0.9538   0.9522   0.9551   0.9552   0.9524  
0.35       0.9533   0.9532   0.9566   0.9504   0.9501   0.9563  
0.40       0.9542   0.9540   0.9590   0.9499   0.9494   0.9586  
0.45       0.9547   0.9544   0.9619   0.9483   0.9475   0.9613   <--
0.50       0.9521   0.9514   0.9665   0.9395   0.9378   0.9655  
0.55       0.9527   0.9520   0.9680   0.9393   0.9375   0.9670  
0.60       0.9487   0.9475   0.9704   0.9300   0.9269   0.9691  
0.65       0.9395   0.9375   0.9723   0.9125   0.9068   0.9703  
0.70       0.9361   0.9334   0.9762   0.9037   0.8960   0.9741  
0.75       0.9232   0.9185   0.9799   0.8800   0.8664   0.9774  
0.80       0.9144   0.9076   0.9881   0.8611   0.8406   0.9861  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.9547, F1=0.9544, Normal Recall=0.9619, Normal Precision=0.9483, Attack Recall=0.9475, Attack Precision=0.9613

```


## Threshold Tuning (saved_model_pruned_10x5_qat)

Model: `models/tflite/saved_model_pruned_10x5_qat.tflite`

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
Loading model: models/tflite/saved_model_pruned_10x5_qat.tflite

Test set: Normal 90% : Attack 10%  (n=460,810, N=414,729, A=46,081)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9191   0.7095   0.9114   0.9986   0.9881   0.5534  
0.20       0.9243   0.7229   0.9173   0.9985   0.9875   0.5702  
0.25       0.9492   0.7901   0.9483   0.9950   0.9570   0.6728  
0.30       0.9530   0.8024   0.9528   0.9948   0.9548   0.6919  
0.35       0.9564   0.8132   0.9571   0.9942   0.9498   0.7109  
0.40       0.9583   0.8199   0.9593   0.9941   0.9492   0.7217  
0.45       0.9605   0.8276   0.9620   0.9939   0.9472   0.7348  
0.50       0.9638   0.8382   0.9668   0.9928   0.9372   0.7581  
0.55       0.9651   0.8431   0.9683   0.9928   0.9369   0.7664  
0.60       0.9663   0.8460   0.9707   0.9917   0.9267   0.7782  
0.65       0.9659   0.8420   0.9725   0.9895   0.9073   0.7854  
0.70       0.9684   0.8500   0.9764   0.9883   0.8964   0.8082  
0.75       0.9684   0.8458   0.9798   0.9851   0.8663   0.8262  
0.80       0.9734   0.8635   0.9882   0.9823   0.8402   0.8882   <--
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.8
  At threshold 0.8: Accuracy=0.9734, F1=0.8635, Normal Recall=0.9882, Normal Precision=0.9823, Attack Recall=0.8402, Attack Precision=0.8882

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
Loading model: models/tflite/saved_model_pruned_10x5_qat.tflite

Test set: Normal 80% : Attack 20%  (n=425,865, N=340,692, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9267   0.8435   0.9113   0.9967   0.9881   0.7359  
0.20       0.9313   0.8518   0.9172   0.9966   0.9875   0.7488  
0.25       0.9502   0.8849   0.9483   0.9889   0.9575   0.8225  
0.30       0.9533   0.8910   0.9528   0.9884   0.9552   0.8349  
0.35       0.9557   0.8955   0.9570   0.9871   0.9501   0.8469  
0.40       0.9573   0.8989   0.9592   0.9870   0.9494   0.8535  
0.45       0.9591   0.9025   0.9620   0.9865   0.9475   0.8616  
0.50       0.9609   0.9056   0.9667   0.9842   0.9378   0.8756  
0.55       0.9621   0.9082   0.9682   0.9841   0.9375   0.8807   <--
0.60       0.9619   0.9069   0.9707   0.9815   0.9269   0.8876  
0.65       0.9594   0.8993   0.9725   0.9766   0.9068   0.8919  
0.70       0.9604   0.9004   0.9765   0.9741   0.8960   0.9049  
0.75       0.9572   0.8900   0.9799   0.9670   0.8664   0.9149  
0.80       0.9588   0.8908   0.9883   0.9612   0.8406   0.9474  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9621, F1=0.9082, Normal Recall=0.9682, Normal Precision=0.9841, Attack Recall=0.9375, Attack Precision=0.8807

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
Loading model: models/tflite/saved_model_pruned_10x5_qat.tflite

Test set: Normal 70% : Attack 30%  (n=283,910, N=198,737, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9342   0.9001   0.9111   0.9944   0.9881   0.8265  
0.20       0.9381   0.9054   0.9169   0.9942   0.9875   0.8359  
0.25       0.9511   0.9215   0.9483   0.9812   0.9575   0.8881  
0.30       0.9535   0.9250   0.9528   0.9803   0.9552   0.8966  
0.35       0.9550   0.9268   0.9571   0.9781   0.9501   0.9046  
0.40       0.9563   0.9288   0.9593   0.9779   0.9494   0.9091  
0.45       0.9577   0.9308   0.9621   0.9772   0.9475   0.9146  
0.50       0.9580   0.9305   0.9667   0.9732   0.9378   0.9234  
0.55       0.9590   0.9321   0.9683   0.9731   0.9375   0.9268   <--
0.60       0.9576   0.9291   0.9707   0.9688   0.9269   0.9313  
0.65       0.9528   0.9201   0.9725   0.9605   0.9068   0.9339  
0.70       0.9523   0.9185   0.9764   0.9563   0.8960   0.9421  
0.75       0.9459   0.9057   0.9799   0.9448   0.8664   0.9487  
0.80       0.9441   0.9002   0.9884   0.9354   0.8406   0.9689  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9590, F1=0.9321, Normal Recall=0.9683, Normal Precision=0.9731, Attack Recall=0.9375, Attack Precision=0.9268

```

### Normal 60% : Attack 40%

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
Loading model: models/tflite/saved_model_pruned_10x5_qat.tflite

Test set: Normal 60% : Attack 40%  (n=212,930, N=127,758, A=85,172)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9414   0.9310   0.9103   0.9914   0.9881   0.8801  
0.20       0.9447   0.9346   0.9162   0.9910   0.9875   0.8870  
0.25       0.9517   0.9407   0.9479   0.9710   0.9575   0.9245  
0.30       0.9536   0.9427   0.9524   0.9696   0.9552   0.9305  
0.35       0.9540   0.9430   0.9567   0.9664   0.9501   0.9360  
0.40       0.9551   0.9442   0.9590   0.9660   0.9494   0.9391  
0.45       0.9561   0.9453   0.9619   0.9649   0.9475   0.9431   <--
0.50       0.9549   0.9433   0.9664   0.9588   0.9378   0.9490  
0.55       0.9558   0.9443   0.9680   0.9587   0.9375   0.9513  
0.60       0.9530   0.9404   0.9704   0.9522   0.9269   0.9543  
0.65       0.9461   0.9308   0.9723   0.9399   0.9068   0.9561  
0.70       0.9441   0.9276   0.9762   0.9337   0.8960   0.9617  
0.75       0.9344   0.9135   0.9798   0.9167   0.8664   0.9661  
0.80       0.9291   0.9047   0.9881   0.9029   0.8406   0.9793  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.9561, F1=0.9453, Normal Recall=0.9619, Normal Precision=0.9649, Attack Recall=0.9475, Attack Precision=0.9431

```

### Normal 50% : Attack 50%

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
Loading model: models/tflite/saved_model_pruned_10x5_qat.tflite

Test set: Normal 50% : Attack 50%  (n=170,346, N=85,173, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9493   0.9512   0.9105   0.9871   0.9881   0.9169  
0.20       0.9519   0.9535   0.9163   0.9865   0.9875   0.9219  
0.25       0.9528   0.9530   0.9480   0.9571   0.9575   0.9485  
0.30       0.9537   0.9538   0.9522   0.9551   0.9552   0.9524  
0.35       0.9533   0.9532   0.9566   0.9504   0.9501   0.9563  
0.40       0.9542   0.9540   0.9590   0.9499   0.9494   0.9586  
0.45       0.9547   0.9544   0.9619   0.9483   0.9475   0.9613   <--
0.50       0.9521   0.9514   0.9665   0.9395   0.9378   0.9655  
0.55       0.9527   0.9520   0.9680   0.9393   0.9375   0.9670  
0.60       0.9487   0.9475   0.9704   0.9300   0.9269   0.9691  
0.65       0.9395   0.9375   0.9723   0.9125   0.9068   0.9703  
0.70       0.9361   0.9334   0.9762   0.9037   0.8960   0.9741  
0.75       0.9232   0.9185   0.9799   0.8800   0.8664   0.9774  
0.80       0.9144   0.9076   0.9881   0.8611   0.8406   0.9861  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.9547, F1=0.9544, Normal Recall=0.9619, Normal Precision=0.9483, Attack Recall=0.9475, Attack Precision=0.9613

```


## Threshold Tuning (saved_model_pruned_10x2_qat)

Model: `models/tflite/saved_model_pruned_10x2_qat.tflite`

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
Loading model: models/tflite/saved_model_pruned_10x2_qat.tflite

Test set: Normal 90% : Attack 10%  (n=460,810, N=414,729, A=46,081)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9264   0.7265   0.9207   0.9973   0.9778   0.5780  
0.20       0.9369   0.7554   0.9326   0.9970   0.9750   0.6165  
0.25       0.9455   0.7803   0.9429   0.9963   0.9683   0.6534  
0.30       0.9469   0.7844   0.9449   0.9960   0.9656   0.6605  
0.35       0.9657   0.8453   0.9690   0.9928   0.9364   0.7704  
0.40       0.9664   0.8471   0.9704   0.9921   0.9308   0.7772  
0.45       0.9687   0.8536   0.9750   0.9901   0.9121   0.8021  
0.50       0.9710   0.8603   0.9796   0.9881   0.8936   0.8293  
0.55       0.9751   0.8696   0.9911   0.9814   0.8307   0.9123   <--
0.60       0.9743   0.8604   0.9947   0.9772   0.7912   0.9428  
0.65       0.9750   0.8625   0.9960   0.9766   0.7852   0.9566  
0.70       0.9756   0.8648   0.9971   0.9762   0.7816   0.9678  
0.75       0.9747   0.8590   0.9974   0.9750   0.7702   0.9711  
0.80       0.9748   0.8590   0.9977   0.9749   0.7686   0.9736  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9751, F1=0.8696, Normal Recall=0.9911, Normal Precision=0.9814, Attack Recall=0.8307, Attack Precision=0.9123

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
Loading model: models/tflite/saved_model_pruned_10x2_qat.tflite

Test set: Normal 80% : Attack 20%  (n=425,865, N=340,692, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9320   0.8519   0.9205   0.9941   0.9780   0.7546  
0.20       0.9410   0.8685   0.9324   0.9934   0.9751   0.7829  
0.25       0.9480   0.8817   0.9429   0.9917   0.9686   0.8091  
0.30       0.9490   0.8833   0.9448   0.9910   0.9656   0.8139  
0.35       0.9625   0.9089   0.9689   0.9839   0.9366   0.8829   <--
0.40       0.9626   0.9087   0.9703   0.9827   0.9315   0.8869  
0.45       0.9624   0.9066   0.9750   0.9780   0.9122   0.9011  
0.50       0.9623   0.9047   0.9795   0.9736   0.8937   0.9159  
0.55       0.9592   0.8908   0.9912   0.9592   0.8315   0.9593  
0.60       0.9542   0.8738   0.9947   0.9504   0.7925   0.9737  
0.65       0.9541   0.8726   0.9960   0.9491   0.7864   0.9802  
0.70       0.9542   0.8724   0.9971   0.9483   0.7825   0.9854  
0.75       0.9524   0.8663   0.9975   0.9459   0.7719   0.9870  
0.80       0.9522   0.8657   0.9977   0.9455   0.7702   0.9882  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.9625, F1=0.9089, Normal Recall=0.9689, Normal Precision=0.9839, Attack Recall=0.9366, Attack Precision=0.8829

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
Loading model: models/tflite/saved_model_pruned_10x2_qat.tflite

Test set: Normal 70% : Attack 30%  (n=283,910, N=198,737, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9374   0.9036   0.9199   0.9899   0.9780   0.8396  
0.20       0.9449   0.9139   0.9319   0.9887   0.9751   0.8599  
0.25       0.9504   0.9214   0.9426   0.9859   0.9686   0.8786  
0.30       0.9508   0.9218   0.9445   0.9846   0.9656   0.8818  
0.35       0.9591   0.9322   0.9688   0.9727   0.9366   0.9279   <--
0.40       0.9585   0.9309   0.9701   0.9706   0.9315   0.9303  
0.45       0.9560   0.9256   0.9748   0.9628   0.9122   0.9393  
0.50       0.9536   0.9203   0.9792   0.9555   0.8937   0.9486  
0.55       0.9432   0.8978   0.9911   0.9321   0.8315   0.9755  
0.60       0.9340   0.8781   0.9947   0.9179   0.7925   0.9845  
0.65       0.9331   0.8758   0.9960   0.9158   0.7864   0.9882  
0.70       0.9327   0.8747   0.9971   0.9145   0.7825   0.9915  
0.75       0.9298   0.8684   0.9975   0.9108   0.7719   0.9925  
0.80       0.9295   0.8676   0.9977   0.9101   0.7702   0.9932  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.9591, F1=0.9322, Normal Recall=0.9688, Normal Precision=0.9727, Attack Recall=0.9366, Attack Precision=0.9279

```

### Normal 60% : Attack 40%

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
Loading model: models/tflite/saved_model_pruned_10x2_qat.tflite

Test set: Normal 60% : Attack 40%  (n=212,930, N=127,758, A=85,172)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9427   0.9317   0.9191   0.9843   0.9780   0.8896  
0.20       0.9489   0.9385   0.9314   0.9825   0.9751   0.9046  
0.25       0.9529   0.9427   0.9424   0.9783   0.9686   0.9181  
0.30       0.9528   0.9425   0.9444   0.9763   0.9656   0.9204  
0.35       0.9561   0.9446   0.9691   0.9582   0.9366   0.9528   <--
0.40       0.9548   0.9428   0.9703   0.9551   0.9315   0.9544  
0.45       0.9498   0.9356   0.9749   0.9433   0.9122   0.9604  
0.50       0.9450   0.9285   0.9791   0.9325   0.8937   0.9662  
0.55       0.9272   0.9013   0.9910   0.8982   0.8315   0.9840  
0.60       0.9138   0.8803   0.9946   0.8779   0.7925   0.9898  
0.65       0.9121   0.8774   0.9959   0.8749   0.7864   0.9922  
0.70       0.9112   0.8758   0.9970   0.8731   0.7825   0.9943  
0.75       0.9072   0.8694   0.9974   0.8677   0.7719   0.9950  
0.80       0.9066   0.8684   0.9976   0.8669   0.7701   0.9954  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.9561, F1=0.9446, Normal Recall=0.9691, Normal Precision=0.9582, Attack Recall=0.9366, Attack Precision=0.9528

```

### Normal 50% : Attack 50%

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
Loading model: models/tflite/saved_model_pruned_10x2_qat.tflite

Test set: Normal 50% : Attack 50%  (n=170,346, N=85,173, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9485   0.9500   0.9190   0.9767   0.9780   0.9235  
0.20       0.9532   0.9542   0.9313   0.9740   0.9751   0.9342  
0.25       0.9554   0.9560   0.9423   0.9677   0.9686   0.9438   <--
0.30       0.9548   0.9553   0.9441   0.9648   0.9656   0.9453  
0.35       0.9529   0.9521   0.9692   0.9386   0.9366   0.9682  
0.40       0.9509   0.9500   0.9703   0.9341   0.9315   0.9691  
0.45       0.9435   0.9416   0.9748   0.9173   0.9122   0.9731  
0.50       0.9363   0.9335   0.9790   0.9020   0.8937   0.9770  
0.55       0.9112   0.9035   0.9909   0.8547   0.8315   0.9891  
0.60       0.8935   0.8816   0.9945   0.8274   0.7925   0.9931  
0.65       0.8911   0.8784   0.9958   0.8234   0.7864   0.9947  
0.70       0.8898   0.8766   0.9971   0.8210   0.7825   0.9962  
0.75       0.8847   0.8700   0.9974   0.8139   0.7719   0.9966  
0.80       0.8839   0.8690   0.9976   0.8127   0.7702   0.9969  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.25
  At threshold 0.25: Accuracy=0.9554, F1=0.9560, Normal Recall=0.9423, Normal Precision=0.9677, Attack Recall=0.9686, Attack Precision=0.9438

```


## Threshold Tuning (saved_model_pruned_5x10_qat)

Model: `models/tflite/saved_model_pruned_5x10_qat.tflite`

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
Loading model: models/tflite/saved_model_pruned_5x10_qat.tflite

Test set: Normal 90% : Attack 10%  (n=460,810, N=414,729, A=46,081)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9370   0.7577   0.9317   0.9982   0.9845   0.6157  
0.20       0.9397   0.7653   0.9349   0.9980   0.9828   0.6266  
0.25       0.9624   0.8352   0.9635   0.9946   0.9526   0.7435  
0.30       0.9646   0.8425   0.9666   0.9939   0.9467   0.7589  
0.35       0.9663   0.8489   0.9685   0.9939   0.9462   0.7697  
0.40       0.9676   0.8535   0.9703   0.9936   0.9435   0.7792  
0.45       0.9706   0.8627   0.9757   0.9915   0.9245   0.8087  
0.50       0.9709   0.8639   0.9763   0.9913   0.9226   0.8122  
0.55       0.9709   0.8610   0.9788   0.9888   0.9003   0.8251  
0.60       0.9729   0.8677   0.9825   0.9874   0.8872   0.8489  
0.65       0.9760   0.8737   0.9922   0.9814   0.8306   0.9216   <--
0.70       0.9764   0.8735   0.9945   0.9796   0.8137   0.9427  
0.75       0.9761   0.8702   0.9954   0.9784   0.8023   0.9507  
0.80       0.9762   0.8709   0.9956   0.9784   0.8019   0.9529  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.65
  At threshold 0.65: Accuracy=0.9760, F1=0.8737, Normal Recall=0.9922, Normal Precision=0.9814, Attack Recall=0.8306, Attack Precision=0.9216

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
Loading model: models/tflite/saved_model_pruned_5x10_qat.tflite

Test set: Normal 80% : Attack 20%  (n=425,865, N=340,692, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9422   0.8719   0.9315   0.9959   0.9846   0.7824  
0.20       0.9443   0.8760   0.9347   0.9954   0.9828   0.7901  
0.25       0.9613   0.9079   0.9634   0.9880   0.9530   0.8669  
0.30       0.9625   0.9100   0.9665   0.9864   0.9468   0.8760  
0.35       0.9640   0.9132   0.9684   0.9863   0.9464   0.8823  
0.40       0.9649   0.9149   0.9702   0.9857   0.9437   0.8879   <--
0.45       0.9654   0.9145   0.9756   0.9811   0.9248   0.9045  
0.50       0.9655   0.9146   0.9762   0.9806   0.9229   0.9065  
0.55       0.9630   0.9069   0.9787   0.9752   0.9004   0.9136  
0.60       0.9634   0.9065   0.9824   0.9721   0.8874   0.9266  
0.65       0.9600   0.8926   0.9922   0.9592   0.8314   0.9636  
0.70       0.9585   0.8870   0.9945   0.9554   0.8145   0.9738  
0.75       0.9569   0.8817   0.9954   0.9529   0.8031   0.9774  
0.80       0.9570   0.8819   0.9956   0.9528   0.8027   0.9784  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9649, F1=0.9149, Normal Recall=0.9702, Normal Precision=0.9857, Attack Recall=0.9437, Attack Precision=0.8879

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
Loading model: models/tflite/saved_model_pruned_5x10_qat.tflite

Test set: Normal 70% : Attack 30%  (n=283,910, N=198,737, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9471   0.9179   0.9311   0.9930   0.9846   0.8596  
0.20       0.9488   0.9201   0.9342   0.9922   0.9828   0.8649  
0.25       0.9602   0.9349   0.9632   0.9795   0.9530   0.9174  
0.30       0.9604   0.9349   0.9663   0.9770   0.9468   0.9233  
0.35       0.9617   0.9367   0.9682   0.9768   0.9464   0.9273  
0.40       0.9621   0.9372   0.9700   0.9757   0.9437   0.9308   <--
0.45       0.9601   0.9329   0.9752   0.9680   0.9248   0.9412  
0.50       0.9600   0.9326   0.9759   0.9673   0.9229   0.9425  
0.55       0.9550   0.9232   0.9785   0.9582   0.9004   0.9472  
0.60       0.9537   0.9201   0.9822   0.9532   0.8874   0.9553  
0.65       0.9439   0.8989   0.9921   0.9321   0.8314   0.9783  
0.70       0.9405   0.8915   0.9945   0.9260   0.8145   0.9845  
0.75       0.9377   0.8854   0.9953   0.9218   0.8031   0.9866  
0.80       0.9377   0.8855   0.9955   0.9217   0.8027   0.9872  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9621, F1=0.9372, Normal Recall=0.9700, Normal Precision=0.9757, Attack Recall=0.9437, Attack Precision=0.9308

```

### Normal 60% : Attack 40%

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
Loading model: models/tflite/saved_model_pruned_5x10_qat.tflite

Test set: Normal 60% : Attack 40%  (n=212,930, N=127,758, A=85,172)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9523   0.9429   0.9308   0.9891   0.9846   0.9046  
0.20       0.9535   0.9441   0.9339   0.9879   0.9828   0.9084  
0.25       0.9592   0.9492   0.9632   0.9685   0.9530   0.9453  
0.30       0.9585   0.9481   0.9664   0.9646   0.9468   0.9494  
0.35       0.9596   0.9494   0.9684   0.9644   0.9464   0.9524   <--
0.40       0.9595   0.9491   0.9701   0.9628   0.9437   0.9546  
0.45       0.9550   0.9427   0.9752   0.9511   0.9248   0.9613  
0.50       0.9546   0.9421   0.9757   0.9500   0.9229   0.9621  
0.55       0.9472   0.9317   0.9784   0.9364   0.9004   0.9652  
0.60       0.9442   0.9271   0.9821   0.9290   0.8874   0.9706  
0.65       0.9277   0.9020   0.9919   0.8982   0.8314   0.9857  
0.70       0.9224   0.8936   0.9944   0.8894   0.8145   0.9897  
0.75       0.9183   0.8872   0.9952   0.8835   0.8031   0.9911  
0.80       0.9183   0.8872   0.9954   0.8833   0.8027   0.9915  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.9596, F1=0.9494, Normal Recall=0.9684, Normal Precision=0.9644, Attack Recall=0.9464, Attack Precision=0.9524

```

### Normal 50% : Attack 50%

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
Loading model: models/tflite/saved_model_pruned_5x10_qat.tflite

Test set: Normal 50% : Attack 50%  (n=170,346, N=85,173, A=85,173)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9575   0.9586   0.9304   0.9837   0.9846   0.9340  
0.20       0.9582   0.9592   0.9336   0.9819   0.9828   0.9367   <--
0.25       0.9582   0.9580   0.9635   0.9535   0.9530   0.9631  
0.30       0.9567   0.9563   0.9666   0.9478   0.9468   0.9660  
0.35       0.9576   0.9571   0.9688   0.9476   0.9464   0.9680  
0.40       0.9570   0.9564   0.9702   0.9452   0.9437   0.9694  
0.45       0.9501   0.9488   0.9753   0.9284   0.9248   0.9740  
0.50       0.9494   0.9481   0.9760   0.9268   0.9229   0.9746  
0.55       0.9395   0.9370   0.9786   0.9076   0.9004   0.9768  
0.60       0.9348   0.9316   0.9823   0.8971   0.8874   0.9804  
0.65       0.9117   0.9040   0.9920   0.8547   0.8314   0.9904  
0.70       0.9044   0.8950   0.9944   0.8428   0.8145   0.9932  
0.75       0.8992   0.8885   0.9953   0.8348   0.8031   0.9942  
0.80       0.8991   0.8883   0.9955   0.8346   0.8027   0.9944  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.2
  At threshold 0.2: Accuracy=0.9582, F1=0.9592, Normal Recall=0.9336, Normal Precision=0.9819, Attack Recall=0.9828, Attack Precision=0.9367

```

