# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 8 models (same as compression_analysis) |
| **Config** | `./config/federated_local_sky.yaml` |
| **Generated** | 2026-03-16 06:30:31 |

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
| Original (TFLite) | 0.0001 | 0.1002 | 0.2001 | 0.3001 | 0.4001 | 0.5001 | 0.6001 | 0.7000 | 0.8000 | 0.9000 | 1.0000 |
| noQAT+PTQ | 0.0577 | 0.1519 | 0.2462 | 0.3404 | 0.4347 | 0.5289 | 0.6226 | 0.7170 | 0.8112 | 0.9057 | 1.0000 |
| Traditional+QAT (no QAT in FL, QAT fine-tune) | 0.9515 | 0.9550 | 0.9572 | 0.9593 | 0.9620 | 0.9644 | 0.9664 | 0.9693 | 0.9716 | 0.9736 | 0.9763 |
| QAT+PTQ | 0.9440 | 0.9472 | 0.9494 | 0.9515 | 0.9538 | 0.9564 | 0.9585 | 0.9614 | 0.9634 | 0.9654 | 0.9682 |
| Compressed (QAT) | 0.9717 | 0.9684 | 0.9648 | 0.9613 | 0.9579 | 0.9547 | 0.9513 | 0.9481 | 0.9446 | 0.9412 | 0.9379 |
| saved_model_pruned_10x5_qat | 0.9717 | 0.9684 | 0.9648 | 0.9613 | 0.9579 | 0.9547 | 0.9513 | 0.9481 | 0.9446 | 0.9412 | 0.9379 |
| saved_model_pruned_10x2_qat | 0.9720 | 0.9688 | 0.9651 | 0.9614 | 0.9579 | 0.9546 | 0.9512 | 0.9480 | 0.9442 | 0.9406 | 0.9374 |
| saved_model_pruned_5x10_qat | 0.9746 | 0.9713 | 0.9673 | 0.9634 | 0.9596 | 0.9559 | 0.9521 | 0.9484 | 0.9443 | 0.9403 | 0.9367 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1818 | 0.3334 | 0.4616 | 0.5715 | 0.6667 | 0.7500 | 0.8235 | 0.8889 | 0.9474 | 1.0000 |
| noQAT+PTQ | 0.0000 | 0.1908 | 0.3467 | 0.4763 | 0.5859 | 0.6797 | 0.7607 | 0.8319 | 0.8945 | 0.9502 | 1.0000 |
| Traditional+QAT (no QAT in FL, QAT fine-tune) | 0.0000 | 0.8130 | 0.9013 | 0.9351 | 0.9536 | 0.9648 | 0.9721 | 0.9781 | 0.9821 | 0.9852 | 0.9880 |
| QAT+PTQ | 0.0000 | 0.7861 | 0.8844 | 0.9229 | 0.9437 | 0.9569 | 0.9655 | 0.9723 | 0.9769 | 0.9806 | 0.9838 |
| Compressed (QAT) | 0.0000 | 0.8560 | 0.9142 | 0.9356 | 0.9469 | 0.9540 | 0.9586 | 0.9620 | 0.9644 | 0.9663 | 0.9680 |
| saved_model_pruned_10x5_qat | 0.0000 | 0.8560 | 0.9142 | 0.9356 | 0.9469 | 0.9540 | 0.9586 | 0.9620 | 0.9644 | 0.9663 | 0.9680 |
| saved_model_pruned_10x2_qat | 0.0000 | 0.8574 | 0.9149 | 0.9358 | 0.9469 | 0.9538 | 0.9584 | 0.9619 | 0.9642 | 0.9660 | 0.9677 |
| saved_model_pruned_5x10_qat | 0.0000 | 0.8673 | 0.9198 | 0.9388 | 0.9488 | 0.9551 | 0.9591 | 0.9621 | 0.9642 | 0.9658 | 0.9673 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0001 | 0.0002 | 0.0002 | 0.0002 | 0.0001 | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0000 |
| noQAT+PTQ | 0.0577 | 0.0577 | 0.0577 | 0.0578 | 0.0578 | 0.0578 | 0.0566 | 0.0568 | 0.0564 | 0.0570 | 0.0000 |
| Traditional+QAT (no QAT in FL, QAT fine-tune) | 0.9515 | 0.9525 | 0.9525 | 0.9521 | 0.9525 | 0.9525 | 0.9516 | 0.9531 | 0.9527 | 0.9494 | 0.0000 |
| QAT+PTQ | 0.9440 | 0.9448 | 0.9447 | 0.9443 | 0.9443 | 0.9446 | 0.9441 | 0.9457 | 0.9441 | 0.9408 | 0.0000 |
| Compressed (QAT) | 0.9717 | 0.9716 | 0.9715 | 0.9713 | 0.9712 | 0.9716 | 0.9715 | 0.9721 | 0.9714 | 0.9710 | 0.0000 |
| saved_model_pruned_10x5_qat | 0.9717 | 0.9716 | 0.9715 | 0.9713 | 0.9712 | 0.9716 | 0.9715 | 0.9721 | 0.9714 | 0.9710 | 0.0000 |
| saved_model_pruned_10x2_qat | 0.9720 | 0.9721 | 0.9720 | 0.9718 | 0.9716 | 0.9719 | 0.9719 | 0.9730 | 0.9718 | 0.9701 | 0.0000 |
| saved_model_pruned_5x10_qat | 0.9746 | 0.9751 | 0.9750 | 0.9748 | 0.9748 | 0.9752 | 0.9752 | 0.9756 | 0.9750 | 0.9731 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 1.0000 |
| 90 | 10 | 460,810 | 0.1002 | 0.1000 | 1.0000 | 0.1818 | 0.0002 | 0.9861 |
| 80 | 20 | 425,865 | 0.2001 | 0.2000 | 1.0000 | 0.3334 | 0.0002 | 0.9661 |
| 70 | 30 | 283,910 | 0.3001 | 0.3000 | 1.0000 | 0.4616 | 0.0002 | 0.9375 |
| 60 | 40 | 212,930 | 0.4001 | 0.4000 | 1.0000 | 0.5715 | 0.0001 | 0.9000 |
| 50 | 50 | 170,346 | 0.5001 | 0.5000 | 1.0000 | 0.6667 | 0.0002 | 0.8889 |
| 40 | 60 | 141,955 | 0.6001 | 0.6000 | 1.0000 | 0.7500 | 0.0002 | 0.8571 |
| 30 | 70 | 121,672 | 0.7000 | 0.7000 | 1.0000 | 0.8235 | 0.0002 | 0.7500 |
| 20 | 80 | 106,465 | 0.8000 | 0.8000 | 1.0000 | 0.8889 | 0.0002 | 0.6667 |
| 10 | 90 | 94,630 | 0.9000 | 0.9000 | 1.0000 | 0.9474 | 0.0002 | 0.5000 |
| 0 | 100 | 85,173 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### noQAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0577 | 0.0000 | 0.0000 | 0.0000 | 0.0577 | 1.0000 |
| 90 | 10 | 460,810 | 0.1519 | 0.1055 | 1.0000 | 0.1908 | 0.0577 | 1.0000 |
| 80 | 20 | 425,865 | 0.2462 | 0.2097 | 1.0000 | 0.3467 | 0.0577 | 0.9998 |
| 70 | 30 | 283,910 | 0.3404 | 0.3126 | 1.0000 | 0.4763 | 0.0578 | 0.9997 |
| 60 | 40 | 212,930 | 0.4347 | 0.4144 | 1.0000 | 0.5859 | 0.0578 | 0.9995 |
| 50 | 50 | 170,346 | 0.5289 | 0.5149 | 1.0000 | 0.6797 | 0.0578 | 0.9992 |
| 40 | 60 | 141,955 | 0.6226 | 0.6139 | 1.0000 | 0.7607 | 0.0566 | 0.9988 |
| 30 | 70 | 121,672 | 0.7170 | 0.7121 | 1.0000 | 0.8319 | 0.0568 | 0.9981 |
| 20 | 80 | 106,465 | 0.8112 | 0.8091 | 1.0000 | 0.8945 | 0.0564 | 0.9967 |
| 10 | 90 | 94,630 | 0.9057 | 0.9052 | 1.0000 | 0.9502 | 0.0570 | 0.9926 |
| 0 | 100 | 85,173 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### Traditional+QAT (no QAT in FL, QAT fine-tune)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9515 | 0.0000 | 0.0000 | 0.0000 | 0.9515 | 1.0000 |
| 90 | 10 | 460,810 | 0.9550 | 0.6959 | 0.9774 | 0.8130 | 0.9525 | 0.9974 |
| 80 | 20 | 425,865 | 0.9572 | 0.8370 | 0.9763 | 0.9013 | 0.9525 | 0.9938 |
| 70 | 30 | 283,910 | 0.9593 | 0.8972 | 0.9763 | 0.9351 | 0.9521 | 0.9894 |
| 60 | 40 | 212,930 | 0.9620 | 0.9320 | 0.9763 | 0.9536 | 0.9525 | 0.9837 |
| 50 | 50 | 170,346 | 0.9644 | 0.9536 | 0.9763 | 0.9648 | 0.9525 | 0.9757 |
| 40 | 60 | 141,955 | 0.9664 | 0.9680 | 0.9763 | 0.9721 | 0.9516 | 0.9640 |
| 30 | 70 | 121,672 | 0.9693 | 0.9798 | 0.9763 | 0.9781 | 0.9531 | 0.9452 |
| 20 | 80 | 106,465 | 0.9716 | 0.9880 | 0.9763 | 0.9821 | 0.9527 | 0.9095 |
| 10 | 90 | 94,630 | 0.9736 | 0.9943 | 0.9763 | 0.9852 | 0.9494 | 0.8166 |
| 0 | 100 | 85,173 | 0.9763 | 1.0000 | 0.9763 | 0.9880 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9440 | 0.0000 | 0.0000 | 0.0000 | 0.9440 | 1.0000 |
| 90 | 10 | 460,810 | 0.9472 | 0.6611 | 0.9695 | 0.7861 | 0.9448 | 0.9964 |
| 80 | 20 | 425,865 | 0.9494 | 0.8139 | 0.9682 | 0.8844 | 0.9447 | 0.9916 |
| 70 | 30 | 283,910 | 0.9515 | 0.8817 | 0.9682 | 0.9229 | 0.9443 | 0.9858 |
| 60 | 40 | 212,930 | 0.9538 | 0.9205 | 0.9682 | 0.9437 | 0.9443 | 0.9780 |
| 50 | 50 | 170,346 | 0.9564 | 0.9459 | 0.9682 | 0.9569 | 0.9446 | 0.9674 |
| 40 | 60 | 141,955 | 0.9585 | 0.9629 | 0.9682 | 0.9655 | 0.9441 | 0.9519 |
| 30 | 70 | 121,672 | 0.9614 | 0.9765 | 0.9682 | 0.9723 | 0.9457 | 0.9272 |
| 20 | 80 | 106,465 | 0.9634 | 0.9858 | 0.9682 | 0.9769 | 0.9441 | 0.8812 |
| 10 | 90 | 94,630 | 0.9654 | 0.9933 | 0.9682 | 0.9806 | 0.9408 | 0.7666 |
| 0 | 100 | 85,173 | 0.9682 | 1.0000 | 0.9682 | 0.9838 | 0.0000 | 0.0000 |

### Compressed (QAT)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9717 | 0.0000 | 0.0000 | 0.0000 | 0.9717 | 1.0000 |
| 90 | 10 | 460,810 | 0.9684 | 0.7861 | 0.9395 | 0.8560 | 0.9716 | 0.9931 |
| 80 | 20 | 425,865 | 0.9648 | 0.8917 | 0.9379 | 0.9142 | 0.9715 | 0.9843 |
| 70 | 30 | 283,910 | 0.9613 | 0.9333 | 0.9379 | 0.9356 | 0.9713 | 0.9733 |
| 60 | 40 | 212,930 | 0.9579 | 0.9560 | 0.9379 | 0.9469 | 0.9712 | 0.9591 |
| 50 | 50 | 170,346 | 0.9547 | 0.9706 | 0.9379 | 0.9540 | 0.9716 | 0.9399 |
| 40 | 60 | 141,955 | 0.9513 | 0.9801 | 0.9379 | 0.9586 | 0.9715 | 0.9125 |
| 30 | 70 | 121,672 | 0.9481 | 0.9874 | 0.9379 | 0.9620 | 0.9721 | 0.8703 |
| 20 | 80 | 106,465 | 0.9446 | 0.9924 | 0.9379 | 0.9644 | 0.9714 | 0.7963 |
| 10 | 90 | 94,630 | 0.9412 | 0.9966 | 0.9379 | 0.9663 | 0.9710 | 0.6346 |
| 0 | 100 | 85,173 | 0.9379 | 1.0000 | 0.9379 | 0.9680 | 0.0000 | 0.0000 |

### saved_model_pruned_10x5_qat

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9717 | 0.0000 | 0.0000 | 0.0000 | 0.9717 | 1.0000 |
| 90 | 10 | 460,810 | 0.9684 | 0.7861 | 0.9395 | 0.8560 | 0.9716 | 0.9931 |
| 80 | 20 | 425,865 | 0.9648 | 0.8917 | 0.9379 | 0.9142 | 0.9715 | 0.9843 |
| 70 | 30 | 283,910 | 0.9613 | 0.9333 | 0.9379 | 0.9356 | 0.9713 | 0.9733 |
| 60 | 40 | 212,930 | 0.9579 | 0.9560 | 0.9379 | 0.9469 | 0.9712 | 0.9591 |
| 50 | 50 | 170,346 | 0.9547 | 0.9706 | 0.9379 | 0.9540 | 0.9716 | 0.9399 |
| 40 | 60 | 141,955 | 0.9513 | 0.9801 | 0.9379 | 0.9586 | 0.9715 | 0.9125 |
| 30 | 70 | 121,672 | 0.9481 | 0.9874 | 0.9379 | 0.9620 | 0.9721 | 0.8703 |
| 20 | 80 | 106,465 | 0.9446 | 0.9924 | 0.9379 | 0.9644 | 0.9714 | 0.7963 |
| 10 | 90 | 94,630 | 0.9412 | 0.9966 | 0.9379 | 0.9663 | 0.9710 | 0.6346 |
| 0 | 100 | 85,173 | 0.9379 | 1.0000 | 0.9379 | 0.9680 | 0.0000 | 0.0000 |

### saved_model_pruned_10x2_qat

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9720 | 0.0000 | 0.0000 | 0.0000 | 0.9720 | 1.0000 |
| 90 | 10 | 460,810 | 0.9688 | 0.7888 | 0.9390 | 0.8574 | 0.9721 | 0.9931 |
| 80 | 20 | 425,865 | 0.9651 | 0.8934 | 0.9374 | 0.9149 | 0.9720 | 0.9841 |
| 70 | 30 | 283,910 | 0.9614 | 0.9343 | 0.9374 | 0.9358 | 0.9718 | 0.9731 |
| 60 | 40 | 212,930 | 0.9579 | 0.9566 | 0.9374 | 0.9469 | 0.9716 | 0.9588 |
| 50 | 50 | 170,346 | 0.9546 | 0.9709 | 0.9374 | 0.9538 | 0.9719 | 0.9395 |
| 40 | 60 | 141,955 | 0.9512 | 0.9804 | 0.9374 | 0.9584 | 0.9719 | 0.9119 |
| 30 | 70 | 121,672 | 0.9480 | 0.9878 | 0.9374 | 0.9619 | 0.9730 | 0.8694 |
| 20 | 80 | 106,465 | 0.9442 | 0.9925 | 0.9374 | 0.9642 | 0.9718 | 0.7950 |
| 10 | 90 | 94,630 | 0.9406 | 0.9965 | 0.9374 | 0.9660 | 0.9701 | 0.6324 |
| 0 | 100 | 85,173 | 0.9374 | 1.0000 | 0.9374 | 0.9677 | 0.0000 | 0.0000 |

### saved_model_pruned_5x10_qat

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9746 | 0.0000 | 0.0000 | 0.0000 | 0.9746 | 1.0000 |
| 90 | 10 | 460,810 | 0.9713 | 0.8068 | 0.9376 | 0.8673 | 0.9751 | 0.9929 |
| 80 | 20 | 425,865 | 0.9673 | 0.9035 | 0.9367 | 0.9198 | 0.9750 | 0.9840 |
| 70 | 30 | 283,910 | 0.9634 | 0.9410 | 0.9367 | 0.9388 | 0.9748 | 0.9729 |
| 60 | 40 | 212,930 | 0.9596 | 0.9612 | 0.9367 | 0.9488 | 0.9748 | 0.9585 |
| 50 | 50 | 170,346 | 0.9559 | 0.9742 | 0.9367 | 0.9551 | 0.9752 | 0.9390 |
| 40 | 60 | 141,955 | 0.9521 | 0.9827 | 0.9367 | 0.9591 | 0.9752 | 0.9112 |
| 30 | 70 | 121,672 | 0.9484 | 0.9890 | 0.9367 | 0.9621 | 0.9756 | 0.8685 |
| 20 | 80 | 106,465 | 0.9443 | 0.9934 | 0.9367 | 0.9642 | 0.9750 | 0.7938 |
| 10 | 90 | 94,630 | 0.9403 | 0.9968 | 0.9367 | 0.9658 | 0.9731 | 0.6306 |
| 0 | 100 | 85,173 | 0.9367 | 1.0000 | 0.9367 | 0.9673 | 0.0000 | 0.0000 |


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
0.20       0.1001   0.1818   0.0001   0.9818   1.0000   0.1000  
0.25       0.1001   0.1818   0.0001   0.9841   1.0000   0.1000  
0.30       0.1002   0.1818   0.0002   0.9861   1.0000   0.1000  
0.35       0.1004   0.1819   0.0005   0.9947   1.0000   0.1000  
0.40       0.1009   0.1820   0.0010   0.9931   0.9999   0.1001  
0.45       0.1023   0.1821   0.0026   0.9669   0.9992   0.1002  
0.50       0.5327   0.2124   0.5219   0.9270   0.6301   0.1277  
0.55       0.8618   0.4214   0.9017   0.9423   0.5032   0.3625   <--
0.60       0.9027   0.2575   0.9842   0.9142   0.1687   0.5431  
0.65       0.9076   0.2185   0.9941   0.9113   0.1292   0.7075  
0.70       0.9097   0.1917   0.9988   0.9097   0.1071   0.9116  
0.75       0.9015   0.0396   0.9994   0.9018   0.0203   0.7800  
0.80       0.8995   0.0000   0.9995   0.9000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.8618, F1=0.4214, Normal Recall=0.9017, Normal Precision=0.9423, Attack Recall=0.5032, Attack Precision=0.3625

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
0.15       0.2000   0.3333   0.0000   0.9000   1.0000   0.2000  
0.20       0.2001   0.3334   0.0001   0.9583   1.0000   0.2000  
0.25       0.2001   0.3334   0.0002   0.9636   1.0000   0.2000  
0.30       0.2001   0.3334   0.0002   0.9683   1.0000   0.2000  
0.35       0.2004   0.3334   0.0005   0.9878   1.0000   0.2001  
0.40       0.2008   0.3335   0.0010   0.9860   0.9999   0.2002  
0.45       0.2018   0.3336   0.0025   0.9097   0.9990   0.2002  
0.50       0.5437   0.3564   0.5217   0.8500   0.6317   0.2482  
0.55       0.8220   0.5306   0.9016   0.8789   0.5032   0.5612   <--
0.60       0.8211   0.2732   0.9843   0.8256   0.1682   0.7280  
0.65       0.8209   0.2227   0.9941   0.8202   0.1283   0.8443  
0.70       0.8202   0.1903   0.9988   0.8171   0.1056   0.9568  
0.75       0.8034   0.0388   0.9993   0.8031   0.0199   0.8835  
0.80       0.7996   0.0000   0.9995   0.7999   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.8220, F1=0.5306, Normal Recall=0.9016, Normal Precision=0.8789, Attack Recall=0.5032, Attack Precision=0.5612

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
0.15       0.3000   0.4615   0.0000   0.7500   1.0000   0.3000  
0.20       0.3001   0.4616   0.0001   0.9167   1.0000   0.3000  
0.25       0.3001   0.4616   0.0001   0.9231   1.0000   0.3000  
0.30       0.3001   0.4616   0.0002   0.9375   1.0000   0.3000  
0.35       0.3003   0.4616   0.0004   0.9778   1.0000   0.3001  
0.40       0.3006   0.4618   0.0009   0.9738   0.9999   0.3002  
0.45       0.3015   0.4618   0.0025   0.8527   0.9990   0.3003  
0.50       0.5545   0.4597   0.5214   0.7676   0.6317   0.3613  
0.55       0.7821   0.5809   0.9017   0.8090   0.5032   0.6868   <--
0.60       0.7397   0.2794   0.9847   0.7342   0.1682   0.8250  
0.65       0.7344   0.2247   0.9942   0.7269   0.1283   0.9039  
0.70       0.7308   0.1906   0.9988   0.7227   0.1056   0.9741  
0.75       0.7055   0.0389   0.9993   0.7041   0.0199   0.9276  
0.80       0.6996   0.0000   0.9995   0.6999   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.7821, F1=0.5809, Normal Recall=0.9017, Normal Precision=0.8090, Attack Recall=0.5032, Attack Precision=0.6868

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
0.15       0.4000   0.5714   0.0000   0.5000   1.0000   0.4000  
0.20       0.4001   0.5714   0.0001   0.8667   1.0000   0.4000  
0.25       0.4001   0.5714   0.0001   0.8750   1.0000   0.4000  
0.30       0.4001   0.5715   0.0001   0.9048   1.0000   0.4000  
0.35       0.4003   0.5715   0.0005   0.9672   1.0000   0.4001  
0.40       0.4005   0.5716   0.0009   0.9600   0.9999   0.4002  
0.45       0.4011   0.5716   0.0025   0.7871   0.9990   0.4004  
0.50       0.5659   0.5379   0.5220   0.6801   0.6318   0.4684  
0.55       0.7423   0.6097   0.9017   0.7314   0.5032   0.7735   <--
0.60       0.6580   0.2823   0.9846   0.6397   0.1682   0.8792  
0.65       0.6477   0.2256   0.9940   0.6311   0.1283   0.9346  
0.70       0.6415   0.1908   0.9988   0.6262   0.1056   0.9827  
0.75       0.6075   0.0389   0.9993   0.6046   0.0199   0.9521  
0.80       0.5997   0.0000   0.9994   0.5999   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.7423, F1=0.6097, Normal Recall=0.9017, Normal Precision=0.7314, Attack Recall=0.5032, Attack Precision=0.7735

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
0.15       0.5000   0.6667   0.0000   0.5000   1.0000   0.5000  
0.20       0.5000   0.6667   0.0001   0.8182   1.0000   0.5000  
0.25       0.5000   0.6667   0.0001   0.8182   1.0000   0.5000  
0.30       0.5001   0.6667   0.0001   0.8571   1.0000   0.5000  
0.35       0.5002   0.6668   0.0005   0.9524   1.0000   0.5001  
0.40       0.5004   0.6669   0.0010   0.9419   0.9999   0.5002   <--
0.45       0.5008   0.6668   0.0025   0.7152   0.9990   0.5004  
0.50       0.5774   0.5992   0.5230   0.5868   0.6317   0.5698  
0.55       0.7023   0.6283   0.9014   0.6447   0.5032   0.8361  
0.60       0.5762   0.2841   0.9843   0.5420   0.1682   0.9145  
0.65       0.5611   0.2262   0.9940   0.5328   0.1283   0.9555  
0.70       0.5521   0.1908   0.9986   0.5275   0.1056   0.9874  
0.75       0.5096   0.0389   0.9992   0.5048   0.0199   0.9635  
0.80       0.4997   0.0000   0.9994   0.4998   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.5004, F1=0.6669, Normal Recall=0.0010, Normal Precision=0.9419, Attack Recall=0.9999, Attack Precision=0.5002

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
0.15       0.1069   0.1830   0.0076   0.9997   1.0000   0.1007  
0.20       0.1135   0.1841   0.0151   0.9997   1.0000   0.1014  
0.25       0.1277   0.1865   0.0307   0.9998   1.0000   0.1028  
0.30       0.1519   0.1908   0.0577   0.9999   1.0000   0.1055  
0.35       0.1668   0.1936   0.0743   0.9999   1.0000   0.1072  
0.40       0.1784   0.1958   0.0871   0.9999   1.0000   0.1085  
0.45       0.2029   0.2006   0.1143   1.0000   1.0000   0.1115  
0.50       0.9325   0.6348   0.9710   0.9548   0.5865   0.6917  
0.55       0.9567   0.7296   0.9982   0.9557   0.5835   0.9731  
0.60       0.9571   0.7304   0.9988   0.9555   0.5817   0.9811  
0.65       0.9573   0.7311   0.9992   0.9554   0.5805   0.9872   <--
0.70       0.9573   0.7307   0.9994   0.9553   0.5790   0.9904  
0.75       0.9575   0.7307   0.9997   0.9551   0.5772   0.9954  
0.80       0.9573   0.7295   0.9998   0.9549   0.5754   0.9964  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.65
  At threshold 0.65: Accuracy=0.9573, F1=0.7311, Normal Recall=0.9992, Normal Precision=0.9554, Attack Recall=0.5805, Attack Precision=0.9872

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
0.15       0.2061   0.3350   0.0076   0.9992   1.0000   0.2012  
0.20       0.2120   0.3367   0.0150   0.9994   1.0000   0.2024  
0.25       0.2246   0.3403   0.0307   0.9996   1.0000   0.2050  
0.30       0.2460   0.3466   0.0575   0.9998   1.0000   0.2096  
0.35       0.2593   0.3506   0.0741   0.9998   1.0000   0.2126  
0.40       0.2695   0.3538   0.0868   0.9999   1.0000   0.2149  
0.45       0.2912   0.3607   0.1141   0.9998   0.9999   0.2201  
0.50       0.8941   0.6895   0.9708   0.9040   0.5877   0.8340  
0.55       0.9155   0.7345   0.9982   0.9058   0.5846   0.9878   <--
0.60       0.9155   0.7339   0.9987   0.9054   0.5826   0.9913  
0.65       0.9156   0.7337   0.9991   0.9052   0.5813   0.9942  
0.70       0.9154   0.7327   0.9994   0.9048   0.5796   0.9956  
0.75       0.9154   0.7320   0.9997   0.9045   0.5780   0.9979  
0.80       0.9150   0.7306   0.9998   0.9042   0.5761   0.9984  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9155, F1=0.7345, Normal Recall=0.9982, Normal Precision=0.9058, Attack Recall=0.5846, Attack Precision=0.9878

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
0.15       0.3053   0.4634   0.0076   0.9987   1.0000   0.3016  
0.20       0.3105   0.4653   0.0151   0.9990   1.0000   0.3032  
0.25       0.3216   0.4693   0.0309   0.9993   1.0000   0.3066  
0.30       0.3403   0.4763   0.0576   0.9997   1.0000   0.3126  
0.35       0.3520   0.4807   0.0742   0.9997   1.0000   0.3164  
0.40       0.3607   0.4841   0.0867   0.9998   1.0000   0.3194  
0.45       0.3797   0.4917   0.1139   0.9997   0.9999   0.3260  
0.50       0.8558   0.7097   0.9707   0.8460   0.5877   0.8957  
0.55       0.8741   0.7358   0.9982   0.8486   0.5846   0.9927   <--
0.60       0.8739   0.7349   0.9987   0.8481   0.5826   0.9950  
0.65       0.8738   0.7343   0.9991   0.8478   0.5813   0.9965  
0.70       0.8734   0.7332   0.9993   0.8473   0.5796   0.9974  
0.75       0.8732   0.7322   0.9997   0.8468   0.5780   0.9988  
0.80       0.8727   0.7308   0.9998   0.8462   0.5761   0.9990  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.8741, F1=0.7358, Normal Recall=0.9982, Normal Precision=0.8486, Attack Recall=0.5846, Attack Precision=0.9927

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
0.15       0.4045   0.5733   0.0075   0.9979   1.0000   0.4018  
0.20       0.4091   0.5751   0.0151   0.9984   1.0000   0.4037  
0.25       0.4186   0.5791   0.0311   0.9990   1.0000   0.4076  
0.30       0.4347   0.5859   0.0579   0.9995   1.0000   0.4144  
0.35       0.4447   0.5903   0.0746   0.9996   1.0000   0.4187  
0.40       0.4524   0.5936   0.0873   0.9996   1.0000   0.4221  
0.45       0.4686   0.6008   0.1143   0.9995   0.9999   0.4294  
0.50       0.8176   0.7205   0.9709   0.7794   0.5877   0.9309  
0.55       0.8327   0.7366   0.9982   0.7828   0.5846   0.9953   <--
0.60       0.8323   0.7354   0.9988   0.7821   0.5826   0.9968  
0.65       0.8320   0.7346   0.9991   0.7816   0.5813   0.9977  
0.70       0.8314   0.7334   0.9993   0.7810   0.5796   0.9982  
0.75       0.8310   0.7323   0.9997   0.7804   0.5780   0.9991  
0.80       0.8303   0.7309   0.9997   0.7796   0.5762   0.9992  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.8327, F1=0.7366, Normal Recall=0.9982, Normal Precision=0.7828, Attack Recall=0.5846, Attack Precision=0.9953

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
0.15       0.5038   0.6684   0.0076   0.9969   1.0000   0.5019  
0.20       0.5077   0.6701   0.0154   0.9977   1.0000   0.5039  
0.25       0.5157   0.6737   0.0314   0.9985   1.0000   0.5080  
0.30       0.5289   0.6798   0.0578   0.9992   1.0000   0.5149  
0.35       0.5372   0.6836   0.0745   0.9994   1.0000   0.5193  
0.40       0.5437   0.6867   0.0874   0.9995   1.0000   0.5228  
0.45       0.5571   0.6931   0.1144   0.9993   0.9999   0.5303  
0.50       0.7792   0.7269   0.9707   0.7019   0.5877   0.9524  
0.55       0.7913   0.7369   0.9980   0.7061   0.5846   0.9967   <--
0.60       0.7906   0.7356   0.9986   0.7052   0.5826   0.9977  
0.65       0.7902   0.7348   0.9990   0.7047   0.5813   0.9984  
0.70       0.7894   0.7335   0.9992   0.7039   0.5796   0.9987  
0.75       0.7888   0.7324   0.9996   0.7032   0.5780   0.9994  
0.80       0.7879   0.7309   0.9997   0.7023   0.5761   0.9995  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.7913, F1=0.7369, Normal Recall=0.9980, Normal Precision=0.7061, Attack Recall=0.5846, Attack Precision=0.9967

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
0.15       0.9517   0.8026   0.9483   0.9979   0.9823   0.6784  
0.20       0.9533   0.8077   0.9501   0.9979   0.9817   0.6861  
0.25       0.9538   0.8096   0.9508   0.9978   0.9816   0.6890  
0.30       0.9550   0.8126   0.9525   0.9973   0.9767   0.6958  
0.35       0.9750   0.8830   0.9787   0.9935   0.9422   0.8308  
0.40       0.9752   0.8837   0.9789   0.9934   0.9418   0.8323  
0.45       0.9746   0.8794   0.9799   0.9917   0.9265   0.8368  
0.50       0.9771   0.8816   0.9912   0.9836   0.8509   0.9145  
0.55       0.9789   0.8880   0.9945   0.9822   0.8379   0.9445   <--
0.60       0.9783   0.8829   0.9962   0.9800   0.8173   0.9599  
0.65       0.9786   0.8842   0.9965   0.9800   0.8171   0.9633  
0.70       0.9788   0.8849   0.9971   0.9797   0.8144   0.9688  
0.75       0.9780   0.8783   0.9984   0.9777   0.7946   0.9818  
0.80       0.9774   0.8745   0.9985   0.9769   0.7877   0.9829  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9789, F1=0.8880, Normal Recall=0.9945, Normal Precision=0.9822, Attack Recall=0.8379, Attack Precision=0.9445

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
0.15       0.9550   0.8972   0.9482   0.9953   0.9822   0.8258  
0.20       0.9563   0.8999   0.9500   0.9952   0.9816   0.8308  
0.25       0.9568   0.9010   0.9507   0.9951   0.9814   0.8327  
0.30       0.9572   0.9013   0.9524   0.9938   0.9763   0.8369  
0.35       0.9714   0.9294   0.9786   0.9855   0.9426   0.9166  
0.40       0.9715   0.9296   0.9788   0.9855   0.9422   0.9173   <--
0.45       0.9692   0.9234   0.9798   0.9818   0.9271   0.9197  
0.50       0.9633   0.9028   0.9911   0.9641   0.8522   0.9597  
0.55       0.9634   0.9017   0.9945   0.9611   0.8391   0.9744  
0.60       0.9607   0.8928   0.9962   0.9565   0.8187   0.9816  
0.65       0.9609   0.8933   0.9965   0.9564   0.8185   0.9831  
0.70       0.9608   0.8928   0.9970   0.9559   0.8159   0.9856  
0.75       0.9578   0.8829   0.9984   0.9513   0.7955   0.9918  
0.80       0.9565   0.8787   0.9985   0.9497   0.7884   0.9923  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9715, F1=0.9296, Normal Recall=0.9788, Normal Precision=0.9855, Attack Recall=0.9422, Attack Precision=0.9173

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
0.15       0.9580   0.9335   0.9477   0.9920   0.9822   0.8895  
0.20       0.9591   0.9351   0.9495   0.9918   0.9816   0.8927  
0.25       0.9595   0.9357   0.9501   0.9917   0.9814   0.8940  
0.30       0.9592   0.9349   0.9519   0.9894   0.9763   0.8968  
0.35       0.9675   0.9456   0.9781   0.9755   0.9426   0.9486  
0.40       0.9675   0.9457   0.9784   0.9753   0.9422   0.9491   <--
0.45       0.9637   0.9388   0.9794   0.9691   0.9271   0.9507  
0.50       0.9493   0.9098   0.9909   0.9399   0.8522   0.9757  
0.55       0.9478   0.9061   0.9944   0.9352   0.8391   0.9846  
0.60       0.9429   0.8959   0.9961   0.9277   0.8187   0.9890  
0.65       0.9431   0.8961   0.9965   0.9276   0.8185   0.9900  
0.70       0.9427   0.8952   0.9970   0.9267   0.8159   0.9916  
0.75       0.9375   0.8842   0.9984   0.9193   0.7955   0.9952  
0.80       0.9355   0.8799   0.9985   0.9167   0.7884   0.9955  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9675, F1=0.9457, Normal Recall=0.9784, Normal Precision=0.9753, Attack Recall=0.9422, Attack Precision=0.9491

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
0.15       0.9613   0.9531   0.9474   0.9876   0.9822   0.9257  
0.20       0.9622   0.9541   0.9492   0.9873   0.9816   0.9280  
0.25       0.9625   0.9544   0.9499   0.9871   0.9814   0.9289   <--
0.30       0.9615   0.9530   0.9516   0.9837   0.9763   0.9308  
0.35       0.9639   0.9543   0.9781   0.9623   0.9426   0.9663  
0.40       0.9639   0.9543   0.9783   0.9621   0.9422   0.9667  
0.45       0.9584   0.9469   0.9793   0.9527   0.9271   0.9676  
0.50       0.9354   0.9135   0.9909   0.9096   0.8522   0.9843  
0.55       0.9322   0.9083   0.9943   0.9026   0.8391   0.9899  
0.60       0.9251   0.8974   0.9960   0.8918   0.8187   0.9928  
0.65       0.9252   0.8975   0.9964   0.8917   0.8185   0.9934  
0.70       0.9245   0.8964   0.9970   0.8904   0.8159   0.9944  
0.75       0.9172   0.8849   0.9983   0.8799   0.7955   0.9968  
0.80       0.9144   0.8805   0.9984   0.8762   0.7884   0.9970  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.25
  At threshold 0.25: Accuracy=0.9625, F1=0.9544, Normal Recall=0.9499, Normal Precision=0.9871, Attack Recall=0.9814, Attack Precision=0.9289

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
0.15       0.9645   0.9651   0.9469   0.9815   0.9822   0.9487  
0.20       0.9651   0.9657   0.9486   0.9810   0.9816   0.9503  
0.25       0.9654   0.9659   0.9493   0.9808   0.9814   0.9509   <--
0.30       0.9637   0.9642   0.9511   0.9757   0.9763   0.9523  
0.35       0.9604   0.9596   0.9781   0.9445   0.9426   0.9773  
0.40       0.9603   0.9596   0.9784   0.9442   0.9422   0.9776  
0.45       0.9532   0.9520   0.9793   0.9308   0.9271   0.9781  
0.50       0.9216   0.9157   0.9909   0.8702   0.8522   0.9895  
0.55       0.9167   0.9097   0.9943   0.8607   0.8391   0.9932  
0.60       0.9073   0.8983   0.9959   0.8460   0.8187   0.9950  
0.65       0.9074   0.8983   0.9962   0.8459   0.8185   0.9954  
0.70       0.9063   0.8970   0.9968   0.8441   0.8159   0.9961  
0.75       0.8969   0.8852   0.9982   0.8300   0.7955   0.9977  
0.80       0.8934   0.8809   0.9983   0.8251   0.7884   0.9979  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.25
  At threshold 0.25: Accuracy=0.9654, F1=0.9659, Normal Recall=0.9493, Normal Precision=0.9808, Attack Recall=0.9814, Attack Precision=0.9509

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
0.15       0.9369   0.7545   0.9333   0.9964   0.9694   0.6176  
0.20       0.9401   0.7638   0.9368   0.9964   0.9692   0.6303  
0.25       0.9436   0.7745   0.9407   0.9963   0.9689   0.6450  
0.30       0.9471   0.7856   0.9448   0.9963   0.9684   0.6608  
0.35       0.9524   0.8025   0.9507   0.9962   0.9673   0.6857  
0.40       0.9608   0.8285   0.9624   0.9938   0.9462   0.7367  
0.45       0.9664   0.8455   0.9717   0.9908   0.9191   0.7828  
0.50       0.9664   0.8455   0.9717   0.9908   0.9191   0.7828  
0.55       0.9677   0.8483   0.9750   0.9890   0.9022   0.8005  
0.60       0.9695   0.8460   0.9842   0.9820   0.8376   0.8546  
0.65       0.9737   0.8585   0.9931   0.9780   0.7988   0.9279  
0.70       0.9746   0.8604   0.9960   0.9763   0.7822   0.9561   <--
0.75       0.9734   0.8520   0.9966   0.9744   0.7646   0.9620  
0.80       0.9732   0.8498   0.9972   0.9737   0.7577   0.9673  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.7
  At threshold 0.7: Accuracy=0.9746, F1=0.8604, Normal Recall=0.9960, Normal Precision=0.9763, Attack Recall=0.7822, Attack Precision=0.9561

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
0.15       0.9404   0.8667   0.9332   0.9918   0.9693   0.7838  
0.20       0.9431   0.8721   0.9367   0.9918   0.9690   0.7928  
0.25       0.9462   0.8781   0.9406   0.9917   0.9687   0.8030  
0.30       0.9493   0.8843   0.9446   0.9916   0.9682   0.8138  
0.35       0.9539   0.8935   0.9506   0.9914   0.9671   0.8303  
0.40       0.9590   0.9023   0.9623   0.9862   0.9460   0.8624  
0.45       0.9613   0.9048   0.9716   0.9798   0.9200   0.8901   <--
0.50       0.9613   0.9048   0.9716   0.9798   0.9200   0.8901  
0.55       0.9607   0.9019   0.9749   0.9759   0.9038   0.9001  
0.60       0.9550   0.8818   0.9841   0.9606   0.8386   0.9296  
0.65       0.9546   0.8758   0.9932   0.9522   0.8004   0.9669  
0.70       0.9536   0.8712   0.9960   0.9486   0.7840   0.9802  
0.75       0.9505   0.8610   0.9967   0.9445   0.7659   0.9830  
0.80       0.9495   0.8573   0.9972   0.9429   0.7586   0.9854  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.9613, F1=0.9048, Normal Recall=0.9716, Normal Precision=0.9798, Attack Recall=0.9200, Attack Precision=0.8901

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
0.15       0.9436   0.9117   0.9327   0.9861   0.9693   0.8605  
0.20       0.9460   0.9150   0.9362   0.9860   0.9690   0.8668  
0.25       0.9487   0.9188   0.9401   0.9859   0.9687   0.8739  
0.30       0.9513   0.9227   0.9441   0.9858   0.9682   0.8812  
0.35       0.9552   0.9284   0.9501   0.9854   0.9671   0.8926  
0.40       0.9571   0.9297   0.9618   0.9765   0.9460   0.9139   <--
0.45       0.9559   0.9260   0.9713   0.9659   0.9200   0.9321  
0.50       0.9559   0.9260   0.9713   0.9659   0.9200   0.9321  
0.55       0.9533   0.9208   0.9746   0.9594   0.9038   0.9384  
0.60       0.9402   0.8938   0.9838   0.9343   0.8386   0.9569  
0.65       0.9353   0.8812   0.9931   0.9207   0.8004   0.9803  
0.70       0.9325   0.8745   0.9961   0.9150   0.7840   0.9885  
0.75       0.9275   0.8637   0.9967   0.9085   0.7659   0.9901  
0.80       0.9256   0.8596   0.9972   0.9060   0.7586   0.9915  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9571, F1=0.9297, Normal Recall=0.9618, Normal Precision=0.9765, Attack Recall=0.9460, Attack Precision=0.9139

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
0.15       0.9469   0.9359   0.9320   0.9785   0.9693   0.9048  
0.20       0.9490   0.9383   0.9358   0.9784   0.9690   0.9095  
0.25       0.9513   0.9409   0.9398   0.9782   0.9687   0.9147  
0.30       0.9536   0.9435   0.9439   0.9780   0.9682   0.9200  
0.35       0.9568   0.9471   0.9498   0.9774   0.9671   0.9278   <--
0.40       0.9554   0.9444   0.9617   0.9639   0.9460   0.9428  
0.45       0.9508   0.9374   0.9714   0.9480   0.9200   0.9554  
0.50       0.9508   0.9374   0.9714   0.9480   0.9200   0.9554  
0.55       0.9462   0.9308   0.9746   0.9382   0.9038   0.9595  
0.60       0.9256   0.9002   0.9837   0.9014   0.8386   0.9716  
0.65       0.9159   0.8839   0.9929   0.8818   0.8004   0.9868  
0.70       0.9111   0.8759   0.9958   0.8737   0.7840   0.9921  
0.75       0.9042   0.8648   0.9965   0.8646   0.7659   0.9931  
0.80       0.9017   0.8606   0.9970   0.8610   0.7586   0.9941  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.9568, F1=0.9471, Normal Recall=0.9498, Normal Precision=0.9774, Attack Recall=0.9671, Attack Precision=0.9278

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
0.15       0.9506   0.9515   0.9319   0.9681   0.9693   0.9343  
0.20       0.9523   0.9530   0.9355   0.9679   0.9690   0.9376  
0.25       0.9541   0.9548   0.9396   0.9677   0.9687   0.9413  
0.30       0.9559   0.9564   0.9436   0.9674   0.9682   0.9450  
0.35       0.9583   0.9587   0.9495   0.9665   0.9671   0.9504   <--
0.40       0.9538   0.9534   0.9616   0.9468   0.9460   0.9610  
0.45       0.9457   0.9443   0.9715   0.9239   0.9200   0.9699  
0.50       0.9457   0.9443   0.9715   0.9239   0.9200   0.9699  
0.55       0.9392   0.9370   0.9746   0.9101   0.9038   0.9727  
0.60       0.9110   0.9041   0.9835   0.8590   0.8386   0.9807  
0.65       0.8966   0.8856   0.9929   0.8326   0.8004   0.9912  
0.70       0.8899   0.8769   0.9958   0.8218   0.7840   0.9947  
0.75       0.8811   0.8657   0.9964   0.8097   0.7659   0.9953  
0.80       0.8778   0.8613   0.9970   0.8051   0.7586   0.9961  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.9583, F1=0.9587, Normal Recall=0.9495, Normal Precision=0.9665, Attack Recall=0.9671, Attack Precision=0.9504

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
0.15       0.9472   0.7859   0.9447   0.9964   0.9695   0.6607  
0.20       0.9496   0.7928   0.9481   0.9957   0.9633   0.6735  
0.25       0.9673   0.8514   0.9705   0.9930   0.9382   0.7793  
0.30       0.9682   0.8549   0.9716   0.9929   0.9374   0.7857  
0.35       0.9685   0.8538   0.9739   0.9909   0.9196   0.7967  
0.40       0.9695   0.8552   0.9771   0.9888   0.9008   0.8139  
0.45       0.9702   0.8512   0.9835   0.9835   0.8513   0.8511  
0.50       0.9704   0.8517   0.9839   0.9833   0.8494   0.8539  
0.55       0.9750   0.8641   0.9952   0.9774   0.7932   0.9489   <--
0.60       0.9753   0.8626   0.9973   0.9757   0.7766   0.9701  
0.65       0.9754   0.8628   0.9979   0.9753   0.7730   0.9763  
0.70       0.9754   0.8625   0.9981   0.9752   0.7713   0.9781  
0.75       0.9755   0.8627   0.9982   0.9751   0.7709   0.9793  
0.80       0.9714   0.8361   0.9983   0.9707   0.7290   0.9799  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9750, F1=0.8641, Normal Recall=0.9952, Normal Precision=0.9774, Attack Recall=0.7932, Attack Precision=0.9489

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
0.15       0.9496   0.8850   0.9446   0.9921   0.9698   0.8139  
0.20       0.9511   0.8873   0.9480   0.9904   0.9634   0.8223  
0.25       0.9640   0.9126   0.9704   0.9845   0.9387   0.8879  
0.30       0.9648   0.9141   0.9715   0.9843   0.9379   0.8916   <--
0.35       0.9630   0.9086   0.9739   0.9797   0.9194   0.8980  
0.40       0.9618   0.9041   0.9770   0.9752   0.9008   0.9074  
0.45       0.9571   0.8883   0.9834   0.9638   0.8521   0.9278  
0.50       0.9571   0.8879   0.9838   0.9633   0.8502   0.9292  
0.55       0.9553   0.8767   0.9952   0.9511   0.7954   0.9766  
0.60       0.9535   0.8699   0.9973   0.9473   0.7781   0.9864  
0.65       0.9532   0.8688   0.9979   0.9465   0.7745   0.9892  
0.70       0.9530   0.8681   0.9981   0.9462   0.7729   0.9901  
0.75       0.9531   0.8681   0.9982   0.9461   0.7726   0.9907  
0.80       0.9449   0.8414   0.9983   0.9369   0.7311   0.9910  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.9648, F1=0.9141, Normal Recall=0.9715, Normal Precision=0.9843, Attack Recall=0.9379, Attack Precision=0.8916

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
0.15       0.9518   0.9235   0.9441   0.9865   0.9698   0.8814  
0.20       0.9522   0.9236   0.9474   0.9837   0.9634   0.8870  
0.25       0.9606   0.9347   0.9700   0.9737   0.9387   0.9306  
0.30       0.9612   0.9355   0.9712   0.9733   0.9379   0.9331   <--
0.35       0.9574   0.9283   0.9736   0.9658   0.9194   0.9373  
0.40       0.9540   0.9215   0.9767   0.9583   0.9008   0.9432  
0.45       0.9439   0.9011   0.9832   0.9394   0.8521   0.9561  
0.50       0.9436   0.9004   0.9836   0.9387   0.8502   0.9570  
0.55       0.9352   0.8805   0.9951   0.9190   0.7954   0.9859  
0.60       0.9315   0.8721   0.9973   0.9129   0.7781   0.9921  
0.65       0.9309   0.8706   0.9979   0.9117   0.7745   0.9937  
0.70       0.9305   0.8697   0.9981   0.9111   0.7729   0.9942  
0.75       0.9305   0.8696   0.9982   0.9110   0.7726   0.9946  
0.80       0.9182   0.8428   0.9984   0.8965   0.7311   0.9948  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.9612, F1=0.9355, Normal Recall=0.9712, Normal Precision=0.9733, Attack Recall=0.9379, Attack Precision=0.9331

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
0.15       0.9542   0.9443   0.9438   0.9791   0.9698   0.9201  
0.20       0.9537   0.9433   0.9472   0.9749   0.9634   0.9240  
0.25       0.9577   0.9466   0.9703   0.9596   0.9387   0.9547  
0.30       0.9580   0.9470   0.9714   0.9591   0.9379   0.9563   <--
0.35       0.9521   0.9389   0.9739   0.9477   0.9194   0.9591  
0.40       0.9465   0.9309   0.9769   0.9366   0.9008   0.9630  
0.45       0.9308   0.9079   0.9833   0.9088   0.8521   0.9715  
0.50       0.9303   0.9070   0.9837   0.9078   0.8502   0.9720  
0.55       0.9152   0.8824   0.9951   0.8794   0.7954   0.9908  
0.60       0.9096   0.8731   0.9972   0.8708   0.7781   0.9947  
0.65       0.9085   0.8713   0.9978   0.8691   0.7745   0.9957  
0.70       0.9079   0.8704   0.9979   0.8683   0.7729   0.9960  
0.75       0.9079   0.8702   0.9981   0.8681   0.7726   0.9962  
0.80       0.8913   0.8433   0.9982   0.8477   0.7311   0.9963  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.9580, F1=0.9470, Normal Recall=0.9714, Normal Precision=0.9591, Attack Recall=0.9379, Attack Precision=0.9563

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
0.15       0.9567   0.9572   0.9436   0.9689   0.9698   0.9450   <--
0.20       0.9552   0.9556   0.9470   0.9628   0.9634   0.9478  
0.25       0.9545   0.9538   0.9703   0.9406   0.9387   0.9694  
0.30       0.9547   0.9539   0.9715   0.9399   0.9379   0.9705  
0.35       0.9468   0.9453   0.9741   0.9236   0.9194   0.9726  
0.40       0.9388   0.9364   0.9768   0.9078   0.9008   0.9749  
0.45       0.9176   0.9118   0.9831   0.8692   0.8521   0.9805  
0.50       0.9168   0.9109   0.9835   0.8678   0.8502   0.9810  
0.55       0.8952   0.8836   0.9951   0.8294   0.7954   0.9938  
0.60       0.8876   0.8738   0.9972   0.8180   0.7781   0.9964  
0.65       0.8861   0.8718   0.9977   0.8157   0.7745   0.9971  
0.70       0.8854   0.8709   0.9979   0.8146   0.7729   0.9973  
0.75       0.8853   0.8707   0.9980   0.8144   0.7726   0.9975  
0.80       0.8646   0.8437   0.9981   0.7877   0.7311   0.9975  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9567, F1=0.9572, Normal Recall=0.9436, Normal Precision=0.9689, Attack Recall=0.9698, Attack Precision=0.9450

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
0.15       0.9472   0.7859   0.9447   0.9964   0.9695   0.6607  
0.20       0.9496   0.7928   0.9481   0.9957   0.9633   0.6735  
0.25       0.9673   0.8514   0.9705   0.9930   0.9382   0.7793  
0.30       0.9682   0.8549   0.9716   0.9929   0.9374   0.7857  
0.35       0.9685   0.8538   0.9739   0.9909   0.9196   0.7967  
0.40       0.9695   0.8552   0.9771   0.9888   0.9008   0.8139  
0.45       0.9702   0.8512   0.9835   0.9835   0.8513   0.8511  
0.50       0.9704   0.8517   0.9839   0.9833   0.8494   0.8539  
0.55       0.9750   0.8641   0.9952   0.9774   0.7932   0.9489   <--
0.60       0.9753   0.8626   0.9973   0.9757   0.7766   0.9701  
0.65       0.9754   0.8628   0.9979   0.9753   0.7730   0.9763  
0.70       0.9754   0.8625   0.9981   0.9752   0.7713   0.9781  
0.75       0.9755   0.8627   0.9982   0.9751   0.7709   0.9793  
0.80       0.9714   0.8361   0.9983   0.9707   0.7290   0.9799  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.9750, F1=0.8641, Normal Recall=0.9952, Normal Precision=0.9774, Attack Recall=0.7932, Attack Precision=0.9489

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
0.15       0.9496   0.8850   0.9446   0.9921   0.9698   0.8139  
0.20       0.9511   0.8873   0.9480   0.9904   0.9634   0.8223  
0.25       0.9640   0.9126   0.9704   0.9845   0.9387   0.8879  
0.30       0.9648   0.9141   0.9715   0.9843   0.9379   0.8916   <--
0.35       0.9630   0.9086   0.9739   0.9797   0.9194   0.8980  
0.40       0.9618   0.9041   0.9770   0.9752   0.9008   0.9074  
0.45       0.9571   0.8883   0.9834   0.9638   0.8521   0.9278  
0.50       0.9571   0.8879   0.9838   0.9633   0.8502   0.9292  
0.55       0.9553   0.8767   0.9952   0.9511   0.7954   0.9766  
0.60       0.9535   0.8699   0.9973   0.9473   0.7781   0.9864  
0.65       0.9532   0.8688   0.9979   0.9465   0.7745   0.9892  
0.70       0.9530   0.8681   0.9981   0.9462   0.7729   0.9901  
0.75       0.9531   0.8681   0.9982   0.9461   0.7726   0.9907  
0.80       0.9449   0.8414   0.9983   0.9369   0.7311   0.9910  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.9648, F1=0.9141, Normal Recall=0.9715, Normal Precision=0.9843, Attack Recall=0.9379, Attack Precision=0.8916

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
0.15       0.9518   0.9235   0.9441   0.9865   0.9698   0.8814  
0.20       0.9522   0.9236   0.9474   0.9837   0.9634   0.8870  
0.25       0.9606   0.9347   0.9700   0.9737   0.9387   0.9306  
0.30       0.9612   0.9355   0.9712   0.9733   0.9379   0.9331   <--
0.35       0.9574   0.9283   0.9736   0.9658   0.9194   0.9373  
0.40       0.9540   0.9215   0.9767   0.9583   0.9008   0.9432  
0.45       0.9439   0.9011   0.9832   0.9394   0.8521   0.9561  
0.50       0.9436   0.9004   0.9836   0.9387   0.8502   0.9570  
0.55       0.9352   0.8805   0.9951   0.9190   0.7954   0.9859  
0.60       0.9315   0.8721   0.9973   0.9129   0.7781   0.9921  
0.65       0.9309   0.8706   0.9979   0.9117   0.7745   0.9937  
0.70       0.9305   0.8697   0.9981   0.9111   0.7729   0.9942  
0.75       0.9305   0.8696   0.9982   0.9110   0.7726   0.9946  
0.80       0.9182   0.8428   0.9984   0.8965   0.7311   0.9948  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.9612, F1=0.9355, Normal Recall=0.9712, Normal Precision=0.9733, Attack Recall=0.9379, Attack Precision=0.9331

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
0.15       0.9542   0.9443   0.9438   0.9791   0.9698   0.9201  
0.20       0.9537   0.9433   0.9472   0.9749   0.9634   0.9240  
0.25       0.9577   0.9466   0.9703   0.9596   0.9387   0.9547  
0.30       0.9580   0.9470   0.9714   0.9591   0.9379   0.9563   <--
0.35       0.9521   0.9389   0.9739   0.9477   0.9194   0.9591  
0.40       0.9465   0.9309   0.9769   0.9366   0.9008   0.9630  
0.45       0.9308   0.9079   0.9833   0.9088   0.8521   0.9715  
0.50       0.9303   0.9070   0.9837   0.9078   0.8502   0.9720  
0.55       0.9152   0.8824   0.9951   0.8794   0.7954   0.9908  
0.60       0.9096   0.8731   0.9972   0.8708   0.7781   0.9947  
0.65       0.9085   0.8713   0.9978   0.8691   0.7745   0.9957  
0.70       0.9079   0.8704   0.9979   0.8683   0.7729   0.9960  
0.75       0.9079   0.8702   0.9981   0.8681   0.7726   0.9962  
0.80       0.8913   0.8433   0.9982   0.8477   0.7311   0.9963  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.3
  At threshold 0.3: Accuracy=0.9580, F1=0.9470, Normal Recall=0.9714, Normal Precision=0.9591, Attack Recall=0.9379, Attack Precision=0.9563

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
0.15       0.9567   0.9572   0.9436   0.9689   0.9698   0.9450   <--
0.20       0.9552   0.9556   0.9470   0.9628   0.9634   0.9478  
0.25       0.9545   0.9538   0.9703   0.9406   0.9387   0.9694  
0.30       0.9547   0.9539   0.9715   0.9399   0.9379   0.9705  
0.35       0.9468   0.9453   0.9741   0.9236   0.9194   0.9726  
0.40       0.9388   0.9364   0.9768   0.9078   0.9008   0.9749  
0.45       0.9176   0.9118   0.9831   0.8692   0.8521   0.9805  
0.50       0.9168   0.9109   0.9835   0.8678   0.8502   0.9810  
0.55       0.8952   0.8836   0.9951   0.8294   0.7954   0.9938  
0.60       0.8876   0.8738   0.9972   0.8180   0.7781   0.9964  
0.65       0.8861   0.8718   0.9977   0.8157   0.7745   0.9971  
0.70       0.8854   0.8709   0.9979   0.8146   0.7729   0.9973  
0.75       0.8853   0.8707   0.9980   0.8144   0.7726   0.9975  
0.80       0.8646   0.8437   0.9981   0.7877   0.7311   0.9975  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9567, F1=0.9572, Normal Recall=0.9436, Normal Precision=0.9689, Attack Recall=0.9698, Attack Precision=0.9450

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
0.15       0.9479   0.7876   0.9457   0.9961   0.9670   0.6644  
0.20       0.9497   0.7931   0.9482   0.9958   0.9637   0.6738  
0.25       0.9675   0.8525   0.9707   0.9931   0.9391   0.7806  
0.30       0.9686   0.8565   0.9721   0.9929   0.9373   0.7885  
0.35       0.9696   0.8603   0.9735   0.9927   0.9352   0.7966  
0.40       0.9700   0.8618   0.9739   0.9926   0.9351   0.7992  
0.45       0.9727   0.8710   0.9785   0.9911   0.9206   0.8264  
0.50       0.9719   0.8641   0.9807   0.9880   0.8930   0.8370  
0.55       0.9752   0.8765   0.9857   0.9867   0.8804   0.8725  
0.60       0.9768   0.8762   0.9942   0.9803   0.8206   0.9399  
0.65       0.9775   0.8777   0.9962   0.9791   0.8087   0.9595   <--
0.70       0.9773   0.8762   0.9968   0.9784   0.8024   0.9651  
0.75       0.9774   0.8763   0.9969   0.9784   0.8018   0.9661  
0.80       0.9769   0.8729   0.9973   0.9775   0.7936   0.9698  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.65
  At threshold 0.65: Accuracy=0.9775, F1=0.8777, Normal Recall=0.9962, Normal Precision=0.9791, Attack Recall=0.8087, Attack Precision=0.9595

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
0.15       0.9499   0.8852   0.9456   0.9913   0.9668   0.8163  
0.20       0.9511   0.8873   0.9480   0.9904   0.9633   0.8224  
0.25       0.9643   0.9131   0.9706   0.9845   0.9390   0.8886  
0.30       0.9651   0.9148   0.9720   0.9841   0.9374   0.8933  
0.35       0.9658   0.9162   0.9734   0.9836   0.9353   0.8980  
0.40       0.9661   0.9170   0.9739   0.9836   0.9351   0.8995  
0.45       0.9670   0.9178   0.9784   0.9803   0.9213   0.9144   <--
0.50       0.9631   0.9064   0.9807   0.9734   0.8930   0.9202  
0.55       0.9646   0.9087   0.9857   0.9706   0.8804   0.9390  
0.60       0.9597   0.8908   0.9943   0.9571   0.8215   0.9728  
0.65       0.9589   0.8874   0.9963   0.9544   0.8095   0.9819  
0.70       0.9581   0.8847   0.9968   0.9530   0.8032   0.9845  
0.75       0.9580   0.8844   0.9969   0.9528   0.8025   0.9849  
0.80       0.9567   0.8800   0.9973   0.9510   0.7943   0.9865  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.45
  At threshold 0.45: Accuracy=0.9670, F1=0.9178, Normal Recall=0.9784, Normal Precision=0.9803, Attack Recall=0.9213, Attack Precision=0.9144

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
0.15       0.9516   0.9230   0.9451   0.9852   0.9668   0.8829  
0.20       0.9522   0.9237   0.9475   0.9837   0.9633   0.8871  
0.25       0.9609   0.9351   0.9703   0.9738   0.9390   0.9312  
0.30       0.9614   0.9358   0.9717   0.9731   0.9374   0.9341  
0.35       0.9617   0.9362   0.9731   0.9723   0.9353   0.9370  
0.40       0.9620   0.9366   0.9735   0.9722   0.9351   0.9380   <--
0.45       0.9610   0.9341   0.9780   0.9667   0.9213   0.9473  
0.50       0.9541   0.9211   0.9803   0.9553   0.8930   0.9511  
0.55       0.9539   0.9197   0.9854   0.9505   0.8804   0.9627  
0.60       0.9424   0.8953   0.9941   0.9286   0.8215   0.9836  
0.65       0.9402   0.8903   0.9962   0.9243   0.8095   0.9891  
0.70       0.9387   0.8872   0.9968   0.9220   0.8032   0.9908  
0.75       0.9386   0.8869   0.9969   0.9218   0.8025   0.9911  
0.80       0.9364   0.8823   0.9973   0.9188   0.7943   0.9921  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9620, F1=0.9366, Normal Recall=0.9735, Normal Precision=0.9722, Attack Recall=0.9351, Attack Precision=0.9380

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
0.15       0.9535   0.9433   0.9447   0.9771   0.9668   0.9210  
0.20       0.9536   0.9432   0.9472   0.9748   0.9633   0.9240  
0.25       0.9578   0.9468   0.9703   0.9598   0.9390   0.9548  
0.30       0.9580   0.9469   0.9717   0.9588   0.9374   0.9567  
0.35       0.9580   0.9469   0.9732   0.9575   0.9353   0.9587  
0.40       0.9582   0.9471   0.9736   0.9575   0.9351   0.9594   <--
0.45       0.9554   0.9429   0.9781   0.9491   0.9213   0.9656  
0.50       0.9454   0.9290   0.9803   0.9322   0.8930   0.9680  
0.55       0.9433   0.9255   0.9853   0.9251   0.8804   0.9755  
0.60       0.9250   0.8976   0.9940   0.8931   0.8215   0.9891  
0.65       0.9214   0.8918   0.9960   0.8869   0.8095   0.9927  
0.70       0.9193   0.8885   0.9967   0.8837   0.8032   0.9939  
0.75       0.9191   0.8881   0.9968   0.8833   0.8025   0.9941  
0.80       0.9160   0.8833   0.9972   0.8791   0.7943   0.9947  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.4
  At threshold 0.4: Accuracy=0.9582, F1=0.9471, Normal Recall=0.9736, Normal Precision=0.9575, Attack Recall=0.9351, Attack Precision=0.9594

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
0.15       0.9557   0.9562   0.9446   0.9661   0.9668   0.9458   <--
0.20       0.9551   0.9555   0.9469   0.9627   0.9633   0.9478  
0.25       0.9548   0.9541   0.9705   0.9409   0.9390   0.9696  
0.30       0.9546   0.9538   0.9718   0.9394   0.9374   0.9708  
0.35       0.9543   0.9534   0.9733   0.9376   0.9353   0.9723  
0.40       0.9545   0.9536   0.9738   0.9376   0.9351   0.9728  
0.45       0.9498   0.9483   0.9783   0.9256   0.9213   0.9770  
0.50       0.9368   0.9339   0.9805   0.9016   0.8930   0.9787  
0.55       0.9328   0.9291   0.9853   0.8917   0.8804   0.9836  
0.60       0.9078   0.8991   0.9940   0.8478   0.8215   0.9928  
0.65       0.9028   0.8928   0.9961   0.8395   0.8095   0.9952  
0.70       0.9000   0.8893   0.9968   0.8352   0.8032   0.9960  
0.75       0.8997   0.8889   0.9969   0.8347   0.8025   0.9961  
0.80       0.8958   0.8840   0.9972   0.8290   0.7943   0.9965  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9557, F1=0.9562, Normal Recall=0.9446, Normal Precision=0.9661, Attack Recall=0.9668, Attack Precision=0.9458

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
0.15       0.9444   0.7791   0.9403   0.9978   0.9812   0.6461  
0.20       0.9669   0.8517   0.9685   0.9945   0.9520   0.7706  
0.25       0.9706   0.8647   0.9741   0.9931   0.9389   0.8014  
0.30       0.9712   0.8667   0.9751   0.9928   0.9365   0.8066  
0.35       0.9719   0.8696   0.9759   0.9928   0.9360   0.8121  
0.40       0.9719   0.8678   0.9775   0.9912   0.9220   0.8196  
0.45       0.9718   0.8651   0.9795   0.9891   0.9032   0.8302  
0.50       0.9747   0.8755   0.9841   0.9877   0.8899   0.8616  
0.55       0.9762   0.8744   0.9928   0.9810   0.8269   0.9277  
0.60       0.9770   0.8768   0.9948   0.9800   0.8173   0.9457  
0.65       0.9775   0.8784   0.9958   0.9795   0.8127   0.9557   <--
0.70       0.9762   0.8689   0.9972   0.9769   0.7874   0.9692  
0.75       0.9764   0.8694   0.9975   0.9768   0.7865   0.9718  
0.80       0.9756   0.8639   0.9979   0.9755   0.7748   0.9762  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.65
  At threshold 0.65: Accuracy=0.9775, F1=0.8784, Normal Recall=0.9958, Normal Precision=0.9795, Attack Recall=0.8127, Attack Precision=0.9557

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
0.15       0.9482   0.8834   0.9401   0.9949   0.9805   0.8038  
0.20       0.9652   0.9162   0.9685   0.9878   0.9520   0.8830  
0.25       0.9671   0.9195   0.9741   0.9846   0.9391   0.9006  
0.30       0.9673   0.9197   0.9750   0.9840   0.9367   0.9034  
0.35       0.9679   0.9211   0.9759   0.9839   0.9361   0.9066   <--
0.40       0.9664   0.9165   0.9774   0.9805   0.9224   0.9107  
0.45       0.9642   0.9098   0.9795   0.9758   0.9030   0.9167  
0.50       0.9653   0.9111   0.9841   0.9728   0.8899   0.9334  
0.55       0.9600   0.8921   0.9929   0.9585   0.8280   0.9670  
0.60       0.9595   0.8898   0.9948   0.9563   0.8181   0.9754  
0.65       0.9594   0.8890   0.9959   0.9552   0.8133   0.9801  
0.70       0.9554   0.8760   0.9972   0.9495   0.7880   0.9862  
0.75       0.9554   0.8760   0.9975   0.9494   0.7872   0.9873  
0.80       0.9536   0.8699   0.9979   0.9469   0.7762   0.9893  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.35
  At threshold 0.35: Accuracy=0.9679, F1=0.9211, Normal Recall=0.9759, Normal Precision=0.9839, Attack Recall=0.9361, Attack Precision=0.9066

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
0.15       0.9520   0.9245   0.9397   0.9912   0.9805   0.8746  
0.20       0.9633   0.9397   0.9682   0.9792   0.9520   0.9277   <--
0.25       0.9633   0.9388   0.9736   0.9739   0.9391   0.9386  
0.30       0.9632   0.9385   0.9745   0.9729   0.9367   0.9404  
0.35       0.9636   0.9392   0.9755   0.9727   0.9361   0.9424  
0.40       0.9607   0.9336   0.9771   0.9671   0.9224   0.9452  
0.45       0.9563   0.9254   0.9792   0.9593   0.9030   0.9489  
0.50       0.9556   0.9233   0.9838   0.9542   0.8899   0.9593  
0.55       0.9434   0.8977   0.9928   0.9309   0.8280   0.9802  
0.60       0.9418   0.8940   0.9948   0.9273   0.8181   0.9854  
0.65       0.9411   0.8923   0.9958   0.9256   0.8133   0.9882  
0.70       0.9344   0.8782   0.9972   0.9165   0.7880   0.9918  
0.75       0.9344   0.8780   0.9974   0.9162   0.7872   0.9925  
0.80       0.9314   0.8716   0.9979   0.9123   0.7762   0.9938  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.2
  At threshold 0.2: Accuracy=0.9633, F1=0.9397, Normal Recall=0.9682, Normal Precision=0.9792, Attack Recall=0.9520, Attack Precision=0.9277

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
0.15       0.9560   0.9469   0.9396   0.9864   0.9805   0.9155  
0.20       0.9618   0.9522   0.9683   0.9680   0.9520   0.9524   <--
0.25       0.9598   0.9492   0.9736   0.9600   0.9391   0.9596  
0.30       0.9594   0.9486   0.9745   0.9585   0.9367   0.9608  
0.35       0.9596   0.9489   0.9754   0.9581   0.9361   0.9620  
0.40       0.9551   0.9427   0.9769   0.9497   0.9224   0.9639  
0.45       0.9486   0.9336   0.9791   0.9380   0.9029   0.9664  
0.50       0.9461   0.9297   0.9836   0.9306   0.8899   0.9732  
0.55       0.9268   0.9005   0.9927   0.8965   0.8280   0.9869  
0.60       0.9240   0.8960   0.9946   0.8913   0.8181   0.9903  
0.65       0.9227   0.8939   0.9957   0.8889   0.8133   0.9921  
0.70       0.9135   0.8793   0.9971   0.8759   0.7880   0.9945  
0.75       0.9133   0.8790   0.9973   0.8755   0.7872   0.9949  
0.80       0.9092   0.8724   0.9978   0.8699   0.7762   0.9958  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.2
  At threshold 0.2: Accuracy=0.9618, F1=0.9522, Normal Recall=0.9683, Normal Precision=0.9680, Attack Recall=0.9520, Attack Precision=0.9524

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
0.15       0.9600   0.9608   0.9395   0.9797   0.9805   0.9419   <--
0.20       0.9602   0.9599   0.9684   0.9528   0.9520   0.9679  
0.25       0.9564   0.9556   0.9736   0.9412   0.9391   0.9727  
0.30       0.9556   0.9547   0.9745   0.9390   0.9367   0.9735  
0.35       0.9557   0.9549   0.9754   0.9385   0.9361   0.9744  
0.40       0.9496   0.9482   0.9769   0.9264   0.9224   0.9756  
0.45       0.9410   0.9387   0.9791   0.9098   0.9030   0.9774  
0.50       0.9367   0.9336   0.9835   0.8993   0.8899   0.9818  
0.55       0.9103   0.9023   0.9926   0.8523   0.8280   0.9911  
0.60       0.9063   0.8972   0.9946   0.8454   0.8181   0.9934  
0.65       0.9045   0.8949   0.9957   0.8421   0.8133   0.9948  
0.70       0.8925   0.8800   0.9971   0.8247   0.7880   0.9963  
0.75       0.8923   0.8796   0.9973   0.8242   0.7872   0.9966  
0.80       0.8870   0.8730   0.9978   0.8168   0.7762   0.9972  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9600, F1=0.9608, Normal Recall=0.9395, Normal Precision=0.9797, Attack Recall=0.9805, Attack Precision=0.9419

```

