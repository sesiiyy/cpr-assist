# Experiment: S0 RGB classifier (CPR_Dataset_S0, 14 classes)

**Task:** Single-frame RGB image classification — 14 technique/error labels on **CPR_Dataset_S0**.

## At a glance

This experiment builds a **single-frame RGB classifier** for CPR coaching: each image is assigned one of **14** labels (correct technique vs. specific error types such as wrong hand position, insufficient depth, wrong rate, etc.), using the **CPR_Dataset_S0** split policy in the main project.

**Source of record:** All metrics and logs below come from the **`cpr_s0_runs`** export (single training run).

**What ran:** A **ResNet-style** network was trained for **10 epochs** on the training split, with validation every epoch. After training, **per-class** precision, recall, and F1 were computed on the **validation** set (**20 872** images) and on the **test** set (**20 336** images). The training log records loss, validation accuracy, and timings per epoch so you can see whether the model was still improving or oscillating.

**Headline outcome:** On the **test** set, **overall accuracy is about 99.94%**, with near-perfect scores on many classes and slightly lower recall on a few (e.g. **SlowFrequency**, **ExcessivePressing**) visible in the per-class table below. On **validation**, **overall accuracy is about 99.02%** — slightly lower than test here, which is common when splits differ. The **best validation accuracy in the saved epoch log** occurs at **epoch 10** (about **99.02%**).

**This folder:** Metrics and tables only; **`.pt` checkpoints are not included**. Optional JSON/text copies of the same run may sit beside this README; the markdown tables are the main human-readable report.

---

## 1. Weights / checkpoints

Trained **`.pt`** checkpoints are not included in this share.

---

## 2. Aggregated test metrics (`cpr_s0_runs`)

| Field | Value |
|-------|------:|
| test_acc | 0.9994099134539732 |
| test_n | 20336 |
| test_sec | 91.1416 |
| test_batches | 636 |

---

## 3. Per-class metrics — validation (n = 20872)

| Class | Precision | Recall | F1-score | Support |
|-------|----------:|-------:|---------:|--------:|
| Correct | 1.0000 | 1.0000 | 1.0000 | 1236 |
| OverlapHands | 1.0000 | 1.0000 | 1.0000 | 1140 |
| ClenchingHands | 1.0000 | 0.9900 | 0.9950 | 1196 |
| SingleHand | 0.9890 | 1.0000 | 0.9945 | 1080 |
| BendingArms | 1.0000 | 1.0000 | 1.0000 | 984 |
| TiltingArms | 1.0000 | 0.9986 | 0.9993 | 1408 |
| JumpPressing | 0.9979 | 1.0000 | 0.9989 | 1408 |
| Squatting | 1.0000 | 1.0000 | 1.0000 | 1408 |
| Standing | 1.0000 | 0.9993 | 0.9996 | 1404 |
| WrongPosition | 1.0000 | 1.0000 | 1.0000 | 1732 |
| InsufficientPressing | 1.0000 | 0.9153 | 0.9558 | 732 |
| SlowFrequency | 0.9988 | 0.9710 | 0.9847 | 4248 |
| ExcessivePressing | 0.8839 | 0.9972 | 0.9371 | 1412 |
| RandomPositionPressing | 1.0000 | 0.9993 | 0.9997 | 1484 |
| **accuracy** | — | — | **0.9902** | **20872** |
| macro avg | 0.9907 | 0.9908 | 0.9903 | 20872 |
| weighted avg | 0.9912 | 0.9902 | 0.9903 | 20872 |

---

## 4. Per-class metrics — test (n = 20336)

| Class | Precision | Recall | F1-score | Support |
|-------|----------:|-------:|---------:|--------:|
| Correct | 1.0000 | 1.0000 | 1.0000 | 1272 |
| OverlapHands | 1.0000 | 1.0000 | 1.0000 | 1132 |
| ClenchingHands | 1.0000 | 1.0000 | 1.0000 | 1124 |
| SingleHand | 1.0000 | 1.0000 | 1.0000 | 1148 |
| BendingArms | 1.0000 | 1.0000 | 1.0000 | 1012 |
| TiltingArms | 1.0000 | 1.0000 | 1.0000 | 1392 |
| JumpPressing | 1.0000 | 1.0000 | 1.0000 | 1392 |
| Squatting | 1.0000 | 1.0000 | 1.0000 | 1392 |
| Standing | 1.0000 | 1.0000 | 1.0000 | 1388 |
| WrongPosition | 1.0000 | 1.0000 | 1.0000 | 1756 |
| InsufficientPressing | 1.0000 | 1.0000 | 1.0000 | 740 |
| SlowFrequency | 1.0000 | 0.9968 | 0.9984 | 3756 |
| ExcessivePressing | 0.9915 | 1.0000 | 0.9957 | 1396 |
| RandomPositionPressing | 1.0000 | 1.0000 | 1.0000 | 1436 |
| **accuracy** | — | — | **0.9994** | **20336** |
| macro avg | 0.9994 | 0.9998 | 0.9996 | 20336 |
| weighted avg | 0.9994 | 0.9994 | 0.9994 | 20336 |

---

## 5. Training — every epoch

| Epoch | train_loss | train_batches | train_sec | val_acc | val_n | val_batches | val_sec | epoch_sec |
|------:|-----------:|--------------:|----------:|--------:|------:|------------:|--------:|----------:|
| 1 | 0.22090633505435817 | 5088 | 856.8881 | 0.9568800306630894 | 20872 | 653 | 91.0619 | 947.95 |
| 2 | 0.08703292592414279 | 5088 | 853.9857 | 0.9546761211192027 | 20872 | 653 | 49.1316 | 903.1173 |
| 3 | 0.06562039563908648 | 5088 | 853.6594 | 0.975804906094289 | 20872 | 653 | 49.3014 | 902.9608 |
| 4 | 0.05598246169429059 | 5088 | 853.5458 | 0.9850996550402453 | 20872 | 653 | 51.7207 | 905.2665 |
| 5 | 0.04861462512877171 | 5088 | 853.6030 | 0.9752778842468379 | 20872 | 653 | 49.4142 | 903.0172 |
| 6 | 0.041923104962516854 | 5088 | 852.7386 | 0.9492621694135684 | 20872 | 653 | 51.2717 | 904.0102 |
| 7 | 0.038656256718276126 | 5088 | 853.3252 | 0.9806439248754312 | 20872 | 653 | 54.8561 | 908.1814 |
| 8 | 0.034241016434116076 | 5088 | 853.5409 | 0.9736968187044844 | 20872 | 653 | 50.3016 | 903.8425 |
| 9 | 0.03225123786350903 | 5088 | 853.7036 | 0.9494538137217324 | 20872 | 653 | 52.1457 | 905.8493 |
| 10 | 0.031306494058566586 | 5088 | 853.7295 | 0.9901782292065926 | 20872 | 653 | 54.0942 | 907.8237 |

Highest val_acc in this log: epoch 10 (0.9901782292065926).

---

## 6. Plots

Optional artifacts not included here: `training_curves.png`, `confusion_val.png`, `confusion_test.png`.
