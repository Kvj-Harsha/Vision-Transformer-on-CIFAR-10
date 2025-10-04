# Vision Transformer (ViT) on CIFAR-10

This repository implements a **Vision Transformer (ViT)** trained from scratch on the **CIFAR-10** dataset. The model is Colab-ready, GPU-optimized, and achieves competitive accuracy without using pretrained weights.

---

## 🚀 Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/)
2. Set runtime: **Runtime → Change runtime type → GPU**
3. Copy-paste the provided script into **separate cells** following the structured sections in the notebook.
4. Run all cells — the best model checkpoint (`best_vit_cifar10.pth`) will be saved automatically.

---

## ⚙️ Best Model Configuration

| Parameter         | Value             |
| ----------------- | ----------------- |
| Image size        | 32×32             |
| Patch size        | 4×4               |
| Embedding dim     | 192               |
| Transformer depth | 8                 |
| Attention heads   | 3                 |
| Dropout (rate)    | 0.1               |
| Optimizer         | AdamW             |
| LR                | 3e-4              |
| Weight decay      | 0.05              |
| Scheduler         | CosineAnnealingLR |
| Epochs            | 50                |
| Batch size        | 128               |
| Label smoothing   | 0.1               |
| Stochastic depth  | 0.1               |

---

## 🛠 Components & Code Structure

* **PatchEmbed**: Converts input images into flattened patch embeddings using a Conv2d layer.
* **MultiHeadSelfAttention (MHSA)**: Implements self-attention over patch embeddings, split into multiple heads for better context capture.
* **MLP**: Feed-forward network inside each Transformer block, with GELU activation and dropout.
* **TransformerEncoderBlock**: Combines MHSA and MLP with residual connections, LayerNorm, and optional stochastic depth.
* **VisionTransformer**: Combines patch embedding, learnable CLS token, positional embeddings, multiple Transformer blocks, final LayerNorm, and classification head.
* **Training Utilities**: Includes device-aware mixed precision, cosine LR scheduler, optional warmup, and checkpointing.
* **Evaluation**: Functions for computing loss, accuracy, confusion matrix, and classification metrics.

---

## 📊 Final Results

| Split | Loss   | Accuracy   |
| ----- | ------ | ---------- |
| Train | 0.7616 | 88.27%     |
| Val   | 0.9758 | 79.86%     |
| Test  | 0.9453 | **81.16%** |

---

## 🧮 Confusion Matrix (Test Set)

```
[[843  24  16  10  10   3   8  14  33  39]
 [ 16 911   0   3   0   1   0   3   9  57]
 [ 56   3 756  36  47  26  32  30   2  12]
 [ 19   5  50 641  29 155  40  30  14  17]
 [  8   3  64  30 743  28  28  83   9   4]
 [ 11   5  31 125  25 736  10  49   1   7]
 [  9   7  31  47  25  23 848   6   3   1]
 [ 15   2  18  15  13  39   1 884   5   8]
 [ 58  32   8   4   2   3   1   0 866  26]
 [ 24  55   1   5   2   2   1   7  15 888]]
```

---

## 📈 Classification Report (Test Set)

| Class | Precision | Recall | F1-score | Support |
| ----- | --------- | ------ | -------- | ------- |
| 0     | 0.7960    | 0.8430 | 0.8188   | 1000    |
| 1     | 0.8701    | 0.9110 | 0.8901   | 1000    |
| 2     | 0.7754    | 0.7560 | 0.7656   | 1000    |
| 3     | 0.6998    | 0.6410 | 0.6691   | 1000    |
| 4     | 0.8292    | 0.7430 | 0.7838   | 1000    |
| 5     | 0.7244    | 0.7360 | 0.7302   | 1000    |
| 6     | 0.8751    | 0.8480 | 0.8614   | 1000    |
| 7     | 0.7993    | 0.8840 | 0.8395   | 1000    |
| 8     | 0.9049    | 0.8660 | 0.8850   | 1000    |
| 9     | 0.8385    | 0.8880 | 0.8626   | 1000    |

**Overall accuracy:** 81.16%

---

## 🔎 Concise Analysis

* **Patch size**: 4×4 patches retained sufficient local details for small 32×32 images.
* **Depth & embedding**: 8 layers with embedding dim 192 offered a good trade-off between capacity and overfitting.
* **Regularization**: Label smoothing (0.1) + stochastic depth (0.1) improved generalization.
* **Augmentation**: Random crop + horizontal flip sufficed; stronger augmentations like AutoAugment could give minor gains.
* **Optimizer & Scheduler**: AdamW + cosine annealing enabled stable convergence. Optional LR warmup helps in noisy runs.
* **Observations**: Classes with high inter-class similarity (e.g., cats vs. dogs) benefited from deeper attention layers and positional embeddings.

---

## 📝 Notes

* The code is modular, allowing easy adjustments to **patch size, depth, embedding dimension, number of heads**, and **dropout rates**.
* Checkpoints save **model state, optimizer state, and scaler** for seamless training continuation.
* Mixed-precision training via `torch.amp` improves GPU efficiency and reduces memory footprint.
