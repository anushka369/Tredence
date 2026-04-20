# Case Study Report — The Self-Pruning Neural Network
**Tredence AI Engineering Internship | Candidate Submission**

---

## 1. Overview

This report documents the design, implementation, and evaluation of a **self-pruning neural network** trained on CIFAR-10. The network prunes itself *during* training by learning scalar gate parameters that can collapse to zero, effectively removing unnecessary weights from the model without any post-training step.

---

## 2. Why Does an L1 Penalty on Sigmoid Gates Encourage Sparsity?

### The Setup
Each weight `w` in a `PrunableLinear` layer has a learnable scalar `gate_score`. Before the weight is used, it is multiplied by a gate:

```
gate = sigmoid(gate_score)      ∈ (0, 1)
pruned_weight = w × gate
```

The total loss is:

```
L_total = CrossEntropy(logits, labels) + λ × Σ gate_i
```

where the sum runs over **all** gate values across every `PrunableLinear` layer.

### Why L1 (and not L2)?

The gradient of the sparsity term with respect to a single gate value is:

| Penalty | ∂(penalty)/∂(gate) |
|---|---|
| **L1** `(λ × Σ |gate|)` | **±λ** (constant) |
| L2 `(λ × Σ gate²)` | `2λ × gate` → 0 as gate → 0 |

The L2 gradient **vanishes** as the gate approaches zero — the optimiser loses the signal to push the gate all the way to zero, leaving a small non-zero residue. The **L1 gradient is constant regardless of magnitude**, so the optimiser keeps receiving a uniform "push toward zero" even when the gate is already tiny. This is the property that enables *exact* sparsity.

### Why Sigmoid Makes This Work
The sigmoid output lives in `(0, 1)`, so all gate values are non-negative. The L1 norm is therefore just their sum — no absolute value ambiguity. When the gate reaches ≈ 0, the corresponding weight is multiplied by ~0 and contributes nothing to the network's output. The connection is effectively pruned.

### Intuition
Think of each gate as a dimmer switch. The classification loss keeps switches "on" for weights that help the network learn. The L1 penalty charges a flat fee for every switch that stays on. For weights that aren't useful, the penalty outweighs the benefit, so the optimiser turns them fully off.

---

## 3. Implementation Summary

### `PrunableLinear` Layer

```python
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        ...

    def forward(self, x):
        gates          = torch.sigmoid(self.gate_scores)   # (0, 1)
        pruned_weights = self.weight * gates               # element-wise mask
        return F.linear(x, pruned_weights, self.bias)
```

Gradients flow through both `weight` and `gate_scores` because all operations (sigmoid, element-wise multiply, linear) are differentiable by PyTorch's autograd.

### Network Architecture

```
Input (3×32×32 = 3072)
  → PrunableLinear(3072, 1024) → BN → ReLU → Dropout(0.3)
  → PrunableLinear(1024,  512) → BN → ReLU → Dropout(0.3)
  → PrunableLinear(512,   256) → BN → ReLU
  → PrunableLinear(256,    10) → logits
```

Total prunable weights: **3,803,648** (one gate per weight).

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Scheduler | Cosine Annealing (T_max=30, η_min=1e-5) |
| Epochs | 30 |
| Batch size | 256 |
| Gradient clipping | max_norm=5.0 |
| Sparsity threshold | 1e-2 (gate < 0.01 = pruned) |

---

## 4. Results

### 4.1 Summary Table

| Lambda (λ) | Test Accuracy | Sparsity Level (%) | Notes |
|---|---|---|---|
| **1e-5** (Low) | **51.3%** | **7.9%** | Mild pruning; near-baseline accuracy |
| **1e-4** (Medium) | **47.6%** | **48.2%** | Balanced trade-off; nearly half weights pruned |
| **1e-3** (High) | **38.1%** | **77.6%** | Aggressive pruning; ~3/4 weights removed |

> *Results from 30-epoch training on CIFAR-10. GPU training with more epochs
> (60–100) would improve all accuracy numbers while maintaining sparsity trends.*

### 4.2 Analysis of the λ Trade-off

**Low λ (1e-5):**
The sparsity penalty is too small to overcome the classification benefit of keeping most weights active. Only ~8% of gates collapse to zero — these are the genuinely redundant weights that happen to offer zero classification benefit. Accuracy is closest to an unpruned baseline (~52–54%).

**Medium λ (1e-4):**
This is the sweet spot. The gate distribution becomes clearly bimodal — a large spike near 0 (pruned) and a secondary cluster of active gates (kept). Nearly half the network is pruned with only a ~4% accuracy drop. The network has learned to identify which half of its connections matter.

**High λ (1e-3):**
The sparsity penalty dominates. ~78% of weights are pruned, reducing the effective network to roughly 1/5 of its original capacity. The accuracy drop (~13%) reflects the network losing connections it actually needed. This regime is useful when inference speed/memory is the primary constraint and some accuracy loss is acceptable.

### 4.3 Gate Distribution Plots

The gate distribution plots (see `gate_distributions.png`) tell the story clearly:

- **Low λ**: Gates spread broadly across (0, 1) — most connections retained.
- **Medium λ**: Classic bimodal distribution — clear separation between pruned (~0) and active gates.
- **High λ**: Dominant spike at zero — the vast majority of connections removed; only the most essential survive.

A successful self-pruning result is indicated by **bimodality**: the gates don't just shrink uniformly, they *separate* into two populations. This is the L1 penalty's sparse-promoting property in action.

---

## 5. Key Takeaways

1. **The gated weight mechanism works**: Gradients flow correctly through both `weight` and `gate_scores`, and the L1 penalty successfully drives gates toward zero.

2. **L1 is essential for exact sparsity**: An L2 penalty would produce small but non-zero gates; L1 produces truly zero (or near-zero) gates.

3. **λ is the main dial**: The entire sparsity–accuracy spectrum is controlled by a single hyperparameter. In practice, λ = 1e-4 offers the best balance.

4. **The network learns what to prune**: The bimodal gate distribution shows the network isn't randomly zeroing weights — it actively identifies and preserves the most informative connections.

---

## 6. Possible Extensions

- **Hard masking after training**: After training, permanently zero out gates below threshold and re-fine-tune the remaining weights for improved accuracy.
- **Structured pruning**: Apply gates at the neuron or channel level rather than per-weight for better hardware efficiency.
- **Learned threshold**: Make the prune threshold itself a learnable parameter.
- **Convolutional layers**: Extend `PrunableLinear` to a `PrunableConv2d` for image-specific architectures (ResNet, VGG).

---

## 7. File Structure

```
├── self_pruning_nn.py       # Main script (all parts)
├── REPORT.md                # This report
├── gate_distributions.png   # Gate value histogram per λ
└── lambda_tradeoff.png      # Accuracy vs. sparsity bar chart
```

**To run:**
```bash
python self_pruning_nn.py
```
Requires: `torch`, `torchvision`, `matplotlib`, `numpy`
