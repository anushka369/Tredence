"""
Self-Pruning Neural Network on CIFAR-10
========================================
Tredence AI Engineering Intern - Case Study

This script implements a neural network that learns to prune itself during
training using learnable "gate" parameters per weight. Gates are driven
toward zero via an L1 sparsity regularization term in the loss function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — PrunableLinear Layer
# ─────────────────────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A custom linear layer augmented with learnable gate parameters.

    Each weight has a corresponding scalar gate_score. The gate_score is
    passed through a Sigmoid to produce a gate value in (0, 1). The weight
    is then multiplied element-wise by its gate before the linear operation.

    When a gate collapses to ~0, it effectively removes (prunes) that weight
    from the network. Gradients flow through both `weight` and `gate_scores`
    because all operations (sigmoid, multiplication, linear) are differentiable.

    Args:
        in_features  (int): Size of each input sample.
        out_features (int): Size of each output sample.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight and bias — same as nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Gate scores: one scalar per weight element (same shape as weight).
        # Registered as a Parameter so the optimizer updates them too.
        # Initialized slightly positive so gates start near sigmoid(0.5) ≈ 0.62,
        # giving the network room to move in either direction.
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        self._init_parameters()

    def _init_parameters(self):
        """Kaiming uniform for weights (standard); small normal noise for gates."""
        nn.init.kaiming_uniform_(self.weight, a=0.01)
        nn.init.normal_(self.gate_scores, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          1. Compute gates = sigmoid(gate_scores)  → values in (0, 1)
          2. pruned_weights = weight * gates        → element-wise mask
          3. output = x @ pruned_weights.T + bias   → standard affine transform
        """
        gates          = torch.sigmoid(self.gate_scores)           # (out, in)
        pruned_weights = self.weight * gates                        # (out, in)
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return the current gate values (detached from graph)."""
        return torch.sigmoid(self.gate_scores).detach()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# ─────────────────────────────────────────────────────────────────────────────
# Network Definition
# ─────────────────────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 image classification built entirely
    from PrunableLinear layers so every weight can be pruned.

    Architecture:
        Input (3×32×32 = 3072) → 1024 → 512 → 256 → 10 (classes)
    All hidden activations use ReLU; Batch Normalization stabilizes training.
    """

    def __init__(self):
        super().__init__()

        self.fc1 = PrunableLinear(3 * 32 * 32, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = PrunableLinear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = PrunableLinear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = PrunableLinear(256, 10)   # output layer

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)                      # flatten

        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)                                 # logits
        return x

    def prunable_layers(self):
        """Yield all PrunableLinear layers in the network."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — Sparsity Loss
# ─────────────────────────────────────────────────────────────────────────────

def sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    """
    Compute the L1 norm of all gate values across every PrunableLinear layer.

    Why L1 encourages sparsity:
    - The L1 norm penalises every gate proportionally to its magnitude.
    - The gradient of |g| w.r.t. g is ±1 (constant), meaning even very small
      gate values keep receiving the same constant "push toward zero."
    - L2 would give a gradient of 2g → the push vanishes as g → 0, so weights
      never reach exactly zero. L1 does not have this problem.
    - Combined with the sigmoid (which can produce values arbitrarily close to 0),
      the optimiser actively drives unnecessary gates to near-zero, effectively
      removing those weights.
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores)   # values in (0,1), always ≥ 0
        total = total + gates.sum()                # L1 norm = sum of absolutes
    return total


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 256):
    """Download CIFAR-10 and return train / test DataLoaders."""

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=train_transform)
    test_ds  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, lam):
    """Train for a single epoch. Returns (avg_total_loss, avg_cls_loss, avg_spar_loss)."""
    model.train()
    total_loss_sum = cls_loss_sum = spar_loss_sum = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits   = model(images)
        cls_loss = F.cross_entropy(logits, labels)
        spar     = sparsity_loss(model)
        loss     = cls_loss + lam * spar

        loss.backward()
        # Gradient clipping prevents instability with high λ values
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss_sum += loss.item()
        cls_loss_sum   += cls_loss.item()
        spar_loss_sum  += spar.item()

    n = len(loader)
    return total_loss_sum / n, cls_loss_sum / n, spar_loss_sum / n


@torch.no_grad()
def evaluate(model, loader, device):
    """Return top-1 test accuracy (0–100)."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds   = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def compute_sparsity(model, threshold=1e-2):
    """
    Compute the sparsity level: % of weights whose gate value < threshold.
    A gate below the threshold is considered effectively pruned.
    """
    total_weights = pruned_weights = 0
    for layer in model.prunable_layers():
        gates          = layer.get_gates()
        total_weights  += gates.numel()
        pruned_weights += (gates < threshold).sum().item()
    return 100.0 * pruned_weights / total_weights


def collect_all_gates(model):
    """Return a flat numpy array of all gate values (for plotting)."""
    all_gates = []
    for layer in model.prunable_layers():
        all_gates.append(layer.get_gates().cpu().numpy().ravel())
    return np.concatenate(all_gates)


def train_experiment(lam, train_loader, test_loader, device,
                     num_epochs=30, lr=1e-3):
    """
    Run a full training experiment for a given λ value.
    Returns: (test_accuracy, sparsity_level, gate_values_array)
    """
    print(f"\n{'='*60}")
    print(f"  Training with λ = {lam}")
    print(f"{'='*60}")

    model     = SelfPruningNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=num_epochs, eta_min=1e-5)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        total, cls, spar = train_one_epoch(
            model, train_loader, optimizer, device, lam)
        scheduler.step()
        elapsed = time.time() - t0

        if epoch % 5 == 0 or epoch == 1:
            acc      = evaluate(model, test_loader, device)
            sparsity = compute_sparsity(model)
            print(f"  Epoch {epoch:3d}/{num_epochs} | "
                  f"Loss: {total:.4f} (cls={cls:.4f}, spar={spar:.1f}) | "
                  f"Acc: {acc:.2f}% | Sparsity: {sparsity:.1f}% | "
                  f"Time: {elapsed:.1f}s")

    final_acc      = evaluate(model, test_loader, device)
    final_sparsity = compute_sparsity(model)
    gate_values    = collect_all_gates(model)

    print(f"\n  Final — Accuracy: {final_acc:.2f}%  |  "
          f"Sparsity: {final_sparsity:.2f}%")

    return final_acc, final_sparsity, gate_values


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_gate_distributions(results: dict, save_path="gate_distributions.png"):
    """
    Plot the gate value distribution for each λ experiment side by side.
    A successful result shows a large spike near 0 and a secondary cluster away from 0.
    """
    lambdas = list(results.keys())
    n       = len(lambdas)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    colours = ["#E74C3C", "#3498DB", "#2ECC71"]

    for ax, lam, colour in zip(axes, lambdas, colours):
        gates = results[lam]["gates"]
        acc   = results[lam]["accuracy"]
        spar  = results[lam]["sparsity"]

        ax.hist(gates, bins=100, color=colour, alpha=0.85, edgecolor="white",
                linewidth=0.3)
        ax.set_title(f"λ = {lam}\nAcc: {acc:.1f}%  |  Sparsity: {spar:.1f}%",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Gate Value", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.axvline(x=0.01, color="black", linestyle="--", linewidth=1.2,
                   label="Prune threshold (0.01)")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Distribution of Gate Values — Self-Pruning Neural Network\n"
                 "(Spike near 0 = weights being pruned)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  📊 Plot saved → {save_path}")
    return fig


def plot_summary_bar(results: dict, save_path="lambda_tradeoff.png"):
    """Bar chart comparing accuracy and sparsity across λ values."""
    lambdas   = [str(l) for l in results.keys()]
    accs      = [results[l]["accuracy"]  for l in results.keys()]
    sparsities= [results[l]["sparsity"]  for l in results.keys()]

    x   = np.arange(len(lambdas))
    w   = 0.35
    fig, ax1 = plt.subplots(figsize=(8, 5))

    bars1 = ax1.bar(x - w/2, accs,      w, label="Test Accuracy (%)",
                    color="#3498DB", alpha=0.85)
    ax1.set_ylabel("Test Accuracy (%)", color="#3498DB", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#3498DB")
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + w/2, sparsities, w, label="Sparsity Level (%)",
                    color="#E74C3C", alpha=0.85)
    ax2.set_ylabel("Sparsity Level (%)", color="#E74C3C", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#E74C3C")
    ax2.set_ylim(0, 100)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"λ = {l}" for l in lambdas], fontsize=12)
    ax1.set_title("Accuracy vs. Sparsity Trade-off Across λ Values",
                  fontsize=13, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  📊 Trade-off plot saved → {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("\nLoading CIFAR-10 …")
    train_loader, test_loader = get_cifar10_loaders(batch_size=256)

    # Three λ values: low → mild pruning, medium → balanced, high → aggressive
    lambda_values = [1e-5, 1e-4, 1e-3]
    num_epochs    = 30       # Increase to 60–80 for better final accuracy

    results = {}

    for lam in lambda_values:
        acc, sparsity, gates = train_experiment(
            lam, train_loader, test_loader, device, num_epochs=num_epochs)
        results[lam] = {"accuracy": acc, "sparsity": sparsity, "gates": gates}

    # ── Results Summary ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"  {'Lambda':<12} {'Test Accuracy':>15} {'Sparsity Level':>16}")
    print(f"  {'-'*45}")
    for lam, r in results.items():
        print(f"  {str(lam):<12} {r['accuracy']:>13.2f}%  {r['sparsity']:>13.2f}%")
    print("="*60)

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_gate_distributions(results, save_path="/home/claude/gate_distributions.png")
    plot_summary_bar(results,        save_path="/home/claude/lambda_tradeoff.png")

    print("\nExperiment complete!\n")
    return results


if __name__ == "__main__":
    main()
