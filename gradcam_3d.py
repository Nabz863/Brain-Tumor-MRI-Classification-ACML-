"""
gradcam_3d.py  —  Grad-CAM + 3D Surface Visualisation
=======================================================
Generates for each test class:
  1. Original MRI image
  2. 2D Grad-CAM heatmap overlay
  3. 3D surface plot of the activation map
  4. Combined 4-panel figure (one per class)

Usage:
  python gradcam_3d.py

Outputs saved to:  plots/gradcam/
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
TEST_DIR  = r"Dataset\test"
MODEL_PATH = "Models/brain_tumor_cnn.pth"
OUT_DIR    = "Plots/gradcam"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE      = torch.device("cpu")
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE    = 256

# ── Model definition (must match train_improved.py) ───────────────────────────
class BetterBrainTumorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   # 0
            nn.ReLU(),                                      # 1
            nn.MaxPool2d(2),                                # 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),   # 3
            nn.ReLU(),                                      # 4
            nn.MaxPool2d(2),                                # 5
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 6
            nn.ReLU(),                                      # 7
            nn.MaxPool2d(2),                                # 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 9
            nn.ReLU(),                                      # 10
            nn.MaxPool2d(2),                                # 11
            nn.AdaptiveAvgPool2d((1, 1))                    # 12
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Hooks into the last conv layer (features[9]) and computes the
    class-discriminative localisation map.
    """
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global average pool the gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(cam,
                            size=(input_tensor.shape[2], input_tensor.shape[3]),
                            mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


# ── Denormalise helper ────────────────────────────────────────────────────────
def denormalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    t = tensor.clone()
    for i in range(3):
        t[i] = t[i] * std[i] + mean[i]
    return t.clamp(0, 1)


# ── 4-panel figure per sample ─────────────────────────────────────────────────
def make_gradcam_figure(img_tensor, cam, true_label, pred_label, confidence):
    """
    Panel layout:
      [Original MRI] [2D Grad-CAM overlay] [Heatmap only] [3D Surface]
    """
    img_np = denormalize(img_tensor.squeeze(0)).permute(1, 2, 0).detach().numpy()
    gray    = img_np.mean(axis=2)           # grayscale for 3-D surface
    H, W    = cam.shape

    fig = plt.figure(figsize=(18, 5), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(1, 4, figure=fig,
                            wspace=0.05, left=0.02, right=0.98,
                            top=0.82, bottom=0.08)

    # ── 1. Original ───────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(img_np)
    ax0.set_title("Original MRI", color="white", fontsize=11, pad=6)
    ax0.axis("off")

    # ── 2. Grad-CAM overlay ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(img_np)
    ax1.imshow(cam, cmap="jet", alpha=0.45,
               vmin=0, vmax=1, extent=[0, W, H, 0])
    ax1.set_title("Grad-CAM Overlay", color="white", fontsize=11, pad=6)
    ax1.axis("off")

    # ── 3. Heatmap only ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    im  = ax2.imshow(cam, cmap="inferno", vmin=0, vmax=1)
    ax2.set_title("Activation Heatmap", color="white", fontsize=11, pad=6)
    ax2.axis("off")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    cbar.set_label("Activation", color="white", fontsize=8)

    # ── 4. 3-D surface ────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[3], projection="3d")
    ax3.set_facecolor("#0d0d0d")
    ax3.patch.set_facecolor("#0d0d0d")

    # Downsample for performance
    step  = max(1, H // 64)
    xs    = np.arange(0, W, step)
    ys    = np.arange(0, H, step)
    X, Y  = np.meshgrid(xs, ys)
    Z_cam = cam[::step, ::step][:Y.shape[0], :X.shape[1]]
    Z_img = gray[::step, ::step][:Y.shape[0], :X.shape[1]]

    # Surface coloured by activation, height = activation
    facecolors = cm.jet(Z_cam)
    facecolors[..., 3] = 0.9   # slight transparency

    ax3.plot_surface(X, Y, Z_cam,
                     facecolors=facecolors,
                     linewidth=0, antialiased=True, shade=True)

    ax3.set_zlim(0, 1.4)
    ax3.set_xlabel("x", color="white", fontsize=7, labelpad=-4)
    ax3.set_ylabel("y", color="white", fontsize=7, labelpad=-4)
    ax3.set_zlabel("activation", color="white", fontsize=7, labelpad=-4)
    ax3.tick_params(colors="white", labelsize=6, pad=-4)
    for pane in [ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane]:
        pane.fill  = False
        pane.set_edgecolor("#333333")
    ax3.set_title("3-D Activation Surface", color="white", fontsize=11, pad=6)
    ax3.view_init(elev=35, azim=-55)

    # ── Overall title ─────────────────────────────────────────────────────────
    correct = "✓" if true_label == pred_label else "✗"
    title   = (f"True: {CLASS_NAMES[true_label].upper()}   "
               f"Predicted: {CLASS_NAMES[pred_label].upper()}  "
               f"({confidence*100:.1f}% conf)  {correct}")
    colour  = "#4ade80" if true_label == pred_label else "#f87171"
    fig.suptitle(title, color=colour, fontsize=13,
                 fontweight="bold", y=0.97)

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load model
    model = BetterBrainTumorCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded.\n")

    # Attach Grad-CAM to last conv layer (features[9] = Conv2d 64→128)
    gradcam = GradCAM(model, model.features[9])

    # Transform (same as evaluate.py)
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(root=TEST_DIR, transform=tf)

    # Pick one sample per class (the first correctly classified one)
    picked = {}          # class_idx → (img_tensor, true_label)
    random.seed(42)

    # Group indices by class
    by_class = {c: [] for c in range(4)}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class[label].append(idx)
    for c in range(4):
        random.shuffle(by_class[c])

    print("Selecting one sample per class and generating visualisations...\n")
    for class_idx in range(4):
        for sample_idx in by_class[class_idx]:
            img_tensor, true_label = dataset[sample_idx]
            inp = img_tensor.unsqueeze(0)

            with torch.no_grad():
                logits = model(inp)
                probs  = F.softmax(logits, dim=1)

            pred_label  = probs.argmax(dim=1).item()
            confidence  = probs[0, pred_label].item()

            # Prefer correctly classified samples
            if pred_label == true_label:
                picked[class_idx] = (img_tensor, true_label, pred_label, confidence)
                break
        else:
            # Fall back to any sample if none correct
            sample_idx = by_class[class_idx][0]
            img_tensor, true_label = dataset[sample_idx]
            inp    = img_tensor.unsqueeze(0)
            logits = model(inp)
            probs  = F.softmax(logits, dim=1)
            pred_label = probs.argmax(1).item()
            confidence = probs[0, pred_label].item()
            picked[class_idx] = (img_tensor, true_label, pred_label, confidence)

    for class_idx, (img_tensor, true_label, pred_label, confidence) in picked.items():
        inp = img_tensor.unsqueeze(0)
        cam, _ = gradcam.generate(inp, class_idx=pred_label)

        fig  = make_gradcam_figure(inp, cam, true_label, pred_label, confidence)
        path = os.path.join(OUT_DIR, f"gradcam_3d_{CLASS_NAMES[class_idx]}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Saved: {path}")

    # ── Summary panel: all 4 classes side-by-side (just the overlays) ─────────
    print("\nGenerating summary panel...")
    fig_summary, axes = plt.subplots(2, 4, figsize=(20, 10),
                                      facecolor="#0d0d0d")

    row_titles = ["Grad-CAM Overlay", "3-D Activation Surface"]
    for col, class_idx in enumerate(range(4)):
        img_tensor, true_label, pred_label, confidence = picked[class_idx]
        inp = img_tensor.unsqueeze(0)
        cam, _ = gradcam.generate(inp, class_idx=pred_label)

        img_np = denormalize(img_tensor).permute(1, 2, 0).numpy()
        H, W   = cam.shape

        # Row 0 — overlay
        ax = axes[0, col]
        ax.imshow(img_np)
        ax.imshow(cam, cmap="jet", alpha=0.5,
                  vmin=0, vmax=1, extent=[0, W, H, 0])
        ax.set_title(CLASS_NAMES[class_idx].upper(),
                     color="white", fontsize=12, fontweight="bold")
        ax.axis("off")

        # Row 1 — heatmap
        ax2 = axes[1, col]
        ax2.imshow(cam, cmap="inferno", vmin=0, vmax=1)
        ax2.axis("off")

    for r, title in enumerate(row_titles):
        axes[r, 0].set_ylabel(title, color="white", fontsize=10, rotation=90,
                               labelpad=8)

    fig_summary.suptitle("Grad-CAM: Model Attention Across All Four Tumour Classes",
                          color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    summary_path = os.path.join(OUT_DIR, "gradcam_summary.png")
    fig_summary.savefig(summary_path, dpi=150, bbox_inches="tight",
                         facecolor=fig_summary.get_facecolor())
    plt.close(fig_summary)
    print(f"  Saved: {summary_path}")

    print(f"\nAll Grad-CAM outputs saved to: {OUT_DIR}/")
    print("Files generated:")
    for cls in CLASS_NAMES:
        print(f"  gradcam_3d_{cls}.png  — 4-panel (original | overlay | heatmap | 3D surface)")
    print(f"  gradcam_summary.png  — all 4 classes side-by-side")


if __name__ == "__main__":
    main()
