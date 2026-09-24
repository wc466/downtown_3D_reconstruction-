"""Plotting helpers (headless)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_matches(img0, img1, xy0, xy1, out_path, title=None, max_lines=300, seed=0):
    """Side-by-side images with a random subset of correspondences drawn as lines.

    img0, img1: (H, W, 3) float in [0, 1]; xy0, xy1: (N, 2) pixel coords (x, y).
    """
    h = max(img0.shape[0], img1.shape[0])
    canvas = np.ones((h, img0.shape[1] + img1.shape[1], 3))
    canvas[:img0.shape[0], :img0.shape[1]] = img0
    canvas[:img1.shape[0], img0.shape[1]:] = img1

    fig, ax = plt.subplots(figsize=(canvas.shape[1] / 100, canvas.shape[0] / 100 + 0.4), dpi=100)
    ax.imshow(canvas)
    if len(xy0):
        idx = np.random.default_rng(seed).permutation(len(xy0))[:max_lines]
        colors = plt.cm.hsv(np.linspace(0, 1, len(idx), endpoint=False))
        for c, i in zip(colors, idx):
            ax.plot([xy0[i, 0], xy1[i, 0] + img0.shape[1]], [xy0[i, 1], xy1[i, 1]], "-", color=c, lw=0.6)
    ax.set_title(title or f"{len(xy0)} matches (showing {min(len(xy0), max_lines)})", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
