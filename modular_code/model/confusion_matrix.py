import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# ==== INPUTS (your numbers) ====
cm = np.array([[1043, 163],
               [136, 241]], dtype=int)  # [[TN, FP], [FN, TP]]
labels = ["Low-Threat (0)", "High-Threat (1)"]

# Metrics (from your report)
precision_0, recall_0, f1_0, support_0 = 0.8461538461538461, 0.61875, 0.7148014440433214, 1600
precision_1, recall_1, f1_1, support_1 = 0.32671081677704195, 0.6218487394957983, 0.4283646888567294, 476
accuracy = 0.6194605009633911

# ==== DERIVED VALUES ====
totals = cm.sum()
row_sums = cm.sum(axis=1, keepdims=True)  # actual totals
col_sums = cm.sum(axis=0, keepdims=True)  # predicted totals

# Normalized by row (recall view)
cm_row_pct = cm / row_sums

# ==== PLOTTING ====
fig = plt.figure(figsize=(9, 8), dpi=160)
gs = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[4, 1.6], height_ratios=[4, 1.2], wspace=0.25, hspace=0.25)

# Heatmap (top-left)
ax = fig.add_subplot(gs[0,0])
cmap = colors.LinearSegmentedColormap.from_list("", ["#f7faf7", "#a7d1b9", "#2d7f5e"])
im = ax.imshow(cm_row_pct, cmap=cmap, vmin=0, vmax=1)

# Grid lines
for i in range(cm.shape[0] + 1):
    ax.axhline(i - 0.5, color="#2f5f4d", lw=1, alpha=0.6)
    ax.axvline(i - 0.5, color="#2f5f4d", lw=1, alpha=0.6)

# Ticks/labels
ax.set_xticks(np.arange(2), labels=[f"Pred {l}" for l in labels])
ax.set_yticks(np.arange(2), labels=[f"Actual {l}" for l in labels])
ax.tick_params(axis="both", which="both", length=0)

# Cell annotations: count + row %
for i in range(2):
    for j in range(2):
        count = cm[i, j]
        pct = cm_row_pct[i, j] * 100
        ax.text(j, i, f"{count}\n{pct:.1f}%",
                ha="center", va="center",
                fontsize=12, fontweight="bold",
                color="#0f2e25")

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Row-normalized (%)", rotation=90)

ax.set_title("Confusion Matrix — CatBoost Model", pad=12, fontsize=14, fontweight="bold")




plt.show()