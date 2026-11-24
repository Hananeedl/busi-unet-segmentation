# ============================================================
# SDSC6018 Group 13 — 结果分析脚本
# 用途：读取实验脚本(unet_busi_local.py)生成的三个 CSV，
#       一键生成期末报告要用的汇总表和图。
# 运行：python analyze_results.py
# 输入：rq1_results.csv / rq2_results.csv / rq3_cases.csv
# 输出：table_rq1_summary.csv, table_rq2_summary.csv,
#       fig_rq1_model_comparison.png, fig_rq2_learning_curve.png,
#       fig_rq3_risk_coverage.png, fig_rq3_uncertainty_scatter.png
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 不弹窗，直接存图
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

# 无论从哪里启动，都在本脚本所在目录读写
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------- RQ1: U-Net vs Attention U-Net ----------
r1 = pd.read_csv("rq1_results.csv")
rq1_summary = r1.groupby("model")[["dice", "iou"]].agg(["mean", "std"])
print("========== RQ1: model comparison ==========")
print(rq1_summary.round(4))
rq1_summary.round(4).to_csv("table_rq1_summary.csv")

fig, ax = plt.subplots(figsize=(6, 4))
models = ["UNet", "AttUNet"]
labels = ["U-Net", "Attention U-Net"]
means = [r1[r1.model == m]["dice"].mean() for m in models]
stds = [r1[r1.model == m]["dice"].std() for m in models]
ax.bar(labels, means, yerr=stds, capsize=6, color=["#4C72B0", "#DD8452"], width=0.5)
for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.02, f"{m:.3f}±{s:.3f}", ha="center", fontsize=10)
ax.set_ylabel("Dice coefficient")
ax.set_title("RQ1: U-Net vs Attention U-Net (5-fold CV)")
ax.set_ylim(0, 1)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("fig_rq1_model_comparison.png", dpi=200)
plt.close()

# ---------- RQ2: data-efficiency learning curve ----------
r2 = pd.read_csv("rq2_results.csv")
rq2_summary = r2.groupby(["fraction", "augment"])[["test_dice", "test_iou"]].agg(["mean", "std"])
print("\n========== RQ2: data efficiency ==========")
print(rq2_summary.round(4))
rq2_summary.round(4).to_csv("table_rq2_summary.csv")

fig, ax = plt.subplots(figsize=(6, 4))
fracs = sorted(r2["fraction"].unique())
for aug, color, marker, lab in [
    (False, "#4C72B0", "o", "without augmentation"),
    (True, "#DD8452", "s", "with augmentation"),
]:
    sub = r2[r2.augment == aug].groupby("fraction")["test_dice"]
    m = [sub.get_group(f).mean() for f in fracs]
    s = [sub.get_group(f).std() for f in fracs]
    ax.errorbar([f * 100 for f in fracs], m, yerr=s, marker=marker, color=color,
                capsize=4, linewidth=2, label=lab)
ax.set_xlabel("Training data used (%)")
ax.set_ylabel("Test Dice coefficient")
ax.set_title("RQ2: Data-efficiency learning curve")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_rq2_learning_curve.png", dpi=200)
plt.close()

# ---------- RQ3: MC Dropout uncertainty ----------
r3 = pd.read_csv("rq3_cases.csv")
u = r3["uncertainty"].values
d = r3["dice"].values

corr, p_val = spearmanr(u, d)
failed = (d < 0.5).astype(int)
if 0 < failed.sum() < len(failed):
    auroc = roc_auc_score(failed, u)
else:
    auroc = float("nan")
print("\n========== RQ3: uncertainty ==========")
print(f"cases: {len(r3)}, failed (Dice<0.5): {failed.sum()}")
print(f"Spearman correlation (uncertainty vs Dice): {corr:.4f} (p={p_val:.4e})")
print(f"AUROC for predicting failed cases: {auroc:.4f}")

# 风险-覆盖率（弃权）曲线：按不确定性从低到高保留病例，看保留病例的平均 Dice
order = np.argsort(u)  # 最自信的在前
d_sorted = d[order]
coverage = np.arange(1, len(d_sorted) + 1) / len(d_sorted)
kept_mean_dice = np.cumsum(d_sorted) / np.arange(1, len(d_sorted) + 1)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(coverage * 100, kept_mean_dice, color="#4C72B0", linewidth=2)
ax.axhline(d.mean(), ls="--", color="gray",
           label=f"mean Dice of all cases = {d.mean():.3f}")
ax.set_xlabel("Coverage: % of most confident cases kept")
ax.set_ylabel("Mean Dice of kept cases")
ax.set_title("RQ3: Risk-coverage (abstention) curve")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_rq3_risk_coverage.png", dpi=200)
plt.close()

# 不确定性-Dice 散点图
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(u, d, s=16, alpha=0.6, color="#4C72B0", edgecolors="none")
ax.set_xlabel("Mean uncertainty (MC Dropout pixel variance)")
ax.set_ylabel("Test Dice coefficient")
ax.set_title(f"RQ3: Uncertainty vs Dice (Spearman={corr:.2f}, p={p_val:.1e})")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig_rq3_uncertainty_scatter.png", dpi=200)
plt.close()

print("\n已生成文件：")
print("  表格: table_rq1_summary.csv, table_rq2_summary.csv")
print("  图:   fig_rq1_model_comparison.png, fig_rq2_learning_curve.png,")
print("        fig_rq3_risk_coverage.png, fig_rq3_uncertainty_scatter.png")
