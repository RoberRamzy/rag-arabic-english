"""
Visualize Cross-lingual Similarity Results
--------------------------------------------
يقرأ النتائج اللي اتحفظت من bge_m3_similarity_test.py:
    - outputs/full_similarity_matrix.csv
    - outputs/within_group_similarity.csv
ويطلع رسومات واضحة تلخص النتائج:
    - grouped_bar_per_topic.png
    - avg_similarity_per_topic.png
    - mini_heatmaps_per_group.png
(من غير الـ full 15x15 heatmap)

Requirements:
    pip install pandas matplotlib seaborn --break-system-packages
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 1) تحميل النتائج ----------
full_sim = pd.read_csv(os.path.join(OUTPUT_DIR, "full_similarity_matrix.csv"), index_col=0)
within_df = pd.read_csv(os.path.join(OUTPUT_DIR, "within_group_similarity.csv"))

sns.set_theme(style="whitegrid", font_scale=1.0)

# ---------- 2) Grouped Bar Chart: مقارنة التلات metrics لكل موضوع ----------
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(within_df))
width = 0.25

ax.bar(x - width, within_df["ar_vs_en"], width, label="AR vs EN", color="#4C72B0")
ax.bar(x,          within_df["ar_vs_mixed"], width, label="AR vs Mixed", color="#DD8452")
ax.bar(x + width,  within_df["en_vs_mixed"], width, label="EN vs Mixed", color="#55A868")

ax.set_xticks(x)
ax.set_xticklabels(within_df["topic"], rotation=20, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("Cosine Similarity")
ax.set_title("Within-group Similarity per Topic (BAAI/bge-m3)")
ax.legend()

# رقم فوق كل عمود
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "grouped_bar_per_topic.png"), dpi=200)
plt.close()

# ---------- 3) Bar chart لمتوسط كل موضوع (avg) ----------
fig, ax = plt.subplots(figsize=(8, 5))
colors = sns.color_palette("viridis", len(within_df))
bars = ax.bar(within_df["topic"], within_df["avg"], color=colors)
ax.set_ylim(0, 1)
ax.set_ylabel("Average Cosine Similarity")
ax.set_title("Average Cross-lingual Similarity per Topic")
ax.bar_label(bars, fmt="%.3f", padding=3)
overall_avg = within_df["avg"].mean()
ax.axhline(overall_avg, color="red", linestyle="--", linewidth=1)
ax.text(len(within_df) - 0.5, overall_avg + 0.015,
        f"Overall avg = {overall_avg:.3f}", color="red", ha="right", fontsize=9)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "avg_similarity_per_topic.png"), dpi=200)
plt.close()

# ---------- 4) Mini heatmaps: كل موضوع لوحده (3x3) ----------
topics = within_df["id"].tolist()
n = len(topics)
cols = 3
rows = int(np.ceil(n / cols))

fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
axes = np.array(axes).reshape(-1)

for i, gid in enumerate(topics):
    labels_group = [f"{gid}_ar", f"{gid}_en", f"{gid}_mixed"]
    sub = full_sim.loc[labels_group, labels_group]
    topic_name = within_df.loc[within_df["id"] == gid, "topic"].values[0]

    sns.heatmap(
        sub, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0.6, vmax=1,
        square=True, cbar=False, ax=axes[i],
        xticklabels=["AR", "EN", "Mixed"], yticklabels=["AR", "EN", "Mixed"],
    )
    axes[i].set_title(f"Group {gid}: {topic_name}")

# إخفاء أي subplot زيادة لو عدد المواضيع مش pos مضبوط مع الشبكة
for j in range(n, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mini_heatmaps_per_group.png"), dpi=200)
plt.close()

print("Saved visualizations in ./outputs :")
print(" - grouped_bar_per_topic.png   (مقارنة AR/EN/Mixed لكل موضوع)")
print(" - avg_similarity_per_topic.png (متوسط كل موضوع)")
print(" - mini_heatmaps_per_group.png  (heatmap صغير 3x3 لكل موضوع)")