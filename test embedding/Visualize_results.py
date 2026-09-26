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
    - language_pair_averages.png

Languages:
    AR    = Arabic
    EN    = English
    FR    = French
    Mixed = Arabic/English mixed

Requirements:
    pip install pandas matplotlib seaborn --break-system-packages
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------- Configuration ----------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(
    style="whitegrid",
    font_scale=1.0
)


# ---------- 1) تحميل النتائج ----------

full_sim = pd.read_csv(
    OUTPUT_DIR / "full_similarity_matrix.csv",
    index_col=0,
)

within_df = pd.read_csv(
    OUTPUT_DIR / "within_group_similarity.csv"
)


# ============================================================
# 2) Grouped Bar Chart
# ============================================================

"""
مقارنة الـ 6 language pairs لكل topic
"""

fig, ax = plt.subplots(
    figsize=(15, 7)
)

x = np.arange(len(within_df))

pair_columns = [
    ("ar_vs_en", "AR vs EN"),
    ("ar_vs_fr", "AR vs FR"),
    ("ar_vs_mixed", "AR vs Mixed"),
    ("en_vs_fr", "EN vs FR"),
    ("en_vs_mixed", "EN vs Mixed"),
    ("fr_vs_mixed", "FR vs Mixed"),
]

width = 0.12

for i, (column, label) in enumerate(pair_columns):

    offset = (
        i - (len(pair_columns) - 1) / 2
    ) * width

    bars = ax.bar(
        x + offset,
        within_df[column],
        width,
        label=label
    )

    ax.bar_label(
        bars,
        fmt="%.2f",
        fontsize=7,
        padding=2,
        rotation=90
    )


ax.set_xticks(x)

ax.set_xticklabels(
    within_df["topic"],
    rotation=20,
    ha="right"
)

ax.set_ylim(0, 1)

ax.set_ylabel(
    "Cosine Similarity"
)

ax.set_xlabel(
    "Topic"
)

ax.set_title(
    "Cross-lingual Similarity per Topic (BAAI/bge-m3)"
)

ax.legend(
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12)
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "grouped_bar_per_topic.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 3) Average Similarity per Topic
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

bars = ax.bar(
    within_df["topic"],
    within_df["avg"]
)

ax.set_ylim(0, 1)

ax.set_ylabel(
    "Average Cosine Similarity"
)

ax.set_xlabel(
    "Topic"
)

ax.set_title(
    "Average Cross-lingual Similarity per Topic"
)

ax.bar_label(
    bars,
    fmt="%.3f",
    padding=3
)

overall_avg = within_df["avg"].mean()

ax.axhline(
    overall_avg,
    linestyle="--",
    linewidth=1
)

ax.text(
    len(within_df) - 0.5,
    overall_avg + 0.015,
    f"Overall avg = {overall_avg:.3f}",
    ha="right",
    fontsize=9
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "avg_similarity_per_topic.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 4) Language Pair Average
# ============================================================

"""
متوسط similarity لكل language pair عبر كل الـ topics
"""

pair_labels = [
    "AR vs EN",
    "AR vs FR",
    "AR vs Mixed",
    "EN vs FR",
    "EN vs Mixed",
    "FR vs Mixed",
]

pair_values = [
    within_df["ar_vs_en"].mean(),
    within_df["ar_vs_fr"].mean(),
    within_df["ar_vs_mixed"].mean(),
    within_df["en_vs_fr"].mean(),
    within_df["en_vs_mixed"].mean(),
    within_df["fr_vs_mixed"].mean(),
]


fig, ax = plt.subplots(
    figsize=(10, 6)
)

bars = ax.bar(
    pair_labels,
    pair_values
)

ax.set_ylim(0, 1)

ax.set_ylabel(
    "Average Cosine Similarity"
)

ax.set_xlabel(
    "Language Pair"
)

ax.set_title(
    "Average Similarity by Language Pair"
)

ax.bar_label(
    bars,
    fmt="%.3f",
    padding=3
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "language_pair_averages.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 5) Mini Heatmaps per Group
# ============================================================

"""
كل topic له 4x4 similarity matrix:

        AR     EN     FR     Mixed
AR
EN
FR
Mixed

وبالتالي نقدر نشوف العلاقات بين الأربع نسخ
"""

topics = within_df["id"].tolist()

n = len(topics)

cols = 3

rows = int(
    np.ceil(n / cols)
)


fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(
        5 * cols,
        4.5 * rows
    )
)

axes = np.array(
    axes
).reshape(-1)


for i, gid in enumerate(topics):

    labels_group = [
        f"{gid}_ar",
        f"{gid}_en",
        f"{gid}_fr",
        f"{gid}_mixed",
    ]

    sub = full_sim.loc[
        labels_group,
        labels_group
    ]

    topic_name = within_df.loc[
        within_df["id"] == gid,
        "topic"
    ].values[0]


    sns.heatmap(
        sub,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        square=True,
        cbar=False,
        ax=axes[i],
        xticklabels=[
            "AR",
            "EN",
            "FR",
            "Mixed"
        ],
        yticklabels=[
            "AR",
            "EN",
            "FR",
            "Mixed"
        ],
    )

    axes[i].set_title(
        f"Group {gid}: {topic_name}"
    )

    axes[i].set_xlabel("")
    axes[i].set_ylabel("")


# إخفاء الـ subplots الزائدة

for j in range(
    n,
    len(axes)
):
    axes[j].axis("off")


plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "mini_heatmaps_per_group.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 6) Print Summary
# ============================================================

print(
    "\nSaved visualizations in ./outputs:"
)

print(
    " - grouped_bar_per_topic.png"
    "       (6 language pairs per topic)"
)

print(
    " - avg_similarity_per_topic.png"
    "       (average similarity per topic)"
)

print(
    " - language_pair_averages.png"
    "       (average similarity per language pair)"
)

print(
    " - mini_heatmaps_per_group.png"
    "       (4x4 heatmap for each topic)"
)

print(
    "\nAverage similarity by language pair:"
)

for label, value in zip(
    pair_labels,
    pair_values
):

    print(
        f" - {label:15s}: {value:.4f}"
    )