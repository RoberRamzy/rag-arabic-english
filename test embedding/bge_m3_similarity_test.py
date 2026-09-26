"""
Cross-lingual Semantic Similarity Test using BAAI/bge-m3
-----------------------------------------------------------
يقرأ ملف sentences.json
(نفس الجملة بـ 4 نسخ: عربي / إنجليزي / فرنسي / مخلوط)

يعمل embeddings باستخدام BAAI/bge-m3
يحسب cosine similarity بين كل جملة وكل جملة تانية
يطبع جدول مقارنة لكل مجموعة
ويحفظ CSV بالنتائج (من غير رسم heatmap)

Requirements:
    pip install -U sentence-transformers numpy pandas --break-system-packages
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ---------- 1) تحميل الداتا ----------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "sentences.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)["sentences"]


# اللغات الموجودة في الداتا
LANGS = ["ar", "en", "fr", "mixed"]


texts = []
labels = []
topics = []

for item in raw:
    for lang in LANGS:
        texts.append(item[lang])
        labels.append(f"{item['id']}_{lang}")
        topics.append(item.get("topic", str(item["id"])))


print(f"Loaded {len(texts)} sentences from {len(raw)} groups.")
print(f"Languages: {', '.join(LANGS)}\n")


# ---------- 2) تحميل الموديل وعمل الـ embeddings ----------

print("Loading BAAI/bge-m3 ...")

model = SentenceTransformer("BAAI/bge-m3")

# BGE-M3 لا يحتاج instruction prefix
# لأننا نعمل symmetric semantic similarity
embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True,
)

embeddings = np.asarray(embeddings)


# ---------- 3) حساب مصفوفة الـ cosine similarity ----------

# بما أن embeddings معمولة normalize:
# cosine similarity = dot product

sim_matrix = embeddings @ embeddings.T

sim_df = pd.DataFrame(
    sim_matrix,
    index=labels,
    columns=labels,
)


# ---------- 4) Within-group similarity ----------

print("\n" + "=" * 80)
print("Within-group semantic similarity")
print("Same sentence expressed in different languages")
print("=" * 80)


within_rows = []

for item in raw:

    gid = item["id"]
    topic = item.get("topic", str(gid))

    ar_label = f"{gid}_ar"
    en_label = f"{gid}_en"
    fr_label = f"{gid}_fr"
    mixed_label = f"{gid}_mixed"

    # Pairwise similarities
    ar_en = sim_df.loc[ar_label, en_label]
    ar_fr = sim_df.loc[ar_label, fr_label]
    ar_mixed = sim_df.loc[ar_label, mixed_label]

    en_fr = sim_df.loc[en_label, fr_label]
    en_mixed = sim_df.loc[en_label, mixed_label]

    fr_mixed = sim_df.loc[fr_label, mixed_label]

    values = np.asarray([
        ar_en,
        ar_fr,
        ar_mixed,
        en_fr,
        en_mixed,
        fr_mixed,
    ], dtype=float)

    within_rows.append({
        "id": gid,
        "topic": topic,

        "ar_vs_en": round(float(ar_en), 4),
        "ar_vs_fr": round(float(ar_fr), 4),
        "ar_vs_mixed": round(float(ar_mixed), 4),

        "en_vs_fr": round(float(en_fr), 4),
        "en_vs_mixed": round(float(en_mixed), 4),

        "fr_vs_mixed": round(float(fr_mixed), 4),

        "avg": round(float(np.mean(values)), 4),
    })


within_df = pd.DataFrame(within_rows)

print(within_df.to_string(index=False))


# ---------- 5) Overall metrics ----------

print("\n" + "=" * 80)
print("Overall metrics")
print("=" * 80)


pair_columns = [
    "ar_vs_en",
    "ar_vs_fr",
    "ar_vs_mixed",
    "en_vs_fr",
    "en_vs_mixed",
    "fr_vs_mixed",
]


for column in pair_columns:

    mean_value = within_df[column].mean()

    print(
        f"{column:15s}: "
        f"{mean_value:.4f}"
    )


avg_overall = within_df[pair_columns].values.mean()

print("-" * 80)
print(f"Overall average: {avg_overall:.4f}")


# ---------- 6) Most similar sentence ----------

print("\n" + "=" * 80)
print("Most similar sentence to each sentence")
print("(excluding itself)")
print("=" * 80)


for label in labels:

    row = sim_df.loc[label].drop(label)

    best_match = row.idxmax()

    print(
        f"{label:15s} -> "
        f"{best_match:15s} "
        f"(sim = {row[best_match]:.4f})"
    )


# ---------- 7) Cross-language averages ----------

print("\n" + "=" * 80)
print("Average similarity by language pair")
print("=" * 80)


pair_results = {
    "Arabic ↔ English": within_df["ar_vs_en"].mean(),
    "Arabic ↔ French": within_df["ar_vs_fr"].mean(),
    "Arabic ↔ Mixed": within_df["ar_vs_mixed"].mean(),
    "English ↔ French": within_df["en_vs_fr"].mean(),
    "English ↔ Mixed": within_df["en_vs_mixed"].mean(),
    "French ↔ Mixed": within_df["fr_vs_mixed"].mean(),
}


for pair, score in pair_results.items():

    print(
        f"{pair:20s}: "
        f"{score:.4f}"
    )


# ---------- 8) حفظ النتائج ----------

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


sim_df.to_csv(
    OUTPUT_DIR / "full_similarity_matrix.csv",
    encoding="utf-8-sig",
)


within_df.to_csv(
    OUTPUT_DIR / "within_group_similarity.csv",
    index=False,
    encoding="utf-8-sig",
)


# حفظ متوسطات أزواج اللغات في ملف منفصل
pair_df = pd.DataFrame(
    [
        {
            "language_pair": pair,
            "average_similarity": score,
        }
        for pair, score in pair_results.items()
    ]
)


pair_df.to_csv(
    OUTPUT_DIR / "language_pair_averages.csv",
    index=False,
    encoding="utf-8-sig",
)


print("\nSaved:")
print(" - full_similarity_matrix.csv")
print(" - within_group_similarity.csv")
print(" - language_pair_averages.csv")