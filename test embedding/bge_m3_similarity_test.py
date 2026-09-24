"""
Cross-lingual Semantic Similarity Test using BAAI/bge-m3
-----------------------------------------------------------
يقرأ ملف sentences.json (نفس الجملة بـ 3 نسخ: عربي / إنجليزي / مخلوط)
يعمل embeddings باستخدام BAAI/bge-m3
يحسب cosine similarity بين كل جملة وكل جملة تانية
يطبع جدول مقارنة لكل مجموعة + يحفظ CSV بالنتائج (من غير رسم heatmap)

Requirements:
    pip install -U sentence-transformers numpy pandas --break-system-packages
    (أو FlagEmbedding لو حابب تستخدم مكتبة BAAI الرسمية)
"""

import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------- 1) تحميل الداتا ----------
DATA_PATH = "sentences.json"   # غيّر المسار لو مختلف عندك

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)["sentences"]

LANGS = ["ar", "en", "mixed"]

texts = []
labels = []          # مثال: "1_ar", "1_en", "1_mixed"
topics = []           # عشان نعرف كل جملة تبع أي مجموعة

for item in raw:
    for lang in LANGS:
        texts.append(item[lang])
        labels.append(f"{item['id']}_{lang}")
        topics.append(item.get("topic", str(item["id"])))

print(f"Loaded {len(texts)} sentences from {len(raw)} groups.\n")

# ---------- 2) تحميل الموديل وعمل الـ embeddings ----------
print("Loading BAAI/bge-m3 ... ")
model = SentenceTransformer("BAAI/bge-m3")

# bge-m3 مبيحتاجش instruction prefix زي bge الأقدم (bge-large مثلا)
# لكن لو حبيت تحسن نتائج الـ retrieval ممكن تضيف "query: " / "passage: "
# هنا بما إننا بنقارن جمل ببعضها (symmetric similarity) هنسيبها من غير prefix
embeddings = model.encode(
    texts,
    normalize_embeddings=True,   # مهم عشان cosine similarity = dot product مباشرة
    show_progress_bar=True,
)

embeddings = np.array(embeddings)

# ---------- 3) حساب مصفوفة الـ cosine similarity ----------
# بما إن الـ embeddings متعمولها normalize فالـ dot product = cosine similarity
sim_matrix = embeddings @ embeddings.T

sim_df = pd.DataFrame(sim_matrix, index=labels, columns=labels)

# ---------- 4) طباعة جدول مقارنة داخل كل مجموعة (within-group) ----------
print("\n" + "=" * 60)
print("Within-group similarity (the same sentence with diffrent language")
print("=" * 60)

within_rows = []
for item in raw:
    gid = item["id"]
    topic = item.get("topic", str(gid))
    ar_label, en_label, mixed_label = f"{gid}_ar", f"{gid}_en", f"{gid}_mixed"

    ar_en = sim_df.loc[ar_label, en_label]
    ar_mixed = sim_df.loc[ar_label, mixed_label]
    en_mixed = sim_df.loc[en_label, mixed_label]

    within_rows.append({
        "id": gid,
        "topic": topic,
        "ar_vs_en": round(ar_en, 4),
        "ar_vs_mixed": round(ar_mixed, 4),
        "en_vs_mixed": round(en_mixed, 4),
        "avg": round(np.mean([ar_en, ar_mixed, en_mixed]), 4),
    })

within_df = pd.DataFrame(within_rows)
print(within_df.to_string(index=False))

avg_overall = within_df["avg"].mean()
print(f"\nMetric average across all groups: {avg_overall:.4f}")

# ---------- 5) طباعة أقرب/أبعد جملة لكل جملة (اختياري بس مفيد) ----------
print("\n" + "=" * 60)
print("Most similar sentence to each sentence (excluding itself)")
print("=" * 60)
for label in labels:
    row = sim_df.loc[label].drop(label)
    best_match = row.idxmax()
    print(f"{label:12s} -> most similar: {best_match:12s} (sim = {row[best_match]:.4f})")

# ---------- 6) حفظ النتائج ----------
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sim_df.to_csv(os.path.join(OUTPUT_DIR, "full_similarity_matrix.csv"), encoding="utf-8-sig")
within_df.to_csv(os.path.join(OUTPUT_DIR, "within_group_similarity.csv"), index=False, encoding="utf-8-sig")

print("\nSaved:")
print(" - full_similarity_matrix.csv")
print(" - within_group_similarity.csv")