import os
import pandas as pd
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm

# Set your embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

def load_corpus_from_parquet(corpus_path):
    """
    Loads the document corpus from the parquet file.
    Each row is treated as a unique document containing metadata about its language.
    """
    df = pd.read_parquet(corpus_path)
    corpus_data = []
    
    for idx, row in df.iterrows():
        # Use 'law_id' or index to form a unique tracking ID
        doc_id = f"doc_{row['law_id']}_{row['language']}" if 'law_id' in df.columns else f"doc_{idx}_{row['language']}"
        
        corpus_data.append({
            "id": doc_id,
            "text": row["content"],
            "lang": row["language"],  # 'en' or 'ar'
            "law_id": row.get("law_id", idx)
        })
        
    return corpus_data

def load_queries_from_parquet(queries_path):
    """
    Loads evaluation queries from the benchmark file.
    """
    if not os.path.exists(queries_path):
        print(f"⚠️ Queries file not found at {queries_path}. Generation/Evaluation will be mock-simulated.")
        return []
        
    df = pd.read_parquet(queries_path)
    print(f"\n--- Detected Columns in Queries File ({queries_path}) ---")
    print(list(df.columns))
    
    queries_data = []
    # Adjust column strings here depending on the benchmark file's exact schema
    # Commonly: 'query', 'query_language', 'ground_truth_law_id'
    for idx, row in df.iterrows():
        queries_data.append({
            "query": row.get("query") or row.get("text"),
            "query_lang": row.get("language") or row.get("query_lang"),
            "expected_law_id": row.get("law_id") or row.get("ground_truth_id")
        })
    return queries_data

def build_bilingual_index(corpus):
    documents = [
        Document(text=doc["text"], doc_id=doc["id"], extra_info={"lang": doc["lang"], "law_id": doc["law_id"]})
        for doc in corpus
    ]
    index = VectorStoreIndex.from_documents(documents)
    return index.as_retriever(similarity_top_k=10)

def evaluate_pipeline(retriever, queries):
    results = []
    for q in tqdm(queries, desc="Evaluating RAG cross-lingual pathways"):
        retrieved_nodes = retriever.retrieve(q["query"])
        
        expected_law_id = q["expected_law_id"]
        
        hit = 0
        mrr = 0.0
        doc_lang = "unknown"
        
        for rank, node in enumerate(retrieved_nodes, start=1):
            # Check match against original law piece reference id
            if str(node.node.metadata.get("law_id")) == str(expected_law_id):
                hit = 1
                mrr = 1.0 / rank
                doc_lang = node.node.metadata.get("lang", "unknown")
                break
                
        results.append({
            "user_lang": q["query_lang"],
            "doc_lang": doc_lang,
            "hit@10": hit,
            "mrr": mrr
        })
    return pd.DataFrame(results)

# --- Run Pipeline ---
if __name__ == "__main__":
    CORPUS_FILE = "corpora/legal_corpus.parquet"
    # Look for a companion queries or benchmark file in your folder structure
    QUERIES_FILE = "corpora/legal_queries.parquet" 
    
    print(f"Loading corpus data from: {CORPUS_FILE}...")
    corpus = load_corpus_from_parquet(CORPUS_FILE)
    print(f"Loaded {len(corpus)} documents into corpus pipeline.")
    
    print(f"Loading evaluation queries...")
    queries = load_queries_from_parquet(QUERIES_FILE)
    
    if not queries:
        print("\nSkipping retrieval step. To execute the full 2x2 matrix analysis, verify if there is a companion queries file (e.g. legal_queries.parquet) in your download directory.")
    else:
        print(f"Indexing documents into Vector Store...")
        retriever = build_bilingual_index(corpus)
        
        print("Running cross-lingual bias evaluation...")
        df_results = evaluate_pipeline(retriever, queries)
        
        # Aggregate to build the Paper's 2x2 Retrieval Performance Matrix
        matrix = df_results.groupby(["user_lang", "doc_lang"])[["hit@10", "mrr"]].mean() * 100
        print("\n=== CROSS-LINGUAL DISPARITY MATRIX ===")
        print(matrix.round(2))
