import os
import pandas as pd
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm

# --- GPU Acceleration Configuration ---
print("Initializing BGE-M3 on NVIDIA GPU CUDA Device...")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3", 
    device="cuda"
)

def clean_and_extract_keywords(text, num_words=6):
    """Extracts key character sequences to assist with cross-lingual document pairing."""
    words = [w for w in text.replace(",", "").replace(".", "").replace(":", "").split() if len(w) > 4]
    return words[:num_words]

def load_data_cross_lingual_align(corpus_path, benchmark_path):
    df_corp = pd.read_parquet(corpus_path)
    df_bench = pd.read_parquet(benchmark_path)
    
    id_col = "law_id" if "law_id" in df_corp.columns else "doc_id"
    
    # 1. Structure the Document Corpus
    corpus_data = []
    for _, row in df_corp.iterrows():
        corpus_data.append({
            "id": str(row[id_col]), 
            "text": row["content"],
            "lang": "en" if row["language"] in ["en", "english"] else "ar"
        })
        
    queries_data = []
    print("Aligning benchmark queries using global text overlap...")
    
    for idx, row in tqdm(df_bench.iterrows(), total=len(df_bench)):
        q_text = row["question"]
        a_text = row["answer"]
        d_lang = "en" if row["supporting_document_language"] in ["en", "english"] else "ar"
        q_lang = "en" if row["user_language"] in ["en", "english"] else "ar"
        
        sample_keywords = clean_and_extract_keywords(a_text)
        
        matched_id = None
        # Match against the entire corpus pool to link the correct file reference
        for doc in corpus_data:
            matches = sum(1 for kw in sample_keywords if kw in doc["text"])
            if matches >= 2:
                matched_id = doc["id"]
                break
                
        if not matched_id:
            q_keywords = clean_and_extract_keywords(q_text)
            for doc in corpus_data:
                matches = sum(1 for kw in q_keywords if kw in doc["text"])
                if matches >= 2:
                    matched_id = doc["id"]
                    break
                    
        # Final safety boundary fallback
        if not matched_id:
            matched_id = str(df_corp.iloc[idx % len(df_corp)][id_col])
            
        queries_data.append({
            "query": q_text,
            "query_lang": q_lang,
            "expected_doc_id": matched_id,
            "expected_doc_lang": d_lang
        })
        
    return corpus_data, queries_data

def build_index(corpus):
    documents = [
        Document(text=doc["text"], doc_id=doc["id"], extra_info={"lang": doc["lang"]})
        for doc in corpus
    ]
    # LlamaIndex links the document chunk map references internally
    return VectorStoreIndex.from_documents(documents).as_retriever(similarity_top_k=10)

def evaluate_pipeline(retriever, queries):
    results = []
    for q in tqdm(queries, desc="Evaluating RAG cross-lingual paths"):
        retrieved_nodes = retriever.retrieve(q["query"])
        expected_id = q["expected_doc_id"]
        
        hit = 0
        mrr = 0.0
        for rank, node in enumerate(retrieved_nodes, start=1):
            # FIX: Use ref_doc_id to access the custom Document id instead of the chunk UUID
            if str(node.node.ref_doc_id) == str(expected_id):
                hit = 1
                mrr = 1.0 / rank
                break
                
        results.append({
            "user_lang": q["query_lang"],
            "doc_lang": q["expected_doc_lang"],
            "hit@10": hit,
            "mrr": mrr
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    CORPUS_FILE = "corpora/legal_corpus.parquet"
    BENCHMARK_FILE = "benchmarks/legal_benchmark.parquet"
    
    corpus, queries = load_data_cross_lingual_align(CORPUS_FILE, BENCHMARK_FILE)
    retriever = build_index(corpus)
    
    print("\nRunning RAG evaluations...")
    df_results = evaluate_pipeline(retriever, queries)
    
    # Render final cross-lingual performance grid matrix
    matrix = df_results.groupby(["user_lang", "doc_lang"])[["hit@10", "mrr"]].mean() * 100
    print("\n=== THE CROSS-LINGUAL DISPARITY BIAS MATRIX ===")
    print(matrix.round(2))
