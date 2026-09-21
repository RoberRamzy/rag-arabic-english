import os
import pandas as pd
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import BaseRetriever
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from tqdm import tqdm

# --- 1. GPU Acceleration Configuration ---
print("Initializing BGE-M3 and BGE-Reranker on CUDA...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", device="cuda")

# Load the Cross-Encoder Reranker
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"
rerank_tokenizer = AutoTokenizer.from_pretrained(RERANKER_NAME)
rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_NAME).cuda().eval()

# --- 2. Custom Balanced Retriever Implementation ---
class BalancedRerankedRetriever(BaseRetriever):
    def __init__(self, arabic_corpus, english_corpus, similarity_top_k=10):
        super().__init__()
        # Isolate indices into native language partitions
        ar_docs = [Document(text=d["text"], doc_id=d["id"], metadata={"law_id": d["id"], "lang": d["lang"]}) for d in arabic_corpus]
        en_docs = [Document(text=d["text"], doc_id=d["id"], metadata={"law_id": d["id"], "lang": d["lang"]}) for d in english_corpus]
        
        # Grab K candidates from each side to feed into the cross-encoder pipeline
        self.ar_retriever = VectorStoreIndex.from_documents(ar_docs).as_retriever(similarity_top_k=similarity_top_k)
        self.en_retriever = VectorStoreIndex.from_documents(en_docs).as_retriever(similarity_top_k=similarity_top_k)
        self.final_top_k = similarity_top_k

    def _retrieve(self, query_bundle):
        query_str = query_bundle.query_str
        
        # Gather candidate representations from both spaces
        ar_candidates = self.ar_retriever.retrieve(query_str)
        en_candidates = self.en_retriever.retrieve(query_str)
        combined_nodes = ar_candidates + en_candidates
        
        if not combined_nodes:
            return []
            
        # Cross-Encoder Reranking Phase
        pairs = [[query_str, node.node.text] for node in combined_nodes]
        with torch.no_grad():
            inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to("cuda")
            scores = rerank_model(**inputs).logits.view(-1).cpu().numpy()
            
        # Sort nodes according to real cross-attention scores
        for node, score in zip(combined_nodes, scores):
            node.score = float(score)
            
        combined_nodes.sort(key=lambda x: x.score, reverse=True)
        return combined_nodes[:self.final_top_k]

# --- 3. Run Pipeline Evaluation ---
if __name__ == "__main__":
    from test import load_data_cross_lingual_align, evaluate_pipeline  # Re-uses your working text loader
    
    CORPUS_FILE = "corpora/legal_corpus.parquet"
    BENCHMARK_FILE = "benchmarks/legal_benchmark.parquet"
    
    corpus, queries = load_data_cross_lingual_align(CORPUS_FILE, BENCHMARK_FILE)
    
    # Split the clean corpus by language boundaries
    ar_corp = [d for d in corpus if d["lang"] == "ar"]
    en_corp = [d for d in corpus if d["lang"] == "en"]
    
    print("\nBuilding Balanced Language Store Index...")
    mitigated_retriever = BalancedRerankedRetriever(ar_corp, en_corp, similarity_top_k=10)
    
    print("\nRunning RAG evaluations with Cross-Encoder Mitigation...")
    df_results = evaluate_pipeline(mitigated_retriever, queries)
    
    matrix = df_results.groupby(["user_lang", "doc_lang"])[["hit@10", "mrr"]].mean() * 100
    print("\n=== THE FIXED (BALANCED + RERANKED) DISPARITY MATRIX ===")
    print(matrix.round(2))
