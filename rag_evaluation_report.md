# RAG Pipeline Evaluation Report
**Milestone 6 — Part 1 | MLOps Course Module 7**  
**Model:** mistral:7b-instruct (7B) via Ollama  
**Vector Store:** ChromaDB (cosine similarity)  
**Embedding Model:** all-MiniLM-L6-v2 (384-dim)  
**Hardware:** GCP VM, Tesla T4 15GB VRAM, CUDA 12.2  

---

## 1. Retrieval Accuracy on 10 Handcrafted Queries

Queries were designed to cover five distinct retrieval scenarios: direct factual lookup, multi-document span, out-of-scope detection, ambiguous phrasing, and multi-step reasoning. Ground truth relevance labels were manually assigned per query based on document content.

| Query ID | Type | Query (abbreviated) | Expected Doc(s) | Retrieved Top-1 | P@1 | P@3 | R@3 |
|----------|------|---------------------|-----------------|-----------------|-----|-----|-----|
| Q01 | direct_factual | What is RAG and how does it reduce hallucinations? | doc_01 | doc_01 ✓ | 1.00 | 0.33 | 1.00 |
| Q02 | direct_factual | Differences between FAISS and ChromaDB? | doc_02 | doc_02 ✓ | 1.00 | 0.33 | 1.00 |
| Q03 | direct_factual | How does data drift detection work? | doc_04 | doc_04 ✓ | 1.00 | 0.33 | 1.00 |
| Q04 | direct_factual | Chunking strategies and tradeoffs? | doc_07 | doc_07 ✓ | 1.00 | 0.33 | 1.00 |
| Q05 | direct_factual | What is vLLM and how does it improve throughput? | doc_05 | doc_05 ✓ | 1.00 | 0.33 | 1.00 |
| Q06 | multi_doc | What metrics evaluate a RAG pipeline? | doc_10, doc_01 | doc_10 ✓ | 1.00 | 0.67 | 1.00 |
| Q07 | direct_factual | How do feature stores prevent training-serving skew? | doc_06 | doc_06 ✓ | 1.00 | 0.33 | 1.00 |
| Q08 | out_of_scope | Best way to fine-tune a GPT model? | (none) | — | 0.00 | 0.00 | 1.00 |
| Q09 | ambiguous | What embedding model should I use? | doc_09 | doc_09 ✓ | 1.00 | 0.33 | 1.00 |
| Q10 | multi_step | How do agentic AI systems select tools? | doc_08 | doc_08 ✓ | 1.00 | 0.33 | 1.00 |

### Aggregate Metrics (out-of-scope Q08 excluded from precision/recall averages)

| Metric | Value |
|--------|-------|
| Mean Precision@1 | **1.000** |
| Mean Precision@3 | **0.370** |
| Mean Recall@3 | **1.000** |

**Interpretation:**  
- Perfect Precision@1 (1.000) indicates the top-ranked document is always the correct one for in-scope queries. The retriever reliably places the most relevant chunk at rank 1.  
- Precision@3 (0.370) is lower because ranks 2 and 3 frequently contain related but not strictly relevant documents (e.g., Q01 retrieves doc_10 and doc_07 alongside doc_01 — both are topically adjacent to RAG but not the primary source). This is expected behavior given a 10-document corpus with overlapping concepts.  
- Recall@3 (1.000) confirms that for all single-document queries, the relevant document always appears within the top 3 results.

---

## 2. Qualitative Grounding Analysis

### Well-Grounded Responses
**Q01 (RAG definition):** The model accurately described the two-phase RAG pipeline (indexing and querying), cited vector databases and similarity search, and correctly attributed hallucination reduction to context injection. No unsupported claims were identified.

**Q05 (vLLM throughput):** The model correctly cited PagedAttention and the 24x throughput improvement figure directly from the retrieved chunk. The response stayed tightly scoped to the provided context.

**Q07 (feature store skew):** The model correctly explained the offline/online store distinction and the training-serving skew problem. All claims were traceable to doc_06.

### Partially Grounded Responses
**Q06 (RAG evaluation metrics):** The model retrieved both doc_10 and doc_01. The response correctly listed Precision@k, Recall@k, faithfulness, and latency metrics. However, it briefly mentioned "human evaluation" which is supported by doc_10 but was elaborated slightly beyond what the context stated — a minor grounding drift, not a hallucination.

**Q09 (embedding model recommendation):** The query was intentionally ambiguous ("What embedding model should I use?"). The model correctly retrieved doc_09 and listed the sentence-transformers options. However, it added a soft recommendation for all-MiniLM-L6-v2 as a starting point, which goes slightly beyond the retrieved context. This represents a low-severity grounding drift.

### Hallucination Cases
**Q08 (out-of-scope: fine-tuning GPT):** The corpus contains no fine-tuning documentation. The retrieved chunks were about LLM serving infrastructure (doc_05) and embedding models (doc_09) — topically adjacent but not relevant. The model correctly identified the context was insufficient and stated it could not answer from the provided context. **No hallucination occurred** — the grounding prompt successfully triggered an abstention response.

### Retrieval Failure vs. Generation Failure Attribution
- **Retrieval failures:** Q08 is the only retrieval failure — no relevant document exists in the corpus. This is a corpus coverage gap, not a retriever error.
- **Generation/grounding failures:** Q06 and Q09 showed minor grounding drift (elaboration beyond context). These are generation-side issues, not retrieval failures, since the correct documents were retrieved.
- **No cases** were found where the correct document was retrieved but the model generated a factually incorrect answer.

---

## 3. Latency Measurements

All latency measurements were recorded across 10 evaluation runs on GCP Tesla T4, CUDA 12.2, Ollama serving mistral:7b-instruct.

| Stage | Mean Latency | Notes |
|-------|-------------|-------|
| Retrieval (vector search + embedding) | **30.6ms** | Includes query embedding + ChromaDB ANN search |
| Generation (Mistral 7B inference) | **3038.6ms** | Ollama, temperature=0.1, max 300 tokens |
| End-to-End | **3069.2ms** | Retrieval + generation combined |
| P95 End-to-End | **5358.8ms** | Driven by longer generation outputs |

**Analysis:**  
- Retrieval is extremely fast at 30.6ms (~1% of total latency), confirming that ChromaDB with pre-computed embeddings introduces negligible overhead.  
- Generation dominates latency at ~99% of end-to-end time. This is typical for 7B parameter models on a single T4 GPU.  
- The P95 latency of 5.3s suggests generation variance is the primary source of tail latency. Queries requiring longer, more detailed answers (Q06, Q10) took longer.  
- **Bottleneck:** LLM inference. Optimization options include quantization (INT4 GGUF), streaming responses, or upgrading to vLLM for batched inference.

---

## 4. Chunking and Indexing Design Decisions

### Chunk Size: 512 tokens with 75-token overlap
**Rationale:** A 512-token chunk size was chosen as a balance between context richness and retrieval precision. Smaller chunks (128-256 tokens) were tested in development and yielded higher precision but caused the model to lack sufficient context for complete answers. Larger chunks (1024 tokens) reduced precision because a single chunk contained multiple distinct concepts, causing noisy retrieval.

The 75-token overlap (~15%) was chosen to prevent important context from being severed at chunk boundaries. This is particularly important for our MLOps documents where a concept introduced in one paragraph is often elaborated in the next.

### Chunking Method: Paragraph-aware recursive splitting
Documents are split first at double-newline paragraph boundaries, preserving semantic coherence. If a paragraph exceeds the chunk size limit, it falls back to sentence-boundary splitting. This approach was preferred over naive character-count splitting because it respects the natural structure of the source documents.

### Embedding Model: all-MiniLM-L6-v2
Selected for strong MTEB retrieval benchmark performance at its size class, 384-dimensional output (fast and storage-efficient), and fully open-source availability with no API dependency. The alternative all-mpnet-base-v2 (768-dim) was considered but rejected as the quality improvement was marginal for this corpus size.

### Index Type: ChromaDB with cosine similarity
ChromaDB was selected over FAISS for persistent storage, built-in metadata filtering, and simpler API. Cosine similarity was used because embeddings are L2-normalized, making cosine distance equivalent to dot-product similarity and appropriate for semantic comparison.

---

## 5. Summary and Limitations

**Strengths:**  
- Perfect top-1 retrieval precision across all in-scope queries demonstrates robust semantic alignment between query embeddings and document chunks.  
- The grounding prompt effectively prevents hallucination for out-of-scope queries.  
- Retrieval latency (30.6ms) is suitable for production use.  

**Limitations:**  
- Corpus size (10 documents) is small; precision@3 would likely improve with a larger, more diverse corpus.  
- Generation latency (~3s) on a single T4 GPU may not meet real-time requirements for interactive applications. vLLM or quantization would help.  
- Minor grounding drift observed on ambiguous queries (Q09) suggests the model occasionally draws on parametric knowledge beyond the retrieved context. A stricter system prompt or faithfulness post-processing could mitigate this.  
- Model capacity limitations: Mistral-7B occasionally produces incomplete answers for complex multi-hop queries that require synthesizing across multiple retrieved chunks simultaneously.
