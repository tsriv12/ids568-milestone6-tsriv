# RAG Pipeline Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG PIPELINE                                     │
│                                                                          │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│   │   INDEXING   │    │  RETRIEVAL   │    │      GENERATION          │  │
│   │   (offline)  │    │  (online)    │    │      (online)            │  │
│   └──────────────┘    └──────────────┘    └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Indexing (Offline)

```
 Raw Documents (10 MLOps topic docs)
         │
         ▼
┌─────────────────────────────────┐
│         DOCUMENT LOADER         │
│  - Reads plain text documents   │
│  - Assigns doc_id and title     │
│  - Passes to chunker            │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│    CHUNKER (paragraph-aware)    │
│                                 │
│  Strategy: Fixed-size + overlap │
│  Chunk size:  512 tokens        │
│  Overlap:      75 tokens        │
│  Split order: paragraph →       │
│               sentence →        │
│               word              │
│                                 │
│  Output: N chunks with          │
│          doc_id metadata        │
└─────────────────────────────────┘
         │
         │  List[str] chunks
         ▼
┌─────────────────────────────────┐
│     EMBEDDER                    │
│                                 │
│  Model: all-MiniLM-L6-v2        │
│  Dimensions: 384                │
│  Normalization: L2              │
│  Batch size: 32                 │
│                                 │
│  Input:  text chunk             │
│  Output: 384-dim float vector   │
└─────────────────────────────────┘
         │
         │  List[List[float]] embeddings
         ▼
┌─────────────────────────────────┐
│     VECTOR STORE (ChromaDB)     │
│                                 │
│  Collection: mlops_rag          │
│  Similarity: cosine             │
│  Storage: persistent (./chroma) │
│  Index type: HNSW (via Chroma)  │
│                                 │
│  Stored per chunk:              │
│    - chunk_id (primary key)     │
│    - embedding vector           │
│    - raw text                   │
│    - metadata (doc_id, title,   │
│      chunk_index, word_count)   │
└─────────────────────────────────┘
```

---

## Phase 2: Querying (Online)

```
  User Query (natural language string)
         │
         ▼
┌─────────────────────────────────┐
│     QUERY EMBEDDER              │
│                                 │
│  Same model as indexing:        │
│  all-MiniLM-L6-v2               │
│  L2-normalized output           │
│                                 │
│  Latency: ~5-15ms               │
└─────────────────────────────────┘
         │
         │  384-dim query vector
         ▼
┌─────────────────────────────────┐
│     RETRIEVER                   │
│                                 │
│  ChromaDB collection.query()    │
│  Similarity: cosine             │
│  Top-k: 3 chunks                │
│                                 │
│  Returns per chunk:             │
│    - rank (1, 2, 3)             │
│    - raw text content           │
│    - doc_id + doc_title         │
│    - similarity score           │
│                                 │
│  Total retrieval latency: ~31ms │
└─────────────────────────────────┘
         │
         │  Top-3 chunks + metadata
         ▼
┌─────────────────────────────────┐
│     PROMPT BUILDER              │
│                                 │
│  Injects retrieved chunks into  │
│  a grounding prompt template:   │
│                                 │
│  "Answer using ONLY the         │
│   provided context. If the      │
│   context is insufficient,      │
│   say so explicitly."           │
│                                 │
│  Context format:                │
│  [Source: <doc_title>]          │
│  <chunk_text>                   │
│  --- (separator)                │
│  [Source: <doc_title>]          │
│  <chunk_text>                   │
└─────────────────────────────────┘
         │
         │  Grounded prompt string
         ▼
┌─────────────────────────────────┐
│     GENERATOR (Mistral 7B)      │
│                                 │
│  Model: mistral:7b-instruct     │
│  Serving: Ollama                │
│  Hardware: Tesla T4 (15GB)      │
│  Temperature: 0.1               │
│  Max tokens: 300                │
│                                 │
│  Generation latency: ~3039ms    │
└─────────────────────────────────┘
         │
         │  Generated answer string
         ▼
  Final Grounded Answer → User
```

---

## Decision Points and Data Transformations

```
┌──────────────────────────────────────────────────────┐
│              DECISION POINTS                         │
│                                                      │
│  DP1: Chunk boundary detection                       │
│       IF paragraph fits in chunk_size → keep whole  │
│       ELSE → split at sentence boundary             │
│                                                      │
│  DP2: Overlap application                           │
│       IF current_chunk > overlap → carry last 75    │
│       ELSE → carry all of current_chunk             │
│                                                      │
│  DP3: Out-of-scope detection (generation-side)      │
│       IF similarity scores < threshold for all k    │
│          → model prompted to abstain                │
│       ELSE → proceed with generation               │
│                                                      │
│  DP4: Answer grounding enforcement                  │
│       System prompt explicitly restricts model to   │
│       retrieved context only                        │
└──────────────────────────────────────────────────────┘
```

---

## Data Flow Summary

```
Raw Text
   → [Chunker] → Text Chunks
   → [Embedder] → Float Vectors (384-dim)
   → [ChromaDB] → Indexed Vector Store

Query String
   → [Embedder] → Query Vector (384-dim)
   → [ChromaDB] → Top-3 Chunks + Similarity Scores
   → [Prompt Builder] → Grounded Prompt
   → [Mistral 7B / Ollama] → Answer String
```

---

## Key Components Summary

| Component | Implementation | Config |
|-----------|---------------|--------|
| Chunker | Custom paragraph-aware splitter | 512 tokens, 75 overlap |
| Embedder | sentence-transformers all-MiniLM-L6-v2 | 384-dim, L2-normalized |
| Vector Store | ChromaDB PersistentClient | cosine similarity, HNSW |
| Retriever | ChromaDB collection.query() | top-k=3 |
| Generator | mistral:7b-instruct via Ollama | temp=0.1, max_tokens=300 |
| Hardware | GCP Tesla T4 15GB VRAM | CUDA 12.2 |

---

## Latency Profile

```
┌─────────────────────────────────────────────────────┐
│  Query received                              t=0ms  │
│  ├── Query embedding                       +15ms   │
│  ├── ChromaDB vector search                +16ms   │
│  │   Total retrieval:                    ~31ms     │
│  │                                                 │
│  ├── Prompt construction                    +1ms   │
│  ├── Mistral 7B generation              +3039ms   │
│  │                                                 │
│  Answer returned                        ~3069ms   │
└─────────────────────────────────────────────────────┘
  Retrieval: 1% of total latency
  Generation: 99% of total latency
  Bottleneck: LLM inference on single T4 GPU
```
