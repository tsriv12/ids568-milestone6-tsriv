# Milestone 6: RAG Pipeline & Agentic AI System
**MLOps Course — Module 7**  
**Author:** Tanya Srivastava  
**Model:** mistral:7b-instruct (7B) | **Vector Store:** ChromaDB | **Serving:** Ollama

---

## Architecture Overview

This project implements two tightly integrated components:

**Part 1 — RAG Pipeline**  
A complete retrieval-augmented generation system that chunks and indexes MLOps documents, retrieves relevant context using semantic search, and generates grounded answers using a local Mistral 7B model.

**Part 2 — Multi-Tool Agent Controller**  
A LLM-driven agent that intelligently selects between three tools (Retriever, Summarizer, KeywordExtractor) to solve multi-step tasks. Tool selection decisions are made by Mistral 7B at each step and logged in full detail for observability.

```
Documents → Chunker → Embedder → ChromaDB (index)
                                      ↕
User Query → Embedder → ChromaDB (search) → Top-k Chunks → Mistral 7B → Answer

Agent Task → Mistral 7B (tool selection) → [Retriever | Summarizer | KeywordExtractor]
           → Accumulate context → Mistral 7B (final answer generation)
```

---

## Model Deployment

| Property | Value |
|----------|-------|
| Model name | `mistral:7b-instruct` |
| Size class | 7B parameters |
| Serving stack | Ollama |
| Hardware | GCP VM, NVIDIA Tesla T4, 15GB VRAM |
| CUDA version | 12.2 |
| Avg retrieval latency | ~31ms |
| Avg generation latency | ~3039ms |
| Avg end-to-end latency | ~3069ms (RAG) / ~22434ms (Agent) |

---

## Setup Instructions

### Prerequisites
- Python 3.11+
- NVIDIA GPU with 8GB+ VRAM (tested on Tesla T4 15GB)
- CUDA 12.x

### Step 1: Clone the repository
```bash
git clone https://github.com/<your-username>/ids568-milestone6-tsriv.git
cd ids568-milestone6-tsriv
```

### Step 2: Create and activate virtual environment
```bash
python3 -m venv venv --without-pip
source venv/bin/activate
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
rm get-pip.py
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install and start Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Ollama starts automatically as a service. Verify it is running:
```bash
ollama list
```

### Step 5: Pull the required model
```bash
ollama pull mistral:7b-instruct
```

Verify the model works:
```bash
ollama run mistral:7b-instruct "In one sentence, what is RAG?"
```

### Step 6: Register the Jupyter kernel (for notebook)
```bash
python -m ipykernel install --user --name=milestone6 --display-name "Milestone 6"
```

---


> **Important:** Run `rag_pipeline.ipynb` first before running `agent_controller.py`.  
> The agent expects the local Chroma index in `./chroma_db` and the `mlops_rag` collection created by the RAG pipeline.

## Usage

### Part 1: Run the RAG Pipeline (Notebook)

Open `rag_pipeline.ipynb` in VS Code or JupyterLab, select the **Milestone 6** kernel, and run all cells. The notebook will:

1. Load and chunk 10 MLOps documents
2. Generate embeddings and build the ChromaDB index
3. Run 10 evaluation queries with precision/recall metrics
4. Print aggregate latency measurements
5. Save results to `eval_results.json`

```bash
# To run headlessly
jupyter nbconvert --to notebook --execute rag_pipeline.ipynb --output rag_pipeline_executed.ipynb
```

### Part 2: Run the Agent Controller

```bash
cd ids568-milestone6-tsriv
source venv/bin/activate
python agent_controller.py
```

The agent will:
1. Run 10 multi-step evaluation tasks
2. Print live tool selection decisions and reasoning at each step
3. Save individual traces to `agent_traces/task_XX.json`
4. Print a full evaluation summary
5. Save combined results to `agent_evaluation_summary.json`

**Expected runtime:** ~4-6 minutes for all 10 tasks on a T4 GPU.

**Sample output:**
```
TASK task_01: What is retrieval-augmented generation...
  Step 1:
    Tool selected: [retriever]
    Reasoning: Retrieving relevant information about RAG from the knowledge base.
    Result: Retrieved 3 chunks. Top: 'Retrieval-Augmented Generation (RAG)' (sim=0.891)
  Step 2:
    Tool selected: [summarizer]
    Reasoning: Summarizing the retrieved content to distill key points.
    Result: Summary (42 words): RAG enhances LLMs by grounding responses in retrieved context...
  FINAL ANSWER: Retrieval-Augmented Generation (RAG) is a technique...
  Trace saved: ./agent_traces/task_01.json
```

---

## Repository Structure

```
ids568-milestone6-tsriv/
├── rag_pipeline.ipynb          # Part 1: RAG pipeline implementation
├── agent_controller.py         # Part 2: Agent controller implementation
├── rag_evaluation_report.md    # Part 1: Evaluation metrics and analysis
├── rag_pipeline_diagram.md     # Part 1: Architecture diagram
├── agent_report.md             # Part 2: Agent analysis and failure cases
├── agent_traces/               # Part 2: 10 JSON trace files
│   ├── task_01.json
│   ├── task_02.json
│   └── ... (10 total)
├── requirements.txt            # Pinned dependencies
├── README.md                   # This file
└── chroma_db/                  # ChromaDB persistent index (auto-generated)
```

---


## Challenge Extensions

This repository also includes optional challenge extensions beyond the base Milestone 6 requirements:

- `rag_extensions.ipynb`: evaluates reranking, hybrid search, and query expansion for the RAG pipeline.
- `agent_extensions.py`: implements agent recovery mechanisms, confidence scoring, and parallel tool execution.
- `extension_results.json`: stores extension evaluation outputs and before/after comparison results.
- `agent_traces/extension_recovery_task.json`, `agent_traces/extension_confidence_task.json`, and `agent_traces/extension_parallel_task.json`: provide trace examples for the agent extensions.

## Known Limitations

1. **Generation latency:** Average ~3 seconds per query on a single T4 GPU. Not suitable for real-time interactive applications without streaming or vLLM upgrade.

2. **Corpus size:** The knowledge base contains 10 documents covering MLOps topics. Queries about topics outside this corpus (e.g., fine-tuning, reinforcement learning) will not retrieve relevant content. The model correctly abstains rather than hallucinating.

3. **Precision@3 (0.37):** Retrieval ranks 2 and 3 frequently return topically adjacent but not strictly relevant documents due to semantic overlap between MLOps concepts. This is expected for a small corpus with related topics.

4. **Agent tool coverage:** On 3 of 10 tasks, Mistral-7B did not invoke all expected tools (summarizer/keyword_extractor skipped). The model converges to retrieval-only when retrieved content is already sufficient for an answer. Final answer quality was not affected.

5. **JSON parsing:** Mistral-7B occasionally produces tool selection output with preamble text before the JSON block. The agent handles this with regex-based extraction and a rule-based fallback, but a larger model (14B) or structured output enforcement would improve reliability.

6. **Single GPU serving:** The current setup uses Ollama on a single T4. For production, vLLM with continuous batching on an A100 would provide significantly higher throughput.

---

## Reproducing Results

All results can be reproduced by following the Setup Instructions above and running:

```bash
# Part 1
jupyter nbconvert --to notebook --execute rag_pipeline.ipynb

# Part 2
python agent_controller.py
```

No proprietary API keys are required. All inference is local via Ollama + mistral:7b-instruct.
