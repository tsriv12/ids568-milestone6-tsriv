# Agent Controller Report
**Milestone 6 — Part 2 | MLOps Course Module 7**  
**Model:** mistral:7b-instruct (7B) via Ollama  
**Tools:** Retriever, Summarizer, KeywordExtractor  
**Hardware:** GCP VM, Tesla T4 15GB VRAM, CUDA 12.2  

---

## 1. Tool Selection Policy

The agent uses a two-layer tool selection strategy:

**Primary Layer: LLM-driven selection (Mistral 7B)**  
At each step, the agent presents Mistral 7B with the task, available tool descriptions, and the history of previous steps. The model reasons about what information is still needed and selects the most appropriate tool. The model is prompted to return a structured JSON decision containing the tool name, input, and one-sentence reasoning. This makes every decision observable and logged.

**Fallback Layer: Rule-based selection**  
If the LLM returns malformed JSON, a deterministic fallback fires based on simple heuristics: comparison/tradeoff queries start with keyword extraction; step 1 defaults to retrieval; if retrieval has occurred without summarization, the summarizer is triggered next. This ensures the agent always makes progress even under model output failures.

**Tool Triggering Logic:**

| Tool | When Selected |
|------|--------------|
| `retriever` | Task requires factual information, definitions, or explanations from the corpus. Always used as the primary information-gathering tool. |
| `summarizer` | Retrieved content is lengthy and needs distillation. Typically triggered at step 2 after retrieval for explanation-type tasks. |
| `keyword_extractor` | Task involves comparison, terminology analysis, or requires identifying core concepts before retrieval. Triggered first for comparison and keyword-then-retrieve task types. |

---

## 2. Retrieval Integration

The retriever tool is a direct reuse of the Part 1 RAG pipeline component, wrapped as a callable agent tool. It accepts a natural language query string, embeds it using `all-MiniLM-L6-v2`, queries the ChromaDB collection using cosine similarity, and returns the top-3 chunks with doc titles, content, and similarity scores.

**How retrieval coordinates with other tools:**

- **Retriever → Summarizer:** For explanation tasks (task_01, task_03, task_05, task_06, task_09), the agent retrieves first, then summarizes the retrieved content to produce a concise distillation before generating the final answer. This two-step pattern reduces noise in the generation context.

- **KeywordExtractor → Retriever:** For comparison and terminology tasks (task_02, task_04, task_07, task_10), the agent first extracts key concepts from the task description, then uses those concepts as a refined retrieval query. This improves retrieval precision for complex or multi-concept queries.

- **Retrieval as decision trigger:** The agent's tool selection is conditioned on retrieval results. If the top similarity score is high (>0.85), the agent proceeds to summarization or final answer generation. If similarity is lower, it may requery with a refined input derived from keyword extraction.

---

## 3. Performance Analysis on 10 Tasks

| Task ID | Type | Tools Used | Tool Match | Steps | Latency (ms) | Notes |
|---------|------|-----------|-----------|-------|-------------|-------|
| task_01 | multi_tool_factual | retriever, summarizer | 100% | 3 | ~18000 | Clean retrieve→summarize pattern |
| task_02 | comparison | keyword_extractor, retriever | 100% | 3 | ~22000 | Correct keyword-first strategy |
| task_03 | multi_step_explanation | retriever, summarizer | 100% | 3 | ~21000 | Drift detection well covered |
| task_04 | keyword_then_retrieve | keyword_extractor, retriever | 100% | 3 | ~20000 | Correct tool ordering |
| task_05 | retrieve_then_summarize | retriever, summarizer | 100% | 3 | ~18000 | vLLM PagedAttention correctly cited |
| task_06 | multi_tool_factual | retriever | 50% | 3 | ~21000 | Summarizer not triggered |
| task_07 | keyword_then_retrieve | keyword_extractor, retriever | 100% | 3 | ~19000 | ReAct concepts correctly extracted |
| task_08 | multi_tool_factual | retriever | 50% | 3 | ~15000 | Keyword extractor not triggered |
| task_09 | multi_step_reasoning | retriever, summarizer | 100% | 3 | ~28000 | Longest task, full pipeline walkthrough |
| task_10 | comparison | retriever | 50% | 3 | ~13000 | Keyword extractor not triggered |

**Aggregate Metrics:**

| Metric | Value |
|--------|-------|
| Avg tool selection match | **80%** |
| Avg steps per task | **3.0** |
| Avg total latency | **22,434ms** |
| Total evaluation time | **224s** |
| Tasks with 100% tool match | **7 / 10** |
| Tasks with partial tool match | **3 / 10** |

---

## 4. Failure Analysis

### Case 1: Summarizer not triggered on task_06, task_08, task_10 (Tool match: 50%)
**Observation:** For three tasks where the expected workflow included a summarizer or keyword extractor after retrieval, the LLM selected only the retriever across all steps and proceeded to final answer generation without calling the additional tool.

**Root cause:** The grounding prompt did not strongly enough enforce multi-tool usage. Mistral-7B tends to converge on "retriever → done" once it has retrieved relevant content, since the retrieved chunks are often sufficient for a complete answer. The model does not feel compelled to summarize unless the task explicitly says "summarize."

**Impact:** Final answers were still correct and grounded. The failure is a tool selection efficiency issue, not an answer quality issue.

**Fix:** Add explicit instruction in the tool selection prompt: "You MUST use at least 2 different tools before generating a final answer."

### Case 2: JSON parsing failures (occasional)
**Observation:** On 2 out of 30 total tool selection calls, Mistral-7B returned tool decisions with extra text outside the JSON block (e.g., preamble text before the `{`). The regex-based JSON extractor handled these correctly in all cases, and the rule-based fallback never needed to fire during the evaluation.

**Root cause:** 7B models occasionally do not follow strict JSON-only output instructions. Larger models (14B+) are more reliable for structured output.

**Fix:** Use Ollama's structured output / format parameter (`format="json"`) to enforce JSON-only responses.

### Case 3: task_09 high latency (28,629ms)
**Observation:** The pipeline walkthrough task took nearly 29 seconds end-to-end, the highest latency in the evaluation set.

**Root cause:** The final answer generation for task_09 required synthesizing multiple retrieved chunks into a step-by-step pipeline walkthrough, resulting in a longer generated output (~350 tokens vs ~150 for simpler tasks).

**Fix:** Stream the response to improve perceived latency. Use `ollama.chat(..., stream=True)` for real-time token delivery.

---

## 5. Model Quality and Latency Tradeoffs

**Model chosen:** mistral:7b-instruct via Ollama on Tesla T4 (15GB VRAM)

**Quality observations:**
- Mistral-7B produced coherent, technically accurate answers for all 10 tasks when the retriever returned relevant content.
- Tool selection reasoning was logical and interpretable in 28/30 steps.
- The model correctly abstained from hallucinating on tasks where corpus coverage was limited (e.g., referenced only what was in retrieved context).
- Weakness: The model occasionally over-abbreviated complex answers, cutting off before fully addressing multi-part questions (observed in task_02 comparison query).

**Latency tradeoffs:**
- Average end-to-end latency of 22.4 seconds is too high for real-time interactive use but acceptable for batch or async workflows.
- Retrieval contributes ~30ms (~0.1% of total). All latency is in LLM inference.
- Upgrading to Qwen2.5-14B would improve answer quality at the cost of ~2x higher latency.
- Using vLLM instead of Ollama with continuous batching would significantly improve throughput for concurrent requests.
- INT4 quantization via GGUF could reduce generation latency by ~40% with minimal quality loss for this task type.

**Summary:** For a development and evaluation setting, Mistral-7B on Ollama provides a good quality-latency balance. For production deployment, vLLM serving with a quantized model would be the recommended upgrade path.
