"""
Milestone 6 — Part 2 Agent Extensions
Implements 3 optional challenge extensions:
1. Recovery Mechanisms — agent reruns retrieval with modified query on failure
2. Confidence Scoring — flags low-confidence tool selections and retrievals
3. Parallel Tool Execution — runs independent tools concurrently
"""

import json
import time
import os
import re
import concurrent.futures
from typing import List, Dict, Any, Tuple
from datetime import datetime

import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# ── Setup (same as agent_controller.py) ──
CHROMA_DIR = "./chroma_db"
TRACES_DIR = "./agent_traces"
os.makedirs(TRACES_DIR, exist_ok=True)

print("Loading models...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection("mlops_rag")
print(f"Ready. Index: {collection.count()} chunks")


# ════════════════════════════════════════════════════
# BASE TOOLS (same as agent_controller.py)
# ════════════════════════════════════════════════════

def tool_retriever(query: str, k: int = 3) -> Dict:
    t0 = time.time()
    qe = embedding_model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=qe, n_results=k,
                               include=["documents", "metadatas", "distances"])
    latency_ms = (time.time() - t0) * 1000
    chunks = [{
        "rank": i+1,
        "doc_id": results["metadatas"][0][i]["doc_id"],
        "doc_title": results["metadatas"][0][i]["doc_title"],
        "content": results["documents"][0][i],
        "similarity": round(1 - results["distances"][0][i], 4)
    } for i in range(len(results["documents"][0]))]
    return {"tool": "retriever", "query": query, "chunks": chunks,
            "top_similarity": chunks[0]["similarity"] if chunks else 0.0,
            "latency_ms": round(latency_ms, 1)}


def tool_summarizer(text: str, max_sentences: int = 3) -> Dict:
    prompt = f"Summarize in {max_sentences} sentences. Be concise and preserve key facts.\n\nTEXT:\n{text}\n\nSUMMARY:"
    t0 = time.time()
    response = ollama.chat(model="mistral:7b-instruct",
                           messages=[{"role": "user", "content": prompt}],
                           options={"temperature": 0.1, "num_predict": 150})
    return {"tool": "summarizer", "summary": response["message"]["content"].strip(),
            "latency_ms": round((time.time()-t0)*1000, 1)}


def tool_keyword_extractor(text: str, max_keywords: int = 8) -> Dict:
    prompt = f"Extract {max_keywords} key technical keywords as a comma-separated list. No explanations.\n\nTEXT:\n{text}\n\nKEYWORDS:"
    t0 = time.time()
    response = ollama.chat(model="mistral:7b-instruct",
                           messages=[{"role": "user", "content": prompt}],
                           options={"temperature": 0.1, "num_predict": 80})
    raw = response["message"]["content"].strip()
    keywords = [k.strip() for k in raw.split(",") if k.strip()][:max_keywords]
    return {"tool": "keyword_extractor", "keywords": keywords,
            "latency_ms": round((time.time()-t0)*1000, 1)}


# ════════════════════════════════════════════════════
# EXTENSION 1: RECOVERY MECHANISMS
# Agent detects low-quality retrieval and reruns with
# a reformulated query before proceeding
# ════════════════════════════════════════════════════

SIMILARITY_THRESHOLD = 0.60  # below this → retrieval is considered low quality

def reformulate_query(original_query: str, failed_result: Dict) -> Tuple[str, float]:
    """
    Uses Mistral 7B to reformulate a query when initial retrieval quality is low.
    Returns: (reformulated_query, latency_ms)
    """
    top_sim = failed_result.get("top_similarity", 0)
    top_doc = failed_result["chunks"][0]["doc_title"] if failed_result.get("chunks") else "unknown"

    prompt = f"""The following search query returned low-quality results (top similarity: {top_sim:.3f}).
The best match found was about: "{top_doc}"

Original query: {original_query}

Reformulate this query to be more specific and use different terminology.
Return ONLY the reformulated query. No explanations.

Reformulated query:"""

    t0 = time.time()
    response = ollama.chat(
        model="mistral:7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3, "num_predict": 60}
    )
    latency_ms = (time.time() - t0) * 1000
    reformulated = response["message"]["content"].strip().strip('"').strip("'")
    return reformulated, latency_ms


def retrieve_with_recovery(query: str, k: int = 3, threshold: float = SIMILARITY_THRESHOLD,
                            max_retries: int = 2) -> Dict:
    """
    Retrieval with automatic recovery:
    1. Run initial retrieval
    2. If top similarity < threshold → reformulate query and retry
    3. Return best result across all attempts
    Recovery trace is logged for observability.
    """
    attempts = []
    best_result = None
    current_query = query

    for attempt in range(1 + max_retries):
        result = tool_retriever(current_query, k=k)
        result["attempt"] = attempt + 1
        result["query_used"] = current_query
        attempts.append(result)

        if result["top_similarity"] >= threshold:
            # Good result — no recovery needed
            result["recovery_triggered"] = False
            result["recovery_reason"] = None
            best_result = result
            break

        if attempt < max_retries:
            # Low quality — trigger recovery
            print(f"    ⚠ Recovery triggered (attempt {attempt+1}): "
                  f"similarity={result['top_similarity']:.3f} < {threshold}")
            reformulated, ref_ms = reformulate_query(query, result)
            print(f"    → Reformulated: {reformulated[:80]}")
            current_query = reformulated

    if best_result is None:
        # Use the attempt with highest top similarity
        best_result = max(attempts, key=lambda x: x["top_similarity"])
        best_result["recovery_triggered"] = True
        best_result["recovery_reason"] = f"All attempts below threshold {threshold}"

    best_result["all_attempts"] = attempts
    best_result["num_attempts"] = len(attempts)
    return best_result


# ── Test Recovery on 5 representative queries ──
RECOVERY_TEST_QUERIES = [
    {"id": "R01", "query": "What is RAG?", "expected_doc": "doc_01"},
    {"id": "R02", "query": "vLLM PagedAttention GPU memory", "expected_doc": "doc_05"},
    {"id": "R03", "query": "model decay in production", "expected_doc": "doc_04"},  # slightly ambiguous
    {"id": "R04", "query": "fine-tuning transformers on custom data", "expected_doc": None},  # out-of-scope
    {"id": "R05", "query": "embedding dimensions semantic similarity", "expected_doc": "doc_09"},
]

def run_recovery_evaluation():
    print("\n" + "="*60)
    print("EXTENSION 1: RECOVERY MECHANISMS")
    print("="*60)
    print(f"Similarity threshold: {SIMILARITY_THRESHOLD}")

    recovery_results = []
    for q in RECOVERY_TEST_QUERIES:
        print(f"\n{q['id']}: {q['query']}")
        result = retrieve_with_recovery(q["query"], threshold=SIMILARITY_THRESHOLD)

        final_sim = result["top_similarity"]
        recovered = result["num_attempts"] > 1
        top_doc = result["chunks"][0]["doc_title"] if result["chunks"] else "none"

        print(f"  Attempts: {result['num_attempts']} | Final similarity: {final_sim:.3f}")
        print(f"  Recovery triggered: {recovered}")
        print(f"  Top result: {top_doc}")

        recovery_results.append({
            "query_id": q["id"],
            "original_query": q["query"],
            "expected_doc": q["expected_doc"],
            "num_attempts": result["num_attempts"],
            "recovery_triggered": recovered,
            "final_similarity": final_sim,
            "top_doc": top_doc,
            "attempts": [{"query": a["query_used"], "similarity": a["top_similarity"]}
                         for a in result["all_attempts"]]
        })

    # Save trace
    trace_path = os.path.join(TRACES_DIR, "extension_recovery.json")
    with open(trace_path, "w") as f:
        json.dump({"extension": "recovery_mechanisms",
                   "threshold": SIMILARITY_THRESHOLD,
                   "results": recovery_results}, f, indent=2)
    print(f"\nRecovery trace saved: {trace_path}")

    recovered_count = sum(1 for r in recovery_results if r["recovery_triggered"])
    print(f"\nSummary: {recovered_count}/{len(recovery_results)} queries triggered recovery")
    return recovery_results


# ════════════════════════════════════════════════════
# EXTENSION 2: CONFIDENCE SCORING
# Agent assigns a confidence score to each tool selection
# and retrieval result, flagging low-confidence decisions
# ════════════════════════════════════════════════════

def score_tool_selection_confidence(task: str, tool_name: str,
                                     reasoning: str, retrieval_result: Dict = None) -> Dict:
    """
    Computes a confidence score (0.0-1.0) for a tool selection decision.
    Combines:
    - Retrieval confidence: top similarity score (if tool=retriever)
    - Reasoning confidence: LLM self-assessed confidence
    - Keyword overlap: between task and tool description
    """
    confidence_components = {}

    # Component 1: Retrieval similarity confidence (only for retriever tool)
    if tool_name == "retriever" and retrieval_result:
        sim = retrieval_result.get("top_similarity", 0.5)
        confidence_components["retrieval_similarity"] = sim
    else:
        confidence_components["retrieval_similarity"] = None

    # Component 2: Keyword overlap between task and tool triggers
    TOOL_KEYWORDS = {
        "retriever": ["what", "how", "explain", "define", "describe", "find", "retrieve"],
        "summarizer": ["summarize", "summary", "condense", "brief", "key points", "overview"],
        "keyword_extractor": ["keywords", "terms", "concepts", "identify", "extract", "topics"]
    }
    task_lower = task.lower()
    triggers = TOOL_KEYWORDS.get(tool_name, [])
    overlap = sum(1 for t in triggers if t in task_lower) / max(len(triggers), 1)
    confidence_components["keyword_overlap"] = round(overlap, 3)

    # Component 3: LLM self-assessment
    prompt = f"""Rate your confidence (0.0 to 1.0) that using the '{tool_name}' tool
is the right choice for this task: "{task}"
Reasoning given: "{reasoning}"
Return ONLY a number between 0.0 and 1.0."""

    response = ollama.chat(
        model="mistral:7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "num_predict": 10}
    )
    raw_conf = response["message"]["content"].strip()
    try:
        llm_confidence = float(re.search(r'0?\.\d+|[01]\.?\d*', raw_conf).group())
        llm_confidence = max(0.0, min(1.0, llm_confidence))
    except Exception:
        llm_confidence = 0.5
    confidence_components["llm_self_assessment"] = round(llm_confidence, 3)

    # Weighted average
    weights = {"retrieval_similarity": 0.5, "keyword_overlap": 0.2, "llm_self_assessment": 0.3}
    total_weight = 0
    weighted_sum = 0
    for comp, val in confidence_components.items():
        if val is not None:
            weighted_sum += val * weights[comp]
            total_weight += weights[comp]
    final_confidence = weighted_sum / total_weight if total_weight > 0 else 0.5

    is_low_confidence = final_confidence < 0.5
    return {
        "tool": tool_name,
        "confidence_score": round(final_confidence, 3),
        "is_low_confidence": is_low_confidence,
        "components": confidence_components,
        "flag": "⚠ LOW CONFIDENCE" if is_low_confidence else "✓ HIGH CONFIDENCE"
    }


def run_confidence_evaluation():
    print("\n" + "="*60)
    print("EXTENSION 2: CONFIDENCE SCORING")
    print("="*60)

    test_cases = [
        ("What is RAG?", "retriever", "Task requires factual lookup from knowledge base"),
        ("Summarize the retrieved content about vLLM", "summarizer", "Need to condense retrieved info"),
        ("What is the best pizza recipe?", "retriever", "Trying retriever for out-of-scope query"),
        ("Extract key concepts from this MLOps text", "keyword_extractor", "Task explicitly asks for keyword extraction"),
        ("How does data drift work?", "summarizer", "Using summarizer for a retrieval-type task"),
    ]

    confidence_results = []
    for task, tool, reasoning in test_cases:
        print(f"\nTask: {task[:60]}")
        print(f"Tool: {tool} | Reasoning: {reasoning}")

        retrieval = tool_retriever(task) if tool == "retriever" else None
        conf = score_tool_selection_confidence(task, tool, reasoning, retrieval)

        print(f"Confidence: {conf['confidence_score']:.3f} — {conf['flag']}")
        print(f"Components: {conf['components']}")
        confidence_results.append({"task": task, "tool": tool, **conf})

    # Save trace
    trace_path = os.path.join(TRACES_DIR, "extension_confidence.json")
    with open(trace_path, "w") as f:
        json.dump({"extension": "confidence_scoring",
                   "threshold": 0.5,
                   "results": confidence_results}, f, indent=2)
    print(f"\nConfidence trace saved: {trace_path}")

    low_conf = sum(1 for r in confidence_results if r["is_low_confidence"])
    print(f"\nSummary: {low_conf}/{len(confidence_results)} decisions flagged as low-confidence")
    return confidence_results


# ════════════════════════════════════════════════════
# EXTENSION 3: PARALLEL TOOL EXECUTION
# Runs independent tools concurrently using ThreadPoolExecutor
# Useful when multiple tools can be called at the same time
# (e.g., retrieve + extract keywords simultaneously)
# ════════════════════════════════════════════════════

def run_tools_parallel(tool_calls: List[Dict]) -> Tuple[List[Dict], float]:
    """
    Executes multiple tool calls in parallel using ThreadPoolExecutor.
    Each tool_call: {"tool": name, "input": query_or_text}
    Returns: (results list, total_wall_time_ms)
    """
    TOOL_MAP = {
        "retriever": tool_retriever,
        "summarizer": tool_summarizer,
        "keyword_extractor": tool_keyword_extractor
    }

    def execute_single(call: Dict) -> Dict:
        tool_fn = TOOL_MAP[call["tool"]]
        result = tool_fn(call["input"])
        result["parallel_input"] = call["input"]
        return result

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
        futures = {executor.submit(execute_single, call): call for call in tool_calls}
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    wall_time_ms = (time.time() - t0) * 1000

    return results, wall_time_ms


def run_parallel_evaluation():
    print("\n" + "="*60)
    print("EXTENSION 3: PARALLEL TOOL EXECUTION")
    print("="*60)

    parallel_tasks = [
        {
            "task_id": "P01",
            "description": "Retrieve AND extract keywords simultaneously for a complex query",
            "tool_calls": [
                {"tool": "retriever", "input": "How does RAG reduce hallucinations in LLMs?"},
                {"tool": "keyword_extractor", "input": "How does RAG reduce hallucinations in LLMs?"}
            ]
        },
        {
            "task_id": "P02",
            "description": "Retrieve from two different angles simultaneously",
            "tool_calls": [
                {"tool": "retriever", "input": "ChromaDB vector database features"},
                {"tool": "retriever", "input": "FAISS approximate nearest neighbor search"}
            ]
        },
        {
            "task_id": "P03",
            "description": "Extract keywords AND summarize retrieved content simultaneously",
            "tool_calls": [
                {"tool": "keyword_extractor", "input": "MLOps pipeline stages including data ingestion, feature engineering, model training, model serving and monitoring"},
                {"tool": "summarizer", "input": "MLOps pipeline stages including data ingestion, feature engineering, model training, model serving and monitoring"}
            ]
        }
    ]

    parallel_results = []
    for pt in parallel_tasks:
        print(f"\n{pt['task_id']}: {pt['description']}")
        print(f"  Tools to run in parallel: {[c['tool'] for c in pt['tool_calls']]}")

        # Run in parallel
        par_results, par_time = run_tools_parallel(pt["tool_calls"])

        # Run sequentially for comparison
        seq_times = []
        for call in pt["tool_calls"]:
            t0 = time.time()
            TOOL_MAP = {"retriever": tool_retriever,
                        "summarizer": tool_summarizer,
                        "keyword_extractor": tool_keyword_extractor}
            TOOL_MAP[call["tool"]](call["input"])
            seq_times.append((time.time() - t0) * 1000)
        seq_total = sum(seq_times)

        speedup = seq_total / par_time if par_time > 0 else 1.0

        print(f"  Sequential time: {seq_total:.0f}ms")
        print(f"  Parallel time:   {par_time:.0f}ms")
        print(f"  Speedup:         {speedup:.2f}x")

        for r in par_results:
            if r["tool"] == "retriever":
                print(f"  Retriever result: {r['chunks'][0]['doc_title'] if r.get('chunks') else 'none'}")
            elif r["tool"] == "keyword_extractor":
                print(f"  Keywords: {r.get('keywords', [])[:4]}")
            elif r["tool"] == "summarizer":
                print(f"  Summary: {r.get('summary', '')[:80]}...")

        parallel_results.append({
            "task_id": pt["task_id"],
            "description": pt["description"],
            "tools": [c["tool"] for c in pt["tool_calls"]],
            "parallel_time_ms": round(par_time, 1),
            "sequential_time_ms": round(seq_total, 1),
            "speedup": round(speedup, 2)
        })

    # Save trace
    trace_path = os.path.join(TRACES_DIR, "extension_parallel.json")
    with open(trace_path, "w") as f:
        json.dump({"extension": "parallel_tool_execution",
                   "results": parallel_results}, f, indent=2)
    print(f"\nParallel execution trace saved: {trace_path}")

    avg_speedup = sum(r["speedup"] for r in parallel_results) / len(parallel_results)
    print(f"\nAverage speedup from parallelization: {avg_speedup:.2f}x")
    return parallel_results


# ════════════════════════════════════════════════════
# MAIN: Run all 3 extensions
# ════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MILESTONE 6 — AGENT EXTENSIONS")
    print("Extensions: Recovery | Confidence | Parallel")
    print("="*60)

    recovery_results   = run_recovery_evaluation()
    confidence_results = run_confidence_evaluation()
    parallel_results   = run_parallel_evaluation()

    # ── Final Summary ──
    print("\n\n" + "="*60)
    print("EXTENSIONS SUMMARY")
    print("="*60)

    recovered = sum(1 for r in recovery_results if r["recovery_triggered"])
    print(f"\nExtension 1 — Recovery Mechanisms:")
    print(f"  Queries tested: {len(recovery_results)}")
    print(f"  Recovery triggered: {recovered}")
    print(f"  Threshold used: {SIMILARITY_THRESHOLD}")

    low_conf = sum(1 for r in confidence_results if r["is_low_confidence"])
    print(f"\nExtension 2 — Confidence Scoring:")
    print(f"  Decisions scored: {len(confidence_results)}")
    print(f"  Low-confidence flags: {low_conf}")

    avg_speedup = sum(r["speedup"] for r in parallel_results) / len(parallel_results)
    print(f"\nExtension 3 — Parallel Execution:")
    print(f"  Tasks parallelized: {len(parallel_results)}")
    print(f"  Average speedup: {avg_speedup:.2f}x")

    print(f"\nAll extension traces saved in: {TRACES_DIR}/")
    print("  extension_recovery.json")
    print("  extension_confidence.json")
    print("  extension_parallel.json")
