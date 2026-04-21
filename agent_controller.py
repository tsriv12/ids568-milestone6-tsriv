"""
Milestone 6 — Part 2: Multi-Tool Agent Controller
Model: mistral:7b-instruct via Ollama
Tools: Retriever, Summarizer, KeywordExtractor
"""

import json
import time
import os
import re
from typing import List, Dict, Any, Tuple
from datetime import datetime

import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# ─────────────────────────────────────────────
# SETUP: Load the same ChromaDB index from Part 1
# ─────────────────────────────────────────────

CHROMA_DIR = "./chroma_db"
TRACES_DIR = "./agent_traces"
os.makedirs(TRACES_DIR, exist_ok=True)

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection("mlops_rag")
print(f"Connected. Index contains {collection.count()} chunks.")


# ─────────────────────────────────────────────
# TOOL 1: RETRIEVER
# Reused directly from Part 1
# ─────────────────────────────────────────────

def tool_retriever(query: str, k: int = 3) -> Dict:
    """
    Retrieves top-k relevant chunks from ChromaDB for a given query.
    Returns structured results with doc titles, content, and similarity scores.
    """
    t0 = time.time()
    query_embedding = embedding_model.encode(
        [query], normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    latency_ms = (time.time() - t0) * 1000

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "rank": i + 1,
            "doc_id": results["metadatas"][0][i]["doc_id"],
            "doc_title": results["metadatas"][0][i]["doc_title"],
            "content": results["documents"][0][i],
            "similarity": round(1 - results["distances"][0][i], 4)
        })

    return {
        "tool": "retriever",
        "query": query,
        "chunks": chunks,
        "top_doc": chunks[0]["doc_title"] if chunks else "none",
        "top_similarity": chunks[0]["similarity"] if chunks else 0.0,
        "latency_ms": round(latency_ms, 1)
    }


# ─────────────────────────────────────────────
# TOOL 2: SUMMARIZER
# Uses Mistral 7B to summarize a passage
# ─────────────────────────────────────────────

def tool_summarizer(text: str, max_sentences: int = 3) -> Dict:
    """
    Summarizes a given text passage into a concise summary using Mistral 7B.
    Used when retrieved content is long and needs distillation before final answer.
    """
    prompt = f"""Summarize the following text in {max_sentences} sentences or fewer.
Be concise and preserve the key technical facts.

TEXT:
{text}

SUMMARY:"""

    t0 = time.time()
    response = ollama.chat(
        model="mistral:7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1, "num_predict": 150}
    )
    latency_ms = (time.time() - t0) * 1000

    summary = response["message"]["content"].strip()
    return {
        "tool": "summarizer",
        "input_length": len(text.split()),
        "summary": summary,
        "output_length": len(summary.split()),
        "latency_ms": round(latency_ms, 1)
    }


# ─────────────────────────────────────────────
# TOOL 3: KEYWORD EXTRACTOR
# Extracts key technical terms from a passage
# ─────────────────────────────────────────────

def tool_keyword_extractor(text: str, max_keywords: int = 8) -> Dict:
    """
    Extracts key technical terms and concepts from a text passage using Mistral 7B.
    Used to identify the core concepts in a query or retrieved passage,
    enabling the agent to refine follow-up retrieval queries.
    """
    prompt = f"""Extract the {max_keywords} most important technical keywords or phrases from the text below.
Return them as a comma-separated list. No explanations, just the keywords.

TEXT:
{text}

KEYWORDS:"""

    t0 = time.time()
    response = ollama.chat(
        model="mistral:7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1, "num_predict": 80}
    )
    latency_ms = (time.time() - t0) * 1000

    raw = response["message"]["content"].strip()
    keywords = [k.strip() for k in raw.split(",") if k.strip()][:max_keywords]

    return {
        "tool": "keyword_extractor",
        "input_length": len(text.split()),
        "keywords": keywords,
        "latency_ms": round(latency_ms, 1)
    }


# ─────────────────────────────────────────────
# TOOL REGISTRY
# ─────────────────────────────────────────────

TOOLS = {
    "retriever": {
        "fn": tool_retriever,
        "description": "Retrieves relevant documents from the MLOps knowledge base. Use when the task requires factual information, definitions, or explanations from the corpus.",
        "triggers": ["what is", "how does", "explain", "describe", "define", "find information", "look up", "retrieve"]
    },
    "summarizer": {
        "fn": tool_summarizer,
        "description": "Summarizes a long passage into a concise summary. Use after retrieval when the retrieved content is lengthy and needs distillation.",
        "triggers": ["summarize", "summary", "brief overview", "in short", "condense", "key points"]
    },
    "keyword_extractor": {
        "fn": tool_keyword_extractor,
        "description": "Extracts key technical terms from text. Use to identify core concepts in a query before retrieval, or to analyze retrieved content.",
        "triggers": ["key terms", "keywords", "concepts", "topics", "terminology", "technical terms"]
    }
}


# ─────────────────────────────────────────────
# AGENT CONTROLLER
# LLM-driven tool selection using Mistral 7B
# ─────────────────────────────────────────────

def select_tool(task: str, step: int, history: List[Dict]) -> Dict:
    """
    Uses Mistral 7B to decide which tool to invoke next.
    Returns: {tool_name, tool_input, reasoning}
    """
    history_text = ""
    if history:
        history_text = "\n\nPREVIOUS STEPS:\n"
        for h in history:
            history_text += f"  Step {h['step']}: Used [{h['tool']}] → {h['result_summary']}\n"

    tool_descriptions = "\n".join([
        f"- {name}: {info['description']}"
        for name, info in TOOLS.items()
    ])

    prompt = f"""You are an AI agent that solves tasks step by step using tools.

AVAILABLE TOOLS:
{tool_descriptions}

TASK: {task}
CURRENT STEP: {step}{history_text}

Decide which tool to use next to make progress on this task.
You MUST respond in this exact JSON format with no other text:
{{
  "tool": "<tool_name>",
  "input": "<input string for the tool>",
  "reasoning": "<one sentence explaining why you chose this tool>"
}}

Valid tool names: retriever, summarizer, keyword_extractor
If you have enough information to answer the task, use tool "retriever" with input "DONE".
"""

    response = ollama.chat(
        model="mistral:7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1, "num_predict": 200}
    )

    raw = response["message"]["content"].strip()

    # Parse JSON from response
    try:
        # Find JSON block in response
        json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if json_match:
            decision = json.loads(json_match.group())
        else:
            decision = json.loads(raw)
    except Exception:
        # Fallback: rule-based selection if LLM output is malformed
        decision = _rule_based_fallback(task, step, history)

    return decision


def _rule_based_fallback(task: str, step: int, history: List[Dict]) -> Dict:
    """
    Fallback rule-based tool selection when LLM output is unparseable.
    Ensures the agent always makes progress.
    """
    task_lower = task.lower()
    used_tools = [h["tool"] for h in history]

    if step == 1:
        # Always start with keyword extraction for complex tasks
        if any(word in task_lower for word in ["compare", "difference", "tradeoff", "versus"]):
            return {"tool": "keyword_extractor", "input": task,
                    "reasoning": "Extracting keywords first to identify key concepts for retrieval."}
        return {"tool": "retriever", "input": task,
                "reasoning": "Starting with retrieval to find relevant information."}

    if "retriever" in used_tools and "summarizer" not in used_tools:
        # Summarize what was retrieved
        last_retrieval = next((h for h in reversed(history) if h["tool"] == "retriever"), None)
        if last_retrieval:
            return {"tool": "summarizer",
                    "input": last_retrieval.get("retrieved_content", task),
                    "reasoning": "Summarizing retrieved content to distill key information."}

    return {"tool": "retriever", "input": task,
            "reasoning": "Retrieving additional information to complete the task."}


def generate_final_answer(task: str, history: List[Dict]) -> Tuple[str, float]:
    """
    Generates the final answer using Mistral 7B, grounded in the agent's accumulated context.
    """
    # Collect all retrieved content and summaries
    context_parts = []
    for step in history:
        if step["tool"] == "retriever" and "chunks" in step.get("raw_result", {}):
            for chunk in step["raw_result"]["chunks"][:2]:
                context_parts.append(f"[Retrieved from: {chunk['doc_title']}]\n{chunk['content']}")
        elif step["tool"] == "summarizer":
            context_parts.append(f"[Summary]\n{step['raw_result'].get('summary', '')}")
        elif step["tool"] == "keyword_extractor":
            kws = step["raw_result"].get("keywords", [])
            context_parts.append(f"[Key Concepts Identified]\n{', '.join(kws)}")

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No context retrieved."

    prompt = f"""You are a helpful AI assistant. Using the context gathered by your tools, answer the task below.
Base your answer on the provided context. Be specific and complete.

CONTEXT:
{context}

TASK: {task}

ANSWER:"""

    t0 = time.time()
    response = ollama.chat(
        model="mistral:7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1, "num_predict": 400}
    )
    latency_ms = (time.time() - t0) * 1000

    return response["message"]["content"].strip(), latency_ms


def run_agent(task: str, task_id: str, max_steps: int = 4) -> Dict:
    """
    Main agent loop. Runs until max_steps or DONE signal.
    Logs full trace to agent_traces/{task_id}.json
    """
    print(f"\n{'='*60}")
    print(f"TASK {task_id}: {task}")
    print(f"{'='*60}")

    trace = {
        "task_id": task_id,
        "task": task,
        "timestamp": datetime.now().isoformat(),
        "model": "mistral:7b-instruct",
        "steps": [],
        "final_answer": "",
        "total_latency_ms": 0,
        "tools_used": []
    }

    history = []
    total_start = time.time()

    for step_num in range(1, max_steps + 1):
        print(f"\n  Step {step_num}:")

        # LLM decides which tool to use
        decision = select_tool(task, step_num, history)
        tool_name = decision.get("tool", "retriever")
        tool_input = decision.get("input", task)
        reasoning = decision.get("reasoning", "No reasoning provided.")

        print(f"    Tool selected: [{tool_name}]")
        print(f"    Reasoning: {reasoning}")
        print(f"    Input: {tool_input[:80]}...")

        # Check for DONE signal
        if tool_input.strip().upper() == "DONE" or step_num == max_steps:
            print(f"    → Agent determined sufficient context gathered.")
            break

        # Execute the selected tool
        if tool_name not in TOOLS:
            tool_name = "retriever"  # safe fallback

        tool_fn = TOOLS[tool_name]["fn"]

        if tool_name == "retriever":
            result = tool_fn(tool_input, k=3)
            result_summary = f"Retrieved {len(result['chunks'])} chunks. Top: '{result['top_doc']}' (sim={result['top_similarity']})"
            retrieved_content = " ".join([c["content"] for c in result["chunks"][:2]])
        elif tool_name == "summarizer":
            # Summarize the most recent retrieved content
            if history:
                last = next((h for h in reversed(history) if h["tool"] == "retriever"), None)
                if last and "chunks" in last.get("raw_result", {}):
                    content_to_summarize = " ".join([c["content"] for c in last["raw_result"]["chunks"][:2]])
                    result = tool_fn(content_to_summarize)
                else:
                    result = tool_fn(tool_input)
            else:
                result = tool_fn(tool_input)
            result_summary = f"Summary ({result['output_length']} words): {result['summary'][:80]}..."
            retrieved_content = result.get("summary", "")
        elif tool_name == "keyword_extractor":
            result = tool_fn(tool_input)
            result_summary = f"Keywords: {', '.join(result['keywords'][:5])}"
            retrieved_content = ", ".join(result["keywords"])

        print(f"    Result: {result_summary}")

        # Log the step
        step_record = {
            "step": step_num,
            "tool": tool_name,
            "tool_input": tool_input,
            "reasoning": reasoning,
            "result_summary": result_summary,
            "raw_result": result,
            "retrieved_content": retrieved_content,
            "latency_ms": result.get("latency_ms", 0)
        }

        history.append(step_record)
        trace["steps"].append(step_record)

        if tool_name not in trace["tools_used"]:
            trace["tools_used"].append(tool_name)

    # Generate final answer
    print(f"\n  Generating final answer...")
    final_answer, gen_latency = generate_final_answer(task, history)
    total_latency = (time.time() - total_start) * 1000

    trace["final_answer"] = final_answer
    trace["final_answer_latency_ms"] = round(gen_latency, 1)
    trace["total_latency_ms"] = round(total_latency, 1)
    trace["num_steps"] = len(history)

    print(f"\n  FINAL ANSWER:\n  {final_answer[:300]}...")
    print(f"\n  Total latency: {total_latency:.0f}ms | Steps: {len(history)} | Tools: {trace['tools_used']}")

    # Save trace to file
    trace_path = os.path.join(TRACES_DIR, f"{task_id}.json")
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"  Trace saved: {trace_path}")

    return trace


# ─────────────────────────────────────────────
# 10 EVALUATION TASKS
# ─────────────────────────────────────────────

EVAL_TASKS = [
    {
        "id": "task_01",
        "task": "What is retrieval-augmented generation and what are the key metrics used to evaluate it?",
        "expected_tools": ["retriever", "summarizer"],
        "type": "multi_tool_factual"
    },
    {
        "id": "task_02",
        "task": "Compare FAISS and ChromaDB as vector databases. What are the key differences and when should each be used?",
        "expected_tools": ["keyword_extractor", "retriever"],
        "type": "comparison"
    },
    {
        "id": "task_03",
        "task": "Explain how data drift is detected in production ML systems and what happens when it is detected.",
        "expected_tools": ["retriever", "summarizer"],
        "type": "multi_step_explanation"
    },
    {
        "id": "task_04",
        "task": "What chunking strategies exist for RAG systems? Extract the key technical terms and then explain the tradeoffs.",
        "expected_tools": ["keyword_extractor", "retriever"],
        "type": "keyword_then_retrieve"
    },
    {
        "id": "task_05",
        "task": "How does vLLM improve LLM serving performance? Summarize the key technical mechanisms.",
        "expected_tools": ["retriever", "summarizer"],
        "type": "retrieve_then_summarize"
    },
    {
        "id": "task_06",
        "task": "What is the role of a feature store in MLOps and how does it prevent training-serving skew?",
        "expected_tools": ["retriever", "summarizer"],
        "type": "multi_tool_factual"
    },
    {
        "id": "task_07",
        "task": "Identify the key technical concepts in this passage and retrieve more information about the most important one: 'Agents use ReAct framework with tool invocations and chain-of-thought reasoning'",
        "expected_tools": ["keyword_extractor", "retriever"],
        "type": "keyword_then_retrieve"
    },
    {
        "id": "task_08",
        "task": "What embedding models are available for semantic search? Which benchmark is used to evaluate them?",
        "expected_tools": ["retriever", "keyword_extractor"],
        "type": "multi_tool_factual"
    },
    {
        "id": "task_09",
        "task": "A user wants to build a RAG system. Walk through the complete pipeline steps they need to implement.",
        "expected_tools": ["retriever", "summarizer"],
        "type": "multi_step_reasoning"
    },
    {
        "id": "task_10",
        "task": "What MLOps tools are used for model versioning and experiment tracking? What problems do they solve?",
        "expected_tools": ["keyword_extractor", "retriever"],
        "type": "comparison"
    }
]


# ─────────────────────────────────────────────
# MAIN: Run all 10 evaluation tasks
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MILESTONE 6 — AGENT CONTROLLER EVALUATION")
    print("Model: mistral:7b-instruct | Tools: 3 | Tasks: 10")
    print("="*60)

    all_traces = []
    eval_start = time.time()

    for task_info in EVAL_TASKS:
        trace = run_agent(
            task=task_info["task"],
            task_id=task_info["id"],
            max_steps=4
        )
        trace["task_type"] = task_info["type"]
        trace["expected_tools"] = task_info["expected_tools"]
        all_traces.append(trace)

    total_eval_time = time.time() - eval_start

    # ── Summary Report ──
    print("\n\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    tool_match_count = 0
    for t in all_traces:
        expected = set(t.get("expected_tools", []))
        used = set(t.get("tools_used", []))
        match = len(expected & used) / len(expected) if expected else 0
        tool_match_count += match
        print(f"\n{t['task_id']} [{t['task_type']}]")
        print(f"  Task:           {t['task'][:70]}...")
        print(f"  Tools used:     {t['tools_used']}")
        print(f"  Expected tools: {t.get('expected_tools', [])}")
        print(f"  Tool match:     {match:.0%}")
        print(f"  Steps taken:    {t['num_steps']}")
        print(f"  Total latency:  {t['total_latency_ms']:.0f}ms")
        print(f"  Answer preview: {t['final_answer'][:100]}...")

    avg_tool_match = tool_match_count / len(all_traces)
    avg_latency = sum(t["total_latency_ms"] for t in all_traces) / len(all_traces)
    avg_steps = sum(t["num_steps"] for t in all_traces) / len(all_traces)

    print(f"\n{'─'*60}")
    print(f"Avg tool selection match: {avg_tool_match:.0%}")
    print(f"Avg steps per task:       {avg_steps:.1f}")
    print(f"Avg total latency:        {avg_latency:.0f}ms")
    print(f"Total evaluation time:    {total_eval_time:.0f}s")
    print(f"Traces saved to:          ./agent_traces/")

    # Save combined summary
    summary = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "model": "mistral:7b-instruct",
        "num_tasks": len(all_traces),
        "avg_tool_match": round(avg_tool_match, 3),
        "avg_steps": round(avg_steps, 2),
        "avg_latency_ms": round(avg_latency, 1),
        "traces": all_traces
    }
    with open("agent_evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Full summary saved: agent_evaluation_summary.json")
