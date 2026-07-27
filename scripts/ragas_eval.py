"""
RAGAS Evaluation — FinAssist AI RAG Quality Assessment
FY 2025-26 / AY 2026-27

Metrics (heuristic, no external API needed):
  - context_hit       : Retrieved context contains key facts from ground truth
  - answer_coverage   : Generated answer covers key terms from ground truth
  - answer_faithfulness: Answer doesn't contradict retrieved context
  - context_density   : Avg length of retrieved chunks (proxy for chunk quality)

Usage:
    python scripts/ragas_eval.py
    python scripts/ragas_eval.py --output scripts/ragas_report.json
"""
import sys
import json
import re
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Ground Truth Q&A Dataset — 5 comprehensive tax scenarios
EVAL_DATASET = [
    {
        "question": (
            "Can I claim deductions under both Section 80C and Section 80D in the old tax regime? "
            "What are the maximum limits and who is eligible?"
        ),
        "ground_truth": (
            "Yes, both can be claimed simultaneously under the Old Tax Regime. "
            "Section 80C: maximum deduction Rs.1,50,000 per year for individuals and HUFs. "
            "Eligible investments include PPF, ELSS, NSC, 5-year FD, life insurance premium, EPF, "
            "home loan principal repayment, and children tuition fees. "
            "Section 80D: maximum Rs.25,000 for health insurance premium for self/spouse/children "
            "(Rs.50,000 if the insured is a senior citizen). "
            "Additional Rs.25,000 for parents health insurance (Rs.50,000 if parents are senior citizens). "
            "Maximum total 80D = Rs.1,00,000 (self plus senior citizen parents). "
            "Both 80C and 80D are Chapter VI-A deductions and are completely blocked under "
            "the New Tax Regime (Section 115BAC). Neither can be claimed in the New Regime."
        ),
        "key_facts": [
            "80C", "1,50,000", "80D", "25,000", "old regime",
            "PPF", "ELSS", "health insurance", "115BAC", "blocked",
        ],
    },
    {
        "question": (
            "I am a salaried employee earning Rs.12 lakh annually with Rs.1.5 lakh invested in PPF "
            "and Rs.25,000 paid as health insurance premium. Which tax regime would result in lower "
            "tax liability and why?"
        ),
        "ground_truth": (
            "New Tax Regime: Standard deduction Rs.75,000. Net taxable income = Rs.11,25,000. "
            "Since Rs.11,25,000 is below Rs.12,00,000, Section 87A rebate fully applies. "
            "Final New Regime tax = Rs.0 (zero). "
            "Old Tax Regime: Standard deduction Rs.50,000. After 80C Rs.1,50,000 (PPF) and "
            "80D Rs.25,000 (health insurance), net taxable income = Rs.9,75,000. "
            "Slab tax on Rs.9,75,000 approximately Rs.72,500 plus 4% cess = approx Rs.75,400. "
            "Conclusion: New Tax Regime results in ZERO tax vs Old Regime approx Rs.75,400. "
            "New Tax Regime is better for this salaried employee."
        ),
        "key_facts": [
            "new regime", "zero", "87A", "standard deduction", "75,000",
            "old regime", "80C", "80D", "PPF", "better",
        ],
    },
    {
        "question": (
            "What deductions are available for interest paid on a home loan under Sections 24(b) "
            "and 80EEA? Can both be claimed together?"
        ),
        "ground_truth": (
            "Section 24(b): For self-occupied property, maximum Rs.2,00,000 per year under Old Regime only. "
            "Not allowed under New Regime for self-occupied property. "
            "For let-out property, actual interest paid (no cap) is deductible under both regimes. "
            "Section 80EEA: Additional deduction of Rs.1,50,000 for first-time home buyers. "
            "Conditions: loan sanctioned between 1 April 2019 and 31 March 2022, "
            "stamp duty value of property must not exceed Rs.45 lakh, "
            "taxpayer must not own any other residential property on loan sanction date. "
            "Both 24(b) and 80EEA can be claimed together for self-occupied property under Old Regime, "
            "giving total interest deduction up to Rs.3,50,000 (Rs.2L plus Rs.1.5L). "
            "Section 80EEA is a Chapter VI-A deduction and is NOT available under New Tax Regime."
        ),
        "key_facts": [
            "24(b)", "2,00,000", "80EEA", "1,50,000", "first-time", "45 lakh",
            "together", "old regime", "new regime", "self-occupied",
        ],
    },
    {
        "question": (
            "My employer deducted TDS, but I also earned interest from fixed deposits and savings "
            "accounts. Do I still need to file an Income Tax Return, and which ITR form should I use?"
        ),
        "ground_truth": (
            "Yes, filing is mandatory if total gross income exceeds the basic exemption limit. "
            "FD interest is taxable as Income from Other Sources. "
            "Bank deducts TDS at 10% under Section 194A if FD interest exceeds Rs.40,000 per year "
            "(Rs.50,000 for senior citizens). Even if TDS is deducted, you must file ITR to report "
            "all income and claim refund if excess TDS was deducted. "
            "Savings account interest: deductible up to Rs.10,000 under Section 80TTA in Old Regime only. "
            "ITR Form: Use ITR-1 (Sahaj) if total income is up to Rs.50 lakh and income sources are "
            "salary, one house property, and other sources (FD interest, savings interest). "
            "Use ITR-2 if income exceeds Rs.50 lakh or if you have capital gains or multiple house properties."
        ),
        "key_facts": [
            "ITR", "file", "FD", "interest", "194A", "TDS",
            "ITR-1", "Sahaj", "80TTA", "10,000", "40,000", "other sources",
        ],
    },
    {
        "question": (
            "What documents are required while filing an Income Tax Return for a salaried individual, "
            "and how should Form 16, AIS, and Form 26AS be used during the filing process?"
        ),
        "ground_truth": (
            "Form 16: TDS certificate from employer. Part A shows TDS deducted and deposited with PAN of employer. "
            "Part B shows salary breakup, allowances, perquisites, and deductions claimed. "
            "Form 26AS: Tax credit statement showing all TDS deducted against your PAN from all sources "
            "(employer, bank, buyer). Always verify Form 16 matches Form 26AS to avoid notices. "
            "Annual Information Statement (AIS): Comprehensive statement of all financial transactions "
            "linked to PAN including salary, FD interest, dividends, mutual fund transactions, property sales. "
            "Other documents: bank statements for interest income, home loan interest certificate for 24(b), "
            "investment proofs for 80C (PPF passbook, ELSS statement), health insurance receipt for 80D, "
            "rent receipts for HRA, Form 16A from bank for FD TDS. "
            "Process: Download Form 26AS and AIS from income tax portal. Pre-fill ITR using these. "
            "Reconcile with Form 16. Report all income including FD interest and dividends."
        ),
        "key_facts": [
            "Form 16", "Form 26AS", "AIS", "TDS", "Part A", "Part B",
            "salary", "PAN", "reconcile", "pre-fill", "interest", "80C",
        ],
    },
]


def build_rag_pipeline():
    from embeddings.embedder import BGEEmbedder
    from vectordb.faiss_store import FAISSStore
    from rag.retriever import HybridRetriever
    from rag.reranker import BGEReranker
    from config.settings import settings
    embedder = BGEEmbedder(settings.embedding_model)
    store = FAISSStore(embedding_dim=embedder.embedding_dim, index_path=settings.faiss_index_path)
    store.load()
    retriever = HybridRetriever(embedder=embedder, faiss_store=store, top_k=15)
    reranker = BGEReranker()
    return embedder, retriever, reranker


def retrieve_contexts(query, embedder, retriever, reranker, top_k=5):
    candidates = retriever.retrieve(query=query, top_k=top_k * 3)
    ranked = reranker.rerank(query=query, candidates=candidates, top_k=top_k)
    return [r.get("text", "") for r in ranked]


def generate_answer(query, contexts):
    import urllib.request
    from config.settings import settings
    context_text = "\n\n".join(f"[{i+1}] {c[:500]}" for i, c in enumerate(contexts))
    prompt = (
        "You are FinAssist AI, an expert Indian tax advisor for FY 2025-26.\n"
        "Answer ONLY using the context below. Be concise and accurate.\n\n"
        f"CONTEXT:\n{context_text}\n\nQUESTION: {query}\n\nANSWER:"
    )
    payload = json.dumps({"model": settings.ollama_model, "prompt": prompt,
                          "stream": False, "options": {"temperature": 0.1}}).encode()
    req = urllib.request.Request(
        f"{settings.ollama_base_url}/api/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read()).get("response", "").strip()


def _tokens(text):
    return set(re.findall(r"[a-z0-9,\.]+", text.lower()))


def score_context_hit(contexts, key_facts):
    if not contexts or not key_facts:
        return 0.0
    combined = " ".join(contexts).lower()
    hits = sum(1 for kf in key_facts if kf.lower() in combined)
    return round(hits / len(key_facts), 3)


def score_answer_coverage(answer, ground_truth):
    if not answer or not ground_truth:
        return 0.0
    gt_tokens = _tokens(ground_truth)
    ans_tokens = _tokens(answer)
    if not gt_tokens:
        return 0.0
    return round(min(len(gt_tokens & ans_tokens) / len(gt_tokens) * 2.5, 1.0), 3)


def score_answer_faithfulness(answer, contexts):
    if not answer or not contexts:
        return 0.0
    ctx_tokens = _tokens(" ".join(contexts))
    meaningful = {t for t in _tokens(answer) if len(t) > 3}
    if not meaningful:
        return 0.0
    return round(sum(1 for t in meaningful if t in ctx_tokens) / len(meaningful), 3)


def score_context_density(contexts):
    if not contexts:
        return 0.0
    return round(min(sum(len(c) for c in contexts) / len(contexts) / 500.0, 1.0), 3)


def run_ragas_evaluation(output_path=None):
    print("=" * 60)
    print("  FinAssist AI - RAG Quality Evaluation")
    print("  FY 2025-26 | Heuristic Scoring")
    print("=" * 60)
    print()

    print("[1/4] Building RAG pipeline...")
    embedder, retriever, reranker = build_rag_pipeline()
    print("      Pipeline ready.\n")

    print("[2/4] Checking Ollama connection...")
    from config.settings import settings
    import urllib.request, urllib.error
    ollama_ok = False
    try:
        urllib.request.urlopen(settings.ollama_base_url, timeout=5)
        ollama_ok = True
        print(f"      Ollama reachable at {settings.ollama_base_url}\n")
    except Exception as e:
        print(f"      Ollama not reachable ({e}). Answers will be skipped.\n")

    print(f"[3/4] Evaluating {len(EVAL_DATASET)} questions...\n")
    results = []
    for i, item in enumerate(EVAL_DATASET, 1):
        q = item["question"]
        gt = item["ground_truth"]
        key_facts = item.get("key_facts", [])
        print(f"  Q{i}: {q[:68]}...")
        t0 = time.time()

        ctxs = retrieve_contexts(q, embedder, retriever, reranker, top_k=5)

        answer = ""
        if ollama_ok:
            try:
                answer = generate_answer(q, ctxs)
            except Exception as e:
                answer = f"[LLM error: {e}]"
                print(f"       Warning: {e}")

        elapsed = round(time.time() - t0, 1)
        ctx_hit   = score_context_hit(ctxs, key_facts)
        ans_cov   = score_answer_coverage(answer, gt) if answer else 0.0
        ans_faith = score_answer_faithfulness(answer, ctxs) if answer else 0.0
        ctx_dens  = score_context_density(ctxs)

        print(f"       [{elapsed}s] ctx_hit={ctx_hit:.2f} | coverage={ans_cov:.2f} | "
              f"faithfulness={ans_faith:.2f} | density={ctx_dens:.2f}")
        if answer:
            print(f"       Answer: {answer[:110]}...")
        print()

        results.append({
            "question": q, "answer": answer, "ground_truth": gt, "contexts": ctxs,
            "scores": {"context_hit": ctx_hit, "answer_coverage": ans_cov,
                       "answer_faithfulness": ans_faith, "context_density": ctx_dens},
        })

    print("[4/4] Summary\n")
    metric_names = ["context_hit", "answer_coverage", "answer_faithfulness", "context_density"]
    agg = {m: round(sum(r["scores"][m] for r in results) / len(results), 3) for m in metric_names}

    print("  Aggregate Scores (0.0 worst - 1.0 best):")
    print(f"  {'Metric':<26} {'Score':>6}  {'Bar':<22} Grade")
    print("  " + "-" * 62)
    for m, score in agg.items():
        bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
        grade = "GOOD" if score >= 0.7 else ("OK" if score >= 0.5 else "POOR")
        print(f"  {m:<26} {score:>6.3f}  [{bar}] {grade}")

    overall = round(sum(agg.values()) / len(agg), 3)
    print()
    print(f"  Overall Average: {overall:.3f}")
    if overall >= 0.7:
        verdict = "GOOD - RAG pipeline is performing well"
    elif overall >= 0.5:
        verdict = "AVERAGE - Some improvements needed"
    else:
        verdict = "POOR - Significant RAG improvements required"
    print(f"  Verdict: {verdict}\n")

    report = {
        "metadata": {"total_questions": len(EVAL_DATASET), "ollama_used": ollama_ok,
                     "scoring": "heuristic (keyword overlap)"},
        "aggregate_scores": agg,
        "overall_average": overall,
        "verdict": verdict,
        "individual_results": results,
    }
    out = output_path or "scripts/ragas_report.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Full report saved: {out}\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG evaluation for FinAssist AI")
    parser.add_argument("--output", "-o", default=None, help="Output JSON report path")
    args = parser.parse_args()
    run_ragas_evaluation(output_path=args.output)