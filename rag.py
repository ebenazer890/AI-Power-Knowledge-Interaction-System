from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import requests


@dataclass
class RetrievedPassage:
    text: str
    score: float


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


class FaissIndex:
    def __init__(self, dim: int) -> None:
        import faiss

        self._faiss = faiss
        self._index = faiss.IndexFlatIP(dim)
        self._passages: List[str] = []

    @property
    def size(self) -> int:
        return len(self._passages)

    def add(self, vectors: np.ndarray, passages: List[str]) -> None:
        if len(passages) != vectors.shape[0]:
            raise ValueError("passages and vectors must have same length")
        self._index.add(vectors)
        self._passages.extend(passages)

    def search(self, query_vector: np.ndarray, top_k: int = 6) -> List[RetrievedPassage]:
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        scores, idxs = self._index.search(query_vector.astype(np.float32), top_k)
        results: List[RetrievedPassage] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0 or idx >= len(self._passages):
                continue
            results.append(RetrievedPassage(text=self._passages[idx], score=float(score)))
        return results


_COMMON_CORRECTIONS = {
    "wther": "whether",
    "wether": "whether",
    "finacial": "financial",
    "finanicial": "financial",
    "summery": "summary",
    "sumarize": "summarize",
    "summarise": "summarize",
}


def normalize_user_question(question: str) -> str:
    q = " ".join((question or "").strip().split())
    if not q:
        return ""

    # conservative word-level corrections (keeps user meaning; helps retrieval)
    tokens = q.split(" ")
    tokens = [_COMMON_CORRECTIONS.get(t.lower(), t) for t in tokens]
    return " ".join(tokens)


def _wants_summary(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in ("summarize", "summary", "summarise"))


def _summary_is_underspecified(q: str) -> bool:
    # If they say “summarize” but don't mention scope, ask a quick clarifying question.
    ql = q.lower()
    if not _wants_summary(ql):
        return False
    scope_markers = ("page", "pages", "section", "chapter", "table", "about", "focus on", "only")
    return not any(m in ql for m in scope_markers)


_MONEY_RE = re.compile(r"(?:[$€£]\s*\d|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b)")
_FINANCE_TERMS = (
    "revenue",
    "income",
    "expense",
    "expenses",
    "cost",
    "costs",
    "profit",
    "loss",
    "balance",
    "debit",
    "credit",
    "interest",
    "tax",
    "invoice",
    "payment",
    "transaction",
    "cash flow",
    "budget",
    "assets",
    "liabilities",
    "equity",
    "fee",
    "charge",
    "refund",
    "amount",
    "total",
    "subtotal",
    "account",
    "statement",
)


def _looks_like_financial_concepts_question(q: str) -> bool:
    ql = q.lower()
    return (
        "financial concept" in ql
        or "financial concepts" in ql
        or ("financial" in ql and any(w in ql for w in ("concept", "concepts", "terms", "topics")))
    )


def _extract_financial_concepts_from_context(context_passages: List[str]) -> List[str]:
    joined = "\n".join(context_passages)
    jl = joined.lower()

    found: List[str] = []
    for term in _FINANCE_TERMS:
        if term in jl:
            found.append(term)

    if _MONEY_RE.search(joined):
        found.append("currency amounts")

    # stable ordering, de-dupe
    out: List[str] = []
    seen = set()
    for x in found:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _build_llm_prompt(question: str, context_passages: List[str]) -> str:
    context = "\n\n".join(context_passages)
    return (
        "You are a helpful PDF RAG assistant.\n"
        "- Use the provided context to answer questions about the PDF.\n"
        "- You MAY use general knowledge for definitions or explanations, but do NOT invent facts that are not supported by the context.\n"
        "- When you cite something from the PDF, prefer quoting short phrases and keep the [page N] tags if present.\n"
        "- Only ask clarifying questions when the user request is genuinely underspecified.\n"
        "\n"
        f"Context:\n{context}\n\n"
        f"User question: {question}\n"
        "Answer:"
    )


def try_openai_chat(prompt: str) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": "You answer user questions about an uploaded PDF using provided context.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception:
        return None


def try_ollama_chat(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    base = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    timeout_s = int(os.getenv("OLLAMA_TIMEOUT", "180"))
    max_tokens = int(os.getenv("OLLAMA_MAX_TOKENS", "256"))

    try:
        r = requests.post(
            f"{base}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
            timeout=timeout_s,
        )
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: {r.text[:300]}"
        data = r.json()
        out = (data.get("response") or "").strip()
        return (out or None), None
    except Exception as e:
        return None, str(e)


def try_gemini_chat(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """Call Google Gemini 2.5 Flash. Returns (answer, error_msg)."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return None, "No GEMINI_API_KEY set."

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    max_tokens = int(os.getenv("GEMINI_MAX_TOKENS", "1024"))
    temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        text = (response.text or "").strip()
        return (text or None), None
    except Exception as e:
        return None, str(e)


def answer_with_llm_or_extract(question: str, retrieved: List[RetrievedPassage]) -> Tuple[str, bool]:
    """Returns (answer, used_llm)."""
    q = normalize_user_question(question)
    context = [p.text for p in retrieved]

    # Intent heuristic 1: underspecified summary => ask a targeted clarifying question.
    if _summary_is_underspecified(q):
        return (
            "Do you want a summary of the whole PDF, or a specific section/pages? "
            "Also, should it be short (5 bullets) or detailed (1–2 paragraphs)?",
            False,
        )

    # Intent heuristic 2: “financial concepts” => answer directly from retrieved context when possible.
    if _looks_like_financial_concepts_question(q):
        concepts = _extract_financial_concepts_from_context(context)
        if concepts:
            concepts_txt = ", ".join(concepts[:10])
            return (
                "Yes — I see financial-related content in the retrieved parts of the PDF. "
                f"Examples of financial concepts/terms mentioned: {concepts_txt}. "
                "If you want, tell me whether you mean *accounting terms*, *financial statements*, or *finance theory*, and I’ll narrow it down.",
                False,
            )

    prompt = _build_llm_prompt(q or question, context)

    provider = os.getenv("LLM_PROVIDER", "None")
    llm_error = None

    if provider == "Gemini":
        ans, llm_error = try_gemini_chat(prompt)
        if ans:
            return ans, True
    elif provider == "OpenAI":
        ans = try_openai_chat(prompt)
        if ans:
            return ans, True
        llm_error = "OpenAI call failed. Check your API key."
    elif provider == "Ollama":
        ans, llm_error = try_ollama_chat(prompt)
        if ans:
            return ans, True

    # Fallback: extractive
    if not retrieved:
        return "I couldn't find anything relevant in the PDF.", False

    top = "\n\n".join([p.text for p in retrieved[:3]])
    if provider == "None" or not provider:
        fallback_msg = (
            "No LLM provider selected. Choose one (Gemini / OpenAI / Ollama) in the sidebar. "
            "For now, here are the most relevant passages:\n\n" + top
        )
    else:
        fallback_msg = (
            f"{provider} did not return an answer. Check your settings in the sidebar."
        )
        if llm_error:
            fallback_msg += f"\n\nError: {llm_error}"
        fallback_msg += "\n\nRelevant passages:\n\n" + top
    return fallback_msg, False
