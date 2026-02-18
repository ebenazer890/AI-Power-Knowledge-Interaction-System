from __future__ import annotations

import hashlib
import os
import tempfile
from typing import List, Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from finance_analysis import (
    advanced_summary,
    aggregate_finance,
    category_breakdown,
    parse_finance_request,
    parse_financial_tables,
    pie_breakdown,
    top_transactions,
    totals,
)
from pdf_utils import extract_pdf_tables, extract_pdf_text, page_chunks_to_passages
from rag import Embedder, FaissIndex, answer_with_llm_or_extract, normalize_user_question


st.set_page_config(page_title="PDF RAG Chatbot + Finance Analyzer", layout="wide")

st.title("PDF RAG Chatbot + Financial Analysis")

with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    max_pages = st.number_input("Max pages to read", min_value=1, max_value=500, value=50)
    st.divider()
    st.header("RAG")
    top_k = st.slider("Top-K passages", min_value=3, max_value=12, value=6)

    llm_provider = st.selectbox("LLM provider", ["None", "Gemini", "OpenAI", "Ollama"], index=0)
    os.environ["LLM_PROVIDER"] = llm_provider

    if llm_provider == "Gemini":
        gemini_key = st.text_input("GEMINI_API_KEY", type="password", value=os.getenv("GEMINI_API_KEY", ""))
        gemini_model = st.text_input("GEMINI_MODEL", value=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
        gemini_max_tokens = st.number_input(
            "GEMINI_MAX_TOKENS",
            min_value=64,
            max_value=8192,
            value=int(os.getenv("GEMINI_MAX_TOKENS", "1024")),
            step=64,
        )
        gemini_temp = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(os.getenv("GEMINI_TEMPERATURE", "0.2")),
            step=0.05,
        )
        if gemini_key:
            os.environ["GEMINI_API_KEY"] = gemini_key
        if gemini_model:
            os.environ["GEMINI_MODEL"] = gemini_model
        os.environ["GEMINI_MAX_TOKENS"] = str(int(gemini_max_tokens))
        os.environ["GEMINI_TEMPERATURE"] = str(gemini_temp)

        if gemini_key:
            try:
                from google import genai
                client = genai.Client(api_key=gemini_key)
                _test = client.models.generate_content(
                    model=gemini_model,
                    contents="Say OK",
                    config={"max_output_tokens": 5},
                )
                if _test.text:
                    st.success(f"Gemini connected ({gemini_model})")
                else:
                    st.warning("Gemini returned empty response.")
            except Exception as e:
                st.warning(f"Gemini error: {e}")
        st.caption("Get a key at [Google AI Studio](https://aistudio.google.com/apikey)")

    elif llm_provider == "OpenAI":
        openai_key = st.text_input("OPENAI_API_KEY", type="password")
        openai_model = st.text_input("OPENAI_MODEL", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
        if openai_model:
            os.environ["OPENAI_MODEL"] = openai_model

    elif llm_provider == "Ollama":
        ollama_url = st.text_input("OLLAMA_URL", value=os.getenv("OLLAMA_URL", "http://localhost:11434"))
        ollama_model = st.text_input("OLLAMA_MODEL", value=os.getenv("OLLAMA_MODEL", "llama3.1:latest"))
        ollama_timeout = st.number_input(
            "OLLAMA_TIMEOUT (seconds)",
            min_value=10,
            max_value=600,
            value=int(os.getenv("OLLAMA_TIMEOUT", "180")),
            step=10,
        )
        ollama_max_tokens = st.number_input(
            "OLLAMA_MAX_TOKENS",
            min_value=16,
            max_value=2048,
            value=int(os.getenv("OLLAMA_MAX_TOKENS", "256")),
            step=16,
        )
        if ollama_url:
            os.environ["OLLAMA_URL"] = ollama_url
        if ollama_model:
            os.environ["OLLAMA_MODEL"] = ollama_model
        os.environ["OLLAMA_TIMEOUT"] = str(int(ollama_timeout))
        os.environ["OLLAMA_MAX_TOKENS"] = str(int(ollama_max_tokens))

        try:
            tags = requests.get(f"{ollama_url}/api/tags", timeout=3).json()
            model_names = {m.get("name") for m in (tags.get("models") or [])}
            if model_names:
                st.success("Ollama connected")
            if ollama_model and ollama_model not in model_names and f"{ollama_model}:latest" not in model_names:
                st.warning(
                    f"Model '{ollama_model}' not found in Ollama. Run: `ollama pull {ollama_model}`",
                )
        except Exception:
            st.warning("Ollama not reachable at this URL. Make sure Ollama is running.")

    st.caption("Tip: If provider is None, the app will show retrieved passages only.")


@st.cache_resource
def _embedder() -> Embedder:
    return Embedder()


def _build_index(passages: List[str]) -> FaissIndex:
    embedder = _embedder()
    vecs = embedder.embed(passages)
    index = FaissIndex(dim=vecs.shape[1])
    index.add(vecs, passages)
    return index


if "chat" not in st.session_state:
    st.session_state.chat = []  # list[dict]

if "index" not in st.session_state:
    st.session_state.index = None

if "passages" not in st.session_state:
    st.session_state.passages = None

if "finance" not in st.session_state:
    st.session_state.finance = None

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None


col_left, col_right = st.columns([1.15, 0.85], gap="large")

with col_left:
    st.subheader("Chat")
    if not uploaded:
        st.info("Upload a PDF to start.")
    else:
        file_bytes = uploaded.getvalue()
        current_hash = hashlib.md5(file_bytes).hexdigest()

        needs_rebuild = (
            st.session_state.pdf_hash != current_hash
            or st.session_state.index is None
            or st.session_state.passages is None
        )

        if needs_rebuild:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(file_bytes)
                st.session_state.pdf_path = f.name
                st.session_state.pdf_hash = current_hash

            with st.spinner("Extracting PDF text..."):
                pages = extract_pdf_text(st.session_state.pdf_path, max_pages=int(max_pages))
                passages = page_chunks_to_passages(pages)
                st.session_state.passages = passages

            with st.spinner("Building vector index..."):
                st.session_state.index = _build_index(passages)

            with st.spinner("Scanning PDF tables for financial data..."):
                raw_tables = extract_pdf_tables(st.session_state.pdf_path, max_pages=int(max_pages))
                tables_only = [t for (_page, t) in raw_tables]
                parsed = parse_financial_tables(tables_only)
                st.session_state.finance = parsed

            st.session_state.chat = []

        st.success(
            f"Indexed {len(st.session_state.passages or [])} passages. "
            + ("Financial table detected." if st.session_state.finance else "No financial table detected.")
        )

        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        question = st.chat_input("Ask a question about the PDF...")
        if question:
            st.session_state.chat.append({"role": "user", "content": question})

            question_norm = normalize_user_question(question)

            embedder = _embedder()
            qv = embedder.embed([question_norm or question])[0]
            retrieved = st.session_state.index.search(qv, top_k=top_k)

            answer, used_llm = answer_with_llm_or_extract(question_norm or question, retrieved)

            with st.chat_message("assistant"):
                st.markdown(answer)
                with st.expander("Retrieved passages"):
                    for p in retrieved:
                        st.markdown(f"**score** {p.score:.3f}  ")
                        st.markdown(p.text)
                        st.divider()
                st.caption("Answered with LLM" if used_llm else "No LLM configured (showing extractive context)")

            st.session_state.chat.append({"role": "assistant", "content": answer})

with col_right:
    st.subheader("Financial analysis")
    parsed = st.session_state.finance
    if not uploaded:
        st.caption("Upload a PDF to enable analysis.")
    elif not parsed:
        st.warning(
            "No financial table detected yet. If your PDF is a statement/table, try increasing max pages or ensure it contains selectable text."
        )
    else:
        t = totals(parsed)
        st.metric("Rows", t["rows"])
        st.metric("Sum", f"{t['sum']:.2f}")
        st.metric("Income (+)", f"{t['income_sum_pos']:.2f}")
        st.metric("Expense (-)", f"{t['expense_sum_neg']:.2f}")

        st.divider()
        st.caption(
            f"Detected columns: datetime='{parsed.datetime_col}', amount='{parsed.amount_col}', category='{parsed.category_col or 'N/A'}'"
        )

        freq = st.selectbox("Aggregate by", ["Hourly", "Daily", "Monthly"], index=1)
        freq_map = {"Hourly": "H", "Daily": "D", "Monthly": "M"}
        agg = aggregate_finance(parsed, freq=freq_map[freq])

        st.markdown("**Time-series totals**")
        st.dataframe(agg, use_container_width=True, height=240)

        fig_line = px.line(agg, x="period", y="total_amount", title=f"Total amount ({freq})")
        st.plotly_chart(fig_line, use_container_width=True)

        pie_df, pie_title = pie_breakdown(parsed)
        fig_pie = px.pie(pie_df, names="label", values="value", title=pie_title)
        st.plotly_chart(fig_pie, use_container_width=True)

        st.divider()
        st.markdown("**Finance assistant**")
        st.caption(
            "Type what you want (examples: 'bar chart by category', 'pie chart', 'trend monthly', 'top expenses')."
        )

        if "finance_query" not in st.session_state:
            st.session_state.finance_query = ""

        with st.form("finance_form", clear_on_submit=False):
            finance_q = st.text_input(
                "Ask about the detected financial table",
                value=st.session_state.finance_query,
                placeholder="e.g., bar chart by category; top 10 expenses; monthly trend",
            )
            submitted = st.form_submit_button("Run")

        if submitted and finance_q.strip():
            st.session_state.finance_query = finance_q
            intent = parse_finance_request(finance_q)

            # Always show advanced numeric summary (finance is the main goal).
            s = advanced_summary(parsed)
            stats_df = pd.DataFrame(
                [
                    {
                        "metric": "net_sum",
                        "value": s["net_sum"],
                    },
                    {"metric": "abs_sum", "value": s["abs_sum"]},
                    {"metric": "mean", "value": s["mean"]},
                    {"metric": "median", "value": s["median"]},
                    {"metric": "std", "value": s["std"]},
                    {"metric": "min", "value": s["min"]},
                    {"metric": "max", "value": s["max"]},
                    {"metric": "income_count", "value": s["income_count"]},
                    {"metric": "expense_count", "value": s["expense_count"]},
                ]
            )
            st.markdown("**Advanced stats**")
            st.dataframe(stats_df, use_container_width=True, hide_index=True, height=280)

            ql = finance_q.lower()
            if any(k in ql for k in ("top expense", "top expenses", "largest expense", "largest expenses", "top income", "top incomes")):
                inc, exp = top_transactions(parsed, n=10)
                st.markdown("**Top income transactions**")
                st.dataframe(inc, use_container_width=True, height=220)
                st.markdown("**Top expense transactions**")
                st.dataframe(exp, use_container_width=True, height=220)

            chart = intent.get("chart", "none")
            group = intent.get("group", "auto")

            if chart == "bar":
                if group == "category" and parsed.category_col:
                    bar_df, bar_title = category_breakdown(parsed, top_n=12)
                    if not bar_df.empty:
                        fig = px.bar(bar_df, x="label", y="value", title=bar_title)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No category column detected to build a category bar chart.")
                else:
                    freq_map = {"hour": "H", "day": "D", "month": "M"}
                    freq = freq_map.get(group, "D")
                    agg2 = aggregate_finance(parsed, freq=freq)
                    fig = px.bar(agg2, x="period", y="total_amount", title=f"Total amount ({freq})")
                    st.plotly_chart(fig, use_container_width=True)

            elif chart == "pie":
                pie_df2, pie_title2 = pie_breakdown(parsed)
                fig = px.pie(pie_df2, names="label", values="value", title=pie_title2)
                st.plotly_chart(fig, use_container_width=True)

            elif chart == "line":
                freq_map = {"hour": "H", "day": "D", "month": "M"}
                freq = freq_map.get(group, "D")
                agg2 = aggregate_finance(parsed, freq=freq)
                fig = px.line(agg2, x="period", y="total_amount", title=f"Total amount ({freq})")
                st.plotly_chart(fig, use_container_width=True)
