# Product Requirements Document (PRD)

## PDF RAG Chatbot + Financial Analysis Platform

| Field            | Detail                                      |
| ---------------- | ------------------------------------------- |
| **Product Name** | PDF RAG Chatbot + Financial Analysis        |
| **Version**      | 1.0                                         |
| **Author**       | Engineering Team                            |
| **Date**         | February 6, 2026                            |
| **Status**       | In Development                              |
| **Stack**        | Python · Streamlit · FAISS · Sentence-Transformers · Plotly |

---

## 1. Executive Summary

A Streamlit-based web application that allows users to **upload PDF documents**, perform **Retrieval-Augmented Generation (RAG) chat** against the document content, and automatically detect and **analyze financial tables** embedded within the PDF. The system combines local vector search with optional LLM integration (OpenAI / Ollama) to provide intelligent, context-aware answers grounded in document content.

---

## 2. Problem Statement

Users working with PDF documents—particularly financial statements, invoices, and transaction reports—need a fast, interactive way to:

1. **Ask natural-language questions** about the contents of a PDF without reading it end-to-end.
2. **Extract, aggregate, and visualize financial data** (income, expenses, time-series trends, category breakdowns) locked inside PDF tables.
3. **Operate without mandatory cloud dependencies**—the tool must work fully offline with a local LLM (Ollama) or even without any LLM at all (extractive fallback mode).

---

## 3. Goals & Success Criteria

| Goal | Success Metric |
| ---- | -------------- |
| Accurate document Q&A | Top-6 retrieved passages contain the answer ≥ 85% of the time |
| Financial table auto-detection | Correctly identifies datetime, amount, and category columns in ≥ 90% of standard bank/financial PDFs |
| Low-latency local usage | Index build + first query completes in < 15 s for a 50-page PDF on commodity hardware |
| Flexible LLM support | Works with OpenAI API, local Ollama, or zero-LLM extractive mode seamlessly |
| Self-contained deployment | Single `streamlit run app.py` command; no external databases or services required |

---

## 4. Target Users

| Persona | Description |
| ------- | ----------- |
| **Financial Analyst** | Uploads bank statements, P&L reports, or invoices; needs quick totals, trend charts, and category breakdowns |
| **Researcher / Student** | Uploads academic or technical PDFs; asks questions and gets passage-level answers |
| **Small Business Owner** | Uploads monthly statements; wants income vs. expense summaries and top transaction lists |
| **Developer / Power User** | Runs the app locally with Ollama for fully private, offline document analysis |

---

## 5. System Architecture

### 5.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Streamlit Frontend (app.py)          │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │  PDF Upload  │  │  Chat Panel  │  │  Finance    │ │
│  │  & Settings  │  │  (RAG Q&A)   │  │  Dashboard  │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘ │
└─────────┼─────────────────┼─────────────────┼────────┘
          │                 │                 │
     ┌────▼────┐     ┌─────▼──────┐   ┌──────▼────────┐
     │pdf_utils│     │  rag.py     │   │finance_       │
     │  .py    │     │ Embedder +  │   │analysis.py    │
     │         │     │ FaissIndex  │   │               │
     └────┬────┘     └─────┬──────┘   └───────────────┘
          │                │
    ┌─────▼──────┐   ┌─────▼──────────────┐
    │ pdfplumber │   │ sentence-transformers│
    │            │   │ FAISS (CPU)          │
    └────────────┘   │ OpenAI / Ollama (opt)│
                     └─────────────────────┘
```

### 5.2 Module Breakdown

| Module | File | Responsibility |
| ------ | ---- | -------------- |
| **UI / Orchestration** | `app.py` (303 lines) | Streamlit layout, session state management, PDF upload handling, chat loop, financial dashboard rendering |
| **PDF Processing** | `pdf_utils.py` (76 lines) | Text extraction via pdfplumber, table extraction, text chunking (1200-char chunks, 150-char overlap), passage tagging with page numbers |
| **RAG Engine** | `rag.py` (275 lines) | Sentence-transformer embeddings, FAISS inner-product index, LLM prompt construction, OpenAI & Ollama integrations, extractive fallback, intent heuristics |
| **Finance Analysis** | `finance_analysis.py` (291 lines) | Heuristic column detection (datetime/amount/category), DataFrame parsing, aggregation (H/D/M), statistical summaries, pie/bar/line chart data, top transactions, NL intent parser |

---

## 6. Functional Requirements

### 6.1 PDF Upload & Processing

| ID | Requirement | Priority |
| -- | ----------- | -------- |
| FR-01 | User can upload a single PDF file (up to 500 pages configurable) | P0 |
| FR-02 | System extracts text page-by-page using pdfplumber | P0 |
| FR-03 | Text is chunked into ~1200-char passages with 150-char overlap and tagged with `[page N]` | P0 |
| FR-04 | System detects when a new/different PDF is uploaded (MD5 hash comparison) and rebuilds the index | P0 |
| FR-05 | System extracts tabular data from each PDF page for financial analysis | P0 |

### 6.2 RAG Chat

| ID | Requirement | Priority |
| -- | ----------- | -------- |
| FR-06 | Passages are embedded using `sentence-transformers/all-MiniLM-L6-v2` (384-dim) | P0 |
| FR-07 | Embeddings are stored in a FAISS `IndexFlatIP` (inner-product / cosine similarity) index | P0 |
| FR-08 | User can configure Top-K retrieval (3–12, default 6) | P1 |
| FR-09 | Retrieved passages are displayed in an expandable section with similarity scores | P1 |
| FR-10 | User questions undergo normalization (whitespace collapse, common typo correction) | P1 |
| FR-11 | System detects underspecified "summarize" requests and asks a clarifying question | P2 |
| FR-12 | System detects "financial concepts" questions and returns term lists from context | P2 |

### 6.3 LLM Integration

| ID | Requirement | Priority |
| -- | ----------- | -------- |
| FR-13 | Support **OpenAI** API (configurable model, default `gpt-4o-mini`) | P0 |
| FR-14 | Support **Ollama** local LLM (configurable URL, model, timeout 10–600s, max tokens 16–2048) | P0 |
| FR-15 | Ollama connectivity check on startup with model availability warning | P1 |
| FR-16 | Graceful fallback to extractive mode (top-3 passages) when no LLM is available | P0 |
| FR-17 | LLM prompt instructs model to use context only, cite `[page N]` tags, and avoid fabrication | P0 |

### 6.4 Financial Analysis

| ID | Requirement | Priority |
| -- | ----------- | -------- |
| FR-18 | Auto-detect datetime, amount, and category columns using heuristic matching (name hints + statistical detection) | P0 |
| FR-19 | Parse monetary values including `$`, commas, and parenthesized negatives | P0 |
| FR-20 | Display summary metrics: row count, sum, income (+), expense (-), mean | P0 |
| FR-21 | Aggregate by Hourly / Daily / Monthly with interactive selector | P0 |
| FR-22 | Render time-series line chart (Plotly) | P0 |
| FR-23 | Render pie chart breakdown (by category or by month if no category) | P0 |
| FR-24 | Finance assistant: natural-language input for chart type and grouping | P1 |
| FR-25 | Advanced stats panel: net sum, abs sum, mean, median, std, min, max, income/expense counts | P1 |
| FR-26 | Top N income/expense transactions display | P1 |
| FR-27 | Bar chart by category or by time period | P1 |

---

## 7. Non-Functional Requirements

| ID | Requirement | Category |
| -- | ----------- | -------- |
| NFR-01 | App runs fully offline when using Ollama (no cloud dependency) | Privacy |
| NFR-02 | No user data is stored on disk beyond the temp PDF file during session | Privacy |
| NFR-03 | Embedding model loads once and is cached across sessions (`@st.cache_resource`) | Performance |
| NFR-04 | FAISS index is rebuilt only when the PDF changes (MD5-based deduplication) | Performance |
| NFR-05 | Single-command startup: `streamlit run app.py` | Usability |
| NFR-06 | Works on Windows, macOS, and Linux | Portability |
| NFR-07 | All dependencies are pip-installable (`requirements.txt`) | Deployment |
| NFR-08 | Responsive two-column layout (chat left, finance right) | UX |

---

## 8. Technical Specifications

### 8.1 Dependencies

| Package | Version | Purpose |
| ------- | ------- | ------- |
| `streamlit` | ≥ 1.32 | Web UI framework |
| `pdfplumber` | ≥ 0.11 | PDF text & table extraction |
| `pypdf` | ≥ 4.0 | PDF parsing support |
| `pandas` | ≥ 2.2 | DataFrame operations & financial analysis |
| `numpy` | ≥ 1.26 | Numerical operations |
| `faiss-cpu` | ≥ 1.8 | Vector similarity search |
| `sentence-transformers` | ≥ 2.6 | Text embedding (all-MiniLM-L6-v2) |
| `plotly` | ≥ 5.19 | Interactive charts |
| `python-dateutil` | ≥ 2.9 | Date parsing |
| `requests` | ≥ 2.31 | HTTP client (Ollama API) |
| `openai` | ≥ 1.40 | OpenAI API client |

### 8.2 Data Flow

```
PDF Upload
  ├─► extract_pdf_text() ─► page_chunks_to_passages() ─► Embedder.embed() ─► FaissIndex
  │                                                                              │
  │                                                                     User Query
  │                                                                         │
  │                                                          normalize_user_question()
  │                                                                         │
  │                                                           Embedder.embed(query)
  │                                                                         │
  │                                                         FaissIndex.search(top_k)
  │                                                                         │
  │                                                     answer_with_llm_or_extract()
  │                                                        ├── OpenAI ──► answer
  │                                                        ├── Ollama ──► answer
  │                                                        └── Fallback ► passages
  │
  └─► extract_pdf_tables() ─► parse_financial_tables()
                                     │
                          FinanceParseResult (DataFrame)
                                     │
              ┌──────────┬───────────┼───────────┬──────────────┐
           totals()  aggregate()  pie_breakdown()  top_transactions()
                                                      advanced_summary()
```

### 8.3 Key Data Structures

| Structure | Location | Fields |
| --------- | -------- | ------ |
| `PdfPageChunk` | `pdf_utils.py` | `page: int`, `text: str` |
| `RetrievedPassage` | `rag.py` | `text: str`, `score: float` |
| `FinanceParseResult` | `finance_analysis.py` | `df: DataFrame`, `datetime_col: str`, `amount_col: str`, `category_col: Optional[str]` |

### 8.4 Embedding & Retrieval Configuration

| Parameter | Value |
| --------- | ----- |
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding Dimension | 384 |
| Similarity Metric | Inner Product (cosine, normalized) |
| Chunk Size | 1200 characters |
| Chunk Overlap | 150 characters |
| Default Top-K | 6 |

---

## 9. User Interface Layout

```
┌─────────────────────────────────────────────────────────────┐
│  PDF RAG Chatbot + Financial Analysis              [title]  │
├──────────────┬──────────────────────────────────────────────┤
│  SIDEBAR     │    LEFT COLUMN (55%)    │  RIGHT COLUMN (45%)│
│              │                         │                    │
│  Upload PDF  │    Chat messages        │  Financial summary │
│  Max pages   │    (user ↔ assistant)   │  Metrics cards     │
│  Top-K       │                         │  Aggregate selector│
│  LLM provider│    Retrieved passages   │  Line chart        │
│  API keys    │    (expandable)         │  Pie chart         │
│              │                         │  Finance assistant  │
│              │    Chat input box       │  (NL query form)   │
│              │                         │  Advanced stats    │
│              │                         │  Top transactions  │
│              │                         │  Dynamic charts    │
└──────────────┴─────────────────────────┴────────────────────┘
```

---

## 10. Session State Management

| Key | Type | Purpose |
| --- | ---- | ------- |
| `chat` | `list[dict]` | Chat message history (`role`, `content`) |
| `index` | `FaissIndex` | Current FAISS vector index |
| `passages` | `list[str]` | Chunked and tagged text passages |
| `finance` | `FinanceParseResult` | Parsed financial data |
| `pdf_hash` | `str` | MD5 hash of current PDF (rebuild detection) |
| `pdf_path` | `str` | Temp file path of uploaded PDF |
| `finance_query` | `str` | Last finance assistant query |

---

## 11. Intent Detection & Heuristics

The system includes lightweight rule-based intent detection:

| Heuristic | Trigger | Behavior |
| --------- | ------- | -------- |
| Underspecified summary | "summarize" without scope markers (page, section, chapter, etc.) | Returns clarifying question instead of answer |
| Financial concepts | "financial concept(s)" or "financial" + terms/topics | Extracts finance terms from context and lists them |
| Finance assistant parser | Keywords: bar/pie/line chart, category/hour/day/month | Returns `{chart, group}` intent dict for chart rendering |
| Typo correction | Dictionary of common misspellings (e.g., "finacial" → "financial") | Applied before embedding query |

---

## 12. Limitations & Known Constraints

| # | Limitation |
| - | ---------- |
| 1 | Single PDF at a time (no multi-document RAG) |
| 2 | Only supports PDFs with selectable/extractable text (not scanned/OCR) |
| 3 | Financial table detection is heuristic-based; may miss unconventional table formats |
| 4 | FAISS index is in-memory only; not persisted across sessions |
| 5 | No authentication or multi-user support |
| 6 | Embedding model is fixed (`all-MiniLM-L6-v2`); not user-configurable |
| 7 | No streaming LLM responses (waits for full completion) |
| 8 | Ollama timeout can be up to 600s for slow hardware, but UX suffers |

---

## 13. Future Enhancements (Roadmap)

| Priority | Enhancement | Description |
| -------- | ----------- | ----------- |
| P1 | **Multi-PDF support** | Upload and query across multiple documents simultaneously |
| P1 | **OCR integration** | Add Tesseract/EasyOCR for scanned PDF support |
| P1 | **Persistent vector store** | Save FAISS index to disk for large documents |
| P2 | **Streaming LLM responses** | Stream tokens from OpenAI/Ollama for better UX |
| P2 | **Configurable embedding model** | Let users choose embedding models |
| P2 | **Export finance reports** | Export charts and summaries to PDF/Excel |
| P3 | **User authentication** | Multi-user support with session isolation |
| P3 | **Conversation memory** | Multi-turn context tracking with chat history in prompts |
| P3 | **Custom prompts** | Allow users to customize the system prompt |

---

## 14. Codebase Metrics

| Metric | Value |
| ------ | ----- |
| Total Python files | 4 |
| Total lines of code | ~945 |
| `app.py` | 303 lines |
| `rag.py` | 275 lines |
| `finance_analysis.py` | 291 lines |
| `pdf_utils.py` | 76 lines |
| External dependencies | 11 packages |
| LLM integrations | 3 (OpenAI, Ollama, Extractive fallback) |

---

## 15. How to Run

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set LLM provider
#    OpenAI: export OPENAI_API_KEY=sk-...
#    Ollama: make sure Ollama is running locally

# 4. Launch
streamlit run app.py
```

---

## 16. Risk Assessment

| Risk | Impact | Mitigation |
| ---- | ------ | ---------- |
| PDF with no extractable text | No content indexed, empty chat | Display clear warning; recommend OCR-capable PDFs |
| Ollama model not pulled | LLM calls fail silently | Connectivity check on sidebar with `ollama pull` instruction |
| Large PDF (500+ pages) | Slow indexing, high memory | Configurable max pages; chunking keeps passage count manageable |
| Unconventional financial table format | Column detection fails | Graceful fallback: "No financial table detected" message |
| OpenAI API rate limits / errors | Answer generation fails | Falls through to Ollama, then to extractive mode |

---

*End of PRD*
