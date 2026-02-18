# Product Requirements Document (PRD) — v2

## PDF RAG Chatbot + Financial Analysis (HTML/CSS/JS + Gemini 2.5 Flash)

| Field            | Detail                                                        |
| ---------------- | ------------------------------------------------------------- |
| **Product Name** | PDF RAG Chatbot + Financial Analysis v2                       |
| **Version**      | 2.0                                                           |
| **Author**       | Engineering Team                                              |
| **Date**         | February 6, 2026                                              |
| **Status**       | Planned                                                       |
| **Stack**        | HTML · CSS · JavaScript · Python (FastAPI) · Gemini 2.5 Flash · FAISS |

---

## 1. Executive Summary

A complete redesign of the PDF RAG Chatbot moving from a Streamlit-only app to a **modern HTML/CSS/JS frontend** with a **Python FastAPI backend**. The LLM provider is replaced with **Google Gemini 2.5 Flash** as the sole AI engine. Users configure their Gemini API key through a built-in **Settings page**, and all chat, retrieval, and financial analysis features are powered by the Gemini 2.5 Flash model.

---

## 2. What Changed from v1

| Aspect | v1 (Current) | v2 (New) |
| ------ | ------------ | -------- |
| Frontend | Streamlit (Python) | HTML + CSS + Vanilla JS |
| LLM Provider | OpenAI / Ollama / Extractive fallback | **Google Gemini 2.5 Flash** only |
| API Key Setup | Sidebar text input per provider | Dedicated **Settings page** with persistent config |
| Backend | Streamlit runtime | **FastAPI** REST API server |
| Communication | In-process (Streamlit session) | Fetch API (JSON over HTTP) |
| Styling | Streamlit defaults | Custom CSS (dark/light theme) |
| Charts | Plotly (Python-rendered) | **Chart.js** (client-side rendering) |
| State | Streamlit `session_state` | Server-side session + `localStorage` |

---

## 3. Goals & Success Criteria

| Goal | Success Metric |
| ---- | -------------- |
| Modern, responsive UI | Works on desktop and tablet (≥768px); smooth interactions |
| Gemini 2.5 Flash integration | All LLM calls routed through Gemini; < 5s average response time |
| Settings persistence | API key saved in `localStorage`; survives page refresh |
| API-first architecture | All features accessible via documented REST endpoints |
| Zero-framework frontend | No React/Vue/Angular — plain HTML/CSS/JS for simplicity |

---

## 4. Target Users

| Persona | Description |
| ------- | ----------- |
| **Financial Analyst** | Uploads bank statements; needs quick totals, charts, and Gemini-powered summaries |
| **Researcher / Student** | Uploads academic PDFs; asks questions via Gemini-powered RAG chat |
| **Small Business Owner** | Uploads invoices/statements; wants income vs. expense analysis |
| **Developer** | Runs the app locally; appreciates clean API endpoints and simple frontend |

---

## 5. System Architecture

### 5.1 High-Level Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    FRONTEND (HTML/CSS/JS)                  │
│                                                           │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐│
│  │  Settings   │  │  Chat Page │  │  Finance Dashboard   ││
│  │  Page       │  │  (RAG Q&A) │  │  (Charts + Tables)   ││
│  │  - API Key  │  │            │  │                      ││
│  │  - Model    │  │            │  │                      ││
│  └─────┬──────┘  └─────┬──────┘  └──────────┬───────────┘│
└────────┼───────────────┼────────────────────┼────────────┘
         │               │                    │
         │        fetch() / JSON              │
         │               │                    │
┌────────▼───────────────▼────────────────────▼────────────┐
│                  BACKEND (Python FastAPI)                  │
│                                                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ /api/    │  │ /api/    │  │ /api/    │  │ /api/    │ │
│  │ settings │  │ upload   │  │ chat     │  │ finance  │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘ │
│       │              │             │              │       │
│  ┌────▼──────────────▼─────────────▼──────────────▼────┐ │
│  │              Core Services                          │ │
│  │  pdf_utils.py  │  rag.py  │  finance_analysis.py    │ │
│  │                │          │                         │ │
│  └────────────────┼──────────┼─────────────────────────┘ │
│                   │          │                            │
│            ┌──────▼──┐  ┌───▼───────────┐                │
│            │  FAISS   │  │ Google Gemini │                │
│            │  Index   │  │ 2.5 Flash API │                │
│            └─────────┘  └───────────────┘                │
└──────────────────────────────────────────────────────────┘
```

### 5.2 File Structure

```
rag-chatbot-v2/
├── frontend/
│   ├── index.html              # Main entry, navigation shell
│   ├── settings.html           # Settings page (Gemini API key config)
│   ├── chat.html               # Chat page (RAG Q&A)
│   ├── finance.html            # Finance dashboard
│   ├── css/
│   │   ├── style.css           # Global styles, layout, theme
│   │   ├── chat.css            # Chat bubble styles
│   │   ├── finance.css         # Dashboard & chart styles
│   │   └── settings.css        # Settings form styles
│   └── js/
│       ├── app.js              # Navigation, global state, init
│       ├── settings.js         # API key management, validation
│       ├── chat.js             # Chat logic, message rendering
│       ├── finance.js          # Chart rendering, table display
│       ├── api.js              # Fetch wrapper for all API calls
│       └── utils.js            # Helpers (formatCurrency, debounce, etc.)
│
├── backend/
│   ├── main.py                 # FastAPI app, CORS, static file serving
│   ├── routes/
│   │   ├── settings.py         # POST /api/settings — validate & store Gemini key
│   │   ├── upload.py           # POST /api/upload — PDF upload & indexing
│   │   ├── chat.py             # POST /api/chat — RAG query + Gemini answer
│   │   └── finance.py          # GET /api/finance/* — aggregation, charts, stats
│   ├── services/
│   │   ├── gemini_service.py   # Gemini 2.5 Flash client wrapper
│   │   ├── rag_service.py      # Embedder + FAISS index management
│   │   ├── pdf_service.py      # PDF text & table extraction
│   │   └── finance_service.py  # Financial analysis logic
│   └── models/
│       └── schemas.py          # Pydantic request/response models
│
├── requirements.txt
└── README.md
```

---

## 6. Gemini 2.5 Flash Integration

### 6.1 Model Specification

| Parameter | Value |
| --------- | ----- |
| **Provider** | Google AI (Generative AI) |
| **Model ID** | `gemini-2.5-flash` |
| **SDK** | `google-genai` (Python) |
| **API Key** | User-provided via Settings page |
| **Temperature** | 0.2 (for factual, grounded answers) |
| **Max Output Tokens** | 1024 (configurable) |
| **Safety Settings** | Default (block harmful content) |

### 6.2 Gemini Service (`gemini_service.py`)

```python
from google import genai

class GeminiService:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": 0.2,
                "max_output_tokens": max_tokens,
            }
        )
        return response.text

    def validate_key(self) -> bool:
        """Quick validation by sending a minimal request."""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents="Say OK",
                config={"max_output_tokens": 5}
            )
            return bool(response.text)
        except Exception:
            return False
```

### 6.3 RAG Prompt Template (Gemini-optimized)

```
You are a helpful PDF RAG assistant powered by Gemini 2.5 Flash.

RULES:
- Answer ONLY based on the provided context passages from the PDF.
- If the answer is not in the context, say "I couldn't find this in the PDF."
- When citing information, reference the [page N] tags from the passages.
- Keep answers concise and accurate.
- Do NOT fabricate facts not present in the context.

CONTEXT:
{retrieved_passages}

USER QUESTION: {question}

ANSWER:
```

---

## 7. Settings Page (Gemini API Initialization)

### 7.1 UI Design

```
┌─────────────────────────────────────────────────┐
│  ⚙️  Settings                                   │
├─────────────────────────────────────────────────┤
│                                                 │
│  Gemini API Configuration                       │
│  ─────────────────────────                      │
│                                                 │
│  API Key:                                       │
│  ┌─────────────────────────────────┐            │
│  │ ●●●●●●●●●●●●●●●●●●●●●●●●●●●●  │  👁️ Show  │
│  └─────────────────────────────────┘            │
│                                                 │
│  Model:                                         │
│  ┌─────────────────────────────────┐            │
│  │ gemini-2.5-flash           ▼   │            │
│  └─────────────────────────────────┘            │
│                                                 │
│  Max Output Tokens:                             │
│  ┌─────────────────────────────────┐            │
│  │ 1024                           │            │
│  └─────────────────────────────────┘            │
│                                                 │
│  Temperature:                                   │
│  ┌─────────────────────────────────┐            │
│  │ 0.2               ◄━━━━━●━━━► │            │
│  └─────────────────────────────────┘            │
│                                                 │
│  ┌──────────────┐  ┌─────────────────┐         │
│  │  ✓ Save Key  │  │  🔄 Test Key    │         │
│  └──────────────┘  └─────────────────┘         │
│                                                 │
│  Status: ✅ Connected — Gemini 2.5 Flash ready  │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 7.2 Settings Flow

```
User opens Settings page
        │
        ▼
  Loads API key from localStorage (if any)
        │
        ▼
  User enters/edits API key
        │
        ▼
  Clicks "Save Key"
        │
        ├──► localStorage.setItem("gemini_api_key", key)
        │
        ▼
  Clicks "Test Key"
        │
        ▼
  POST /api/settings/validate  { api_key: "..." }
        │
        ├── 200 OK ──► ✅ "Connected — Gemini 2.5 Flash ready"
        │
        └── 401/500 ──► ❌ "Invalid API key or connection error"
```

### 7.3 Settings JavaScript (`settings.js`)

```javascript
const API_KEY_STORAGE = "gemini_api_key";
const MODEL_STORAGE = "gemini_model";

function loadSettings() {
    const key = localStorage.getItem(API_KEY_STORAGE) || "";
    const model = localStorage.getItem(MODEL_STORAGE) || "gemini-2.5-flash";
    document.getElementById("api-key-input").value = key;
    document.getElementById("model-select").value = model;
    updateStatus(key ? "Key loaded from storage" : "No API key set", key ? "info" : "warn");
}

async function saveSettings() {
    const key = document.getElementById("api-key-input").value.trim();
    const model = document.getElementById("model-select").value;
    if (!key) {
        updateStatus("API key cannot be empty", "error");
        return;
    }
    localStorage.setItem(API_KEY_STORAGE, key);
    localStorage.setItem(MODEL_STORAGE, model);
    updateStatus("Settings saved", "success");
}

async function testConnection() {
    const key = localStorage.getItem(API_KEY_STORAGE);
    if (!key) {
        updateStatus("Save your API key first", "error");
        return;
    }
    updateStatus("Testing connection...", "info");
    try {
        const res = await fetch("/api/settings/validate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ api_key: key }),
        });
        const data = await res.json();
        if (data.valid) {
            updateStatus("✅ Connected — Gemini 2.5 Flash ready", "success");
        } else {
            updateStatus("❌ Invalid API key or model not accessible", "error");
        }
    } catch (err) {
        updateStatus(`❌ Connection error: ${err.message}`, "error");
    }
}
```

---

## 8. Functional Requirements

### 8.1 Settings & Gemini Initialization

| ID | Requirement | Priority |
| -- | ----------- | -------- |
| FR-01 | Settings page with form fields: API Key (password), Model selector, Max Tokens, Temperature | P0 |
| FR-02 | API key stored in browser `localStorage` — persists across refreshes | P0 |
| FR-03 | "Save" button stores all settings to `localStorage` | P0 |
| FR-04 | "Test Key" button calls `POST /api/settings/validate` to verify Gemini connectivity | P0 |
| FR-05 | Show/hide toggle for API key visibility | P1 |
| FR-06 | Status indicator — green (connected), yellow (untested), red (failed) | P1 |
| FR-07 | API key sent in request header (`X-Gemini-Key`) for every backend call | P0 |
| FR-08 | Backend initializes `google.genai.Client` with the provided key per-request | P0 |
| FR-09 | Default model locked to `gemini-2.5-flash` with option to change | P0 |

### 8.2 PDF Upload & Processing

| ID | Requirement | Priority |
| -- | ----------- | -------- |
| FR-10 | Drag-and-drop PDF upload zone + file picker button | P0 |
| FR-11 | Upload via `POST /api/upload` (multipart form data) | P0 |
| FR-12 | Backend extracts text + tables using pdfplumber | P0 |
| FR-13 | Text chunked into ~1200-char passages with `[page N]` tags | P0 |
| FR-14 | Passages embedded with `all-MiniLM-L6-v2` and indexed in FAISS | P0 |
| FR-15 | Response returns: passage count, whether financial table detected, session ID | P0 |
| FR-16 | Progress bar during upload and indexing | P1 |

### 8.3 RAG Chat

| ID | Requirement | Priority |
| -- | ----------- | -------- |
| FR-17 | Chat interface with user/assistant message bubbles (CSS styled) | P0 |
| FR-18 | User sends question via `POST /api/chat` with query + session ID + API key | P0 |
| FR-19 | Backend retrieves top-K passages from FAISS, builds prompt, calls Gemini 2.5 Flash | P0 |
| FR-20 | Response includes: Gemini answer, retrieved passages with scores | P0 |
| FR-21 | Expandable "Sources" section below each answer showing passage text + scores | P1 |
| FR-22 | Chat history maintained in JS array + rendered in DOM | P0 |
| FR-23 | "Clear Chat" button to reset conversation | P1 |
| FR-24 | Loading spinner while waiting for Gemini response | P1 |
| FR-25 | Auto-scroll to latest message | P1 |
| FR-26 | If no API key is configured, show banner: "Set your Gemini API key in Settings" | P0 |

### 8.4 Financial Analysis Dashboard

| ID | Requirement | Priority |
| -- | ----------- | -------- |
| FR-27 | Summary cards: Total Rows, Net Sum, Income (+), Expense (-) | P0 |
| FR-28 | Time-series line chart (Chart.js) with period selector (Hourly/Daily/Monthly) | P0 |
| FR-29 | Pie chart breakdown by category (or by month fallback) | P0 |
| FR-30 | Bar chart by category or time period | P1 |
| FR-31 | Advanced stats table: mean, median, std, min, max, income/expense counts | P1 |
| FR-32 | Top N income/expense transactions table | P1 |
| FR-33 | Finance NL query box — sends query to Gemini for smart financial insights | P1 |
| FR-34 | All chart data fetched via `GET /api/finance/*` endpoints | P0 |

---

## 9. API Endpoints

### 9.1 Settings

| Method | Endpoint | Body | Response |
| ------ | -------- | ---- | -------- |
| `POST` | `/api/settings/validate` | `{ "api_key": "...", "model": "gemini-2.5-flash" }` | `{ "valid": true/false, "message": "..." }` |

### 9.2 Upload

| Method | Endpoint | Body | Response |
| ------ | -------- | ---- | -------- |
| `POST` | `/api/upload` | `multipart/form-data` (file + max_pages) | `{ "session_id": "...", "passages": 142, "finance_detected": true }` |

### 9.3 Chat

| Method | Endpoint | Body | Response |
| ------ | -------- | ---- | -------- |
| `POST` | `/api/chat` | `{ "session_id": "...", "question": "...", "top_k": 6 }` | `{ "answer": "...", "sources": [{ "text": "...", "score": 0.87 }] }` |

Headers: `X-Gemini-Key: <api_key>`

### 9.4 Finance

| Method | Endpoint | Params | Response |
| ------ | -------- | ------ | -------- |
| `GET` | `/api/finance/summary` | `session_id` | `{ "rows": 150, "net_sum": -2340.50, ... }` |
| `GET` | `/api/finance/aggregate` | `session_id, freq=D` | `[{ "period": "2025-01", "total": 500 }, ...]` |
| `GET` | `/api/finance/pie` | `session_id, top_n=8` | `{ "labels": [...], "values": [...], "title": "..." }` |
| `GET` | `/api/finance/top` | `session_id, n=10` | `{ "income": [...], "expenses": [...] }` |
| `GET` | `/api/finance/stats` | `session_id` | `{ "mean": ..., "median": ..., "std": ..., ... }` |
| `POST` | `/api/finance/ask` | `{ "session_id": "...", "question": "..." }` | `{ "answer": "...", "chart_data": {...} }` |

---

## 10. Frontend Technical Specifications

### 10.1 HTML Pages

| Page | File | Purpose |
| ---- | ---- | ------- |
| Landing / Chat | `index.html` + `chat.html` | PDF upload + RAG chat interface |
| Settings | `settings.html` | Gemini API key configuration |
| Finance | `finance.html` | Financial analysis dashboard |

### 10.2 CSS Architecture

```css
/* style.css — Design Tokens */
:root {
    --bg-primary:    #0f172a;     /* Dark background */
    --bg-secondary:  #1e293b;
    --bg-card:       #334155;
    --text-primary:  #f1f5f9;
    --text-secondary:#94a3b8;
    --accent:        #3b82f6;     /* Blue accent */
    --accent-hover:  #2563eb;
    --success:       #22c55e;
    --warning:       #eab308;
    --error:         #ef4444;
    --border-radius: 12px;
    --shadow:        0 4px 6px -1px rgba(0,0,0,0.3);
    --font-family:   'Inter', system-ui, sans-serif;
}

/* Light theme override */
[data-theme="light"] {
    --bg-primary:    #f8fafc;
    --bg-secondary:  #e2e8f0;
    --bg-card:       #ffffff;
    --text-primary:  #0f172a;
    --text-secondary:#475569;
}
```

| Style File | Scope |
| ---------- | ----- |
| `style.css` | Global layout, navigation, theme variables, typography, responsive grid |
| `chat.css` | Chat bubbles, message list, input area, typing indicator |
| `finance.css` | Chart containers, metric cards, data tables |
| `settings.css` | Form styling, status badges, toggle switches |

### 10.3 JavaScript Modules

| File | Responsibility |
| ---- | -------------- |
| `app.js` | Page navigation (SPA-like), theme toggle, global init |
| `api.js` | Centralized `fetch()` wrapper — auto-injects `X-Gemini-Key` header from `localStorage` |
| `settings.js` | Load/save/test API key, model selection, status display |
| `chat.js` | Send messages, render bubbles, toggle sources, auto-scroll, loading states |
| `finance.js` | Fetch chart data, render Chart.js charts, populate tables |
| `utils.js` | `formatCurrency()`, `debounce()`, `escapeHtml()`, date formatters |

### 10.4 Client-side Libraries (CDN)

| Library | Version | Purpose |
| ------- | ------- | ------- |
| **Chart.js** | 4.x | Line, bar, pie charts |
| **Inter Font** | — | Typography (Google Fonts) |

No build tools, no bundlers — plain `<script>` tags and ES modules.

---

## 11. Backend Technical Specifications

### 11.1 FastAPI Server (`main.py`)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="PDF RAG Chatbot + Finance v2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# API routes
from routes import settings, upload, chat, finance
app.include_router(settings.router, prefix="/api/settings")
app.include_router(upload.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(finance.router, prefix="/api/finance")

# Serve frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
```

### 11.2 Dependencies (`requirements.txt`)

| Package | Version | Purpose |
| ------- | ------- | ------- |
| `fastapi` | ≥ 0.110 | Backend REST framework |
| `uvicorn` | ≥ 0.29 | ASGI server |
| `python-multipart` | ≥ 0.0.9 | File upload handling |
| `google-genai` | ≥ 1.0 | Gemini 2.5 Flash SDK |
| `pdfplumber` | ≥ 0.11 | PDF text & table extraction |
| `pandas` | ≥ 2.2 | Financial data analysis |
| `numpy` | ≥ 1.26 | Numerical operations |
| `faiss-cpu` | ≥ 1.8 | Vector similarity search |
| `sentence-transformers` | ≥ 2.6 | Text embeddings |
| `python-dateutil` | ≥ 2.9 | Date parsing |

### 11.3 Session Management

```
Server-side dict: sessions = {}

sessions[session_id] = {
    "passages": List[str],
    "index": FaissIndex,
    "finance": Optional[FinanceParseResult],
    "pdf_hash": str,
    "created_at": datetime,
}
```

Sessions are keyed by UUID, returned on upload, and required for chat and finance calls.

---

## 12. Gemini API Key Flow (End-to-End)

```
                    FRONTEND                                 BACKEND
                    ────────                                 ───────

1. User navigates to Settings page
2. Enters Gemini API key
3. Clicks "Save"
   └──► localStorage.setItem("gemini_api_key", key)

4. Clicks "Test Connection"
   └──► POST /api/settings/validate ──────────────► Receives { api_key }
        { api_key: "AIza..." }                      │
                                                     ▼
                                              genai.Client(api_key=key)
                                              model.generate_content("Say OK")
                                                     │
        ◄──────────────────────────────────── { valid: true }
   └──► Show ✅ "Connected"

5. User uploads PDF
   └──► POST /api/upload
        Headers: X-Gemini-Key: AIza...  ──────────► Extract text, build index
                                                     Return session_id
        ◄──────────────────────────────────── { session_id: "abc-123", ... }

6. User asks a question
   └──► POST /api/chat
        Headers: X-Gemini-Key: AIza...  ──────────► Retrieve passages from FAISS
        { session_id, question }                     Build prompt with context
                                                     Call Gemini 2.5 Flash
                                                     │
        ◄──────────────────────────────────── { answer: "...", sources: [...] }
   └──► Render answer bubble + sources
```

---

## 13. UI Pages — Wireframes

### 13.1 Navigation Bar

```
┌──────────────────────────────────────────────────────┐
│  📄 PDF RAG Chatbot    │  💬 Chat  │  📊 Finance  │  ⚙️  │
└──────────────────────────────────────────────────────┘
```

### 13.2 Chat Page

```
┌──────────────────────────────────────────────────────┐
│  📄 Upload PDF: [drag & drop zone / Browse button]   │
│  Status: ✅ 142 passages indexed | Finance detected  │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────────────────────────┐  ← User msg   │
│  │  What is the total revenue?      │                │
│  └──────────────────────────────────┘                │
│                                                      │
│  ┌──────────────────────────────────┐  ← AI msg     │
│  │  Based on the PDF [page 3], the  │                │
│  │  total revenue is $2.4M...       │                │
│  │  ▸ View Sources (6 passages)     │                │
│  └──────────────────────────────────┘                │
│                                                      │
│  ┌──────────────────────────────────┐  ← User msg   │
│  │  Summarize page 5               │                │
│  └──────────────────────────────────┘                │
│                                                      │
├──────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────┐  ┌────────┐ │
│  │  Type your question...             │  │  Send  │ │
│  └────────────────────────────────────┘  └────────┘ │
└──────────────────────────────────────────────────────┘
```

### 13.3 Finance Dashboard

```
┌──────────────────────────────────────────────────────┐
│  ┌──────┐  ┌──────┐  ┌──────────┐  ┌────────────┐  │
│  │ Rows │  │ Net  │  │ Income   │  │ Expense    │  │
│  │ 150  │  │-2340 │  │ +12,500  │  │ -14,840    │  │
│  └──────┘  └──────┘  └──────────┘  └────────────┘  │
├──────────────────────────────────────────────────────┤
│  Aggregate: [Hourly ▼] [Daily ▼] [Monthly ▼]       │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │           📈 Time-Series Line Chart           │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌────────────────────┐  ┌──────────────────────┐   │
│  │   🥧 Pie Chart      │  │   📊 Bar Chart       │   │
│  └────────────────────┘  └──────────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  Ask Gemini about your finances:              │   │
│  │  ┌────────────────────────────┐  ┌────────┐  │   │
│  │  │ What's my biggest expense? │  │  Ask   │  │   │
│  │  └────────────────────────────┘  └────────┘  │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

---

## 14. Non-Functional Requirements

| ID | Requirement | Category |
| -- | ----------- | -------- |
| NFR-01 | API key never sent to any server except Google's Gemini API | Security |
| NFR-02 | API key stored only in `localStorage` (client-side); never persisted on server | Security |
| NFR-03 | All API calls include `X-Gemini-Key` header; server does not cache keys | Security |
| NFR-04 | Frontend loads in < 2s (no heavy frameworks) | Performance |
| NFR-05 | Gemini responses arrive in < 5s for standard queries | Performance |
| NFR-06 | Responsive layout for desktop (≥ 1024px) and tablet (≥ 768px) | Usability |
| NFR-07 | Dark and light theme support via CSS variables | Usability |
| NFR-08 | Works on Chrome, Firefox, Edge, Safari (latest) | Compatibility |
| NFR-09 | Single command to start: `uvicorn backend.main:app` | Deployment |
| NFR-10 | No build step required for frontend (vanilla HTML/CSS/JS) | Simplicity |

---

## 15. Limitations & Constraints

| # | Limitation |
| - | ---------- |
| 1 | Requires a valid Google Gemini API key (free tier available) |
| 2 | Single PDF at a time per session |
| 3 | No OCR for scanned PDFs |
| 4 | FAISS index is in-memory only (lost on server restart) |
| 5 | No user authentication (single-user local use) |
| 6 | Gemini 2.5 Flash has rate limits on free tier |
| 7 | No streaming responses (full response returned at once) |
| 8 | Charts rendered client-side — large datasets may slow the browser |

---

## 16. Future Enhancements (Roadmap)

| Priority | Enhancement | Description |
| -------- | ----------- | ----------- |
| P1 | **Streaming Gemini responses** | Use Gemini streaming API for progressive answer display |
| P1 | **Multi-PDF sessions** | Upload and query across multiple PDFs |
| P1 | **OCR support** | Integrate Tesseract for scanned PDFs |
| P2 | **Chat history export** | Download conversation as Markdown/PDF |
| P2 | **Finance report export** | Export charts + stats to PDF/Excel |
| P2 | **Gemini function calling** | Let Gemini invoke finance functions directly |
| P3 | **User authentication** | Google OAuth for multi-user support |
| P3 | **Persistent sessions** | Store sessions in SQLite for server restarts |
| P3 | **Mobile-responsive** | Full mobile layout (< 768px) |

---

## 17. How to Run

```bash
# 1. Clone and setup
cd rag-chatbot-v2
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Install backend dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn backend.main:app --reload --port 8000

# 4. Open browser
# http://localhost:8000

# 5. Go to Settings → Enter your Gemini API key → Test → Start chatting!
```

### Getting a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Click "Create API Key"
3. Copy the key (starts with `AIza...`)
4. Paste into the Settings page

---

## 18. Risk Assessment

| Risk | Impact | Mitigation |
| ---- | ------ | ---------- |
| Gemini API key invalid/expired | All LLM features fail | "Test Key" button + clear error messages |
| Gemini rate limit exceeded | Temporary 429 errors | Show user-friendly "Rate limited, try again in X seconds" |
| Large PDF upload (100+ MB) | Server timeout | Set max file size (50 MB); show progress |
| Browser `localStorage` cleared | API key lost | Settings page shows clear "no key" warning on load |
| Gemini model deprecated | API calls fail | Model selector allows switching; default updated in code |
| CORS issues in development | Frontend can't reach backend | FastAPI CORS middleware configured for all origins |

---

*End of PRD v2*
