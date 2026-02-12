# AI-Powered Smart Advertisement Recommendation System (Offline RAG)

Offline prototype: FastAPI backend + Tailwind UI + Chroma vector DB + local LLM via Ollama.

## 1) Prereqs (offline inference)

- Install **Ollama** (local LLM server)
- Pull a model once (then it runs offline). Examples:
  - `ollama pull qwen3:8b`
  - `ollama pull llama3.1:8b`

## 2) Setup (Windows)

From this folder:

```powershell
C:/Users/91755/Documents/project final year/.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## 3) Run

```powershell
cd backend
C:/Users/91755/Documents/project final year/.venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000
```

Open:
- http://127.0.0.1:8000/ (user dashboard)
- http://127.0.0.1:8000/admin (admin upload)

## 4) Configure

Environment variables (optional):

- `OLLAMA_BASE_URL` (default `http://localhost:11434`)
- `OLLAMA_MODEL` (default `llama3.1:8b`, or set to any installed Ollama model like `qwen3:8b`)
- `EMBEDDING_MODEL` (default `sentence-transformers/all-MiniLM-L6-v2`)
- `CHROMA_DIR` (default `data/chroma`)
- `KB_DIR` (default `data/kb_docs`)

## Notes

- The embedding model may download on first run. After that, it can run offline.
- Upload `.txt`, `.md`, or `.pdf` documents in the admin panel to expand the knowledge base.

Tip: you can also set these in `backend/.env` (already supported by the backend).
