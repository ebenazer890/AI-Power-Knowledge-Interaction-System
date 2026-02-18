# RAG PDF Chatbot + Financial Analysis (Streamlit)

## What it does
- Upload a PDF
- Chat with the PDF (RAG: local embeddings + vector search + optional LLM)
- If the PDF contains financial tables, generate:
  - totals
  - grouping by hour/day/month (based on detected datetime column)
  - pie chart (by category if found, otherwise by time bucket share)

## Run
1) Create/activate venv (already created in this workspace)
2) Install deps:

```bash
".venv/Scripts/python.exe" -m pip install -r requirements.txt
```

3) Start app:

```bash
".venv/Scripts/python.exe" -m streamlit run app.py
```

## LLM options
The app can work in two modes:
- **OpenAI**: set `OPENAI_API_KEY` in your environment.
- **Ollama** (local): install Ollama and pull a model, e.g. `ollama pull llama3.1`. The app will try `http://localhost:11434`.

If neither is available, the app still does retrieval and shows relevant passages, but answers will be extractive.
