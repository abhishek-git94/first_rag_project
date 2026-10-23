# RAG Document Q&A

A Streamlit app for asking questions about PDF research papers. It uses a local Hugging Face sentence-transformer model to embed documents, FAISS for similarity search, and Groq for answer generation.

## Cost and API keys

- Document embeddings run locally with `sentence-transformers/all-MiniLM-L6-v2`; no Hugging Face API key or per-request Hugging Face inference charge is required. The model downloads on first use and consumes the app's CPU, memory, and storage.
- Answer generation requires `GROQ_API_KEY`. Groq has plan-specific limits and pricing; check your Groq console and avoid enabling paid billing if you want to stay within free usage.
- No OpenAI API key is used.

For local use, put the Groq key in a private `.env` file:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

For Streamlit Community Cloud, add `GROQ_API_KEY` in **App settings → Secrets**. Never commit the key to the repository.

## Run

Install the dependencies and start Streamlit:

```bash
pip install -r requirements.txt
streamlit run main.py
```

Put PDF files in `research_papers/`. In the app, click **Build document index**, wait for the local model and index to load, then ask a question. Expand **Source passages** to see the retrieved text.

The document index is held in the Streamlit session and rebuilt when needed; it is not persisted across sessions.
