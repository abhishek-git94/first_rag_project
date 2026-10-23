# RAG Document Q&A

A Streamlit app for asking questions about PDF research papers. It extracts and splits PDF text, indexes the chunks in a FAISS vector store, retrieves relevant passages for each question, and uses Groq's Llama 3 model to generate an answer grounded in those passages.

## Apps

- `main.py` uses OpenAI embeddings and requires an OpenAI API key.
- `app_huggingfaceembedding.py` uses the `all-MiniLM-L6-v2` Hugging Face sentence-transformer model for embeddings instead.

Both apps use Groq (`Llama3-8b-8192`) for answers. The Hugging Face model is downloaded when first used.

## Requirements

- Python 3.9 or newer
- API keys as described below

Install the dependencies:

```bash
pip install streamlit langchain langchain-core langchain-community langchain-groq langchain-openai langchain-huggingface faiss-cpu pypdf python-dotenv openai sentence-transformers
```

## Configure API keys

Create a `.env` file in the project directory:

```dotenv
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_token
```

`GROQ_API_KEY` must be a valid Groq API key. `main.py` also needs a valid `OPENAI_API_KEY`. The Hugging Face app uses the local sentence-transformer model for embeddings, but currently reads `OPENAI_API_KEY` and `HF_TOKEN` from the environment when it starts; define both entries in `.env` (the OpenAI key is not used to generate embeddings in that app).

Keep `.env` private; do not commit API keys.

## Add research papers

Put PDF files in the `research_papers/` directory. This repository includes two example papers. The apps load PDFs from that directory and build the vector index in memory when you click **Document Embedding**.

## Run

From the project directory, start one of the apps:

```bash
streamlit run main.py
```

Or, to use Hugging Face embeddings:

```bash
streamlit run app_huggingfaceembedding.py
```

In the browser:

1. Click **Document Embedding** and wait for the vector database to be created.
2. Enter a question about the papers.
3. Read the answer and expand **Document similarity Search** to inspect the retrieved passages.

The index is held in the Streamlit session and is rebuilt when needed; it is not saved as a persistent vector database.

## How it works

1. `PyPDFDirectoryLoader` reads PDFs from `research_papers/`.
2. `RecursiveCharacterTextSplitter` splits the loaded text into chunks of 1,000 characters with 200 characters of overlap.
3. The selected embedding model converts chunks into vectors stored in FAISS.
4. Relevant chunks are retrieved for a question and passed as context to the Groq chat model.
