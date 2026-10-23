import os
import time

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="RAG Document Q&A", page_icon="📚")
st.title("RAG Document Q&A")
st.caption("Ask questions about the research papers using local Hugging Face embeddings and Groq.")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"

prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the provided context. If the context does not contain
the answer, say you could not find it in the uploaded papers.

<context>
{context}
</context>

Question: {input}
"""
)


@st.cache_resource(show_spinner="Loading the local Hugging Face embedding model…")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_embedding():
    paper_dir = "research_papers"
    if not os.path.isdir(paper_dir):
        st.error(f"The `{paper_dir}/` folder was not found.")
        return False

    with st.spinner("Reading PDFs and building the search index…"):
        docs = PyPDFDirectoryLoader(paper_dir).load()
        if not docs:
            st.error(f"No readable PDF files were found in `{paper_dir}/`.")
            return False
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs[:50])
        st.session_state.vectors = FAISS.from_documents(chunks, get_embeddings())
    return True


if st.button("Build document index", type="primary"):
    if create_vector_embedding():
        st.success("Document index is ready.")

question = st.text_input("Ask a question about the research papers")
if question:
    vectors = st.session_state.get("vectors")
    if vectors is None:
        st.info("Build the document index first.")
    else:
        try:
            groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
        except Exception:
            groq_key = os.getenv("GROQ_API_KEY", "")

        if not groq_key:
            st.error("Add your Groq API key as `GROQ_API_KEY` in Streamlit app Settings → Secrets.")
        else:
            try:
                llm = ChatGroq(groq_api_key=groq_key, model_name=GROQ_MODEL)
                chain = create_retrieval_chain(
                    vectors.as_retriever(), create_stuff_documents_chain(llm, prompt)
                )
                started = time.perf_counter()
                result = chain.invoke({"input": question})
                st.write(result["answer"])
                st.caption(f"Answered in {time.perf_counter() - started:.1f}s")
                with st.expander("Source passages"):
                    for doc in result["context"]:
                        st.write(doc.page_content)
                        st.divider()
            except Exception as exc:
                st.error(f"Could not generate an answer: {exc}")
