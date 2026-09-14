import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import os

st.set_page_config(page_title="ResearchRag", page_icon="🔬", layout="wide")

st.title("  Research2Code - RAG for Research Papers")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password")

    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF research paper", type="pdf")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if uploaded_file is not None and not st.session_state.vector_store:
    with st.spinner("Processing PDF..."):
        # 1. Extract Text
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"

        # 2. Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # 3. Local Embeddings & ChromaDB
        st.write("Generating local embeddings... this may take a moment.")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create vector store in-memory
        vector_store = Chroma.from_texts(chunks, embeddings)
        st.session_state.vector_store = vector_store
        st.success("Document processed and ready for questions!")

# Chat Interface
st.header("Chat with your Paper")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input for new question
if prompt := st.chat_input("Ask a question about the paper (e.g., 'Explain the proposed algorithm')"):
    if not st.session_state.vector_store:
        st.error("Please upload a PDF document first.")
    elif not groq_api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Setup LLM
                    llm = ChatGroq(
                        groq_api_key=groq_api_key,
                        model_name="llama-3.3-70b-versatile",
                        temperature=0.2
                    )

                    # Prompt Template
                    system_prompt = (
                        "You are an assistant for question-answering tasks based on a research paper. "
                        "Use the following pieces of retrieved context to answer the question. "
                        "If you don't know the answer, say that you don't know. "
                        "Keep the answer concise and relevant to the paper. "
                        "Provide references to the context if possible.\n\n"
                        "{context}"
                    )

                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])

                    # Retrieval Chain
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={
                                                                           "k": 3})
                    question_answer_chain = create_stuff_documents_chain(
                        llm, prompt_template)
                    rag_chain = create_retrieval_chain(
                        retriever, question_answer_chain)

                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]

                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer})

                    # Optionally display sources
                    with st.expander("View Source Context"):
                        for i, doc in enumerate(response["context"]):
                            st.write(f"**Source {i+1}:**")
                            st.write(doc.page_content)
                            st.divider()

                except Exception as e:
                    st.error(f"Error generating response: {e}")
