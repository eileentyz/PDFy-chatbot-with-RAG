import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
#from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
#from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os

# Page Setup
st.title("PDFy ᯓ★ˎˊ˗")
st.write("Ask any questions about your PDF!")

# Sidebar for API Key
api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")

# File Uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and api_key: # Ensure both file and API key are provided
    # Set API key for Google Generative AI
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue()) # Save to temp file
        tmp_file_path = tmp_file.name # Get the path

    with st.spinner("Processing PDF..."): # Show spinner during processing
        # Load PDF
        loader = PyPDFLoader(tmp_file_path)
        data = loader.load()

        if not data: # Check if any text was extracted
            st.error("PDF contains no extractable text. Try another PDF.")
            st.stop()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(data)

        if not splits:
            st.error("Failed to split PDF into chunks.")
            st.stop()

        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create FAISS vector store
        vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=embeddings
        )

        retriever = vectorstore.as_retriever()

        # Define LLM - Using Gemini 2.5 Flash
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        def format_docs(docs):
            return "\n\n".join(d.page_content for d in docs)

        rag_chain = (
            retriever
            | RunnableLambda(format_docs)
            | llm
            | StrOutputParser()
        )

        st.success("PDF processed!")

    # QA input
    query = st.text_input("Ask a question about the PDF:")
    if query:
        answer = rag_chain.invoke(query)
        st.write("### Answer")
        st.write(answer)
