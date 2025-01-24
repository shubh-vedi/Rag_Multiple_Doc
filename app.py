import streamlit as st
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader, ExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found in environment variables. Please add it to your `.env` file.")
    st.stop()

# Function to load documents
def load_document(file):
    if file.type == "application/pdf":
        loader = PyPDFLoader(file.name)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        loader = Docx2txtLoader(file.name)
    elif file.type == "text/csv":
        loader = CSVLoader(file.name)
    elif file.type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        loader = ExcelLoader(file.name)
    else:
        st.error("Unsupported file type")
        return None
    return loader.load()

# Function to split text into chunks
def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    return texts

# Function to create a vector store using OpenAI embeddings
def create_vector_store(texts):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

# Function to create the RAG chain with GPT-4o
def create_rag_chain(vectorstore):
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    return qa_chain

# Streamlit UI
def main():
    st.title("RAG Streamlit App with GPT-4o")

    # File uploader
    uploaded_files = st.file_uploader("Upload documents", type=["pdf", "docx", "csv", "xlsx"], accept_multiple_files=True)

    if uploaded_files:
        docs = []
        for file in uploaded_files:
            # Save the file temporarily
            with open(file.name, "wb") as f:
                f.write(file.getbuffer())
            # Load the document
            loaded_docs = load_document(file)
            if loaded_docs:
                docs.extend(loaded_docs)

        if docs:
            # Split text into chunks
            texts = split_text(docs)

            # Create vector store
            vectorstore = create_vector_store(texts)

            if vectorstore:
                st.success("Documents processed successfully!")

                # Question input
                question = st.text_input("Ask a question:")

                if question:
                    # Create RAG chain
                    qa_chain = create_rag_chain(vectorstore)
                    # Get answer
                    answer = qa_chain.run(question)
                    st.write("**Answer:**", answer)

if __name__ == "__main__":
    main()
