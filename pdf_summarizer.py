import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
import os
import tempfile

st.set_page_config(page_title="Document Summarizer Tool", page_icon=":robot_face:", layout="wide")

st.title("Document Summarizer Tool")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file_path = tmp_file.name

    # Load the PDF file
    loader = UnstructuredPDFLoader(file_path=tmp_file_path)
    data = loader.load()

    # Split and chunk the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)

    # Add the chunks to the vector database
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
        collection_name="pdf-text"
    )

    # Set up the LLM
    local_model = "mistral"
    llm = ChatOllama(model=local_model)

    # RAG prompt
    template = """Summarize the key points from the following context:
    {context}
    """

    prompt = PromptTemplate.from_template(template)

    chain = (
        {"context": vector_db.as_retriever()}
        | prompt
        | llm
        | StrOutputParser()
    )

    summary = chain.invoke("Summarize the key points from the PDF")
    st.write("Summary:", summary)

    # Remove the temporary file
    os.remove(tmp_file_path)