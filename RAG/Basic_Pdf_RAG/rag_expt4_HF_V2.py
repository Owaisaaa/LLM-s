"""
PDF-based RAG Chatbot using Hugging Face Models
This scrip uses HF Inference API to load a large language model and embeddings model for processing PDF documents.
HF Inference API is used to avoid the need for local GPU resources, making it easier to run on machines with limited resources.
But this comes with a cost, as the API is not free and has usage limits.

This script allows users to upload a PDF document, extract text, generate embeddings,
and perform question-answering using a Retrieval-Augmented Generation (RAG) approach.
"""

import gradio as gr
import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from markdown import markdown

# Load API keys from .env file
load_dotenv()
hf_api_key = os.getenv("RAG_PDF_API_KEY")

# Define Hugging Face models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Initialize Embeddings model
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Load LLM using Hugging Face Inference API
llm = HuggingFaceHub(
    repo_id=LLM_MODEL,
    model_kwargs={"temperature": 0.7, "max_length": 512},
    huggingfacehub_api_token=hf_api_key
)

# Global variables
VECTOR_DB_NAME = "huggingface-rag"
vector_db = None

# Function to load PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path=file_path)
    return loader.load()

# Function to split text
def split_text(data, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(data)

# Function to create vector database
def create_vector_db(chunks):
    global vector_db
    if not chunks:
        return "Error: No text extracted from the PDF."
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=VECTOR_DB_NAME,
        persist_directory=None # Set to None for in-memory storage -> so that embeddings are not saved to disk
    )
    return "PDF processed successfully! You can now ask questions."

# Function to set up retriever
def get_retriever():
    if not vector_db:
        return None
    
    query_prompt = ChatPromptTemplate.from_template(
        """Generate 2 alternative versions of the question to improve retrieval:
        Original: {question}"""
    )
    
    return MultiQueryRetriever.from_llm(vector_db.as_retriever(), llm, prompt=query_prompt)

# Function to create RAG chain
def create_rag_chain():
    retriever = get_retriever()
    if not retriever:
        return None
    
    prompt = ChatPromptTemplate.from_template(
        """You are an AI assistant that answers questions based only on the given context.
        Provide a well-structured, coherent, and concise response.

        ### Context:
        {context}

        ### Question:
        {question}

        ### Answer:
        """
    )
    
    return (
        {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    )

# Function to process user query
def process_query(question, chat_history):
    if not vector_db:
        return "Error: No vector database found. Please upload and process a PDF first.", chat_history

    chain = create_rag_chain()
    if not chain:
        return "Error: Unable to initialize the RAG chain.", chat_history

    response = chain.invoke(question)
    
    # Extract only the answer from the response
    answer_match = re.search(r"### Answer:\s*(.*)", response, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else response.strip()
    
    chat_history.append((question, markdown(answer)))
    return "", chat_history

# Function to process PDF upload
def process_pdf(file):
    global vector_db
    if not file:
        return "Please upload a valid PDF file."
    
    # Reset the vector DB before adding new embeddings
    vector_db = None
    # Load and process the PDF
    file_path = file.name
    data = load_pdf(file_path)
    chunks = split_text(data)
    
    return create_vector_db(chunks)

# Gradio UI
def gradio_ui():
    with gr.Blocks(theme="soft") as demo:
        gr.Markdown("<h1 style='text-align: center; color: #4A90E2;'>📖 Conversational AI for PDFs</h1>")

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="AI Chat")
                user_input = gr.Textbox(placeholder="Ask me a question...", label="Your Question")

                with gr.Row():
                    ask_button = gr.Button("🔍 Ask", variant="primary")
                    clear_button = gr.Button("🗑️ Clear")

            with gr.Column(scale=1):
                gr.Markdown("### Upload PDFs Here:")
                pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
                status = gr.Textbox(label="Status", interactive=False)

        # Define button actions
        ask_button.click(process_query, inputs=[user_input, chatbot], outputs=[user_input, chatbot])
        clear_button.click(lambda: [], outputs=[chatbot])
        pdf_upload.change(process_pdf, inputs=[pdf_upload], outputs=[status])

    return demo

demo = gradio_ui()
demo.launch()
