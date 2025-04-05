"""" 
This script avoids calling the Inference API, it downloads and run the model locally. 
This avoids API limits and saves costs incurred on HF Spaces
However, Mistral-7B requires significant VRAM (16GB+ recommended). 
If you're running locally, consider using a smaller model like mistralai/Mistral-7B-Instruct-v0.1 or 
mistralai/Mistral-7B-v0.2
""" 

import gradio as gr
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline
from markdown import markdown

# Load API keys from .env file
load_dotenv()
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")

# Define Hugging Face models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Small, fast embeddings
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Initialize Embeddings model
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Load LLM as a text generation pipeline
llm_pipeline = pipeline("text-generation", model=LLM_MODEL, token=hf_api_key)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Global variables
VECTOR_DB_NAME = "huggingface-rag"
vector_db = None

# Function to load PDF
def load_pdf(file_path):
    print(f"Loading PDF: {file_path}")
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
    print("Creating vector database...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=VECTOR_DB_NAME
    )
    return "Vector database created successfully"

# Function to set up retriever
def get_retriever():
    if not vector_db:
        print("Error: Vector database is not initialized.")
        return None

    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="Generate 2 alternative versions of the question to improve retrieval:\nOriginal: {question}"
    )

    return MultiQueryRetriever.from_llm(vector_db.as_retriever(), llm, prompt=query_prompt)

# Function to create RAG chain
def create_rag_chain():
    retriever = get_retriever()
    if not retriever:
        return None

    template = "Answer based ONLY on the following context:\n{context}\nQuestion: {question}"
    prompt = ChatPromptTemplate.from_template(template)

    return (
        {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    )

# Function to query the document
def chat_with_pdf(question):
    global vector_db
    if not vector_db:
        return "Error: No vector database found. Please upload and process a PDF first."

    chain = create_rag_chain()
    if not chain:
        return "Error: Unable to initialize the RAG chain."

    print(f"Processing question: {question}")
    response = chain.invoke(question)
    print(f"Response: {response}")

    return markdown(response)

# Gradio Interface
def gradio_interface(file):
    global vector_db

    if not file:
        return "Please upload a valid PDF file."
    
    # Reset the vector DB before adding new embeddings
    vector_db = None
    # Load and process the PDF
    file_path = file.name
    data = load_pdf(file_path)
    # chunks = split_text(data[:1])
    chunks = split_text(data)
    
    message = create_vector_db(chunks)
    
    if "Error" in message:
        return message
    
    return "PDF processed successfully! You can now ask questions."

# Create Gradio UI
gui = gr.Blocks()
with gui:
    gr.Markdown("## Chat with your PDF using RAG (Hugging Face)")
    file_input = gr.File(label="Upload PDF")
    process_button = gr.Button("Process PDF")
    status_output = gr.Textbox()
    question_input = gr.Textbox(label="Ask a question")
    submit_button = gr.Button("Ask")
    answer_output = gr.Markdown()
    
    process_button.click(gradio_interface, inputs=file_input, outputs=status_output)
    submit_button.click(chat_with_pdf, inputs=question_input, outputs=answer_output)

# Launch the Gradio app
gui.launch()
