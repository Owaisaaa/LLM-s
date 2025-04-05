import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from markdown import markdown

# Global variables
VECTOR_DB_NAME = "local-rag"
local_model = "llama3.2"
llm = ChatOllama(model=local_model)
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
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        collection_name=VECTOR_DB_NAME,
        persist_directory=None  # Set to None for in-memory storage
    )
    return "Vector database created successfully"

# Function to set up retriever
def get_retriever():
    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Generate 2 different versions of the given user question to retrieve relevant documents from a vector database. Provide these alternative questions separated by newlines. Original question: {question}""",
    )
    return MultiQueryRetriever.from_llm(vector_db.as_retriever(), llm, prompt=query_prompt)

# Function to create RAG chain
def create_rag_chain():
    retriever = get_retriever()
    template = """Answer the question based ONLY on the following context:\n{context}\nQuestion: {question}"""
    prompt = ChatPromptTemplate.from_template(template)
    return (
        {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    )

# Function to query the document
def chat_with_pdf(question):
    if not vector_db:
        return "Error: No vector database found. Please upload and process a PDF first."
    chain = create_rag_chain()
    response = chain.invoke(question)
    return markdown(response)

# Gradio Interface
def gradio_interface(file):
    file_path = file.name
    data = load_pdf(file_path)
    chunks = split_text(data[:2])
    create_vector_db(chunks)
    return "PDF processed successfully. You can now ask questions."

gui = gr.Blocks()
with gui:
    gr.Markdown("## Chat with your PDF using RAG")
    file_input = gr.File(label="Upload PDF")
    process_button = gr.Button("Process PDF")
    status_output = gr.Textbox()
    question_input = gr.Textbox(label="Ask a question")
    submit_button = gr.Button("Ask")
    answer_output = gr.Markdown()
    
    process_button.click(gradio_interface, inputs=file_input, outputs=status_output)
    submit_button.click(chat_with_pdf, inputs=question_input, outputs=answer_output)

gui.launch()
