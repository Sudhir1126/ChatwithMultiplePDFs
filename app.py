import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Debug block to check FAISS import
try:
    from langchain.vectorstores import FAISS
    print("FAISS imported successfully")
except ImportError as e:
    print(f"FAISS import failed: {str(e)}")

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Set page config as the very first Streamlit command
st.set_page_config(page_title="Chat With Multiple PDFs", layout="wide", page_icon="\U0001F4DA")

# Load environment variables
load_dotenv()

# Configuration for Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Functions
def get_pdf_text(pdf_docs):
    """Extracts text from the uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    """Splits the extracted text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Generates embeddings for the text chunks and stores them in FAISS."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Returns a conversational chain using LangChain."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, say "Answer is not available in the context."\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def process_question(user_question):
    """Handles user questions and returns the response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        # Allow dangerous deserialization as this is a trusted local file
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response.get("output_text", "No response generated.")
    except Exception as e:
        return f"An error occurred: {str(e)}"

def handle_api_call_with_retry(function, *args, **kwargs):
    for attempt in range(3):  # Retry up to 3 times
        try:
            return function(*args, **kwargs)
        except google.api_core.exceptions.ResourceExhausted as e:
            if attempt < 2:  # Wait before retrying
                time.sleep(2 ** attempt)
            else:
                raise e

# Main app
def main():
    st.title("\U0001F4DA Chat with Multiple PDF Files")
    st.write("Upload PDF files, process them, and ask questions based on their content.")

    with st.sidebar:
        st.header("Menu")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing files..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Files processed successfully! You can now ask questions.")
                    else:
                        st.error("No text extracted from the uploaded files. Please check your PDFs.")
            else:
                st.error("Please upload at least one PDF file.")

    user_question = st.text_input("Ask a question:")
    if user_question:
        with st.spinner("Generating response..."):
            reply = process_question(user_question)
            st.write("### Reply:")
            st.write(reply)

    st.markdown("---")
    st.text("Designed by Sudhir Venkatesh")

if __name__ == "__main__":
    main()
