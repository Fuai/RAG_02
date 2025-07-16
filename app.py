
# Import required libraries
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

# Streamlit page configuration
st.title("RAG System: PDF-Based Question Answering")
st.write("Upload a PDF file in the sidebar and ask questions about its content.")

# Step 1: Sidebar configuration for PDF upload
st.sidebar.title("PDF Upload")
st.sidebar.write("Please upload a PDF file to begin.")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# Initialize session state for tracking PDF changes
if 'current_pdf_name' not in st.session_state:
    st.session_state.current_pdf_name = None

# Handle PDF upload and cleanup
if uploaded_file is not None:
    # Check if we need to clear the cache (new PDF uploaded)
    if st.session_state.current_pdf_name != uploaded_file.name:
        # Clear the cache if PDF changed
        st.cache_resource.clear()
        # Remove old temporary PDF if it exists
        if st.session_state.current_pdf_name and os.path.exists("temp_pdf.pdf"):
            os.remove("temp_pdf.pdf")
        # Update current PDF name
        st.session_state.current_pdf_name = uploaded_file.name
    
    # Save new PDF
    with open("temp_pdf.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    pdf_path = "temp_pdf.pdf"
else:
    st.warning("Please upload a PDF file to begin!")
    st.stop()

# Step 2: Load and process the PDF
@st.cache_resource
def load_and_process_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        st.error("PDF file not found. Please upload a file!")
        return None, None
    
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # Create embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create FAISS vector store
    vector_store = FAISS.from_documents(texts, embedding_model)
    
    return vector_store, documents

# Step 3: Set up the RAG system
@st.cache_resource
def setup_rag_system(_vector_store):
    # Initialize tokenizer and model
    model_name = "facebook/opt-350m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Configure the pipeline with proper text generation settings
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.3,
        top_k=50,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Create LangChain pipeline
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # Create prompt template
    prompt_template = """Context: {context}
    Q: {question}
    A:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_vector_store.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=False,
        chain_type_kwargs={
            "prompt": PROMPT
        }
    )
    
    return qa_chain

# Step 4: Load PDF and initialize RAG
vector_store, documents = load_and_process_pdf(pdf_path)
if vector_store is None:
    st.stop()

qa_chain = setup_rag_system(vector_store)

# Step 5: Main interface - User input and answers
st.subheader("Ask a Question")
query = st.text_input("Enter your question about the uploaded PDF:", "")
if st.button("Submit"):
    if query:
        with st.spinner("Generating answer..."):
            # Get the result and clean it up
            result = qa_chain({"query": query})
            answer = result["result"]
            
            # Clean up the answer
            # Remove common prefixes and formatting
            prefixes_to_remove = [
                "Extract and return ONLY",
                "the most relevant information",
                "from the following context",
                "without any additional text or formatting:",
                "used technique that generalizes",
                "Context:",
                "Question:",
                "Answer:",
                "Q:",
                "A:"
            ]
            
            for prefix in prefixes_to_remove:
                answer = answer.replace(prefix, "")
            
            # Remove any text after "Question:" or "Answer:" if they appear
            answer = answer.split("Question:")[0].split("Answer:")[0]
            
            # Clean up extra whitespace and newlines
            answer = " ".join(answer.split())
            
            # Display the cleaned answer
            st.write(answer)
    else:
        st.warning("Please enter a question!")

# Cleanup temporary file when the app stops
def cleanup():
    if os.path.exists("temp_pdf.pdf"):
        os.remove("temp_pdf.pdf")

# Register cleanup function
import atexit
atexit.register(cleanup)
