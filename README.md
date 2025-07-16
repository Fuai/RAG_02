# PDF Question Answering System with RAG

A Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents and ask questions about their content. The system provides direct, concise answers based on the document's content.

## Features

- PDF document upload and processing
- Question answering using RAG architecture
- Clean, direct answers without unnecessary text
- Automatic cleanup of temporary files
- User-friendly Streamlit interface

## Requirements

- Python 3.8+
- Streamlit
- LangChain
- PyTorch
- Transformers
- FAISS
- sentence-transformers


## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Upload a PDF file using the sidebar
3. Ask questions about the PDF content
4. Get direct, concise answers

## How it Works

The system uses:
- LangChain for document processing and RAG implementation
- FAISS for efficient vector similarity search
- Hugging Face's OPT-350M model for text generation
- sentence-transformers for text embeddings

## Note

The system creates temporary files for PDF processing. These are automatically cleaned up when:
- A new PDF is uploaded
- The application stops running 
