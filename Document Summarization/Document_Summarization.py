import os
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the fine-tuned transformer model from Hugging Face
llm = HuggingFaceHub(repo_id="facebook/bart-large-cnn")

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Prompt template for summarization
prompt_template = PromptTemplate(template="""
Summarize the following document:
{text}
---
Summary:
""", input_variables=["text"])

def load_document(file_path):
    """Loads document content based on file type."""
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file format.")
    
    return loader.load()

def summarize_text(text):
    """Summarizes the given text using the fine-tuned model."""
    texts = text_splitter.split_text(text)
    chain = load_summarize_chain(llm, chain_type="map_reduce", prompt=prompt_template)
    return chain.run(texts)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded."
        
        file = request.files["file"]
        if file.filename == "":
            return "No selected file."
        
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        
        documents = load_document(file_path)
        full_text = "\n".join([doc.page_content for doc in documents])
        summary = summarize_text(full_text)
        
        return f"<h2>Summary:</h2><p>{summary}</p>"
    
    return '''
    <form action="/" method="post" enctype="multipart/form-data">
        <input type="file" name="file">
        <input type="submit" value="Upload and Summarize">
    </form>
    '''

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)
