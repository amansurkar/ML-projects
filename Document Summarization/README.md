# Text Summarization with LangChain

This project provides an automated document summarization system using LangChain and a fine-tuned transformer model from Hugging Face. The system allows users to upload documents and receive concise summaries using Natural Language Processing (NLP).

## Features
- **Document Upload**: Supports PDF, TXT, and DOCX formats.
- **Text Extraction**: Extracts text from uploaded documents.
- **Text Splitting**: Uses `RecursiveCharacterTextSplitter` to process large documents efficiently.
- **Summarization Model**: Utilizes `facebook/bart-large-cnn`, a fine-tuned transformer model for text summarization.
- **Pipeline Execution**: Implements the `map_reduce` technique for effective summarization.
- **Web Interface**: Built using Flask, providing an intuitive UI for document upload and summarization.

## Tech Stack
- **Python**
- **Flask** (for web interface)
- **LangChain** (for text processing and summarization)
- **Hugging Face Transformers** (for model inference)
- **FAISS** (for optional vector storage)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/text-summarization-langchain.git
   cd text-summarization-langchain
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Hugging Face authentication if required:
   ```bash
   export HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key
   ```

## Usage
1. Run the Flask server:
   ```bash
   python app.py
   ```
2. Open your browser and go to `http://127.0.0.1:5000/`.
3. Upload a document and receive a summarized output.

## Project Structure
```
text-summarization-langchain/
│── app.py                # Main application file
│── requirements.txt      # Python dependencies
│── uploads/              # Directory to store uploaded files
│── templates/            # HTML templates for Flask (if extended)
│── README.md             # Project documentation
```

## API Endpoints
| Endpoint   | Method | Description |
|------------|--------|-------------|
| `/`        | GET/POST | Uploads a document and returns a summary |

## Future Enhancements
- **API for batch document processing**
- **Integration with cloud storage (AWS S3, Google Drive, etc.)**
- **Support for multilingual summarization**
- **Enhanced UI with React.js for a modern frontend**

## License
This project is licensed under the MIT License.

## Contributors
- Your Name (your.email@example.com)

Feel free to contribute and enhance the project!

