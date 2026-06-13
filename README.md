# RAG Chatbot with LangChain & Streamlit

This project is a Retrieval-Augmented Generation (RAG) chatbot built with LangChain and Streamlit. It allows users to upload a PDF document and ask questions about its content. To prevent hallucinations, the chatbot will explicitly state "Not found in document" if the answer is not contained within the uploaded PDF.

## 🚀 Features & Pipeline

The pipeline follows these steps as per the assignment requirements:

1. **Setup**: Uses LangChain, FAISS (for vector storage), PyPDF (for parsing), and Streamlit (for the UI).
2. **Load**: Accepts PDF uploads via Streamlit and loads them using `PyPDFLoader`.
3. **Split**: Chunks the extracted text using `RecursiveCharacterTextSplitter` (chunk size 1000, overlap 200) to maintain context.
4. **Embed & Store**: Uses open-source `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2` via `sentence-transformers`) to create vector embeddings and stores them locally using `FAISS`.
5. **Retrieve + Generate**: Uses the FAISS retriever to fetch the top 3 most relevant chunks. A `RetrievalQA` chain combined with `ChatGoogleGenerativeAI` (Gemini 1.5 Flash) generates the answer based *strictly* on the retrieved context.
6. **Hallucination Prevention**: A custom prompt forces the LLM to output exactly "Not found in document" if the context doesn't contain the answer.

## 🛠️ Installation & Setup

1. **Clone or Download** this repository.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up API Key**:
   - Rename `.env.example` to `.env`.
   - Add your Google Gemini API key:
     ```env
     GOOGLE_API_KEY=your_actual_api_key_here
     ```
   *(Note: You can get a free API key from Google AI Studio).*

## 🏃‍♂️ Running the App

Start the Streamlit application by running:
```bash
streamlit run app.py
```

## 🧪 Testing the Chatbot

1. Open the app in your browser (usually `http://localhost:8501`).
2. **Upload a PDF** using the sidebar.
3. Wait for the success message ("Document processed and vector index built!").
4. **Ask questions** in the chat interface.
   - Ask a question clearly answered in the document to test retrieval.
   - Ask a question completely unrelated to the document to verify the "Not found in document" behavior.
