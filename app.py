import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot with LangChain")

# Ensure API Key is set
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.warning("Please add your GOOGLE_API_KEY to the .env file or environment variables.")
    st.stop()

# Initialize session state for chat history and vector store
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# --- Sidebar for setup ---
with st.sidebar:
    st.header("1. Document Setup")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file is not None and st.session_state.vector_store is None:
        with st.spinner("Processing document..."):
            # Step 2: Load - Save uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # Load PDF
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()

                # Step 3: Split
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents(docs)

                # Step 4: Embed & Store
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_store = FAISS.from_documents(splits, embeddings)
                
                # Save to session state so it persists
                st.session_state.vector_store = vector_store
                st.success("Document processed and vector index built!")
            except Exception as e:
                st.error(f"Error processing document: {e}")
            finally:
                # Clean up temp file
                os.unlink(tmp_path)

    if st.session_state.vector_store is not None:
        st.success("Vector store is ready!")
        if st.button("Clear Document"):
            st.session_state.vector_store = None
            st.session_state.messages = []
            st.rerun()

# --- Main Chat Interface ---
st.header("2. Chat with Document")

if st.session_state.vector_store is None:
    st.info("Please upload a PDF document in the sidebar to start chatting.")
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Step 5: Retrieve + Generate
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Setup LLM
                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

                    # Setup Retriever
                    retriever = st.session_state.vector_store.as_retriever(
                        search_kwargs={"k": 3}
                    )

                    # Define the prompt to prevent hallucination
                    # "If it doesn't know the answer, it should say 'Not found in document'"
                    system_prompt = (
                        "You are an assistant for question-answering tasks. "
                        "Use the following pieces of retrieved context to answer the question. "
                        "If the answer is not contained within the context, you MUST say 'Not found in document' and nothing else. "
                        "Do not use outside knowledge. Keep the answer concise.\n\n"
                        "Context:\n{context}"
                    )
                    
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])

                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)

                    # Wire it all together using LCEL
                    rag_chain = (
                        {"context": retriever | format_docs, "input": RunnablePassthrough()}
                        | prompt_template
                        | llm
                        | StrOutputParser()
                    )

                    # Step 6: Test/Execute
                    answer = rag_chain.invoke(prompt)
                    
                    st.markdown(answer)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
