import streamlit as st
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import os
import textwrap
import tempfile
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

from langchain_community.llms import Together
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template("""
You are a helpful AI assistant. Answer the user’s question based on the provided context. 
Be **concise** and **summarize key points** only. If the answer is too long, shorten it to fit within a short paragraph.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer (short and to-the-point):
""")

# 🔐 Load API Key
try:
    TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
except Exception:
    load_dotenv()
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

TOGETHER_MODEL = "meta-llama/Llama-3-8b-chat-hf"

# 📘 Streamlit UI
st.set_page_config(page_title="AskDoc – Conversational RAG", layout="wide")
st.title("📘 AskDoc – Smart Conversational Q&A")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
webpage_url = st.text_input("🌐 Or enter a website URL to extract content")
question = st.text_input("💬 Ask something about the content")
submit = st.button("🔍 Ask")

# 🔧 Text splitting
def load_and_chunk(file):
    text = ""
    temp_path = tempfile.mkstemp()[1]
    with open(temp_path, "wb") as f:
        f.write(file.read())
    if file.name.endswith(".pdf"):
        doc = fitz.open(temp_path)
        for page in doc:
            text += page.get_text()
    else:
        with open(temp_path, "r", encoding="utf-8") as f:
            text = f.read()
    os.remove(temp_path)
    chunks = textwrap.wrap(text, width=500, break_long_words=False)
    return chunks

# 🔧 Web scraping using Selenium
def scrape_and_chunk(url):
    try:
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        driver = webdriver.Chrome(options=chrome_options)

        driver.get(url)
        time.sleep(5)
        text = driver.find_element(By.TAG_NAME, "body").text
        driver.quit()

        chunks = textwrap.wrap(text, width=500, break_long_words=False)
        return chunks
    except Exception as e:
        st.error(f"❌ Failed to fetch content: {e}")
        return []

# 🔍 Vectorstore from chunks
def create_vectorstore(chunks):
    documents = [Document(page_content=chunk) for chunk in chunks]
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
    vectordb = FAISS.from_documents(documents, embedding_model)
    return vectordb

# 🧠 Setup LangChain QA chain
def create_qa_chain(vectordb):
    llm = Together(
        model=TOGETHER_MODEL,
        temperature=0.2,
        together_api_key=TOGETHER_API_KEY
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=False,
        combine_docs_chain_kwargs={
            "prompt": CONDENSE_QUESTION_PROMPT
        }
    )
    return qa_chain

# 🧠 Session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# 📥 File or URL processing
chunks = []
if uploaded_file:
    chunks = load_and_chunk(uploaded_file)
    st.success(f"✅ Indexed {len(chunks)} chunks from uploaded file.")
elif webpage_url:
    chunks = scrape_and_chunk(webpage_url)
    st.success(f"✅ Indexed {len(chunks)} chunks from webpage.")

if chunks:
    vectordb = create_vectorstore(chunks)
    st.session_state.vectorstore = vectordb
    st.session_state.qa_chain = create_qa_chain(vectordb)

# 🧾 Ask Question
if submit and question:
    if not st.session_state.qa_chain:
        st.warning("⚠️ Please upload a document or enter a valid URL first.")
    else:
        with st.spinner("🤖 Thinking..."):
            answer = st.session_state.qa_chain.run(question)
        st.subheader("🟢 Answer:")
        st.markdown(f"> {answer}")
