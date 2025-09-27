import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ✅ Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found in .env file!")

# ✅ Step 1: Load PDF
loader = PyPDFLoader("TamerShaban-resume-ux2025.pdf")
docs = loader.load()

# ✅ Step 2: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
documents = text_splitter.split_documents(docs)

# ✅ Step 3: Embeddings + FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# ✅ Step 4: Retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ✅ Step 5: LLM (use valid model)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# ✅ Step 6: Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ✅ Step 7: RAG Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# -------- Flask API --------
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "❌ 'query' is required"}), 400

    # ✅ Use invoke instead of __call__
    result = qa_chain.invoke({"question": query})
    answer = result["answer"]

    return jsonify({
        "query": query,
        "answer": answer,
        "chat_history": [str(msg.content) for msg in memory.chat_memory.messages]
    })

@app.route("/chat-ui")
def chat_ui():
    return render_template("index.html")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🤖 Resume RAG Chatbot API is running!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
