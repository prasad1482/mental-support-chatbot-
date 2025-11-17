import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# =============== GOOGLE API KEY ===============
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY missing in Render Environment Variables")

# =============== KNOWLEDGE BASE ===============
KNOWLEDGE_BASE_DOCS = []
for file in os.listdir("knowledge_base"):
    with open(f"knowledge_base/{file}", "r", encoding="utf-8") as f:
        KNOWLEDGE_BASE_DOCS.append({
            "content": f.read(),
            "source": file
        })

# =============== CRISIS DETECTION ===============
CRISIS_KEYWORDS = [
    "kill myself", "suicide", "want to die", "end my life",
    "self-harm", "hopeless", "can't go on", "better off dead",
    "take pills", "hang myself"
]

CRISIS_INTERVENTION_RESPONSE = {
    "isCrisis": True,
    "text": "I’m really worried about you. Please reach out to Vandrevala 9999666555 or call 112 immediately."
}

def check_crisis(msg: str) -> bool:
    msg = msg.lower()
    return any(word in msg for word in CRISIS_KEYWORDS)

# =============== BUILD RAG PIPELINE ===============
def build_rag():
    # Lightweight, CPU-only embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # In-memory vector store (IMPORTANT for Render)
    vector_store = Chroma.from_texts(
        texts=[d["content"] for d in KNOWLEDGE_BASE_DOCS],
        embedding=embeddings,
        metadatas=[{"source": d["source"]} for d in KNOWLEDGE_BASE_DOCS],
        collection_name="sparky",
        persist_directory=None  # prevents writing to disk
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    system_prompt = """
You are Sparky, a friendly wellness assistant. Be supportive, warm, never judge.
Context:
{context}
"""

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return chain

rag_chain = None

# =============== FASTAPI SETUP ===============
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.on_event("startup")
async def startup_event():
    global rag_chain
    rag_chain = build_rag()
    print("🔥 RAG Pipeline Ready on Render!")

@app.get("/")
async def main_page():
    return FileResponse("templates/index.html")

@app.post("/chat")
async def chat(req: ChatRequest):
    user_message = req.message.strip()

    if check_crisis(user_message):
        return CRISIS_INTERVENTION_RESPONSE

    try:
        response = rag_chain.invoke(user_message)
        return {"isCrisis": False, "text": response}
    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail="Internal Error")
