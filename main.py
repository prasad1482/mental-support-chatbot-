import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ====== GROQ API KEY ======
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing in Render environment variables")

# ====== LOAD KNOWLEDGE BASE ======
KNOWLEDGE_BASE_DOCS = []
for file in os.listdir("knowledge_base"):
    with open(f"knowledge_base/{file}", "r", encoding="utf-8") as f:
        KNOWLEDGE_BASE_DOCS.append({
            "content": f.read(),
            "source": file
        })

# ====== CRISIS DETECTION ======
CRISIS_WORDS = [
    "suicide", "kill myself", "end my life", "self-harm", "want to die",
    "hopeless", "cut myself", "better off dead", "can't go on"
]

CRISIS_RESPONSE = {
    "isCrisis": True,
    "text": "I’m really concerned about you. Please call Vandrevala Helpline at 9999666555 or emergency number 112 immediately."
}

def is_crisis(msg: str):
    msg = msg.lower()
    return any(word in msg for word in CRISIS_WORDS)

# ====== RAG PIPELINE CREATION ======
def create_rag():
    embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


    vector_store = Chroma.from_texts(
        texts=[d["content"] for d in KNOWLEDGE_BASE_DOCS],
        embedding=embeddings,
        metadatas=[{"source": d["source"]} for d in KNOWLEDGE_BASE_DOCS],
        persist_directory=None  # In-memory only
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    system_prompt = """
You are Sparky, a caring student wellness companion.
Be friendly, empathetic, supportive.
Use the context below to give helpful advice.

CONTEXT:
{context}
"""

    model = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="mixtral-8x7b-32768",
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

# ====== FASTAPI INITIALIZATION ======
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

class ChatRequest(BaseModel):
    message: str

@app.on_event("startup")
async def startup():
    global rag_chain
    print("Starting minimal server...")
    rag_chain = create_rag()
    print("RAG ready!")


@app.get("/")
async def home():
    return FileResponse("templates/index.html")

@app.post("/chat")
async def chat(request: ChatRequest):
    user_message = request.message.strip()

    if is_crisis(user_message):
        return CRISIS_RESPONSE

    try:
        output = rag_chain.invoke(user_message)
        return {"isCrisis": False, "text": output}
    except Exception as e:
        print("ERROR:", e)
        raise HTTPException(status_code=500, detail="Server error")
