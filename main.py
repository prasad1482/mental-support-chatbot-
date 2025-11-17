import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

# --------- LLM (Groq) ----------
from langchain_groq import ChatGroq

# --------- Embeddings (Lightweight) ----------
from langchain_huggingface import HuggingFaceEmbeddings

# --------- Vector Store (FAISS — NO ONNX, NO Torch) ----------
from langchain_community.vectorstores import FAISS

# --------- LangChain Core ----------
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ----------------- Load API KEY -----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing in environment!")


# ----------------- Knowledge Base -----------------
KNOWLEDGE_BASE_DOCS = [
    {
        "content": open("knowledge_base/01_exam_stress.txt").read(),
        "source": "Exam Stress"
    },
    {
        "content": open("knowledge_base/02_anxiety.txt").read(),
        "source": "Anxiety"
    },
    {
        "content": open("knowledge_base/03_depression.txt").read(),
        "source": "Depression"
    },
    {
        "content": open("knowledge_base/04_sleep.txt").read(),
        "source": "Sleep Issues"
    },
    {
        "content": open("knowledge_base/05_homesickness.txt").read(),
        "source": "Homesickness"
    },
    {
        "content": open("knowledge_base/06_time_management.txt").read(),
        "source": "Time Management"
    },
    {
        "content": open("knowledge_base/07_self_care.txt").read(),
        "source": "Self-care"
    },
    {
        "content": open("knowledge_base/08_helplines.txt").read(),
        "source": "Helplines"
    }
]


# ----------------- Crisis Detection -----------------
CRISIS_KEYWORDS = [
    "kill myself", "suicide", "want to die", "end my life", "no reason to live",
    "self-harm", "cut myself", "hopeless", "can't go on", "better off dead",
    "going to jump", "take pills", "hang myself", "don't want to wake up",
    "nobody cares", "burden to everyone", "tired of living",
    "khudkushi", "aatmahatya"
]

CRISIS_RESPONSE = {
    "isCrisis": True,
    "text": """⚠️ **I'm really worried about you.**

You are NOT alone, and your feelings matter.  
Please reach out immediately:

**India 24/7 Helplines:**
📞 Vandrevala Foundation: **9999-666-555**  
📞 iCall (TISS): **9152987821**

If you're on campus, go to the **Counseling Center**.

If you are in immediate danger, please call **112**.

Talking to someone right now can truly help. 💛"""
}


def check_crisis(text: str) -> bool:
    t = text.lower()
    return any(word in t for word in CRISIS_KEYWORDS)


# ----------------- RAG Pipeline -----------------
rag_chain = None


def create_rag():
    print("Loading lightweight embeddings (BGE-small)...")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    print("Building FAISS vector store...")
    docs = [d["content"] for d in KNOWLEDGE_BASE_DOCS]
    metas = [{"source": d["source"]} for d in KNOWLEDGE_BASE_DOCS]

    vector_store = FAISS.from_texts(
        docs, embeddings, metadatas=metas
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    print("Loading Groq LLM (Gemma 2B / Mixtral)...")
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="mixtral-8x7b-32768",     # you can change to gemma-2b
        temperature=0.2
    )

    system_prompt = """
You are Sparky — a warm, empathetic mental wellness assistant for college students.
Use the provided context to give supportive, simple, helpful responses.

CONTEXT:
{context}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    print("RAG ready!")
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


# ----------------- FastAPI -----------------
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
async def startup():
    global rag_chain
    print("Starting app...")
    rag_chain = create_rag()


@app.get("/")
def root():
    return FileResponse("templates/index.html")


@app.post("/chat")
async def chat(request: ChatRequest):
    msg = request.message.strip()

    if check_crisis(msg):
        return CRISIS_RESPONSE

    try:
        response = rag_chain.invoke(msg)
        return {"isCrisis": False, "text": response}
    except Exception as e:
        print("ERROR:", e)
        return {
            "isCrisis": False,
            "text": "Sorry, I am facing an issue. Please try again!"
        }


# -----------------
# LOCAL RUN
# -----------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
