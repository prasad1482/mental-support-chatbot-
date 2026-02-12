import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dotenv import load_dotenv
load_dotenv()

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Global variables for RAG
retriever = None
rag_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - with timeout to prevent hanging
    import asyncio
    global rag_chain
    try:
        # Give RAG setup 30 seconds max
        await asyncio.wait_for(asyncio.to_thread(setup_rag_pipeline), timeout=30.0)
        print("✅ RAG pipeline ready!")
    except asyncio.TimeoutError:
        print("⚠️ RAG setup timed out → using fallback LLM")
        rag_chain = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.5,
        )
    except Exception as e:
        print(f"⚠️ RAG failed → using fallback LLM: {e}")
        rag_chain = ChatGroq(
            model="llama-3.1-8b-instant",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.5,
        )
    yield
    # Shutdown (if needed)


app = FastAPI(lifespan=lifespan)

# For frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------------------------
# 🔥 PROMPT TEMPLATE
# ---------------------------
prompt_template = """
You are Sparky, a warm, caring mental wellness buddy for Indian college students.
YOU MUST ALWAYS RESPOND IN CLEAR, SIMPLE, NATURAL ENGLISH ONLY. 
Never use Hindi or any other language. No Hinglish. No mixed words. ENGLISH ONLY.

Context: {context}
Question: {question}

CRISIS RULE → If the user mentions suicide, self-harm, "want to die", "end life", 
"kill myself", or similar → reply ONLY with:

"I'm really worried about you. Please reach out right now — you are not alone.
Indian 24×7 Free Helplines:
• iCall (TISS): 9152987821
• Vandrevala Foundation: 9999666555
• Sneha Chennai: 044-24640050
• Emergency: 112"

Otherwise:
• Give 1–3 practical, empathetic mental wellness tips based on the context.
• Keep sentences short and simple.
• FINAL RULE: ALWAYS RESPOND IN ENGLISH ONLY.

Answer:
"""


# ---------------------------
# 🔥 LOAD TXT FILES (LIGHTWEIGHT)
# ---------------------------
def load_knowledge_base():
    docs = []
    folder = "knowledge_base"

    for file in os.listdir(folder):
        if file.endswith(".txt"):
            path = os.path.join(folder, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                docs.append(text)
            except Exception as e:
                print(f"Error reading {file}: {e}")

    return docs


# ---------------------------
# 🔥 BUILD RAG PIPELINE
# ---------------------------
def setup_rag_pipeline():
    global retriever, rag_chain

    print("Loading lightweight embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Loading knowledge base (manual)...")
    documents = load_knowledge_base()

    if not documents:
        raise ValueError("No knowledge base documents found.")

    print("Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = []

    for doc in documents:
        pieces = splitter.split_text(doc)
        chunks.extend(pieces)

    print(f"Total chunks: {len(chunks)}")

    print("Building Chroma vector store...")
    vectorstore = Chroma.from_texts(chunks, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    print("Connecting to Groq...")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.4,
    )

    prompt = ChatPromptTemplate.from_template(prompt_template)

    rag_chain = (
        {"context": retriever, "question": lambda x: x}
        | prompt
        | llm
        | (lambda x: x.content)
    )

    print("SPARKY IS READY!")


# ---------------------------
# 🌐 ROUTES
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.get_template("index.html").render({"request": request})


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_msg = data["message"].strip().lower()

    crisis_words = ["suicide", "kill myself", "want to die", "hopeless", "end life", "self harm"]

    if any(word in user_msg for word in crisis_words):
        return {
            "response": """**I'm really worried about you. Please reach out right now — you are not alone.**

Indian 24×7 Free Helplines:
• iCall (TISS): 9152987821  
• Vandrevala Foundation: 9999666555  
• Sneha : 9948939192  
• Emergency: 112"""
        }

    # Normal RAG response
    response = rag_chain.invoke(data["message"])
    return {"response": response}
