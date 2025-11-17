# main.py → FINAL OOM-FIX VERSION (Lightweight Embeddings + Port Binding)
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import os

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # ← Lightweight API embeddings (no download)
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

retriever = None
rag_chain = None

prompt_template = """
You are Sparky, a warm, caring mental wellness buddy for Indian college students.
Use simple English + little Hindi if natural.

Context:
{context}

Question: {question}

CRISIS RULE → If user mentions suicide, self-harm, "want to die", "end life", "hopeless" → reply ONLY with:

**I'm really worried about you. Please reach out right now — you are not alone.**
Indian 24×7 Free Helplines:
• iCall (TISS): 9152987821
• Vandrevala Foundation: 9999666555
• Sneha Chennai: 044-24640050
• Emergency: 112

Otherwise give 1–3 practical tips from context and end with a gentle question.

Answer:
"""

def setup_rag_pipeline():
    global retriever, rag_chain

    print("Loading lightweight embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # ← Lightweight API (no download)

    print("Loading knowledge base...")
    loader = DirectoryLoader("knowledge_base", glob="**/*.txt")
    docs = loader.load()

    print("Splitting into smaller chunks (memory efficient)...")
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)  # ← Smaller chunks for RAM

    print(f"Adding {len(chunks)} chunks to Chroma...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="sparky",
        persist_directory="./chroma_db"  # ← Persist for Render
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # ← k=3 for less RAM

    print("Connecting to Groq...")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.4,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_template(prompt_template)
    rag_chain = (
        {"context": retriever, "question": lambda x: x}
        | prompt
        | llm
        | (lambda x: x.content)
    )

    print("SPARKY IS LIVE!")
    return True

@app.on_event("startup")
async def startup_event():
    success = setup_rag_pipeline()
    if not success:
        print("Startup fallback – basic mode")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.get_template("index.html").render({"request": request})

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data["message"].strip()

    crisis_keywords = ["suicide", "kill myself", "end life", "want to die", "hopeless", "self harm", "no point living"]
    if any(keyword in user_message.lower() for keyword in crisis_keywords):
        return {"response": """**I'm really worried about you. Please reach out right now — you are not alone.**
Indian 24×7 Free & Confidential Helplines:
• iCall (TISS): 9152987821
• Vandrevala Foundation: 9999666555
• Sneha Chennai: 044-24640050
• Emergency: 112"""}

    response = rag_chain.invoke(user_message)
    return {"response": response}