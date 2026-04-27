from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------
# OpenRouter setup
# ---------------------------
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

#loading the data
loader = TextLoader("data/medical.txt")
docs = loader.load()

#split into documents
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(docs)

#creating embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# storing in FAISS
db = FAISS.from_documents(chunks, embeddings)
retriever = db.as_retriever()

while True:
    query = input("\nAsk a medical question (type 'exit' to stop): ")

    if query.lower() == "exit":
        break

    # Retrieve relevant chunks
    docs = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in docs])

    # Prompt
    prompt = f"""
You are a medical assistant.
Answer ONLY using the context below.
If the answer is not in the context, say "I don't know".
This is not medical advice.

Context:
{context}

Question:
{query}
"""

    # LLM call
    response = client.chat.completions.create(
        model="google/gemma-3-4b-it:free",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\nAnswer:\n", response.choices[0].message.content)



