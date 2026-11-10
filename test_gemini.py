
import os
from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env")

# Test the Gemini chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

response = llm.invoke("Say hello in one sentence.")
print("Gemini response:", response.content)

# Test the embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector = embeddings.embed_query("What is the sick leave policy?")

print("Embedding dimensions:", len(vector))
print("First five values:", vector[:5])
print("Both Gemini services are working!")
