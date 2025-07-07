from Retrieve import hybrid_retrieve, bm25_model, collection
import openai
import os
import dotenv
from openai import OpenAI

dotenv.load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

query = "Rocky mountains' climate"

# Get top 5 chunks
top_docs = hybrid_retrieve(query, bm25_model, collection, top_k=5)

# Combine the paragraphs
context = "\n\n".join([doc for _, _, doc, _, _ in top_docs])

prompt = f"""Use only the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:"""

# LLM generate anwser
response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
)

print("GPT：", response.choices[0].message.content)
