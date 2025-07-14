import os
import openai
import dotenv
from openai import OpenAI
from Retrieve import hybrid_retrieve, bm25_model, collection

# Load environment variables
dotenv.load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_answer_from_retrieve(query, top_k=5, model="gpt-3.5-turbo"):

    # Retrieve top_k document chunks
    top_docs = hybrid_retrieve(query, bm25_model, collection, top_k=top_k)

    # Combine retrieved document chunks into context
    context = "\n\n".join([doc for _, _, doc, _, _ in top_docs])

    # Construct prompt
    prompt = f"""Use only the context below to answer the question.

Context:
{context}

Question:
{query}

Answer:"""

    try:
        # Call OpenAI chat model
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1024
        )

        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"[ERROR] Failed to generate answer: {e}")
        return f"Error: {str(e)}"
