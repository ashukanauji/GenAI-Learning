import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

question = "Is this review positive or negative? 'The delivery was late but the product is amazing.'"

techniques = {
    "zero_shot": question,
    "few_shot": f"positive → 'Great product!' | negative → 'Terrible quality.' | {question}",
    "chain_of_thought": f"Think step by step. {question}",
    "role_framed": f"You are a sentiment analysis expert. Classify with high confidence. {question}",
    "structured": f"Classify this review as JSON: {{sentiment, confidence, reasoning}}. {question}"
}

for name, prompt in techniques.items():
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    print(f"\n--- {name} ---\n{response.choices[0].message.content}")