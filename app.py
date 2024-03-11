import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import openai

# Streamlit app title
st.title('RAG with STARBUCKS GPT')

# API key input with error handling
api_key_input = st.text_input("Enter your OpenAI API Key:", type="password")

def validate_api_key(key):
    if not key:
        st.write("Please enter your OpenAI API Key.")
        return False
    return True

# Validate API key before proceeding
if not validate_api_key(api_key_input):
    st.stop()

openai.api_key = api_key_input

# JSON file loading with exception handling
file_path = 'starbucks_data.json'
try:
    with open(file_path, 'r') as file:
        data = json.load(file)
except FileNotFoundError:
    st.error(f"Error: Could not find '{file_path}'. Please ensure the file exists and is in the correct location.")
    st.stop()

# Document processing
docs = [f"{d['Beverage_category']} {d['Beverage']} {d['Beverage_prep']}" for d in data]
model = SentenceTransformer("sentence-transformers/all-MiniLM-l6-v2")
embeddings = model.encode(docs)
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(embeddings).astype('float32'))
faiss.write_index(index, "faiss_index.bin")
index = faiss.read_index("faiss_index.bin")

# User input with placeholder text
question = st.text_input("Ask me anything about Starbucks (e.g., beverage recommendations):", "")

# Button with improved clarity
if st.button('Generate Answer'):
    # Input validation (ensure question is entered)
    if not question:
        st.write("Please enter your question.")
        return

    question_embedding = model.encode([question], convert_to_tensor=True)
    question_embedding = question_embedding.cpu().numpy()
    _, I = index.search(question_embedding, 5)
    related_docs = [docs[i] for i in I[0]]
    prompt = question + "\n\n" + "\n\n".join(related_docs)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    st.write(response.choices[0].message['content'])

