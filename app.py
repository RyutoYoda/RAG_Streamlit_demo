import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import openai

st.title("RAG with STARBUCKS GPT")

# サイドバーでAPIキーの入力を受け付ける
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

# 初期設定
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

# すでにあるメッセージを表示
for message in st.session_state.messages:
    if message["role"] == "user":
        st.text_area("User", value=message["content"], height=75, disabled=True, key=f"user_{st.session_state.messages.index(message)}")
    elif message["role"] == "assistant":
        st.text_area("Assistant", value=message["content"], height=100, disabled=True, key=f"assistant_{st.session_state.messages.index(message)}")

# ユーザーからの質問を受け取る
prompt = st.text_input("Ask something about Starbucks beverages:")

if st.button('Generate Answer') and prompt and api_key:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # JSONファイルを読み込む
    file_path = 'starbucks_data.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

    docs = [f"{d['Beverage_category']} {d['Beverage']} {d['Beverage_prep']}" for d in data]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-l6-v2")
    embeddings = model.encode(docs)

    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings).astype('float32'))

    question_embedding = model.encode([prompt], convert_to_tensor=True)
    question_embedding = question_embedding.cpu().numpy()

    _, I = index.search(question_embedding, 5)

    related_docs = [docs[i] for i in I[0]]

    prompt_for_gpt = prompt + "\n\n" + "\n\n".join(related_docs)

    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_for_gpt}
        ]
    )

    answer = response.choices[0].message['content']
    st.session_state.messages.append({"role": "assistant", "content": answer})

# アプリのUIを更新するためにページを再読み込み
st.experimental_rerun()
