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

# すでにあるメッセージをチャット形式で表示
for message in st.session_state.messages:
    st.chat_message(message["role"], message["content"])

# ユーザーからの質問を受け取る
prompt = st.chat_input("Ask something about Starbucks beverages:")

if prompt:
    # ユーザーの質問をセッション状態に追加
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # JSONファイルを読み込む
    file_path = 'starbucks_data.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

    # 文書を生成
    docs = [f"{d['Beverage_category']} {d['Beverage']} {d['Beverage_prep']}" for d in data]

    # Hugging Faceモデルをロードして文書をベクトル化
    model = SentenceTransformer("sentence-transformers/all-MiniLM-l6-v2")
    embeddings = model.encode(docs)

    # FAISSインデックスを作成
    d = embeddings.shape[1]  # ベクトルの次元
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings).astype('float32'))

    # 質問をベクトル化
    question_embedding = model.encode([prompt], convert_to_tensor=True)
    question_embedding = question_embedding.cpu().numpy()

    # 関連する文書を検索
    _, I = index.search(question_embedding, 5)

    # 関連する文書の内容を取得
    related_docs = [docs[i] for i in I[0]]

    # 関連する文書を基にして、プロンプトを作成
    prompt_for_gpt = prompt + "\n\n" + "\n\n".join(related_docs)

    # OpenAI GPTを使用してテキスト生成
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_for_gpt}
        ]
    )

    # 生成されたテキストをセッション状態に追加
    answer = response.choices[0].message['content']
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # チャット形式で回答を表示
    answer_str = str(answer)
    st.chat_message("assistant", answer)

