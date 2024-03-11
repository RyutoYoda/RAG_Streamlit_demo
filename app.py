import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import openai

# Streamlitアプリのタイトル
st.title('RAG with STARBUCKS GPT')

# サイドバーでAPIキーの入力
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

# セッション状態でメッセージ管理
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# すでにあるメッセージをチャット形式で表示
for message in st.session_state['messages']:
    with st.chat_message_container():
        role = message["role"]
        st.chat_message(role, message["content"])

# チャット入力
prompt = st.chat_input("Ask something:")

# APIキーと質問が入力されたら処理を実行
if prompt and api_key:
    # ユーザーの質問をセッション状態のメッセージに追加
    st.session_state['messages'].append({"role": "user", "content": prompt})

    # OpenAIのAPIキーを設定
    openai.api_key = api_key

    # JSONファイルを読み込む
    file_path = 'starbucks_data.json'  # JSONファイルへのパス
    with open(file_path, 'r') as file:
        data = json.load(file)

    # 文書を生成
    docs = [f"{d['Beverage_category']} {d['Beverage']} {d['Beverage_prep']}" for d in data]

    # Hugging Faceモデルをロードして文書をベクトル化
    model = SentenceTransformer("sentence-transformers/all-MiniLM-l6-v2")
    embeddings = model.encode(docs)

    # FAISSインデックスを作成
    d = embeddings.shape[1]  # ベクトルの次元
    index = faiss.IndexFlatL2(d)  # L2距離を使ったインデックス
    index.add(np.array(embeddings).astype('float32'))  # ベクトルをインデックスに追加

    # 質問をベクトル化し、FAISSインデックスを使用して関連する文書を検索
    question_embedding = model.encode([prompt], convert_to_tensor=True)
    question_embedding = question_embedding.cpu().numpy()
    _, I = index.search(question_embedding, 5)

    # 関連する文書の内容を取得
    related_docs = [docs[i] for i in I[0]]

    # 関連する文書を基にして、プロンプトを作成
    prompt_for_gpt = prompt + "\n\n" + "\n\n".join(related_docs)

    # OpenAI GPTチャットAPIを使用してテキスト生成
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_for_gpt}
        ]
    )

    # 生成されたテキストをセッション状態のメッセージに追加
    st.session_state['messages'].append({"role": "assistant", "content": response.choices[0].message['content']})

