import streamlit as st
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
prompt = st.text_input("質問を入力してください:", value="", key="prompt")

# APIキーと質問が入力されたら処理を実行
if prompt and api_key:
    # ユーザーの質問をセッション状態のメッセージに追加
    st.session_state['messages'].append({"role": "user", "content": prompt})

    # OpenAIのAPIキーを設定
    openai.api_key = api_key

    # JSONファイルを読み込む
    file_path = 'starbucks_data.json'  # JSONファイルへのパス
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        st.error("JSONファイルが見つかりません。")
        st.stop()
    except json.JSONDecodeError:
        st.error("JSONファイルの形式が不正です。")
        st.stop()

    # 文書を生成
    data: List[str] = [f"{d['Beverage_category']} {d['Beverage']} {d['Beverage_prep']}" for d in data]

    # Hugging Faceモデルをロードして文書をベクトル化
    model = SentenceTransformer("sentence-transformers/all-MiniLM-l6-v2")
    embeddings: np.ndarray = model.encode(data)

    # 質問をベクトル化し、FAISSインデックスを使用して関連する文書を検索
    question_embedding = model.encode([prompt], convert_to_tensor=True)
    question_embedding = question_embedding.cpu().numpy()
    _, I = faiss.IndexFlatL2(embeddings.shape[1]).search(question_embedding, 5)

    # 関連する文書の内容を取得
    related_docs: List[str] = [data[i] for i in I[0]]

    # 関連する文書を基にして、プロンプトを作成
    prompt_for_gpt = prompt + "\n\n" + "\n\n".join(related_docs)

    # OpenAI GPTチャットAPIを使用してテキスト生成
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_for_gpt}
            ]
        )
    except openai.error.OpenAIError as e:
        st.error(e.message)
        st.stop()

    # 生成されたテキストをセッション状態のメッセージに追加
    st.session_state['messages'].append({"role": "assistant", "content": response.choices[0].message['content']})


