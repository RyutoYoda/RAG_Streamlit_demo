import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
import openai
os.environ["TOKENIZERS_PARALLELISM"] = "false"
st.set_page_config(
    page_title="Barista Chat",
    page_icon="☕️"
)
# Streamlitアプリのタイトル
st.title('Barista Chat☕️')
# スタイル設定
st.markdown("""
<style>
body {
    font-family: 'Helvetica Neue', sans-serif;
}
.big-font {
    font-size:50px !important;
    font-weight: bold;
    text-align: center;
    margin-bottom: 30px;
}
.header-font {
    font-size:30px !important;
    font-weight: bold;
    margin-bottom: 20px;
}
.subheader-font {
    font-size:20px !important;
    font-weight: bold;
    margin-bottom: 10px;
}
.container {
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}
.stButton>button {
    font-size: 16px !important;
    font-weight: bold !important;
    border-radius: 5px !important;
    width: 100%;
    padding: 10px;
}
.stTextInput>div>div>input {
    border-radius: 5px !important;
    border: 1px solid !important;
}
ul {
    list-style-type: none;
    padding: 0;
}
li {
    margin: 10px 0;
    padding: 10px;
    border-radius: 5px;
    box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.1);
}
.header-container {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
    text-align: center;
}
.sidebar-img {
    display: block;
    margin-left: auto;
    margin-right: auto;
    margin-bottom: 20px;
}
.main-content {
    display: flex;
    flex-direction: column;
    align-items: center;
}
.card {
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    margin: 20px;
    padding: 20px;
    width: 90%;
    max-width: 700px;
    text-align: left;
}
.card img {
    border-radius: 10px;
}
@media (prefers-color-scheme: dark) {
    body {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .big-font, .header-font {
        color: #61dafb;
    }
    .subheader-font {
        color: #a9a9a9;
    }
    .container, .card, li {
        background-color: #282c34;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5);
    }
    .stButton>button {
        background-color: #61dafb !important;
        color: #282c34 !important;
    }
    .stTextInput>div>div>input {
        border: 1px solid #61dafb !important;
        color: #ffffff !important;
        background-color: #3c3f41 !important;
    }
}
@media (prefers-color-scheme: light) {
    body {
        background-color: #f5f5f5;
        color: #333333;
    }
    .big-font, .header-font {
        color: #007bff;
    }
    .subheader-font {
        color: #666666;
    }
    .container, .card, li {
        background-color: #ffffff;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #007bff !important;
        color: #ffffff !important;
    }
    .stTextInput>div>div>input {
        border: 1px solid #cccccc !important;
    }
}
</style>
""", unsafe_allow_html=True)

# APIキーの入力
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")


# 質問の入力
question = st.text_input("Enter your question:")

# ボタンが押されたら処理を実行
if st.button('Generate Answer'):
    if not api_key:
        st.write("Please enter your API Key.")
    elif not question:
        st.write("Please enter a question.")
    else:
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

        # インデックスをファイルに保存
        faiss.write_index(index, "faiss_index.bin")
        
        # FAISSインデックスを読み込み
        index = faiss.read_index("faiss_index.bin")

        # 質問をベクトル化し、FAISSインデックスを使用して関連する文書を検索
        question_embedding = model.encode([question], convert_to_tensor=True)
        question_embedding = question_embedding.cpu().numpy()
        _, I = index.search(question_embedding, 5)

        # 関連する文書の内容を取得
        related_docs = [docs[i] for i in I[0]]

        # 関連する文書を基にして、プロンプトを作成
        prompt = question + "\n\n" + "\n\n".join(related_docs)

        # OpenAI GPTチャットAPIを使用してテキスト生成
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt},  # 変更: ユーザーメッセージを追加
            ]
        )
        # 生成されたテキストを表示
        st.write(response.choices[0].message['content'])
