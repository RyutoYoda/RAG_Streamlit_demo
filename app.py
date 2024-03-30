import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
import openai
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Streamlitアプリのタイトル
st.title('Barista GPT')

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
