import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import openai

# Streamlitアプリのタイトル
st.title('RAG with STARBUCKS GPT')

# APIキーの入力をサイドバーに移動
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password", key="api_key")

# セッション状態の初期化
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'answers' not in st.session_state:
    st.session_state.answers = []

# 質問の入力をサイドバーに移動
question = st.sidebar.text_input("Enter your question:", key="question")

# ボタンがサイドバーに配置
if st.sidebar.button('Generate Answer'):
    if not api_key:
        st.sidebar.write("Please enter your API Key.")
    elif not question:
        st.sidebar.write("Please enter a question.")
    else:
        # APIキーと質問の処理...
        # ここで質問と回答を処理し、セッション状態に保存
        st.session_state.questions.append(question)
        # 仮の回答をセッション状態に追加（APIからの回答に置き換える）
        st.session_state.answers.append("This is a generated answer for your question.")

# 対話的なチャット画面の表示
for question, answer in zip(st.session_state.questions, st.session_state.answers):
    st.text_area("Question", value=question, height=75, disabled=True)
    st.text_area("Answer", value=answer, height=150, disabled=True)

