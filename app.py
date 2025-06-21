# filepath: d:\chat_bot\app.py
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# ...existing code...

# app.py

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sentence_transformers import SentenceTransformer, util

# ==== App Setup ====
st.set_page_config(page_title="Indian Legal Rights Chatbot", layout="centered")
st.title("🧠 Indian Legal Rights Chatbot")
st.caption("Ask me about your legal rights in India. Example: *What are my rights if I'm arrested?*")

# ==== Load Resources ====
@st.cache_resource
def load_all():
    df = pd.read_csv("ilr_updated.csv")
    model = tf.keras.models.load_model("legal_rights_chat_model.h5")
    tokenizer = pickle.load(open("legal_tokenizer.pkl", "rb"))
    encoder = pickle.load(open("legal_encoder.pkl", "rb"))
    sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    questions = df['Question'].astype(str).tolist()
    tags = df['Tag'].astype(str).tolist()
    answers = df['Answer'].astype(str).tolist()
    qa_pairs = list(zip(questions, tags, answers))
    question_embeddings = sbert.encode(questions, convert_to_tensor=True)
    return df, model, tokenizer, encoder, sbert, qa_pairs, question_embeddings

df, model, tokenizer, lbl_encoder, semantic_model, qa_pairs, question_embeddings = load_all()

vocab_size = 3000
max_len = 40

# ==== Semantic Matching ====
def get_best_semantic_answer(user_input, predicted_tag):
    filtered = [(q, a) for (q, t, a) in qa_pairs if t == predicted_tag]
    if not filtered:
        return "Sorry, I couldn't find anything related."

    f_qs = [q for (q, _) in filtered]
    f_as = [a for (_, a) in filtered]
    f_embeddings = semantic_model.encode(f_qs, convert_to_tensor=True)
    user_embedding = semantic_model.encode(user_input, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(user_embedding, f_embeddings)[0]
    best_idx = int(sims.argmax())
    return f_as[best_idx]

# ==== Chat Logic ====
def get_chatbot_response(user_input):
    seq = tokenizer.texts_to_sequences([user_input])
    padded_seq = pad_sequences(seq, maxlen=max_len, padding='post')
    prediction = model.predict(padded_seq, verbose=0)
    prob = np.max(prediction)
    tag = lbl_encoder.inverse_transform([np.argmax(prediction)])[0]

    if prob > 0.6:
        answer = get_best_semantic_answer(user_input, tag)
        return answer, round(prob * 100, 2)
    else:
        return "I'm not confident enough to answer that. Try rephrasing.", round(prob * 100, 2)

# ==== Chat Interface ====
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:", placeholder="Type your legal question here...")

if st.button("Ask") or (user_input and st.session_state.history and st.session_state.history[-1]["user"] != user_input):
    response, confidence = get_chatbot_response(user_input)
    st.session_state.history.append({"user": user_input, "bot": response, "conf": confidence})

# Display chat history
for entry in st.session_state.history:
    st.markdown(f"**You:** {entry['user']}")
    st.markdown(f"**Bot:** {entry['bot']}")
    st.caption(f"Confidence: {entry['conf']}%")
    st.markdown("---")
