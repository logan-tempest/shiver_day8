# legal_rights_chatbot.py

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sentence_transformers import SentenceTransformer, util

# ============ STEP 1: Load CSV ============
CSV_FILE = "ilr_updated.csv"  # Ensure this file is in your project directory
df = pd.read_csv(CSV_FILE)

# ============ STEP 2: Prepare Data ============
questions = df['Question'].astype(str).tolist()
tags = df['Tag'].astype(str).tolist()
answers = df['Answer'].astype(str).tolist()
qa_pairs = list(zip(questions, tags, answers))

# Encode the tags
lbl_encoder = LabelEncoder()
encoded_tags = lbl_encoder.fit_transform(tags)
num_classes = len(lbl_encoder.classes_)

# Tokenization for classification
vocab_size = 3000
max_len = 40
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(questions)
sequences = tokenizer.texts_to_sequences(questions)
padded = pad_sequences(sequences, padding='post', maxlen=max_len)

# ============ STEP 3: Build & Train Model ============
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64, input_length=max_len),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    padded,
    np.array(encoded_tags),
    epochs=300,
    batch_size=8,
    validation_split=0.15,
    verbose=1
)

# Save model and encoders
model.save("legal_rights_chat_model.h5")
with open("legal_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("legal_encoder.pkl", "wb") as f:
    pickle.dump(lbl_encoder, f)

# ============ STEP 4: Load Semantic Search ============
semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
question_embeddings = semantic_model.encode(questions, convert_to_tensor=True)

# ============ STEP 5: Semantic Match Function ============
def find_best_semantic_match(user_input, predicted_tag):
    filtered = [(q, a) for (q, t, a) in qa_pairs if t == predicted_tag]
    if not filtered:
        return "Sorry, I couldn't find anything for that category."

    filtered_questions = [q for (q, _) in filtered]
    filtered_answers = [a for (_, a) in filtered]
    filtered_embeddings = semantic_model.encode(filtered_questions, convert_to_tensor=True)

    user_embedding = semantic_model.encode(user_input, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, filtered_embeddings)[0]
    best_match_idx = int(similarities.argmax())

    return filtered_answers[best_match_idx]

# ============ STEP 6: Chat Loop ============
def chat():
    print("🧠 Indian Legal Rights Chatbot Ready! Type 'quit' to exit.\n")
    while True:
        user = input("You: ").strip()
        if user.lower() == "quit":
            print("Bot: Thank you for using the Legal Rights Chatbot!")
            break

        seq = tokenizer.texts_to_sequences([user])
        padded_seq = pad_sequences(seq, maxlen=max_len, padding='post')
        prediction = model.predict(padded_seq, verbose=0)
        prob = np.max(prediction)
        predicted_tag = lbl_encoder.inverse_transform([np.argmax(prediction)])[0]

        if prob > 0.6:
            print("Accuracy:", round(prob * 100, 2), "%")
            answer = find_best_semantic_match(user, predicted_tag)
            print("Bot:", answer)
        else:
            print("Accuracy:", round(prob * 100, 2), "%")
            print("Bot: Sorry, I’m not confident enough. Please try rephrasing.")

# Run chatbot
if __name__ == "__main__":
    chat()
