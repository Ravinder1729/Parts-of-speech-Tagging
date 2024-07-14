import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
st.image(r"C:\Users\ravin\Downloads\speech.webp")
Md2 = pickle.load(open('tk_y4.pkl', 'rb'))
Md3 = pickle.load(open('tk_x4.pkl', 'rb'))
Mdl1 = pickle.load(open('pos4.pkl', 'rb'))
test = st.text_input("Enter your input sequence")
if st.button("Predict"):
    test_tokens = test.split() 
    sequences = Md3.texts_to_sequences([test])  
    te = pad_sequences(sequences, maxlen=271, padding="post")
    pred = Mdl1.predict(te)
    pred_sequence = np.argmax(pred[0], axis=1)
    pred_sequence = pred_sequence[pred_sequence != 0]
    prediction_labels = Md2.sequences_to_texts([pred_sequence])[0].split()
    for token, label in zip(test_tokens, prediction_labels):
        st.title(f"{token} - {label}")
