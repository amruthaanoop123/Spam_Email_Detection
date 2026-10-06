
import streamlit as st
import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("Spam Email Detector")

message = st.text_input("Enter your message:")

if st.button("Check Message"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        message_tfidf = vectorizer.transform([message])
        prediction = model.predict(message_tfidf)[0]

        if prediction == 1:
            st.error("This message is Spam")
        else:
            st.success("This message is Ham")
