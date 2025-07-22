import streamlit as st
import joblib

model = joblib.load('spam_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("Spam or Ham Classifier")

st.write("Enter an email or SMS message and find out if it's spam or ham!")

user_input = st.text_area("Enter your message:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Preprocess input
        input_vec = vectorizer.transform([user_input])
        
        # Predict
        prediction = model.predict(input_vec)[0]
        
        if prediction == 1:
            st.error("This is **Spam**.")
        else:
            st.success("This is **Ham**.")
