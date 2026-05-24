import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("dataset/emails.csv")
data['label_num'] = data.label.map({
    'safe': 0,
    'phishing': 1
})
X = data['text']
y = data['label_num']
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)
st.title("📧 AI-Powered Phishing Email Detection")
st.write("Detect whether an email is Safe or Phishing.")
email_input = st.text_area("Enter Email Content")

if st.button("Detect Email"):
    if email_input.strip() == "":
        st.warning("⚠️ Please enter email content")
    
    else:
        email_vector = vectorizer.transform([email_input])
        prediction = model.predict(email_vector)
        if prediction[0] == 1:
            st.error("⚠️ PHISHING EMAIL DETECTED")
        else:
            st.success("✅ SAFE EMAIL")