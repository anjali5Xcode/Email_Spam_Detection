import streamlit as st
import pickle

# Load the saved model and vectorizer
with open('MNB.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    cv = pickle.load(vectorizer_file)

# Add a title to the Streamlit app
st.title("Email Spam Detection")

# Create a text area for user input
input_mail = st.text_area("Enter the email message:")

# Add a button to trigger the prediction
if st.button("Predict"):
    # Transform the user input text
    input_data_features = cv.transform([input_mail])

    # Make a prediction
    prediction = model.predict(input_data_features)

    # Display the prediction result
    if prediction[0] == 1:
        st.error("Spam Mail")
    else:
        st.success("Ham Mail")
