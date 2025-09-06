import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the saved model and vectorizer
# Make sure you have saved both files from your training script
try:
    with open('MNB.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        cv = pickle.load(vectorizer_file)
except FileNotFoundError:
    st.error("Error: Model or vectorizer file not found. Please ensure 'MNB.pkl' and 'vectorizer.pkl' exist.")
    st.stop()


# Initialize the stemmer and stopwords
ps = PorterStemmer()
nltk.download('stopwords')
all_stopwords = stopwords.words('english')
all_stopwords.remove('not') # Keep 'not' for accurate sentiment analysis

# Function to preprocess the text
def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    return review

# Set up the Streamlit app
st.set_page_config(page_title="Email Spam Detector", layout="centered")

st.markdown(
    """
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1a1a1a;
        margin-bottom: 0.5em;
    }
    .st-emotion-cache-1g813z1 {
        padding: 1rem;
        background-color: #f0f2f6;
        border-radius: 12px;
        border: 2px solid #e0e0e0;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 1.25rem;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.75rem 1.5rem;
        width: 100%;
        margin-top: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1.5rem;
        text-align: center;
        font-size: 1.2rem;
    }
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1.5rem;
        text-align: center;
        font-size: 1.2rem;
    }
    .stTextArea label {
        font-size: 1.25rem;
        font-weight: bold;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown('<h1 class="main-header">✉️ Email Spam Detector</h1>', unsafe_allow_html=True)
st.markdown("Enter an email message below to see if it's spam or not.", unsafe_allow_html=True)

# Create a text area for user input
input_mail = st.text_area("Enter the email message:", height=200, help="Paste the full email content here.")

# Add a button to trigger the prediction
if st.button("Predict"):
    if input_mail:
        # 1. Preprocess the input text
        preprocessed_text = preprocess_text(input_mail)

        # 2. Transform the preprocessed text using the loaded CountVectorizer
        input_data_features = cv.transform([preprocessed_text])

        # 3. Make a prediction
        prediction = model.predict(input_data_features)

        # 4. Display the prediction result
        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.error("This is a SPAM Mail!")
        else:
            st.success("This is a HAM (Not Spam) Mail!")
    else:
        st.warning("Please enter some text to analyze.")
