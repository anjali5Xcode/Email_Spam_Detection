✉️ Email Spam Detection with Machine Learning

This project is a machine learning-based application for classifying emails as "spam" or "ham" (not spam). The application is built with Streamlit, providing a simple and interactive web interface where users can paste an email message and get an instant prediction.
The core of the project relies on a Multinomial Naive Bayes classifier trained on a large dataset of email messages. The text data is preprocessed and converted into a numerical format using the Bag of Words model before being fed to the machine learning model.


🌐 Live Application

You can try the live application here:
https://emailspamdetection-kgrawlwnq7fxtefmnde4vp.streamlit.app/#prediction-result


✨ Features

Real-time Prediction: Get instant spam/ham predictions for any email text you input.

Machine Learning Powered: Utilizes a trained Naive Bayes classifier for accuracy.

Intuitive UI: A clean and user-friendly interface built with Streamlit.

Data Preprocessing: Includes steps for cleaning and vectorizing raw text data.


⚙️ How It Works

Model Architecture
The project follows a standard machine learning workflow:

Data Preprocessing: Raw email text is cleaned by removing punctuation, converting to lowercase, and stemming words to their root form.

Feature Extraction: The preprocessed text is converted into a numerical format using a CountVectorizer (Bag of Words model). This creates a matrix where each row represents an email and each column represents a unique word, with values indicating the frequency of that word.

Model Training: A MultinomialNB model is trained on this numerical data to learn the patterns that differentiate spam from ham emails.

Persistence: The trained model (MNB_model.pkl) and the CountVectorizer (vectorizer.pkl) are saved to disk.


Application Logic
The Streamlit application (app.py) performs the following steps in a continuous loop:

Loads the pre-trained model and vectorizer from the .pkl files.

Waits for user input in the text area.

When the "Predict" button is clicked, it preprocesses the new email text using the same function from the training script.

Transforms the preprocessed text into a numerical vector using the loaded CountVectorizer.

Passes this vector to the MultinomialNB model for prediction.

Displays the result ("SPAM" or "HAM") to the user.


🚀 Getting Started
To set up and run this project locally, follow these steps.

Prerequisites
You need to have Python installed on your machine. The following libraries are also required:

streamlit

scikit-learn

pandas

nltk

Installation
Clone the repository to your local machine (if applicable), then install the required libraries:

pip install streamlit scikit-learn pandas nltk


Project Structure

email_spam_detection.py: The script used for training the machine learning model and saving the .pkl files.

app.py: The Streamlit application that provides the web interface.

spam_mail.csv: The dataset used for training the model. (Not included here, but you will need a similar dataset).

MNB_model.pkl: The trained machine learning model.

vectorizer.pkl: The fitted vectorizer object.


Running the Application
Train the Model: First, run the training script to generate the model and vectorizer files.
python email_spam_detection.py
Run the App: Once the .pkl files have been created, run the Streamlit application.

streamlit run app.py

This will launch the web application in your default browser.


🤝 Contribution
Feel free to open an issue or submit a pull request if you have suggestions for improvements.
