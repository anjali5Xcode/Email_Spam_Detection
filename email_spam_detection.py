import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pickle

# --- 1. Load the Dataset ---
try:
    data = pd.read_csv('spam.csv', encoding='latin-1')
except FileNotFoundError:
    print("Error: The file 'spam.csv' was not found.")
    exit()

# --- 2. Preprocessing the Dataset ---
nltk.download('stopwords')
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

corpus = []
for i in range(0, len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['v2'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# --- 3. Creating the Bag of Words model ---
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = data.iloc[:, 0].values
y = pd.get_dummies(y)
y = y.iloc[:, 1].values

# --- 4. Splitting the Dataset into Training and Testing sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# --- 5. Training the Naive Bayes model on the Training set ---
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# --- 6. Predicting the Test set results ---
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# --- 7. Saving the model and vectorizer to disk ---
with open('MNB.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(cv, vectorizer_file)

print("\nTraining complete. Model and vectorizer saved as 'MNB_model.pkl' and 'vectorizer.pkl'.")
