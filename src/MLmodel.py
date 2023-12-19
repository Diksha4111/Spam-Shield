import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load your comment dataset
# For YouTube comments, you would typically fetch comments using the YouTube API
# For demonstration purposes, we have a CSV file with 'comment' and 'label' columns
df = pd.read_csv(r"./data/youtube-comments-data.csv")

# Initialize the Porter Stemmer
stemmer = PorterStemmer()

# Tokenize, remove stop words, and apply stemming
stop_words = set(stopwords.words('english'))
df['comment'] = df['comment'].apply(lambda x: ' '.join([stemmer.stem(word) for word in word_tokenize(x) if word.lower() not in stop_words]))

# Split the dataset into features (X) and labels (y)
X = df['comment']
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction with TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Create an SVM classifier
svm_classifier = SVC(kernel='linear')

# Train the SVM classifier
svm_classifier.fit(X_train_tfidf, y_train)

# Calculate accuracy on the test set
y_pred = svm_classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

# Generate classification result
def classification(input_text):

    # Preprocess the text (tokenization, stop words removal, and stemming)
    preprocessed_text = ' '.join([stemmer.stem(word) for word in word_tokenize(input_text) if word.lower() not in stop_words])

    # Transform the preprocessed text using the TF-IDF vectorizer
    text_tfidf = tfidf_vectorizer.transform([preprocessed_text])

    # Use the trained SVM classifier for prediction
    prediction = svm_classifier.predict(text_tfidf)

    # Return the result as JSON
    if prediction[0] == 1:
        result = "spam"
    else:
        result = "not spam"

    return result