from flask import Flask, render_template, request
import pandas as pd
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

def preprocess_text(text):
    # Tokenize the text into words
    words = nltk.word_tokenize(text.lower())
    
    # Remove stopwords
    stopwords = set(nltk.corpus.stopwords.words('english'))
    words = [word for word in words if word not in stopwords]
    
    # Reconstruct the preprocessed text
    preprocessed_text = " ".join(words)
    return preprocessed_text

df = pd.read_csv("model\\data.csv")

df = df.head(4000)

df['Data'] = df['Data'].apply(preprocess_text)

# Extract features and labels
X = df['Data']
y = df['Value']

# Convert the features into a numerical format using TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(X)


@app.route("/", methods=["GET","POST"])
def predict_sentiment():
    if request.method == "GET":
        return render_template("home.html",sentimentResult = None)
    else:
        with open('model\\model.pkl', 'rb') as f:
            new_classifier = pickle.load(f)

        # Get the input text from the form
        input_text = request.form["input_text"]
        exm = preprocess_text(input_text)
        example_input_vector = tfidf_vectorizer.transform([exm])
        prediction = new_classifier.predict(example_input_vector)
        print(prediction[0])
        # Return the prediction to the template
        
        return render_template("home.html", sentimentResult=prediction[0])

if __name__ == "__main__":
    app.run(debug=True, port=5555)
