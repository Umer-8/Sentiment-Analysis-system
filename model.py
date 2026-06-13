import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


# ===== DATA PREPROCESSING =====

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

df = pd.read_csv("dataset.csv")
df['clean_text'] = df['text'].apply(preprocess_text)


# ===== FEATURE ENGINEERING (TF-IDF) =====

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']


# ===== MODEL TRAINING =====

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# ===== MODEL EVALUATION =====

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# ===== CONFUSION MATRIX VISUALIZATION =====

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# ===== SAVE MODEL & VECTORIZER =====

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Model and vectorizer saved successfully!")


# ===== PREDICTION WITH PROBABILITIES =====

def predict_sentiment_with_proba(text):
    cleaned = preprocess_text(text)
    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]

    print("Predicted Sentiment:", prediction)
    print("\nProbabilities:")
    for label, prob in zip(model.classes_, probabilities):
        print(f"  {label}: {prob:.2%}")

    return prediction, probabilities


# ===== BATCH PREDICTION =====

def batch_predict(sentences):
    cleaned = [preprocess_text(s) for s in sentences]
    vectors = vectorizer.transform(cleaned)

    predictions = model.predict(vectors)
    probabilities = model.predict_proba(vectors)

    results = []
    for sentence, pred, probs in zip(sentences, predictions, probabilities):
        results.append({
            "text": sentence,
            "predicted_sentiment": pred,
            "confidence": f"{max(probs):.2%}"
        })

    return results


# ===== EXPORT RESULTS TO CSV =====

def export_to_csv(results, filename="batch_predictions.csv"):
    df_results = pd.DataFrame(results)
    df_results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


# ===== CLI MENU =====

def cli_menu():
    while True:
        print("\n===== Sentiment Analysis System =====")
        print("1. Predict single sentence")
        print("2. Batch prediction")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ")

        if choice == "1":
            text = input("Enter your sentence: ")
            predict_sentiment_with_proba(text)

        elif choice == "2":
            n = int(input("How many sentences? "))
            sentences = []
            for i in range(n):
                sentences.append(input(f"Sentence {i+1}: "))

            results = batch_predict(sentences)
            for r in results:
                print(r)

            save = input("Export results to CSV? (y/n): ")
            if save.lower() == "y":
                export_to_csv(results)

        elif choice == "3":
            print("Exiting... Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")


cli_menu()