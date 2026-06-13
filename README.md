# Multi-Class Sentiment Analysis System

A text classification system that predicts sentiment as **Positive**, **Negative**, **Neutral**, or **Mixed** using NLP and Machine Learning.

## 🎯 Objective
Build a multi-class sentiment analysis model that can detect not just positive/negative sentiment, but also neutral statements and mixed opinions (text containing both positive and negative views).

## 🛠 Tools & Libraries
- Python
- pandas
- scikit-learn
- NLTK
- matplotlib & seaborn

## 📊 Dataset
A custom dataset (`dataset.csv`) with 100 labeled sentences, 25 per class:
- Positive 😊
- Negative 😡
- Neutral 😐
- Mixed 🤔

## 🔄 Workflow
1. **Data Preprocessing** – lowercase conversion, punctuation removal, tokenization, stopword removal, lemmatization
2. **Feature Engineering** – TF-IDF vectorization
3. **Model Training** – Logistic Regression (80/20 train-test split)
4. **Model Evaluation** – Accuracy, Precision, Recall, F1-score, Confusion Matrix
5. **Prediction System** – CLI-based single and batch prediction with confidence scores

## 📈 Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

## ▶️ How to Run
1. Install dependencies:
```bash
pip install pandas scikit-learn nltk matplotlib seaborn joblib
```

2. Run the script:
```bash
python model.py
```

3. Use the CLI menu to:
   - Predict sentiment for a single sentence
   - Run batch predictions on multiple sentences
   - Export batch results to CSV

## 📝 Example
**Input:**
```
I like the features but the app is very slow
```

**Output:**
```
Predicted Sentiment: Mixed
```

## 💾 Output Files
- `sentiment_model.pkl` – trained Logistic Regression model
- `tfidf_vectorizer.pkl` – fitted TF-IDF vectorizer
- `batch_predictions.csv` – exported batch prediction results
