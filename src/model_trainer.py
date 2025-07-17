# src/model_trainer.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import mlflow
import joblib

class SentimentModel:
    def __init__(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
            ('clf', LogisticRegression(max_iter=200))
        ])
    
    def train(self, X_train, y_train, X_test, y_test):
        with mlflow.start_run():
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(self.model, "sentiment_model")
            print(f"✅ Model trained - Accuracy: {accuracy:.4f}")
            return accuracy
    
    def save(self, path="models/sentiment_model.pkl"):
        joblib.dump(self.model, path)
        print(f"✅ Model saved to {path}")
