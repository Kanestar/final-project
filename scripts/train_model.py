import sys
import os

# Add the root project folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processor import DataProcessor
from src.model_trainer import SentimentModel
import mlflow
# scripts/train_model.py
from src.data_processor import DataProcessor
from src.model_trainer import SentimentModel
import mlflow

def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("MindAI-Experiment")

    # Load and preprocess
    processor = DataProcessor()
    df = processor.load_data("data/sentiment140.csv")
    X_train, X_test, y_train, y_test = processor.split_data(df)

    # Train and save
    model = SentimentModel()
    accuracy = model.train(X_train, y_train, X_test, y_test)
    model.save()

if __name__ == "__main__":
    main()
