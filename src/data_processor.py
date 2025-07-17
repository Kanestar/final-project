# src/data_processor.py
import pandas as pd
import re
from sklearn.model_selection import train_test_split

class DataProcessor:
    def load_data(self, file_path):
        columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
        df = pd.read_csv(file_path, encoding='latin-1', names=columns)
        
        # Clean text
        df['text'] = df['text'].apply(self.clean_text)
        
        # Convert sentiment 4 → 1 (positive), 0 stays as negative
        df['target'] = df['target'].replace(4, 1)
        
        return df[['text', 'target']]

    def clean_text(self, text):
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.lower().strip()

    def split_data(self, df, test_size=0.2):
        X = df['text']
        y = df['target']
        return train_test_split(X, y, test_size=test_size, random_state=42)
