import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml

log_dir='logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_path, on_bad_lines="skip")
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', data_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """Apply TfIdf to the data."""
    try:
        vectorizer=TfidfVectorizer(max_features=max_features)
        X_train =train_data['text'].values
        X_test = test_data['text'].values
        y_train = train_data['target'].values
        y_test = test_data['target'].values
        
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        train_df=pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train
        test_df=pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        logger.debug('TfIdf vectorization applied with max_features=%d', max_features)
        return train_df, test_df
    
    except Exception as e:
        logger.error('Error during TF-IDF vectorization: %s', e)
        raise

def save_data(df: pd.DataFrame,  data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        logger.debug('Data saved to %s', data_path)
    except Exception as e:
        logger.error('Error saving data: %s', e)
        raise

def main():
    try:
        max_features = 50
        
        # Split the data into train and test sets
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')
        
        # Apply TfIdf
        train_df, test_df = apply_tfidf(train_data, test_data, max_features)
        
        # Save the processed data
        save_data(train_df, os.path.join("./data", "processed", "train_tfidf.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_tfidf.csv"))
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()