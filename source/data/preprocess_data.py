import pandas as pd
import numpy as np
import os
import re
import ssl
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data'))

STOP_WORDS = {"i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
              "she", "her", "it", "its", "they", "them", "their", "what", "which",
              "who", "this", "that", "am", "is", "are", "was", "were", "be", "been",
              "have", "has", "had", "do", "does", "did", "a", "an", "the", "and",
              "but", "if", "or", "because", "as", "until", "while", "of", "at",
              "by", "for", "with", "about", "to", "from", "in", "out", "on", "off"}

nltk_data_path = os.path.join(DATA_FOLDER, 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
nltk.download('punkt_tab', download_dir=nltk_data_path, quiet=True)

stemmer = PorterStemmer()

def encode_label(label):
    return 1 if label.lower() == 'spam' else 0

def text_preprocessing_pipeline(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '<NUM>', text)
    text = re.sub(r'[^\w\s<>]', '', text)

    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in STOP_WORDS]
    stemmed_words = [stemmer.stem(w) for w in filtered_words if w.isalnum() or w == '<NUM>']

    return " ".join(stemmed_words)

if __name__ == "__main__":
    data_path = os.path.join(DATA_FOLDER, 'spam.csv')

    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
    else: 
        dataset = pd.read_csv(data_path, encoding='latin-1')
        dataset = dataset[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})

        dataset['label'] = dataset['label'].apply(encode_label)
        dataset['text'] = dataset['text'].apply(text_preprocessing_pipeline)

        output_path = os.path.join(DATA_FOLDER, 'preprocessed_spam.csv')
        dataset.to_csv(output_path, index=False)
        
        print(f"Process completed. File saved at: {output_path}")
        print(dataset.head())