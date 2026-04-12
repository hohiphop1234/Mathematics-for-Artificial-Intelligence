import pandas as pd
import numpy as np
import os
import math
from collections import Counter
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'data'))

def build_vocabulary(docs):
    vocab = set()
    for doc in docs:
        vocab.update(str(doc).split())
    return sorted(list(vocab))

def calculate_idf(docs, vocab):
    num_docs = len(docs)
    df_counts = Counter()
    doc_word_sets = [set(str(doc).split()) for doc in docs]
    for word_set in doc_word_sets:
        df_counts.update(word_set)
    
    return {word: math.log((1 + num_docs) / (1 + df_counts[word])) + 1 for word in vocab}

def calculate_tfidf(doc, vocab, idf):
    words = str(doc).split()
    counts = Counter(words)
    total_words = len(words)
    
    if total_words == 0:
        return [0.0] * len(vocab)
    
    vector = [(counts[word] / total_words) * idf.get(word, 0) for word in vocab]
    norm = math.sqrt(sum(x**2 for x in vector))
    return [x / norm for x in vector] if norm > 0 else vector

if __name__ == "__main__":
    input_path = os.path.join(DATA_FOLDER, 'preprocessed_spam.csv')
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
    else:
        df = pd.read_csv(input_path).fillna('')
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

        print("Processing TF-IDF...")
        vocab = build_vocabulary(train_df['text'])
        idf_scores = calculate_idf(train_df['text'], vocab)

        X_train = np.array([calculate_tfidf(d, vocab, idf_scores) for d in train_df['text']], dtype=np.float32)
        X_test = np.array([calculate_tfidf(d, vocab, idf_scores) for d in test_df['text']], dtype=np.float32)

        save_path = os.path.join(DATA_FOLDER, 'spam_features.npz')
        np.savez_compressed(
            save_path, 
            X_train=X_train, 
            X_test=X_test, 
            y_train=train_df['label'].values, 
            y_test=test_df['label'].values
        )

        print(f"Optimization complete. Data compressed and saved to: {save_path}")
        print(f"Vocabulary size: {len(vocab)}")



# data = np.load(os.path.join(DATA_FOLDER, 'spam_features.npz'))
# X_train, y_train = data['X_train'], data['y_train']