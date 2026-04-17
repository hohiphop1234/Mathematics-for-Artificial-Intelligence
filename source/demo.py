"""
Demo phân loại email Spam / Ham (Not Spam).

Cách chạy (từ thư mục gốc của repo):
    python source/demo.py

Chương trình sẽ:
  1. Đọc dữ liệu đã tiền xử lý ở data/processed_spam.csv
  2. Chia train/test (80/20, stratify theo nhãn)
  3. Xây dựng TF-IDF (thủ công, dùng lại logic của train_test_split.py)
  4. Huấn luyện Logistic Regression (sklearn)
  5. In ra Precision, Recall, F1-score và Confusion Matrix trên tập test
  6. Mở vòng lặp nhập nội dung email để dự đoán Spam / Ham
"""

import os
import sys
import math
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.abspath(os.path.join(BASE_DIR, '..', 'data'))

sys.path.append(os.path.join(BASE_DIR, 'data'))
from preprocess_data import text_preprocessing_pipeline  # noqa: E402


def build_vocabulary(docs):
    vocab = set()
    for doc in docs:
        vocab.update(str(doc).split())
    return sorted(list(vocab))


def calculate_idf(docs, vocab):
    num_docs = len(docs)
    df_counts = Counter()
    for doc in docs:
        df_counts.update(set(str(doc).split()))
    return {w: math.log((1 + num_docs) / (1 + df_counts[w])) + 1 for w in vocab}


def calculate_tfidf(doc, vocab_index, idf):
    words = str(doc).split()
    if not words:
        return np.zeros(len(vocab_index), dtype=np.float32)

    counts = Counter(words)
    total = len(words)
    vec = np.zeros(len(vocab_index), dtype=np.float32)
    for w, c in counts.items():
        idx = vocab_index.get(w)
        if idx is not None:
            vec[idx] = (c / total) * idf[w]

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def load_dataset():
    path = os.path.join(DATA_FOLDER, 'processed_spam.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Hãy chạy source/data/preprocess_data.py trước."
        )
    df = pd.read_csv(path).fillna('')

    if 'processed_text' in df.columns and 'encoded_label' in df.columns:
        df = df[['processed_text', 'encoded_label']].rename(
            columns={'processed_text': 'text', 'encoded_label': 'label'}
        )
    elif 'v1' in df.columns and 'v2' in df.columns:
        df = df.rename(columns={'v1': 'label', 'v2': 'text'})[['text', 'label']]
        df['label'] = df['label'].apply(lambda s: 1 if str(s).lower() == 'spam' else 0)
    else:
        df = df[['text', 'label']]

    df['label'] = df['label'].astype(int)
    return df


def train_model(verbose=True):
    df = load_dataset()

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )

    if verbose:
        print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")
        print("Đang xây dựng TF-IDF...")

    vocab = build_vocabulary(train_df['text'])
    vocab_index = {w: i for i, w in enumerate(vocab)}
    idf = calculate_idf(train_df['text'], vocab)

    X_train = np.vstack([calculate_tfidf(d, vocab_index, idf) for d in train_df['text']])
    X_test = np.vstack([calculate_tfidf(d, vocab_index, idf) for d in test_df['text']])
    y_train = train_df['label'].values
    y_test = test_df['label'].values

    if verbose:
        print(f"Kích thước từ điển: {len(vocab)}")
        print("Đang huấn luyện Logistic Regression...")

    model = LogisticRegression(solver='liblinear', max_iter=10000)
    model.fit(X_train, y_train)

    return model, vocab_index, idf, (X_test, y_test)


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (nhãn dương = SPAM)")
    print("=" * 60)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (trong các email bị gắn SPAM, bao nhiêu đúng)")
    print(f"  Recall    : {rec:.4f}  (trong các email SPAM thật, bao nhiêu bắt được)")
    print(f"  F1-score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print("                 Pred: HAM   Pred: SPAM")
    print(f"  Actual: HAM    {cm[0, 0]:>9d}   {cm[0, 1]:>9d}")
    print(f"  Actual: SPAM   {cm[1, 0]:>9d}   {cm[1, 1]:>9d}")

    print("\nClassification Report đầy đủ:")
    print(classification_report(y_test, y_pred, target_names=['HAM', 'SPAM'], digits=4))


def predict_email(text, model, vocab_index, idf):
    processed = text_preprocessing_pipeline(text)
    vec = calculate_tfidf(processed, vocab_index, idf).reshape(1, -1)

    pred = int(model.predict(vec)[0])
    prob_spam = float(model.predict_proba(vec)[0, 1])
    label = 'SPAM' if pred == 1 else 'HAM (Not Spam)'
    return label, prob_spam, processed


def explain_prediction(processed_text, model, vocab_index, idf, top_k=10):
    """Phân tích đóng góp của từng token vào quyết định logistic regression.

    contribution = tfidf(token) * weight(token)
    - > 0 : đẩy email về phía SPAM
    - < 0 : đẩy email về phía HAM
    """
    weights = model.coef_[0]
    bias = float(model.intercept_[0])

    words = processed_text.split()
    if not words:
        return {'logit': bias, 'bias': bias, 'contributions': [], 'unique_tokens': []}

    counts = Counter(words)
    total = len(words)
    vec = np.zeros(len(vocab_index), dtype=np.float32)
    for w, c in counts.items():
        idx = vocab_index.get(w)
        if idx is not None:
            vec[idx] = (c / total) * idf[w]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    contributions = []
    for w in set(words):
        idx = vocab_index.get(w)
        if idx is None:
            contributions.append((w, None, None, 0.0, False))
            continue
        tfidf_val = float(vec[idx])
        weight_val = float(weights[idx])
        contrib = tfidf_val * weight_val
        contributions.append((w, tfidf_val, weight_val, contrib, True))

    contributions.sort(key=lambda x: x[3], reverse=True)
    logit = bias + float(np.dot(vec, weights))
    return {
        'logit': logit,
        'bias': bias,
        'contributions': contributions,
        'top_k': top_k,
    }


def print_explanation(explanation):
    contribs = explanation['contributions']
    top_k = explanation['top_k']
    in_vocab = [c for c in contribs if c[4]]
    oov = [c for c in contribs if not c[4]]

    spam_push = [c for c in in_vocab if c[3] > 0][:top_k]
    ham_push = sorted([c for c in in_vocab if c[3] < 0], key=lambda x: x[3])[:top_k]

    total_spam_contrib = sum(c[3] for c in in_vocab if c[3] > 0)
    total_ham_contrib = sum(c[3] for c in in_vocab if c[3] < 0)

    print("\n  --- Phân tích đóng góp của từng từ (logit = bias + sum(tfidf*weight)) ---")
    print(f"  bias           = {explanation['bias']:+.4f}")
    print(f"  sum(+) SPAM    = {total_spam_contrib:+.4f}")
    print(f"  sum(-) HAM     = {total_ham_contrib:+.4f}")
    print(f"  logit tổng     = {explanation['logit']:+.4f}"
          f"  (logit > 0 => SPAM, logit < 0 => HAM)")

    if spam_push:
        print(f"\n  Top {len(spam_push)} từ đẩy về SPAM:")
        print(f"    {'token':<20}{'tfidf':>10}{'weight':>10}{'contrib':>12}")
        for w, tfidf_val, weight_val, contrib, _ in spam_push:
            print(f"    {w:<20}{tfidf_val:>10.4f}{weight_val:>+10.4f}{contrib:>+12.4f}")
    else:
        print("\n  Không có từ nào đẩy về phía SPAM.")

    if ham_push:
        print(f"\n  Top {len(ham_push)} từ đẩy về HAM:")
        print(f"    {'token':<20}{'tfidf':>10}{'weight':>10}{'contrib':>12}")
        for w, tfidf_val, weight_val, contrib, _ in ham_push:
            print(f"    {w:<20}{tfidf_val:>10.4f}{weight_val:>+10.4f}{contrib:>+12.4f}")

    if oov:
        oov_tokens = [c[0] for c in oov]
        print(f"\n  OOV (không có trong vocab, bỏ qua): {' '.join(oov_tokens)}")


def interactive_loop(model, vocab_index, idf):
    print("\n" + "=" * 60)
    print("DEMO PHÂN LOẠI EMAIL")
    print("=" * 60)
    print("Nhập nội dung email để phân loại.")
    print("Gõ 'exit' hoặc 'quit' để thoát. Để nhập nhiều dòng, kết thúc bằng 'END'.")
    print("-" * 60)

    while True:
        try:
            first = input("\nEmail > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nThoát demo.")
            break

        if first.lower() in {"exit", "quit"}:
            print("Thoát demo.")
            break
        if not first:
            continue

        if first.upper() == 'MULTI':
            print("(Chế độ nhiều dòng — kết thúc bằng 1 dòng chỉ ghi END)")
            lines = []
            while True:
                line = input()
                if line.strip() == 'END':
                    break
                lines.append(line)
            text = "\n".join(lines)
        else:
            text = first

        label, prob_spam, processed = predict_email(text, model, vocab_index, idf)
        print(f"  Văn bản sau tiền xử lý : {processed[:120]}{'...' if len(processed) > 120 else ''}")
        print(f"  Xác suất SPAM          : {prob_spam:.4f}  ({prob_spam*100:.2f}%)")
        print(f"  => Kết quả dự đoán     : {label}")

        explanation = explain_prediction(processed, model, vocab_index, idf, top_k=10)
        print_explanation(explanation)


def run_builtin_examples(model, vocab_index, idf):
    examples = [
        ("Hi John, are we still meeting for lunch tomorrow at 12?", "HAM"),
        ("Congratulations! You have WON a $1000 Walmart gift card. "
         "Click here to claim now: http://bit.ly/win-prize", "SPAM"),
        ("Reminder: your dentist appointment is scheduled for Friday 3pm.", "HAM"),
        ("URGENT! Your account has been compromised. Verify your password "
         "immediately at secure-login-check.com or it will be suspended.", "SPAM"),
        ("FREE entry into our $250,000 weekly prize draw! Text WIN to 80086 now!", "SPAM"),
    ]

    print("\n" + "=" * 60)
    print("VÍ DỤ MINH HỌA")
    print("=" * 60)
    for text, expected in examples:
        label, prob_spam, _ = predict_email(text, model, vocab_index, idf)
        mark = "OK" if label.startswith(expected) else "!!"
        print(f"[{mark}] Expected: {expected:4s} | Pred: {label:16s} | P(spam)={prob_spam:.3f}")
        print(f"     \"{text[:90]}{'...' if len(text) > 90 else ''}\"")


def main():
    model, vocab_index, idf, (X_test, y_test) = train_model()
    evaluate(model, X_test, y_test)
    run_builtin_examples(model, vocab_index, idf)
    interactive_loop(model, vocab_index, idf)


if __name__ == "__main__":
    main()
