# -*- coding: utf-8 -*-
"""
Demo Web UI phan loai Spam / Ham bang Streamlit.

Cach chay (tu thu muc goc cua repo):
    streamlit run source/demo_streamlit.py
"""

import os
import sys
import math
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))

sys.path.append(os.path.join(BASE_DIR, "data"))
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
    path = os.path.join(DATA_FOLDER, "processed_spam.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Hãy chạy source/data/preprocess_data.py trước."
        )
    df = pd.read_csv(path).fillna("")
    if "processed_text" in df.columns and "encoded_label" in df.columns:
        df = df[["processed_text", "encoded_label"]].rename(
            columns={"processed_text": "text", "encoded_label": "label"}
        )
    df["label"] = df["label"].astype(int)
    return df


@st.cache_resource(show_spinner="Đang huấn luyện model (chỉ lần đầu)...")
def train_pipeline():
    df = load_dataset()
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    vocab = build_vocabulary(train_df["text"])
    vocab_index = {w: i for i, w in enumerate(vocab)}
    idf = calculate_idf(train_df["text"], vocab)

    X_train = np.vstack([calculate_tfidf(d, vocab_index, idf) for d in train_df["text"]])
    X_test = np.vstack([calculate_tfidf(d, vocab_index, idf) for d in test_df["text"]])
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    model = LogisticRegression(solver="liblinear", max_iter=10000)
    model.fit(X_train, y_train)

    y_proba_test = model.predict_proba(X_test)[:, 1]

    return {
        "model": model,
        "vocab_index": vocab_index,
        "idf": idf,
        "vocab_size": len(vocab),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "y_test": y_test,
        "y_proba_test": y_proba_test,
    }


def compute_metrics(y_test, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "cm": confusion_matrix(y_test, y_pred, labels=[0, 1]),
    }


EXAMPLES = [
    ("Hi John, are we still meeting for lunch tomorrow at 12?", "HAM"),
    ("Congratulations! You have WON a $1000 Walmart gift card. "
     "Click here to claim now: http://bit.ly/win-prize", "SPAM"),
    ("Reminder: your dentist appointment is scheduled for Friday 3pm.", "HAM"),
    ("URGENT! Your account has been compromised. Verify your password "
     "immediately at secure-login-check.com or it will be suspended.", "SPAM"),
    ("FREE entry into our $250,000 weekly prize draw! Text WIN to 80086 now!", "SPAM"),
    ("Can you send me the report before EOD? Thanks!", "HAM"),
]


def main():
    st.set_page_config(page_title="Spam Email Classifier", layout="wide")

    st.title("Phân loại Email: Spam vs Not Spam")
    st.caption(
        "Logistic Regression + TF-IDF thủ công. "
        "Dữ liệu: SMS Spam Collection (đã tiền xử lý bằng NLTK + Porter Stemmer)."
    )

    pipeline = train_pipeline()
    model = pipeline["model"]
    vocab_index = pipeline["vocab_index"]
    idf = pipeline["idf"]
    y_test = pipeline["y_test"]
    y_proba_test = pipeline["y_proba_test"]

    with st.sidebar:
        st.header("Cấu hình")
        threshold = st.slider(
            "Threshold phân loại SPAM",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.01,
            help=(
                "Nếu xác suất SPAM >= threshold thì gắn nhãn SPAM. "
                "Hạ threshold -> recall tăng, precision giảm."
            ),
        )
        st.markdown("---")
        st.markdown("**Thông tin dataset**")
        st.write(f"Train: {pipeline['train_size']} mẫu")
        st.write(f"Test: {pipeline['test_size']} mẫu")
        st.write(f"Vocab size: {pipeline['vocab_size']}")

    metrics = compute_metrics(y_test, y_proba_test, threshold)

    st.subheader(f"Hiệu năng trên tập test (threshold = {threshold:.2f})")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("Precision (SPAM)", f"{metrics['precision']:.4f}")
    c3.metric("Recall (SPAM)", f"{metrics['recall']:.4f}")
    c4.metric("F1-score (SPAM)", f"{metrics['f1']:.4f}")

    with st.expander("Confusion Matrix & giải thích", expanded=False):
        cm = metrics["cm"]
        cm_df = pd.DataFrame(
            cm,
            index=["Thực tế: HAM", "Thực tế: SPAM"],
            columns=["Dự đoán: HAM", "Dự đoán: SPAM"],
        )
        st.dataframe(cm_df, width="stretch")
        tn, fp, fn, tp = cm.ravel()
        st.markdown(
            f"""
- **TP (SPAM đúng)** = {tp}
- **FP (HAM bị gắn SPAM)** = {fp} → càng thấp thì **Precision** càng cao
- **FN (SPAM bị bỏ sót)** = {fn} → càng thấp thì **Recall** càng cao
- **TN (HAM đúng)** = {tn}

Công thức:
- Precision = TP / (TP + FP) = {tp} / ({tp} + {fp}) = **{metrics['precision']:.4f}**
- Recall = TP / (TP + FN) = {tp} / ({tp} + {fn}) = **{metrics['recall']:.4f}**
"""
        )

    st.markdown("---")
    st.subheader("Thử dự đoán một email")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        if "email_text" not in st.session_state:
            st.session_state.email_text = EXAMPLES[1][0]

        email_text = st.text_area(
            "Nội dung email",
            value=st.session_state.email_text,
            height=220,
            key="email_input",
        )
        predict_btn = st.button("Phân loại", type="primary")

    with col_right:
        st.markdown("**Ví dụ có sẵn (bấm để điền)**")
        for i, (text, expected) in enumerate(EXAMPLES):
            preview = text if len(text) < 70 else text[:67] + "..."
            if st.button(f"[{expected}] {preview}", key=f"ex_{i}"):
                st.session_state.email_text = text
                st.rerun()

    if predict_btn or email_text:
        if not email_text.strip():
            st.info("Nhập nội dung email rồi bấm **Phân loại**.")
            return

        processed = text_preprocessing_pipeline(email_text)
        vec = calculate_tfidf(processed, vocab_index, idf).reshape(1, -1).astype(np.float32)
        prob_spam = float(model.predict_proba(vec)[0, 1])
        is_spam = prob_spam >= threshold

        st.markdown("### Kết quả")
        r1, r2 = st.columns([1, 2])
        with r1:
            if is_spam:
                st.error(f"SPAM\n\nP(spam) = {prob_spam:.4f}  ({prob_spam*100:.2f}%)")
            else:
                st.success(f"HAM (Not Spam)\n\nP(spam) = {prob_spam:.4f}  ({prob_spam*100:.2f}%)")
        with r2:
            st.progress(prob_spam, text=f"Xác suất SPAM: {prob_spam:.1%}")
            st.caption(f"Threshold hiện tại: {threshold:.2f}")

        # --- Phan tich dong gop tung token ---
        weights = model.coef_[0]
        bias = float(model.intercept_[0])
        vec_flat = vec.ravel()

        token_rows = []
        for w in set(processed.split()):
            idx = vocab_index.get(w)
            if idx is None:
                continue
            tfidf_val = float(vec_flat[idx])
            weight_val = float(weights[idx])
            contrib = tfidf_val * weight_val
            token_rows.append(
                {
                    "token": w,
                    "tfidf": tfidf_val,
                    "weight": weight_val,
                    "contribution": contrib,
                }
            )

        logit = bias + float(np.dot(vec_flat, weights))

        st.markdown("### Các từ đóng góp vào quyết định")
        st.caption(
            "Công thức: logit = bias + Σ tfidf(token) × weight(token). "
            "Contribution > 0 → đẩy email về SPAM, < 0 → đẩy về HAM. "
            "P(spam) = sigmoid(logit)."
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Bias", f"{bias:+.4f}")
        m2.metric("Σ contribution", f"{(logit - bias):+.4f}",
                  help="Tổng đóng góp của tất cả token trong email.")
        m3.metric("Logit", f"{logit:+.4f}",
                  help="logit > 0 -> SPAM, logit < 0 -> HAM")

        if token_rows:
            token_df = pd.DataFrame(token_rows).sort_values(
                "contribution", ascending=False
            )

            spam_df = token_df[token_df["contribution"] > 0].head(10).reset_index(drop=True)
            ham_df = (
                token_df[token_df["contribution"] < 0]
                .sort_values("contribution")
                .head(10)
                .reset_index(drop=True)
            )

            col_spam, col_ham = st.columns(2)
            with col_spam:
                st.markdown("#### Top từ đẩy về SPAM")
                if len(spam_df) == 0:
                    st.write("_Không có token nào đẩy về SPAM._")
                else:
                    st.dataframe(
                        spam_df.style.format(
                            {
                                "tfidf": "{:.4f}",
                                "weight": "{:+.4f}",
                                "contribution": "{:+.4f}",
                            }
                        ).background_gradient(subset=["contribution"], cmap="Reds"),
                        width="stretch",
                        hide_index=True,
                    )

            with col_ham:
                st.markdown("#### Top từ đẩy về HAM")
                if len(ham_df) == 0:
                    st.write("_Không có token nào đẩy về HAM._")
                else:
                    st.dataframe(
                        ham_df.style.format(
                            {
                                "tfidf": "{:.4f}",
                                "weight": "{:+.4f}",
                                "contribution": "{:+.4f}",
                            }
                        ).background_gradient(subset=["contribution"], cmap="Greens_r"),
                        width="stretch",
                        hide_index=True,
                    )

            highlight_tokens = set(spam_df["token"].tolist())
            if highlight_tokens:
                st.markdown("#### Văn bản gốc (highlight từ nghi spam)")
                highlighted = []
                for word in email_text.split():
                    base = text_preprocessing_pipeline(word).strip()
                    if base and base in highlight_tokens:
                        highlighted.append(
                            f"<mark style='background:#ffd6d6;padding:0 3px;border-radius:3px'>{word}</mark>"
                        )
                    else:
                        highlighted.append(word)
                st.markdown(" ".join(highlighted), unsafe_allow_html=True)
        else:
            st.info(
                "Email không chứa token nào có trong vocab -> "
                "mô hình chỉ dựa vào bias."
            )

        with st.expander("Chi tiết tiền xử lý"):
            st.write("Văn bản sau khi tokenize + stem + loại stopwords:")
            st.code(processed or "(trống)")

            words = processed.split()
            in_vocab = [w for w in words if w in vocab_index]
            oov = [w for w in words if w not in vocab_index]
            st.write(f"Token trong vocab ({len(in_vocab)}): `{' '.join(in_vocab)}`")
            if oov:
                st.write(f"Token OOV ({len(oov)}): `{' '.join(oov)}`")


if __name__ == "__main__":
    main()
