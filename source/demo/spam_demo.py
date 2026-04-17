"""Streamlit demo: Spam email detector using ClassWeightLogisticRegression.

Run:
    streamlit run source/demo/spam_demo.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from source.data.preprocess_data import text_preprocessing_pipeline
from source.data.train_test_split import build_vocabulary, calculate_idf, calculate_tfidf
from source.models.logistic_regression_ipv_class_weight import ClassWeightLogisticRegression


PREPROCESSED_CSV = os.path.join(DATA_FOLDER, "preprocessed_spam.csv")
ARTIFACTS_PATH = os.path.join(DATA_FOLDER, "demo_artifacts.pkl")


def _train_artifacts():
    """Train the model from scratch and return (model, vocab, idf, vocab_index, test_acc)."""
    df = pd.read_csv(PREPROCESSED_CSV).fillna("")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    vocab = build_vocabulary(train_df["text"])
    idf = calculate_idf(train_df["text"], vocab)

    X_train = np.array(
        [calculate_tfidf(d, vocab, idf) for d in train_df["text"]], dtype=np.float32
    )
    X_test = np.array(
        [calculate_tfidf(d, vocab, idf) for d in test_df["text"]], dtype=np.float32
    )
    y_train = train_df["label"].values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.float32)

    model = ClassWeightLogisticRegression(
        learning_rate=1.0, iterations=3000, alpha=0.5, beta=0.5
    )
    model.fit(X_train, y_train, X_test, y_test)

    z_test = np.dot(X_test, model.weights) + model.bias
    p_test = 1.0 / (1.0 + np.exp(-np.clip(z_test, -500, 500)))
    test_acc = float(np.mean((p_test >= 0.5).astype(int) == y_test.astype(int)))

    vocab_index = {w: i for i, w in enumerate(vocab)}
    return model, vocab, idf, vocab_index, test_acc


def _save_artifacts(artifacts):
    with open(ARTIFACTS_PATH, "wb") as f:
        pickle.dump(artifacts, f)


def _load_artifacts():
    with open(ARTIFACTS_PATH, "rb") as f:
        return pickle.load(f)


def _artifacts_fresh() -> bool:
    if not os.path.exists(ARTIFACTS_PATH):
        return False
    if not os.path.exists(PREPROCESSED_CSV):
        return False
    return os.path.getmtime(ARTIFACTS_PATH) >= os.path.getmtime(PREPROCESSED_CSV)


@st.cache_resource(show_spinner=False)
def get_artifacts():
    if _artifacts_fresh():
        try:
            return _load_artifacts()
        except Exception:
            pass

    with st.spinner("Đang train mô hình lần đầu, vui lòng đợi vài giây..."):
        artifacts = _train_artifacts()
        try:
            _save_artifacts(artifacts)
        except Exception:
            pass
    return artifacts


def predict_email(raw_text: str, artifacts, top_k: int = 10):
    model, vocab, idf, vocab_index, _ = artifacts

    cleaned = text_preprocessing_pipeline(raw_text)
    tokens = cleaned.split()

    tfidf_vec = np.array(calculate_tfidf(cleaned, vocab, idf), dtype=np.float32)

    z = float(np.dot(tfidf_vec, model.weights) + model.bias)
    prob = float(1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))))

    contributions = {}
    for tok in set(tokens):
        idx = vocab_index.get(tok)
        if idx is None:
            continue
        contrib = float(tfidf_vec[idx] * model.weights[idx])
        if contrib > 0:
            contributions[tok] = contrib

    top = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return prob, cleaned, top


def main():
    st.set_page_config(page_title="Spam Email Demo", page_icon="📧", layout="centered")
    st.title("Spam Email Demo")
    st.caption("Weighted Logistic Regression trên TF-IDF (SMS Spam dataset)")

    artifacts = get_artifacts()
    _, _, _, _, test_acc = artifacts

    with st.sidebar:
        st.header("Cấu hình")
        threshold = st.slider("Ngưỡng phân loại spam", 0.1, 0.95, 0.5, 0.05)
        top_k = st.slider("Số từ spam hiển thị", 3, 20, 10)
        st.markdown("---")
        st.metric("Accuracy trên test", f"{test_acc * 100:.2f}%")
        st.markdown(
            "Mô hình: `ClassWeightLogisticRegression`\n\n"
            "Đặc trưng: TF-IDF (vocab xây từ tập train)"
        )

    default_email = (
        "Congratulations! You have won a FREE iPhone. "
        "Click here to claim your prize now: http://bit.ly/free-prize"
    )
    email_text = st.text_area("Nội dung email", value=default_email, height=200)

    if st.button("Dự đoán", type="primary", use_container_width=True):
        if not email_text.strip():
            st.warning("Vui lòng nhập nội dung email.")
            return

        prob, cleaned, top = predict_email(email_text, artifacts, top_k=top_k)

        col1, col2 = st.columns(2)
        col1.metric("Xác suất spam", f"{prob * 100:.2f}%")
        is_spam = prob >= threshold
        if is_spam:
            col2.error(f"SPAM (>= {threshold:.2f})")
        else:
            col2.success(f"HAM (< {threshold:.2f})")

        st.progress(min(max(prob, 0.0), 1.0))

        st.subheader("Các từ có khả năng cao là spam (trong email)")
        if top:
            df_top = pd.DataFrame(top, columns=["Từ (đã stem)", "Đóng góp"])
            df_top = df_top.set_index("Từ (đã stem)")
            st.bar_chart(df_top)
            st.dataframe(df_top, use_container_width=True)
        else:
            st.info(
                "Không có từ nào trong email đẩy điểm spam tăng lên "
                "(hoặc các từ không có trong vocabulary)."
            )

        with st.expander("Xem nội dung email sau khi tiền xử lý"):
            st.code(cleaned or "(rỗng)")


if __name__ == "__main__":
    main()
