# Mathematics-for-Artificial-Intelligence
Mathematics for Artificial Intelligence Project


# How to Run

1. Data preprocessing using `source/data/preprocess_data.py`
2. Train test split using `source/data/train_test_split.py`
3. Run models using `source/models/<specific model>`

# Chạy demo

Demo Streamlit: nhập email -> in tỉ lệ spam (Weighted Logistic Regression) và liệt kê các từ có khả năng cao là spam.

```bash
pip install streamlit pandas numpy scikit-learn nltk
streamlit run source/demo/spam_demo.py
```

Lần đầu chạy sẽ train model trong vài giây, sau đó cache lại ở `data/demo_artifacts.pkl`. Mở trình duyệt tại `http://localhost:8501`.
