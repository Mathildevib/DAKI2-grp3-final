# bert_svc.py
"""
Træner en SVM på BERT-embeddings (DanskBERT) for at forudsige Product ID (Product) (Product)
Bruger Maybe_final/Dataset/train_dataset.csv som input.
Inspireret af train.py, men med LLM-embeddings og kun SVM.
Bruger GPU hvis tilgængelig.
"""
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import re
from tqdm import tqdm

# --- Metrics functions ---
TOP_K = 5

def precision_at_k(y_true, proba, k=TOP_K):
    topk = np.argsort(proba, axis=1)[:, -k:]
    return np.mean([
        len(set(np.where(y_true[i] == 1)[0]) & set(topk[i])) / k
        for i in range(len(y_true))
    ])

def recall_at_k(y_true, proba, k=TOP_K):
    topk = np.argsort(proba, axis=1)[:, -k:]
    scores = []
    for i in range(len(y_true)):
        true_set = set(np.where(y_true[i] == 1)[0])
        if true_set:
            scores.append(len(true_set & set(topk[i])) / len(true_set))
    return np.mean(scores) if scores else 0.0

def f1_at_k(y_true, proba, k=TOP_K):
    p, r = precision_at_k(y_true, proba, k), recall_at_k(y_true, proba, k)
    return 2 * p * r / (p + r) if (p + r) else 0.0

def weighted_proba_score(y_true, proba, k=TOP_K):
    """
    Weighted probability score: average of predicted probabilities for true labels.
    For each sample, sums proba for true labels in top-K, divided by number of true labels.
    """
    topk = np.argsort(proba, axis=1)[:, -k:]
    scores = []
    for i in range(len(y_true)):
        true_set = set(np.where(y_true[i] == 1)[0])
        if true_set:
            scores.append(sum(proba[i, j] for j in topk[i] if j in true_set) / len(true_set))
    return np.mean(scores) if scores else 0.0

def recall_scorer(y_true, decision_vals):
    return recall_at_k(y_true, decision_vals, k=TOP_K)

def iou_score(y_true, y_pred):
    scores = []
    for i in range(len(y_true)):
        t = set(np.where(y_true[i] == 1)[0])
        p = set(np.where(y_pred[i] == 1)[0])
        scores.append(1.0 if not t and not p else len(t & p) / len(t | p))
    return np.mean(scores)

def accuracy_counts(y_true_cnt, y_pred_cnt):
    mask = y_true_cnt > 0
    return np.mean((y_true_cnt[mask] == y_pred_cnt[mask]).astype(float))

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9åæø\s]", " ", text)
    return text

def combine_text_fields(df):
    combined = []
    for i, row in df.iterrows():
        txt = ""
        if not pd.isna(row.get('Instructions')):
            txt += preprocess_text(row['Instructions']) + " "
        if not pd.isna(row.get('Primær Asset Produkt')):
            txt += preprocess_text(row['Primær Asset Produkt']) + " "
        combined.append(txt.strip())
    return combined

def get_bert_embeddings(texts, model_name="vesteinn/DanskBERT", max_length=128, batch_size=32, use_gpu=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    model = model.to(device)
    model.eval()
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT batches"):
        batch_texts = texts[i:i+batch_size]
        encoded_input = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        with torch.no_grad():
            outputs = model(**encoded_input)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

def main():
    # --- Load data ---
    df = pd.read_csv("Data/trainingdatabentaxnew.csv")
    # Drop rows with missing Product ID
    df = df.dropna(subset=['Product ID (Product) (Product)'])
    # Combine text fields
    X_text = combine_text_fields(df)
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Product ID (Product) (Product)'])
    # --- BERT embeddings ---
    print("Genererer BERT embeddings...")
    X_embeddings = get_bert_embeddings(X_text)
    # --- Train SVM ---
    print("Træner SVM...")
    svm = LinearSVC(max_iter=20000, class_weight='balanced', random_state=42)
    svm.fit(X_embeddings, y)
    # --- Evaluer på træningsdata (cross-val kan tilføjes senere) ---
    y_pred = svm.predict(X_embeddings)
    n = len(y)
    C = len(label_encoder.classes_)
    # One-hot encode y_true and y_pred for metrics
    y_true_bin = np.zeros((n, C), dtype=int)
    y_pred_bin = np.zeros((n, C), dtype=int)
    y_true_bin[np.arange(n), y] = 1
    y_pred_bin[np.arange(n), y_pred] = 1
    # Brug decision_function som "proba"-score
    decision_vals = svm.decision_function(X_embeddings)
    # Hvis kun 2 klasser, reshape
    if len(decision_vals.shape) == 1:
        decision_vals = decision_vals.reshape(-1, 1)
    proba = decision_vals
    # --- Print metrics i ønsket format ---
    print("\n===== Test-set metrics =====")
    print(f"{'precision@'+str(TOP_K):>15}: {precision_at_k(y_true_bin, proba):.3f}")
    print(f"{'recall@'+str(TOP_K):>15}: {recall_at_k(y_true_bin, proba):.3f}")
    print(f"{'f1@'+str(TOP_K):>15}: {f1_at_k(y_true_bin, proba):.3f}")
    print(f"{'weighted':>15}: {weighted_proba_score(y_true_bin, proba):.3f}")
    print(f"{'iou':>15}: {iou_score(y_true_bin, y_pred_bin):.3f}")
    # quantity_acc ikke relevant for single-label, print dummy
    print(f"{'quantity_acc':>15}: {'N/A':>5}")
    # --- Gem model og encoder ---
    import pickle
    with open('bert_svc_model.pkl', 'wb') as f:
        pickle.dump({'model': svm, 'label_encoder': label_encoder}, f)
    print("\nSVM model gemt som 'bert_svc_model.pkl'")

if __name__ == "__main__":
    main()
