import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.calibration")

# Standard library
import re
from pathlib import Path

# Third-party libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
import joblib
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import precision_recall_curve, average_precision_score




# Data configuration
DATA_CSV = Path("dataset") / "train_dataset.csv"
TARGET_COL = "Product ID (Product) (Product)"
GROUP_COL = "Work Order"
QUANTITY_COL = "Quantity"

stemmer = SnowballStemmer("danish")

# --- Load data ---

def load_stemmed_stopwords(path: str) -> set[str]:
    sw = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            w = line.strip().lower()
            w = re.sub(r"[^a-z0-9åæø\s]", "", w)
            if w:
                sw.add(stemmer.stem(w))       
    return sw


STOP_WORDS_FILE = r"DAKI2-grp3-final-main\\Data\\danish_stopwords.txt"
STOP_WORDS = load_stemmed_stopwords(STOP_WORDS_FILE)

# TF-IDF settings
RAW_STEPS = [20000]
NGRAM_RANGE = (1, 3)
MAX_FEATURES = 20000
MIN_DF = 1        
MAX_DF = 0.8      

# Model and CV settings
K_FOLDS = 5
TOP_K = 5
SVM_MAX_ITER = 200000
RANDOM_STATE = 42
EXAMPLES = 5

# --- Metrics functions ---
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

def quantity_precision_recall(y_true_cnt, y_pred_cnt):
    """
    Calculates precision and recall for quantity prediction (nonzero counts).
    Precision: Of all predicted nonzero, how many are correct.
    Recall: Of all true nonzero, how many are correctly predicted nonzero.
    """
    true_nonzero = (y_true_cnt > 0)
    pred_nonzero = (y_pred_cnt > 0)
    tp = np.logical_and(true_nonzero, pred_nonzero).sum()
    precision = tp / pred_nonzero.sum() if pred_nonzero.sum() else 0.0
    recall = tp / true_nonzero.sum() if true_nonzero.sum() else 0.0
    return precision, recall

# --- Preprocessing utilities ---
def parse_part_list(cell):
    import ast
    try:
        lst = ast.literal_eval(cell)
        return lst if isinstance(lst, list) else [lst]
    except Exception:
        return []

    
def preprocess_instruction(text: str) -> str:
    """
    • lower-case
    • remove non-alphanumeric characters (keep å, æ, ø, numbers, spaces)
    • stem each token with the Danish Snowball-STEMMER
    """
    txt = str(text).lower()
    txt = re.sub(r"[^a-z0-9åæø\s]", " ", txt)

    # stem per word
    stemmed = (stemmer.stem(w) for w in txt.split())
    return " ".join(stemmed)




# --- Feature & target construction ---
def make_targets(df):
    X = df[['Instructions', 'Primær Asset Produkt']]
    mlb = MultiLabelBinarizer()
    Y_bin = mlb.fit_transform(df[TARGET_COL])
    def cnt_vec(row):
        mapping = {p:q for p,q in zip(row[TARGET_COL], row[QUANTITY_COL])}
        return [mapping.get(cls,0) for cls in mlb.classes_]
    Y_cnt = np.array([cnt_vec(r) for _,r in df.iterrows()]).astype(int)
    return X, Y_bin, Y_cnt, mlb


def build_preprocessor(stop_words=STOP_WORDS, max_features=MAX_FEATURES):
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=NGRAM_RANGE,
            max_features=max_features,
            stop_words=list(stop_words),
            sublinear_tf=True,
            min_df=MIN_DF,
            max_df=MAX_DF
        ))
    ])
    
    return ColumnTransformer([
        ('text_tfidf', text_pipeline, 'Instructions'),
        ('ohe', OneHotEncoder(handle_unknown='ignore'),
               ['Primær Asset Produkt'])
    ])



# --- Safeguard for quantity ---
def apply_quantity_safeguard(proba, qty_pred, k=TOP_K):
    topk = np.argsort(proba, axis=1)[:, -k:]
    for i, idxs in enumerate(topk):
        for j in idxs:
            if qty_pred[i,j]==0:
                qty_pred[i,j]=1
    return qty_pred

# --- Cross-validation with metrics logging ---
def  cross_validate(Xt, Y_bin, Y_cnt, groups):
    """
    Cross-validate on pre-transformed features, in 2 pipelines:
    TF-IDF+OHE, with calibration and quantity-SVC with Dummy fallback.
    """
    gkf = GroupKFold(n_splits=K_FOLDS)

    proba     = np.zeros(Y_bin.shape)
    qty_pred  = np.zeros(Y_cnt.shape, dtype=int)
    train_metrics, val_metrics = [], []

    for fold, (tr, te) in enumerate(gkf.split(Xt, Y_bin, groups), 1):
        print(f"Fold {fold}: train={len(tr)}, val={len(te)}")
        Xt_tr, Xt_te = Xt[tr], Xt[te]

        # ------- pipeline 1: multilabel classification with calibration -------
        for lbl in range(Y_bin.shape[1]):
            y_tr = Y_bin[tr, lbl]
            if len(np.unique(y_tr)) < 2:          
                continue

            # train one LinearSVC (OvR) for this label
            base = LinearSVC(max_iter=SVM_MAX_ITER, random_state=RANDOM_STATE)
            base.fit(Xt_tr, y_tr)

            # wrap in Platt-scaling calibrator
            try:
                calib = CalibratedClassifierCV(base, cv=3, method="sigmoid", n_jobs=-1)
                calib.fit(Xt_tr, y_tr)

            except ValueError: 
                calib = CalibratedClassifierCV(base, cv="prefit", method="sigmoid", n_jobs=-1)
                calib.fit(Xt_tr, y_tr)

            proba[te, lbl] = calib.predict_proba(Xt_te)[:, 1]

        # --------------- Stage 2: quantity prediction ------------------
        for lbl in range(Y_cnt.shape[1]):
            mask = Y_cnt[tr, lbl] > 0           # only where qty > 0
            yq   = Y_cnt[tr, lbl][mask]

            # ---- Dummy fallback: not enough examples or 1 unique value
            if mask.sum() < 3 or len(np.unique(yq)) < 2: #the check
                constant = 0 if mask.sum() == 0 else int(np.bincount(yq).argmax())
                dummy = DummyClassifier(strategy="constant", constant=constant)
                # same number of features as Xt_tr to avoid shape errors
                dummy.fit(np.zeros((1, Xt_tr.shape[1])), [constant])
                qty_pred[te, lbl] = dummy.predict(Xt_te)
                continue

            #if Enough data → real SVC
            qclf = LinearSVC(max_iter=SVM_MAX_ITER, random_state=RANDOM_STATE)
            qclf.fit(Xt_tr[mask], yq)
            qty_pred[te, lbl] = qclf.predict(Xt_te)

        # Safeguard: prevent 0-quantities for Top-K labels
        qty_pred = apply_quantity_safeguard(proba, qty_pred)

        # ------------------ Metrics ------------------
        pm_tr = evaluate(Y_bin[tr], proba[tr])
        pm_tr["quantity_acc"] = accuracy_counts(Y_cnt[tr], qty_pred[tr])
        # add quantity precision/recall for trainsplit
        qty_prec_tr, qty_rec_tr = quantity_precision_recall(Y_cnt[tr], qty_pred[tr])
        pm_tr["quantity_precision"] = qty_prec_tr
        pm_tr["quantity_recall"] = qty_rec_tr
        train_metrics.append((fold, pm_tr))

        pm_val = evaluate(Y_bin[te], proba[te])
        pm_val["quantity_acc"] = accuracy_counts(Y_cnt[te], qty_pred[te])
        # Tilføj quantity precision/recall for val split
        qty_prec_val, qty_rec_val = quantity_precision_recall(Y_cnt[te], qty_pred[te])
        pm_val["quantity_precision"] = qty_prec_val
        pm_val["quantity_recall"] = qty_rec_val
        val_metrics.append((fold, pm_val))

    print("Cross-validation complete.")
    return proba, qty_pred, train_metrics, val_metrics


# --- Helpers for display ---
def evaluate(y_true, proba):
    y_pred = np.zeros_like(proba,dtype=int)
    for i,idxs in enumerate(np.argsort(proba,axis=1)[:,-TOP_K:]):
        y_pred[i,idxs]=1
    return {
        f'precision@{TOP_K}': precision_at_k(y_true,proba),
        f'recall@{TOP_K}': recall_at_k(y_true,proba),
        f'f1@{TOP_K}': f1_at_k(y_true,proba),
        'weighted': weighted_proba_score(y_true,proba),
        'iou': iou_score(y_true,y_pred)
    }

def train_and_export_final_model(
    df,
    max_features,
    svm_max_iter,
    random_state,
    output_prefix="tfidf_svc"
):

    # 1) Build and fit the preprocessor
    preprocessor = build_preprocessor(max_features=max_features)

    preprocessor.fit(df[['Instructions', 'Primær Asset Produkt']])
    X_final = preprocessor.transform(df[['Instructions', 'Primær Asset Produkt']])

    # 2) Create the targets and the binarizer
    _, Y_bin, Y_cnt, mlb = make_targets(df)

    # 3) Per-label sampling + training with calibration
    class_clfs = {}
    for i, label in enumerate(mlb.classes_):
        y = Y_bin[:, i]
        if len(np.unique(y)) < 2:
            continue

        # 1) train base-SVM
        base = LinearSVC(max_iter=svm_max_iter, random_state=random_state)
        base.fit(X_final, y)

        # 2) try calibration with CV, fall back to prefit on ValueError
        try:
            calibrator = CalibratedClassifierCV(base, cv=3, method='sigmoid', n_jobs=-1)
            calibrator.fit(X_final, y)
        except ValueError:
            calibrator = CalibratedClassifierCV(base, cv='prefit', method='sigmoid', n_jobs=-1)
            calibrator.fit(X_final, y)

        class_clfs[label] = calibrator

    # 4) Train quantity-classifiers per label
    qty_clfs = {}
    for i, label in enumerate(mlb.classes_):
        mask = Y_cnt[:, i] > 0
        uniq_vals = np.unique(Y_cnt[mask, i])

        # === NEW: check for enough data AND >1 unique value ===
        if mask.sum() >= 3 and len(uniq_vals) >= 2:
            qclf = LinearSVC(max_iter=svm_max_iter, random_state=random_state)
            qclf.fit(X_final[mask], Y_cnt[mask, i])
            qty_clfs[label] = qclf
        else:
            # Fallback: constant predictor (most common qty, or 0 if no data)
            constant = int(uniq_vals[0]) if mask.sum() else 0
            dummy = DummyClassifier(strategy="constant", constant=constant)
            dummy.fit(np.zeros((1, 1)), [constant])      # one "fake feature", one target
            qty_clfs[label] = dummy

    # 5) Export all models
    joblib.dump(preprocessor, f"preprocessor_{output_prefix}.joblib")
    joblib.dump(class_clfs,  f"classifiers_{output_prefix}.joblib")
    joblib.dump(mlb,         f"label_binarizer_{output_prefix}.joblib")
    joblib.dump(qty_clfs,    f"quantity_classifiers_{output_prefix}.joblib")
    print(
        f"Saved: preprocessor_{output_prefix}.joblib, "
        f"classifiers_{output_prefix}.joblib, "
        f"label_binarizer_{output_prefix}.joblib, "
        f"quantity_classifiers_{output_prefix}.joblib"
    )

    return preprocessor, class_clfs, mlb, qty_clfs

# --- Main workflow ---
def main():
    # 1) Load CSV and preprocess text columns
    df = pd.read_csv(DATA_CSV)
    df[TARGET_COL]   = df[TARGET_COL].apply(parse_part_list)
    df[QUANTITY_COL] = df[QUANTITY_COL].apply(parse_part_list)
    df['Instructions'] = df['Instructions'].map(preprocess_instruction)

    # 2) Beregn den samlede ordforrådsstørrelse (vocabulary size) for at kunne logge det senere.
    #    Byg samtidig listen 'feature_steps', som bruges til at holde styr på de forskellige trin/features i modellen.
    texts = df['Instructions'].tolist()
    cv = CountVectorizer(ngram_range=NGRAM_RANGE)
    cv.fit(texts)
    vocab_size = len(cv.vocabulary_)
    print(f"Full TF-IDF vocab size: {vocab_size}")
    feature_steps = [s for s in RAW_STEPS if s < vocab_size] #+ [vocab_size]
    print("Sweeping TF-IDF max_features over:", feature_steps)

    records = []  # <-- Initialize records list here

    # 3) Sweep over different TF-IDF sizes
    for mf in feature_steps:
        print(f"\n--- Running sweep for max_features = {mf} ---")

        # 3a) Build & fit the global preprocessor once (TF-IDF -> LDA -> OHE)
        preprocessor = build_preprocessor(
            max_features=mf
        )
        preprocessor.fit(df[['Instructions', 'Primær Asset Produkt']])

        # 3b) Transform full dataset
        X_trans = preprocessor.transform(df[['Instructions', 'Primær Asset Produkt']])

        # 3c) Build target arrays
        _, Y_bin, Y_cnt, mlb = make_targets(df)

        # 3d) Run CV on transformed features
        proba, qty_pred, train_metrics, val_metrics = cross_validate(
            X_trans, Y_bin, Y_cnt, df[GROUP_COL]
        )

        # 3e) Compute mean validation metrics
        val_df = pd.DataFrame([met for _, met in val_metrics])
        mean_val = val_df.mean().to_dict()
        mean_val['quantity_acc'] = mean_val.get('quantity_acc', accuracy_counts(Y_cnt, qty_pred))
        mean_val['max_features'] = mf

        # --- Calculate and print quantity precision and recall ---
        qty_prec, qty_rec = quantity_precision_recall(Y_cnt, qty_pred)
        print(f"quantity_acc precision: {qty_prec:.3f}")
        print(f"quantity_acc recall: {qty_rec:.3f}")

        # 3f) Print out key metrics for this setting
        for metric in [
            f'precision@{TOP_K}',
            f'recall@{TOP_K}',
            f'f1@{TOP_K}',
            'weighted',
            'iou',
            'quantity_acc'
        ]:
            print(f"{metric}: {mean_val[metric]:.3f}")

        records.append(mean_val)

    # 4) export final model
    final_prep, final_clf, mlb, final_qty_clfs = train_and_export_final_model(df, MAX_FEATURES, SVM_MAX_ITER, RANDOM_STATE,output_prefix="tfidf_svc")

if __name__ == "__main__":
    main()
