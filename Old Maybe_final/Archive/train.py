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


STOP_WORDS_FILE = "data/danish_stopwords.txt"
STOP_WORDS = load_stemmed_stopwords(STOP_WORDS_FILE)

# TF-IDF settings
RAW_STEPS = [20000]
NGRAM_RANGE=(1,3)
MAX_FEATURES = 20000

# Model and CV settings
K_FOLDS = 5
TOP_K = 5
SVM_MAX_ITER = 200000
RANDOM_STATE = 42
EXAMPLES = 5
N_TOPICS = 100




# --- Metrics functions ---
def hamming_score(y_true, y_pred):
    return np.mean(np.sum(y_true == y_pred, axis=1) / y_true.shape[1])

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

def partial_coverage_score(y_true, proba, k=TOP_K):
    topk = np.argsort(proba, axis=1)[:, -k:]
    return np.mean([
        len(set(np.where(y_true[i] == 1)[0]) & set(topk[i])) / max(1, np.sum(y_true[i]))
        for i in range(len(y_true))
    ])

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


def build_preprocessor(stop_words=STOP_WORDS, max_features=MAX_FEATURES, n_topics=100):
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=NGRAM_RANGE,
            max_features=max_features,
            stop_words=list(stop_words),
            sublinear_tf=True
        )),
#        ('svd', TruncatedSVD(
#            n_components=n_topics,
#            random_state=RANDOM_STATE
#        )),
#        ('norm', Normalizer(copy=False))
    ])
    
    return ColumnTransformer([
#        ('text_lsa', text_pipeline, 'Instructions'),
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
def  cross_validate_transformed(Xt, Y_bin, Y_cnt, groups):
    """
    Cross-validate on pre-transformed features Xt
    (TF-IDF/SVD/OHE), with calibration and quantity-SVN or Dummy fallback.
    """
    gkf = GroupKFold(n_splits=K_FOLDS)

    proba     = np.zeros(Y_bin.shape)
    qty_pred  = np.zeros(Y_cnt.shape, dtype=int)
    train_metrics, val_metrics = [], []

    for fold, (tr, te) in enumerate(gkf.split(Xt, Y_bin, groups), 1):
        print(f"Fold {fold}: train={len(tr)}, val={len(te)}")
        Xt_tr, Xt_te = Xt[tr], Xt[te]

        # ------- Stage 1: multilabel classification with calibration -------
        for lbl in range(Y_bin.shape[1]):
            y_tr = Y_bin[tr, lbl]
            if len(np.unique(y_tr)) < 2:          # not enough variation
                continue

            base = LinearSVC(max_iter=SVM_MAX_ITER, random_state=RANDOM_STATE)
            base.fit(Xt_tr, y_tr)

            try:
                calib = CalibratedClassifierCV(base, cv=3, method="sigmoid", n_jobs=-1)
                calib.fit(Xt_tr, y_tr)
            except ValueError:  # e.g. not enough pos/neg per fold
                calib = CalibratedClassifierCV(base, cv="prefit", method="sigmoid", n_jobs=-1)
                calib.fit(Xt_tr, y_tr)

            proba[te, lbl] = calib.predict_proba(Xt_te)[:, 1]

        # --------------- Stage 2: quantity prediction ------------------
        for lbl in range(Y_cnt.shape[1]):
            mask = Y_cnt[tr, lbl] > 0           # only where qty > 0
            yq   = Y_cnt[tr, lbl][mask]

            # ---- Dummy fallback: not enough examples or 1 unique value
            if mask.sum() < 3 or len(np.unique(yq)) < 2:
                constant = 0 if mask.sum() == 0 else int(np.bincount(yq).argmax())
                dummy = DummyClassifier(strategy="constant", constant=constant)
                # same number of features as Xt_tr to avoid shape errors
                dummy.fit(np.zeros((1, Xt_tr.shape[1])), [constant])
                qty_pred[te, lbl] = dummy.predict(Xt_te)
                continue

            # ---- Enough data → real SVC
            qclf = LinearSVC(max_iter=SVM_MAX_ITER, random_state=RANDOM_STATE)
            qclf.fit(Xt_tr[mask], yq)
            qty_pred[te, lbl] = qclf.predict(Xt_te)

        # Safeguard: prevent 0-quantities for Top-K labels
        qty_pred = apply_quantity_safeguard(proba, qty_pred)

        # ------------------ Metrics ------------------
        pm_tr = evaluate(Y_bin[tr], proba[tr])
        pm_tr["quantity_acc"] = accuracy_counts(Y_cnt[tr], qty_pred[tr])
        train_metrics.append((fold, pm_tr))

        pm_val = evaluate(Y_bin[te], proba[te])
        pm_val["quantity_acc"] = accuracy_counts(Y_cnt[te], qty_pred[te])
        val_metrics.append((fold, pm_val))

    print("Cross-validation complete.")
    return proba, qty_pred, train_metrics, val_metrics


# --- Helpers for display ---

def display_fold_comparison(train_metrics, val_metrics):
    # Create DataFrames for train and validation metrics
    train_df = pd.DataFrame([met for _, met in train_metrics],
                            index=[f"Fold {fold}" for fold, _ in train_metrics])
    val_df = pd.DataFrame([met for _, met in val_metrics],
                          index=[f"Fold {fold}" for fold, _ in val_metrics])
    # Ensure consistent column order
    col_order = train_df.columns.tolist()
    val_df = val_df.reindex(columns=col_order)

    print("--- Train Metrics per Fold ---")
    print(train_df.to_string())
    print("--- Validation Metrics per Fold ---")
    print(val_df.to_string())


def evaluate(y_true, proba):
    y_pred = np.zeros_like(proba,dtype=int)
    for i,idxs in enumerate(np.argsort(proba,axis=1)[:,-TOP_K:]):
        y_pred[i,idxs]=1
    return {
        f'precision@{TOP_K}': precision_at_k(y_true,proba),
        f'recall@{TOP_K}': recall_at_k(y_true,proba),
        f'f1@{TOP_K}': f1_at_k(y_true,proba),
        'hamming': hamming_score(y_true,y_pred),
        'weighted': weighted_proba_score(y_true,proba),
        'partial_cov': partial_coverage_score(y_true,proba),
        'iou': iou_score(y_true,y_pred)
    }

def print_example_predictions(df, proba, qty_pred, mlb, k=TOP_K, n_examples=3):
    print("\n-- Example predictions --")
    for i in range(min(n_examples, len(df))):
        wo = df[GROUP_COL].iloc[i]
        true_parts = df[TARGET_COL].iloc[i]
        true_qtys  = df[QUANTITY_COL].iloc[i]
        true_map   = {p: q for p, q in zip(true_parts, true_qtys)}
        prob_row   = proba[i]
        qty_row    = qty_pred[i]
        topk_idxs  = np.argsort(prob_row)[-k:][::-1]
        preds = [(mlb.classes_[j], prob_row[j], qty_row[j]) for j in topk_idxs]

        print(f"\nWork Order {wo}")
        print("  True :", [f"{p}×{true_map[p]}" for p in true_parts])
        print("  Pred :", [f"{p} ({s:.3f}) → {q}×" for p, s, q in preds])

# --- Precision-Recall Curve ---
def plot_pr_curve(y_true, proba):
    """
    Plot macro-average Precision-Recall curve across classes.
    """
    # Compute macro-average precision score
    ap = average_precision_score(y_true, proba, average='macro')
    # For PR curve, compute micro-averaged curve for plotting
    y_true_flat = y_true.ravel()
    proba_flat = proba.ravel()
    precision, recall, _ = precision_recall_curve(y_true_flat, proba_flat)
    plt.figure()
    plt.plot(recall, precision, label=f'macro AP = {ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend()
    plt.tight_layout()
    plt.show()

def train_and_export_final_model(
    df,
    max_features,
    n_topics,
    svm_max_iter,
    random_state,
    output_prefix="lda"
):

    # 1) Build and fit the preprocessor
    preprocessor = build_preprocessor(max_features=max_features, n_topics=n_topics)

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

    # 2) Compute full vocabulary size for logging + build feature_steps
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
            max_features=mf,
            n_topics=N_TOPICS
        )
        preprocessor.fit(df[['Instructions', 'Primær Asset Produkt']])

        # 3b) Transform full dataset
        X_trans = preprocessor.transform(df[['Instructions', 'Primær Asset Produkt']])

        # 3c) Build target arrays
        _, Y_bin, Y_cnt, mlb = make_targets(df)

        # 3d) Run CV on transformed features
        proba, qty_pred, train_metrics, val_metrics = cross_validate_transformed(
            X_trans, Y_bin, Y_cnt, df[GROUP_COL]
        )

        # 3e) Compute mean validation metrics
        val_df = pd.DataFrame([met for _, met in val_metrics])
        mean_val = val_df.mean().to_dict()
        mean_val['quantity_acc'] = mean_val.get('quantity_acc', accuracy_counts(Y_cnt, qty_pred))
        mean_val['max_features'] = mf

        # 3f) Print out key metrics for this setting
        for metric in [
            f'precision@{TOP_K}',
            f'recall@{TOP_K}',
            f'f1@{TOP_K}',
            'hamming',
            'weighted',
            'partial_cov',
            'iou',
            'quantity_acc'
        ]:
            print(f"{metric}: {mean_val[metric]:.3f}")

        records.append(mean_val)

    # 4) Summarize sweep results
    df_sweep = pd.DataFrame.from_records(records).set_index('max_features')
    
    print("\n=== Sweep Summary ===")
    print(df_sweep)

    plt.figure()
    plt.plot(df_sweep.index, df_sweep[f'recall@{TOP_K}'], marker='o')
    plt.xscale('log')
    plt.xlabel('TF-IDF max_features')
    plt.ylabel(f'Recall@{TOP_K}')
    plt.title(f'Recall@{TOP_K} vs TF-IDF max_features')
    plt.tight_layout()
    plt.show()

    best_mf = df_sweep[f'recall@{TOP_K}'].idxmax()

    print(f"\nBest max_features by recall@{TOP_K}: {best_mf}")
    print(df_sweep.loc[best_mf])

    final_prep, final_clf, mlb, final_qty_clfs = train_and_export_final_model(df, best_mf, N_TOPICS, SVM_MAX_ITER, RANDOM_STATE, output_prefix="lda")

if __name__ == "__main__":
    main()
