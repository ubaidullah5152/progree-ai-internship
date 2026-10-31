"""Task 2: Multi-class NLP Sentiment Classifier
Pipeline: clean -> stop-word removal -> lemmatization -> TF-IDF -> model -> F1 evaluation"""
import re, nltk, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, f1_score

for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

# ---- 1. Load data ----
df = pd.read_csv("data/reviews.csv")

# ---- 2. Preprocess ----
# keep negations: "not" / "no" change sentiment, so we do not remove them
stop = set(stopwords.words("english")) - {"not", "no", "nor", "never"}
lemm = WordNetLemmatizer()

def preprocess(text):
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    tokens = [lemm.lemmatize(w) for w in text.split() if w not in stop]
    return " ".join(tokens)

df["clean"] = df["text"].apply(preprocess)
print(df[["text", "clean"]].head(3).to_string(), "\n")

# ---- 3. Split (stratified keeps class balance) ----
X_tr, X_te, y_tr, y_te = train_test_split(
    df["clean"], df["label"], test_size=0.2, stratify=df["label"], random_state=42)

# ---- 4. TF-IDF ----
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
Xtr, Xte = tfidf.fit_transform(X_tr), tfidf.transform(X_te)

# ---- 5. Train & compare models ----
models = {"Naive Bayes": MultinomialNB(),
          "Logistic Regression": LogisticRegression(max_iter=1000),
          "Linear SVM": LinearSVC()}
results, best_name, best_f1, best_pred = {}, None, -1, None
for name, m in models.items():
    m.fit(Xtr, y_tr)
    pred = m.predict(Xte)
    f1 = f1_score(y_te, pred, average="macro")
    results[name] = f1
    print(f"{name:20s} macro-F1 = {f1:.3f}")
    if f1 > best_f1: best_name, best_f1, best_pred = name, f1, pred

# ---- 6. Detailed evaluation of best model ----
print(f"\nBest model: {best_name}\n")
print(classification_report(y_te, best_pred, digits=3))
labels = sorted(df["label"].unique())
cm = confusion_matrix(y_te, best_pred, labels=labels)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title(f"Confusion Matrix - {best_name}"); plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.savefig("outputs/confusion_matrix.png", dpi=150)

plt.figure(figsize=(5, 3.5))
plt.bar(results.keys(), results.values(), color=["#6c8ebf", "#82b366", "#d6b656"])
plt.ylim(0, 1.05); plt.ylabel("Macro F1"); plt.title("Model Comparison")
plt.xticks(rotation=15); plt.tight_layout(); plt.savefig("outputs/model_comparison.png", dpi=150)

# ---- 7. Try new sentences ----
best = models[best_name]
for s in ["I love this product, it is great!", "Terrible service, I want my money back.",
          "The package arrived on Monday."]:
    print(f"{s!r} -> {best.predict(tfidf.transform([preprocess(s)]))[0]}")
