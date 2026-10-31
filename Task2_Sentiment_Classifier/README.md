# Task 2: Multi-Class NLP Sentiment Classifier

Classifies text into **positive / negative / neutral** using classic NLP + Machine Learning.

## Pipeline
Raw text -> lowercase & clean -> stop-word removal -> lemmatization -> TF-IDF -> ML model -> F1-score evaluation

## Project Structure
```
task2_sentiment/
├── data/reviews.csv            # dataset (columns: text, label)
├── make_dataset.py             # generates the sample dataset
├── sentiment_classifier.py     # main script
├── outputs/                    # confusion matrix + model comparison charts
├── requirements.txt
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
python make_dataset.py          # optional: creates sample data
python sentiment_classifier.py
```

## Models Compared
Naive Bayes, Logistic Regression, Linear SVM (metric: macro F1-score).

## Dataset (real data)
`make_dataset.py` builds a real 3-class dataset automatically from NLTK corpora (4,277 samples):
- **positive / negative**: real tweets (NLTK `twitter_samples`), emoticons removed to avoid label leakage
- **neutral**: objective factual sentences (NLTK `subjectivity`)

You can also use your own CSV (columns: `text`, `label`) in `data/reviews.csv`, e.g. Kaggle Amazon/Twitter reviews.

## Results (856 test samples)
| Model | Macro F1 |
|---|---|
| Naive Bayes (best) | 0.756 |
| Logistic Regression | 0.743 |
| Linear SVM | 0.744 |

Per class (Naive Bayes): negative F1 0.750, neutral F1 0.850, positive F1 0.670.

## Tech
Python, NLTK, Scikit-Learn, Pandas, Matplotlib, Seaborn
