# Task 2 Report: Multi-Class Sentiment Classifier

**Intern:** Ubaid Ullah  |  **Organization:** Progree  |  **Domain:** Artificial Intelligence

## 1. Objective
Build an NLP classifier that groups unstructured text into three sentiment classes (positive, negative, neutral), using stop-word removal, lemmatization, TF-IDF and F1-score evaluation.

## 2. Dataset
4,277 real English text samples in 3 classes (1,323 positive, 1,454 negative, 1,500 neutral), split 80% train / 20% test (stratified).
- Positive/negative: real tweets from NLTK `twitter_samples`. URLs, mentions, hashtags and emoticons were removed (emoticons were used to originally label these tweets, so keeping them would leak the answer).
- Neutral: objective, factual sentences from the NLTK `subjectivity` corpus.

## 3. Approach
1. **Cleaning:** lowercase, remove punctuation and numbers.
2. **Stop-word removal:** NLTK English list, but negations (not, no, nor, never) are kept because they flip sentiment.
3. **Lemmatization:** WordNet lemmatizer (e.g. "days" -> "day").
4. **Vectorization:** TF-IDF with unigrams and bigrams.
5. **Models:** Multinomial Naive Bayes, Logistic Regression, Linear SVM.
6. **Metrics:** precision, recall, F1 per class, macro F1, confusion matrix.

## 4. Results (856 test samples)
| Model | Macro F1 |
|---|---|
| Naive Bayes (best) | 0.756 |
| Logistic Regression | 0.743 |
| Linear SVM | 0.744 |

Best model per class: negative P 0.738 / R 0.763 / F1 0.750; neutral P 0.773 / R 0.943 / F1 0.850; positive P 0.804 / R 0.574 / F1 0.670. Overall accuracy 76.8%.
Figures: `outputs/confusion_matrix.png`, `outputs/model_comparison.png`.

## 5. Analysis and Limitations
- Neutral is easiest (F1 0.85) because factual sentences use a different style from tweets, which also means part of this score reflects the data source rather than pure sentiment.
- Positive has low recall (0.57): many positive tweets are short or use slang, and get mistaken for negative or neutral.
- The neutral class comes from a different domain, so everyday neutral sentences (e.g. "The package arrived on Monday.") can be misclassified. A neutral set from the same source would be more reliable.
- Bag-of-words TF-IDF cannot understand sarcasm or context.

## 6. Future Work
Use a single-source 3-class dataset (e.g. Kaggle Twitter/Amazon reviews), cross-validation, hyperparameter tuning, and transformer embeddings (BERT).

## 7. Conclusion
The pipeline works end to end and is ready to run on any CSV with `text` and `label` columns. A macro F1 of about 0.76 on real data is a realistic baseline for TF-IDF models.
