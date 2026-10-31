"""Builds a REAL 3-class dataset from NLTK corpora (no manual download needed).
positive / negative -> NLTK twitter_samples (real tweets)
neutral             -> NLTK subjectivity 'obj' sentences (objective, factual statements)
"""
import re, os, nltk, pandas as pd
for p in ["twitter_samples", "subjectivity"]:
    nltk.download(p, quiet=True)
from nltk.corpus import twitter_samples, subjectivity

N = 1500  # samples per class

def clean_tweet(t):
    t = re.sub(r"http\S+|@\w+|#", " ", t)
    # remove emoticons: tweets were originally collected using them, so they would leak the label
    t = re.sub(r"[:;=][\-o\^']?[\)\(DPpd\]\[/\\|]+|\(:|\):|<3|:-?[\)\(DdPp]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

pos = [clean_tweet(t) for t in twitter_samples.strings("positive_tweets.json")]
neg = [clean_tweet(t) for t in twitter_samples.strings("negative_tweets.json")]
neu = [" ".join(s) for s in subjectivity.sents(categories="obj")]

rows = [(t, "positive") for t in pos[:N]] + [(t, "negative") for t in neg[:N]] + [(t, "neutral") for t in neu[:N]]
df = pd.DataFrame(rows, columns=["text", "label"])
df = df[df.text.str.len() > 5].drop_duplicates("text").sample(frac=1, random_state=42)
os.makedirs("data", exist_ok=True)
df.to_csv("data/reviews.csv", index=False)
print(df.label.value_counts())
