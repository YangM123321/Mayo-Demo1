import re, random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def clean_text(t):
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+"," ", t).strip()

data = [
    ("Patient reports polyuria and high fasting glucose.", "diabetes"),
    ("Hb low, fatigue observed, possible anemia.", "anemia"),
    ("Fasting glucose within normal range.", "other"),
    ("Dizziness and low hemoglobin suspected.", "anemia"),
    ("Elevated A1C and thirst.", "diabetes"),
    ("No abnormal findings.", "other"),
]*100
random.shuffle(data)

df = pd.DataFrame(data, columns=["note","label"])
df["note"] = df["note"].apply(clean_text)

Xtr, Xte, ytr, yte = train_test_split(df["note"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
vec = TfidfVectorizer(ngram_range=(1,2), min_df=2)
Xtr = vec.fit_transform(Xtr); Xte = vec.transform(Xte)

clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
pred = clf.predict(Xte)
print(classification_report(yte, pred, digits=3))

