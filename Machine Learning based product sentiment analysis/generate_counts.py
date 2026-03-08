# IMPORTANT: import the SAME class used during training
from app import TextPreprocessor

import pandas as pd
import pickle

print("Loading model...")

# load trained objects
with open('sentiment_model.pkl','rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl','rb') as f:
    vectorizer = pickle.load(f)

with open('text_preprocessor.pkl','rb') as f:
    preprocessor = pickle.load(f)

print("Loading dataset...")
df = pd.read_csv("Reviews.csv")

# use review column
texts = df['Text'].astype(str)

print("Preprocessing reviews (this takes time)...")
cleaned = [preprocessor.preprocess(t) for t in texts]

print("Vectorizing...")
X = vectorizer.transform(cleaned)

print("Predicting sentiments (please wait)...")
predictions = model.predict(X)

print("Counting labels...")

positive = 0
negative = 0
neutral = 0

for p in predictions:
    if p == 1:
        positive += 1
    elif p == 0:
        negative += 1
    else:
        neutral += 1

result = {
    "total_reviews": len(df),
    "positive": positive,
    "negative": negative,
    "neutral": neutral
}

# save counts
with open("real_counts.pkl","wb") as f:
    pickle.dump(result,f)

print("\nDONE SUCCESSFULLY ✅")
print(result)