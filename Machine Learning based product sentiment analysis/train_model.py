"""
Model Training Script
Trains the sentiment analysis model and saves it for use in the Streamlit dashboard
"""

import pandas as pd
import numpy as np
import re
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download required NLTK data
print("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
print("✓ NLTK setup complete!")

# Text Preprocessing Class
class TextPreprocessor:
    """Complete text preprocessing pipeline"""
    
    def __init__(self, use_stemming=False):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer() if use_stemming else None
        
    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text):
        """Remove common stopwords"""
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(filtered_tokens)
    
    def apply_stemming(self, text):
        """Apply stemming to reduce words to root form"""
        if self.stemmer:
            tokens = word_tokenize(text)
            stemmed = [self.stemmer.stem(word) for word in tokens]
            return ' '.join(stemmed)
        return text
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.apply_stemming(text)
        return text

def train_and_save_model(csv_path='Reviews.csv'):
    """Train the model and save it along with the vectorizer and preprocessor"""
    
    print("="*60)
    print("TRAINING SENTIMENT ANALYSIS MODEL")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Text', 'Score'])

    #df = df.sample(n=5000, random_state=42)  # Use a subset for faster training
    print(f"   ✓ Loaded {len(df)} reviews")
    
    # Create sentiment labels
    print("\n2. Creating sentiment labels...")
    text_column = 'Text'
    rating_column = 'Score'
    threshold = 3
    
    # ---------- CREATE 3 SENTIMENT CLASSES ----------
    def get_sentiment(score):
        if score >= 4:
            return "positive"
        elif score == 3:
            return "neutral"
        else:
            return "negative"

    df['sentiment'] = df[rating_column].apply(get_sentiment)
    # ---------- CHECK REAL DISTRIBUTION ----------
    positive_count = (df['sentiment'] == 'positive').sum()
    neutral_count = (df['sentiment'] == 'neutral').sum()
    negative_count = (df['sentiment'] == 'negative').sum()

    print("\nActual Dataset Distribution:")
    print(f"✓ Positive reviews: {positive_count}")
    print(f"✓ Neutral reviews: {neutral_count}")
    print(f"✓ Negative reviews: {negative_count}")

# Shuffle dataset (do NOT balance)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
#     from sklearn.utils import resample

# # ---------------- BALANCE DATA ----------------
#     # ---------- CREATE 3 CLASS DATASETS ----------
#     df_pos = df[df['sentiment'] == 'positive']
#     df_neg = df[df['sentiment'] == 'negative']
#     df_neu = df[df['sentiment'] == 'neutral']

#     print("\nBefore balancing:")
#     print("Positive:", len(df_pos))
#     print("Neutral:", reminding := len(df_neu))
#     print("Negative:", len(df_neg))

#     # ---------- BALANCE DATASET (VERY IMPORTANT) ----------
#     min_class = min(len(df_pos), len(df_neg), len(df_neu))

#     df_pos = df_pos.sample(min_class, random_state=42)
#     df_neg = df_neg.sample(min_class, random_state=42)
#     df_neu = df_neu.sample(min_class, random_state=42)

#     # combine
#     df = pd.concat([df_pos, df_neg, df_neu])

    # ---------- NOW COUNTS ----------
    positive_count = (df['sentiment'] == 'positive').sum()
    neutral_count = (df['sentiment'] == 'neutral').sum()
    negative_count = (df['sentiment'] == 'negative').sum()

    print("\nAfter balancing:")
    print(f"✓ Positive reviews: {positive_count}")
    print(f"✓ Neutral reviews: {neutral_count}")
    print(f"✓ Negative reviews: {negative_count}")
    print(f"   ✓ Positive reviews: {positive_count} ({positive_count/len(df)*100:.2f}%)")
    print(f"   ✓ Negative reviews: {negative_count} ({negative_count/len(df)*100:.2f}%)")
    
    # Preprocess
    print("\n3. Preprocessing reviews...")
    preprocessor = TextPreprocessor(use_stemming=False)
    df['cleaned_review'] = df[text_column].apply(preprocessor.preprocess)
    
    # Remove empty reviews
    df = df[df['cleaned_review'].str.strip() != '']
    print(f"   ✓ Preprocessing complete. {len(df)} reviews remaining")
    
    # Split data
    print("\n4. Splitting data...")
    X = df['cleaned_review']
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    print(f"   ✓ Training set: {len(X_train)} samples")
    print(f"   ✓ Test set: {len(X_test)} samples")
    
    # Create TF-IDF features
    print("\n5. Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1,1),
        min_df=5,
        max_df=0.85
    )

    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"   ✓ Created {len(vectorizer.get_feature_names_out())} features")
    
    # Train model
    # Train model
    print("\n6. Training model...")
    model = LogisticRegression(max_iter=2000, class_weight='balanced')
    model.fit(X_train_tfidf, y_train)
    print("   ✓ Model trained!")
    
    # Evaluate
    print("\n7. Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"   ✓ Test Accuracy: {test_accuracy*100:.2f}%")
    
    # Save model, vectorizer, and preprocessor
    print("\n8. Saving model files...")
    with open('sentiment_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("   ✓ Saved sentiment_model.pkl")
    
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("   ✓ Saved tfidf_vectorizer.pkl")
    
    with open('text_preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print("   ✓ Saved text_preprocessor.pkl")
    
    # Save model metrics for dashboard
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        'accuracy': test_accuracy,
        'confusion_matrix': cm.tolist(),
        'test_size': len(X_test),
        'train_size': len(X_train),
        'total_reviews': len(df),
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,

    }
    
    with open('model_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    print("   ✓ Saved model_metrics.pkl")
    
    print("\n" + "="*60)
    print("✓ MODEL TRAINING COMPLETE!")
    print("="*60)
    print("\nYou can now run the Streamlit dashboard:")
    print("  streamlit run app.py")
    
    return model, vectorizer, preprocessor, metrics

if __name__ == "__main__":
    train_and_save_model()