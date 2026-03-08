"""
Quick test script to debug the preprocessing and prediction
"""

import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

class TextPreprocessor:
    """Complete text preprocessing pipeline"""
    
    def __init__(self, use_stemming=False):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer() if use_stemming else None
        
    def clean_text(self, text):
        """Clean and normalize text"""
        if text == "":
            return ""
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
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

# Test the problematic phrase
test_text = "PRODUCTS NEED IMPROVEMENT"
preprocessor = TextPreprocessor(use_stemming=False)

print("="*60)
print("TESTING PREPROCESSING")
print("="*60)
print(f"Original text: {test_text}")
print(f"After cleaning: {preprocessor.clean_text(test_text)}")
print(f"After stopword removal: {preprocessor.remove_stopwords(preprocessor.clean_text(test_text))}")
print(f"Final preprocessed: {preprocessor.preprocess(test_text)}")
print(f"Words in final: {preprocessor.preprocess(test_text).split()}")

# Check if "need" is a stopword
print("\n" + "="*60)
print("CHECKING STOPWORDS")
print("="*60)
print(f"Is 'need' a stopword? {'need' in preprocessor.stop_words}")
print(f"Is 'improvement' a stopword? {'improvement' in preprocessor.stop_words}")
print(f"Is 'products' a stopword? {'products' in preprocessor.stop_words}")

# Load model and test prediction
try:
    print("\n" + "="*60)
    print("TESTING PREDICTION")
    print("="*60)
    
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    cleaned = preprocessor.preprocess(test_text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    print(f"Preprocessed text: '{cleaned}'")
    print(f"Prediction: {'POSITIVE' if prediction == 1 else 'NEGATIVE'}")
    print(f"Positive probability: {probabilities[1]*100:.2f}%")
    print(f"Negative probability: {probabilities[0]*100:.2f}%")
    
    # Check which features are present
    feature_array = features.toarray()[0]
    non_zero_indices = feature_array.nonzero()[0]
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"\nFeatures found in text:")
    for idx in non_zero_indices[:10]:
        print(f"  - {feature_names[idx]}")
        
except FileNotFoundError:
    print("Model files not found. Please run train_model.py first.")
