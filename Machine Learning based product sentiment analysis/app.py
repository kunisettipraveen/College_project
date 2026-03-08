"""
Streamlit Dashboard for Product Review Sentiment Analysis
Complete interactive dashboard with 5 pages
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from io import StringIO

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Fix Streamlit spacing and make full width dashboard
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100% !important;
    }

    /* remove top blank space */
    .css-18e3th9 {
        padding-top: 0rem;
    }

    /* remove white background behind charts */
    .stPlotlyChart, .stPyplot {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)
# Fix progress bar colors and text visibility
st.markdown("""
<style>
            /* ===== REMOVE STREAMLIT WHITE HEADER ===== */

/* top toolbar */
header[data-testid="stHeader"] {
    display: none;
}

/* hamburger menu */
#MainMenu {
    visibility: hidden;
}

/* footer */
footer {
    visibility: hidden;
}

/* remove top padding created by header */
.block-container {
    padding-top: 0rem !important;
}

/* remove blank space above page */
section.main > div {
    padding-top: 0rem !important;
}

/* progress bar */
div[data-testid="stProgressBar"] > div > div > div {
    background-color: #00c6ff;
}

/* make all markdown text white */
/* Default text color */
section.main {
    color: #EAF3FF;
}

/* Keep headings white */
h1, h2, h3, h4 {
    color: white !important;
}

/* About page card — force black readable text */
.about-card, 
.about-card * {
    color: black !important;
}

/* remove white block behind pyplot charts */
div[data-testid="stPyplot"] {
    background-color: transparent !important;
    padding: 0 !important;
}

/* remove white element containers */
div[data-testid="stMetric"] {
    background-color: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# Custom CSS
# ===== NEW MODERN UI DESIGN =====
st.markdown("""
<style>

/* Full page background */
.stApp {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    background-attachment: fixed;
}

/* Remove white container */
.block-container {
    padding-top: 2rem;
}

/* Header Title */
.main-header {
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    color: white;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}

/* Glass card effect */
.metric-card {
    backdrop-filter: blur(14px);
    background: rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.6rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    transition: 0.3s ease;
}
/* ===== SENTIMENT COLORS ===== */

.metric-card.positive {
    background: rgba(46, 204, 113, 0.18);
    border: 2px solid #2ecc71;
    color: white;
}

.metric-card.negative {
    background: rgba(231, 76, 60, 0.18);
    border: 2px solid #e74c3c;
    color: white;
}

.metric-card.neutral {
    background: rgba(255, 193, 7, 0.18);
    border: 2px solid #ffc107;
    color: white;
} 

/* Hover animation */
.metric-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 16px 40px rgba(0,0,0,0.35);
}

/* Numbers */
.metric-value {
    font-size: 2.7rem;
    font-weight: bold;
    color: #00ffd5;
}

/* Labels */
.metric-label {
    font-size: 1.1rem;
    color: #ffffff;
    opacity: 0.9;
}

/* Positive */
.positive {
    border-left: 6px solid #00ff9c;
}

/* Negative */
.negative {
    border-left: 6px solid #ff4b4b;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#141e30,#243b55);
    color: white;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Text area */
textarea {
    background-color: rgba(255,255,255,0.1) !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    padding: 12px !important;
}

/* Buttons */
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color: white;
    border: none;
    padding: 0.8rem;
    font-size: 1.15rem;
    border-radius: 12px;
    font-weight: bold;
    letter-spacing: 0.5px;
    transition: 0.25s;
}

/* Button hover glow */
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px #00c6ff;
}

/* Prediction result box */
.prediction-box {
    padding: 2rem;
    border-radius: 20px;
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.2);
    text-align: center;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #00c6ff;
    border-radius: 10px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
            /* ===== FORCE ALL TEXT TO BE VISIBLE ===== */

/* main content text */
/* apply white text to whole app EXCEPT about card */
section.main *:not(.about-card):not(.about-card *) {
    color: #EAF3FF !important;
}

/* headings */
h1, h2, h3, h4 {
    color: #FFFFFF !important;
}

/* labels */
label {
    color: #FFFFFF !important;
    font-weight: 500 !important;
}

/* radio buttons */
.stRadio div {
    color: white !important;
}

/* expander text */
details summary {
    color: white !important;
}

/* metric text */
[data-testid="stMetricLabel"] {
    color: #dbeafe !important;
}

/* sidebar */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* progress bar text (Precision Recall F1) */
div[data-testid="stProgressBar"] span {
    color: white !important;
    font-weight: 600 !important;
}
            /* ---------- FIX TEXT AREA (REVIEW BOX) ---------- */
            /* ===== FIX ABOUT PAGE TEXT COLOR ===== */

.about-card {
background: #f0f0f0 !important;
color: #000000 !important;
}
            /* ---- FORCE BLACK TEXT INSIDE ABOUT CARD (OVERRIDE STREAMLIT GLOBAL WHITE) ---- */

.about-card,
.about-card * {
color: #000000 !important;
}

.about-card strong {
color: #000000 !important;
}

.about-card span {
color: #000000 !important;
}


.about-card p,
.about-card h1,
.about-card h2,
.about-card h3,
.about-card h4 {
color: #000000 !important;
}


/* main typing area */
textarea {
    color: #000000 !important;          /* typed text BLACK */
    background-color: #ffffff !important;
    caret-color: #000000 !important;    /* cursor black */
    font-weight: 500 !important;
}

/* when user clicks inside */
textarea:focus {
    color: #000000 !important;
    background-color: #ffffff !important;
}

/* streamlit specific textarea */
div[data-baseweb="textarea"] textarea {
    color: black !important;
    background: white !important;
}

/* placeholder */
textarea::placeholder {
    color: #666666 !important;
    opacity: 1;
}

/* border styling */
div[data-baseweb="textarea"] {
    border: 2px solid #00c6ff !important;
    border-radius: 12px !important;
}
            /* ===== FIX ABOUT PAGE INVISIBLE TEXT ===== */

/* Streamlit markdown text */
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] span {
    color: #FFFFFF !important;
}

/* bullet lists */
div[data-testid="stMarkdownContainer"] ul {
    color: #FFFFFF !important;
}

/* ensure strong text visible */
div[data-testid="stMarkdownContainer"] strong {
    color: #FFFFFF !important;
}
            /* ===== MAKE BACKGROUND FIXED (WEBSITE FEEL) ===== */

/* lock full page */
/* ===== REAL FIXED BACKGROUND FOR STREAMLIT ===== */

/* create background layer */
.stApp::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
    z-index: -1;
}

/* allow normal scrolling */
section.main {
    background: transparent !important;
}

/* streamlit main scroll area */
section.main {
    height: 100vh;
    overflow-y: auto;
    overflow-x: hidden;
}

/* fixed gradient background */
.stApp {
    position: fixed;
    inset: 0;
    z-index: -1;
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}

</style>
""", unsafe_allow_html=True)

# Text Preprocessing Class (same as in training)
class TextPreprocessor:
    """Complete text preprocessing pipeline"""
    
    def __init__(self, use_stemming=False):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer() if use_stemming else None
        
    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text) or text == "":
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
        try:
            tokens = word_tokenize(text)
            filtered_tokens = [word for word in tokens if word not in self.stop_words]
            return ' '.join(filtered_tokens)
        except:
            return text
    
    def apply_stemming(self, text):
        """Apply stemming to reduce words to root form"""
        if self.stemmer:
            try:
                tokens = word_tokenize(text)
                stemmed = [self.stemmer.stem(word) for word in tokens]
                return ' '.join(stemmed)
            except:
                return text
        return text
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.apply_stemming(text)
        return text

# Load model and vectorizer
@st.cache_resource
def load_model():
    """Load the trained model, vectorizer, and preprocessor"""
    try:
        with open('sentiment_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        try:
            with open('text_preprocessor.pkl', 'rb') as f:
                preprocessor = pickle.load(f)
        except:
            preprocessor = TextPreprocessor(use_stemming=False)
        return model, vectorizer, preprocessor
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found! Please run 'train_model.py' first to train and save the model.")
        return None, None, None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None, None, None

@st.cache_data
def load_metrics():
    """Load model metrics"""
    try:
        with open('model_metrics.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <h1>🎯</h1>
        <h2>Sentiment Analyzer</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "📱 Navigation",
        ["🏠 Home", "🔮 Live Predictor", "📊 Model Insights", "📁 Batch Analysis", "ℹ️ About"],
        label_visibility="visible"
    )
    
    st.markdown("---")
    st.markdown("### 🔧 Tech Stack")
    st.markdown("""
    - Python 🐍
    - Scikit-learn
    - NLTK
    - Streamlit
    - TF-IDF
    - Naive Bayes
    """)

# PAGE 1: HOME
if page == "🏠 Home":
    st.markdown('<p class="main-header">🎯 MACHINE LEARNING-BASED SENTIMENT ANALYSIS</p>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>Powered by Machine Learning & NLP</p>", unsafe_allow_html=True)
    
    # Load metrics
    metrics = load_metrics()
    
    # Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5, gap="large")

    
    if metrics:
        total_reviews = metrics.get('total_reviews', 568453)
        accuracy = metrics.get('accuracy', 0.8480) * 100
        positive_count = metrics.get('positive_count', 0)
        negative_count = metrics.get('negative_count', 0)
        neutral_count  = metrics.get('neutral_count', 0)

        positive_percentage = (positive_count / total_reviews) * 100
        negative_percentage = (negative_count / total_reviews) * 100
        neutral_percentage  = (neutral_count / total_reviews) * 100
    else:
        total_reviews = 568453
        accuracy = 84.80
        positive_count = 0
        negative_count = 0
        neutral_count  = 0

    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{total_reviews:,}</p>
            <p class="metric-label">📊 Reviews (100%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        correct_predictions = int((accuracy/100) * total_reviews)

        st.markdown(f"""
        <div class="metric-card positive">
            <p class="metric-value">{accuracy:.2f}%</p>
            <p class="metric-label">⚡ Accuracy ({correct_predictions:,} correct)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card positive">
            <p class="metric-value">{positive_count:,}</p>
            <p class="metric-label">✅ Positive ({positive_percentage:.2f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card negative">
            <p class="metric-value">{negative_count:,}</p>
            <p class="metric-label">❌ Negative ({negative_percentage:.2f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card neutral">
            <p class="metric-value">{neutral_count:,}</p>
            <p class="metric-label">😐 Neutral ({neutral_percentage:.2f}%)</p>
        </div>
        """, unsafe_allow_html=True)

    
    # Confusion Matrix and Metrics
    st.markdown("### 📈 Model Performance")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if metrics and 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix'])
        else:
            # Default values from notebook
            cm = np.array([[8960, 15975], [1307, 87449]])
        
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(8,6))
        fig.patch.set_alpha(0)
        ax.set_facecolor("#0f2027")
        sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', 
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'],
                    cbar_kws={'label': 'Count'},
                    ax=ax)
        ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20, color='white')
        ax.set_ylabel('True Label', fontsize=12, color='white')
        ax.set_xlabel('Predicted Label', fontsize=12, color='white')

        # make tick labels white
        ax.tick_params(colors='white')

        # make numbers inside heatmap white
        for text in ax.texts:
            text.set_color("white")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.markdown("<h3 style='color:white;'>📊 Class Performance Metrics</h3>", unsafe_allow_html=True)
        
        # Negative Class
        st.markdown("**🔴 Negative Class:**")

        st.markdown("<span style='color:white;font-weight:600;'>Precision: 87.27%</span>", unsafe_allow_html=True)
        st.progress(0.8727)

        st.markdown("<span style='color:white;font-weight:600;'>Recall: 35.93%</span>", unsafe_allow_html=True)
        st.progress(0.3593)

        st.markdown("<span style='color:white;font-weight:600;'>F1-Score: 50.91%</span>", unsafe_allow_html=True)
        st.progress(0.5091)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Positive Class
        st.markdown("**🟢 Positive Class:**")
        st.progress(0.8455, text="Precision: 84.55%")
        st.progress(0.9853, text="Recall: 98.53%")
        st.progress(0.9101, text="F1-Score: 91.01%")
    
    st.markdown("---")
    st.markdown("### 🔧 Tech Stack")
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: #f0f0f0; border-radius: 10px;'>
        <p style='font-size: 1.1rem; margin: 0.5rem;'><strong>Python</strong> • <strong>Scikit-learn</strong> • <strong>NLTK</strong> • <strong>Pandas</strong> • <strong>Streamlit</strong></p>
        <p style='font-size: 1.1rem; margin: 0.5rem;'><strong>TF-IDF</strong> • <strong>Naive Bayes</strong> • <strong>NLP</strong></p>
    </div>
    """, unsafe_allow_html=True)

# PAGE 2: LIVE PREDICTOR
elif page == "🔮 Live Predictor":
    st.markdown('<p class="main-header">🔮 LIVE SENTIMENT PREDICTION</p>', unsafe_allow_html=True)
    
    model, vectorizer, preprocessor = load_model()
    
    if model is None:
        st.error("⚠️ Model files not found! Please run 'train_model.py' first to train and save the model.")
        st.info("💡 After training, the model files will be saved and you can use this page.")
    else:
        # Input section
        st.markdown("### ✍️ Enter Your Review")
        user_input = st.text_area(
            "",
            placeholder="Type your review here... (e.g., 'This product is amazing! Best purchase ever!')",
            height=90,
            label_visibility="collapsed"
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("🚀 ANALYZE SENTIMENT", use_container_width=True)
        
        if predict_button and user_input:
            try:
                # Preprocess and predict
                cleaned_text = preprocessor.preprocess(user_input)
                
                if cleaned_text.strip() == "":
                    st.warning("⚠️ Please enter a valid review. The text appears to be empty after preprocessing.")
                else:
                    # Show preprocessing debug info
                    with st.expander("🔍 Debug: Preprocessed Text", expanded=False):
                        st.write(f"**Original:** {user_input}")
                        st.write(f"**Preprocessed:** {cleaned_text}")
                        st.write(f"**Words after preprocessing:** {cleaned_text.split()}")
                    
                    features = vectorizer.transform([cleaned_text])
                    prediction = model.predict(features)[0]
                    probabilities = model.predict_proba(features)[0]
                    prediction = str(prediction)
                    class_index = list(model.classes_).index(prediction)
                    final_confidence = probabilities[class_index]
                    
                    # Rule-based override for clear negative phrases
                    # These phrases are clearly negative but model may misclassify
                    strong_negative_phrases = [
                        'need improvement', 'needs improvement', 'requires improvement',
                        'need fix', 'needs fix', 'needs fixing', 'needs to be fixed',
                        'poor quality', 'bad quality', 'terrible quality',
                        'waste of money', 'complete waste', 'not worth',
                        'very disappointed', 'extremely disappointed'
                    ]
                    
                    text_lower_original = user_input.lower()
                    has_strong_negative = any(phrase in text_lower_original for phrase in strong_negative_phrases)
                    original_prediction = prediction
                    override_applied = False
                    
                    # If strong negative phrase detected and model predicts positive with low confidence, override
                    if has_strong_negative and prediction == "positive" and final_confidence < 0.90:
                        prediction = "negative"
                        override_applied = True
                        # recalculate confidence after override
                        class_index = list(model.classes_).index(prediction)
                        final_confidence = probabilities[class_index]
                    
                    # Check for negative indicators that might be missed
                    negative_indicators = ['need improvement', 'needs improvement', 'needs fix', 'need fix', 
                                         'poor quality', 'bad quality', 'terrible', 'awful', 'worst', 
                                         'disappointed', 'waste', 'broken', 'defective', 'faulty',
                                         'products need', 'product needs', 'requires improvement']
                    text_lower = user_input.lower()
                    has_negative_phrase = any(indicator in text_lower for indicator in negative_indicators)
                    
                    # Show key words that influenced prediction
                    feature_names = vectorizer.get_feature_names_out()
                    feature_array = features.toarray()[0]
                    non_zero_indices = feature_array.nonzero()[0]
                    if len(non_zero_indices) > 0:
                        with st.expander("🔍 Debug: Key Words Used", expanded=False):
                            key_words = [feature_names[idx] for idx in non_zero_indices[:20]]
                            st.write(f"**Features found in text:** {', '.join(key_words[:10])}")
                            if has_negative_phrase:
                                st.warning("⚠️ Detected negative phrase indicators, but model may have misclassified.")
                    
                    st.markdown("---")
                    st.markdown("### 🎯 Prediction Result")
                    
                    # Show override message if applied
                    if override_applied:
                        st.warning(f"⚠️ **Rule-based Override Applied:** Detected strong negative phrase ('need improvement' / 'needs improvement'). Model originally predicted POSITIVE ({probabilities[1]*100:.2f}%), but overriding to NEGATIVE based on phrase analysis.")
                    
                    # Show warning if negative phrases detected but model predicts positive (without override)
                    if has_negative_phrase and prediction == 1 and not override_applied:
                        st.warning("⚠️ **Note:** Your text contains negative indicators (like 'need improvement'), but the model predicted POSITIVE. This may be a misclassification. The model might not have learned these specific negative patterns well.")
                    
                    # Calculate final confidence once
                    class_index = list(model.classes_).index(prediction)
                    final_confidence = probabilities[class_index]
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:

                        if prediction == "positive":
                            st.markdown(f"""
                            <div class="metric-card positive" style="padding: 4rem;">
                                <p style="font-size: 4rem; margin: 0;">😊</p>
                                <p class="metric-value">POSITIVE</p>
                                <p class="metric-label">Confidence: {final_confidence*100:.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)

                        elif prediction == "neutral":
                            st.markdown(f"""
                             <div class="metric-card neutral" style="padding: 4rem;">
                                <p style="font-size: 4rem; margin: 0;">😐</p>
                                <p class="metric-value">NEUTRAL</p>
                                <p class="metric-label">Confidence: {final_confidence*100:.2f}%</p>
                             </div>
                            """, unsafe_allow_html=True)

                        else:
                            st.markdown(f"""
                            <div class="metric-card negative" style="padding: 4rem;">
                                <p style="font-size: 4rem; margin: 0;">😡</p>
                                <p class="metric-value">NEGATIVE</p>
                                <p class="metric-label">Confidence: {final_confidence*100:.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        # Gauge chart - use correct probability
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = final_confidence * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Confidence"},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "#56ab2f" if prediction == 1 else "#eb3349"},
                                'steps': [
                                    {'range': [0, 50], 'color': "#f0f0f0"},
                                    {'range': [50, 75], 'color': "#e0e0e0"},
                                    {'range': [75, 100], 'color': "#d0d0d0"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Probability bars
                    st.markdown("### 📊 Confidence Breakdown")
                    st.progress(probabilities[1], text=f"✅ Positive: {probabilities[1]*100:.2f}%")
                    st.progress(probabilities[0], text=f"❌ Negative: {probabilities[0]*100:.2f}%")
                    
                    # Show manual override suggestion if needed
                    if has_negative_phrase and prediction == 1 and not override_applied:
                        st.info("💡 **Suggestion:** Phrases like 'need improvement' typically indicate negative sentiment. The model may have misclassified this. Consider rephrasing with stronger negative words like 'poor quality', 'terrible', or 'disappointed' for better detection.")
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        
        # Example reviews
        st.markdown("---")
        st.markdown("### 🧪 Try These Examples")
        
        examples = [
            ("✅ Positive", "Excellent quality! Highly recommend! This product exceeded my expectations."),
            ("❌ Negative", "Terrible product. Complete waste of money. Very disappointed with the quality."),
            ("❌ Need Improvement", "PRODUCTS NEED IMPROVEMENT. Poor quality and disappointing."),
            ("🤔 Neutral", "It's okay, nothing special but does the job. Average quality for the price.")
        ]
        
        for label, example_text in examples:
            if st.button(f"📝 {label}: {example_text[:50]}...", key=f"example_{label}"):
                st.session_state.example_text = example_text
                st.rerun()
        
        # Auto-fill if example was clicked
        if 'example_text' in st.session_state:
            user_input = st.text_area(
                "",
                value=st.session_state.example_text,
                height=150,
                label_visibility="collapsed",
                key="example_input"
            )
            del st.session_state.example_text

# PAGE 3: MODEL INSIGHTS
elif page == "📊 Model Insights":
    st.markdown('<p class="main-header">📊 MODEL INSIGHTS & WORD IMPORTANCE</p>', unsafe_allow_html=True)
    
    model, vectorizer, preprocessor = load_model()
    
    if model is None:
        st.error("⚠️ Model files not found! Please run 'train_model.py' first.")
    else:
        # Get top words from model
        # -------- WORD IMPORTANCE FOR LOGISTIC REGRESSION --------
        feature_names = vectorizer.get_feature_names_out()
        n_words = 15

        # coefficients: rows = classes, columns = words
        coefs = model.coef_
        classes = list(model.classes_)

        # get class index safely
        pos_index = classes.index("positive")
        neg_index = classes.index("negative")
        neu_index = classes.index("neutral")

        # positive words
        positive_indices = coefs[pos_index].argsort()[-n_words:][::-1]
        positive_words = [feature_names[i] for i in positive_indices]
        positive_scores = coefs[pos_index][positive_indices]

        # negative words
        negative_indices = coefs[neg_index].argsort()[-n_words:][::-1]
        negative_words = [feature_names[i] for i in negative_indices]
        negative_scores = coefs[neg_index][negative_indices]

        # neutral words
        neutral_indices = coefs[neu_index].argsort()[-n_words:][::-1]
        neutral_words = [feature_names[i] for i in neutral_indices]
        neutral_scores = coefs[neu_index][neutral_indices]
        
        # col1, col2, col3 = st.columns(3)
        
        # with col1:
        #     st.markdown("### 🔴 Top Negative Words")
        #     fig = go.Figure(go.Bar(
        #         x=negative_scores[::-1],
        #         y=negative_words[::-1],
        #         orientation='h',
        #         marker=dict(color='#eb3349'),
        #         text=[f'{score:.2f}' for score in negative_scores[::-1]],
        #         textposition='auto'
        #     ))
        #     fig.update_layout(
        #         height=500, 
        #         xaxis_title="Log Probability",
        #         yaxis_title="",
        #         title="Top 15 Negative Indicators",
        #         showlegend=False
        #     )
        #     st.plotly_chart(fig, use_container_width=True)
        
        # with col2:
        #     st.markdown("### 🟢 Top Positive Words")
        #     fig = go.Figure(go.Bar(
        #         x=positive_scores[::-1],
        #         y=positive_words[::-1],
        #         orientation='h',
        #         marker=dict(color='#56ab2f'),
        #         text=[f'{score:.2f}' for score in positive_scores[::-1]],
        #         textposition='auto'
        #     ))
        #     fig.update_layout(
        #         height=500, 
        #         xaxis_title="Log Probability",
        #         yaxis_title="",
        #         title="Top 15 Positive Indicators",
        #         showlegend=False
        #     )
        #     st.plotly_chart(fig, use_container_width=True)

        # with col3:
        #     st.markdown("### 🟡 Top Neutral Words")
        #     fig = go.Figure(go.Bar(
        #         x=neutral_scores[::-1],
        #         y=neutral_words[::-1],
        #         orientation='h',
        #         marker=dict(color='#f2c94c'),
        #         text=[f'{score:.2f}' for score in neutral_scores[::-1]],
        #         textposition='auto'
        #     ))

        #     fig.update_layout(
        #         height=500,
        #         xaxis_title="Log Probability",
        #         yaxis_title="",
        #         title="Top 15 Neutral Indicators",
        #         showlegend=False
        #     )

        #     st.plotly_chart(fig, use_container_width=True)
                
        # Word Clouds
        st.markdown("---")
        st.markdown("### ☁️ Word Clouds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔴 Negative Word Cloud")
            try:
                # Create word cloud for negative words
                wordcloud_neg = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    colormap='Reds',
                    max_words=100
                ).generate(' '.join(negative_words * 10))  # Multiply for better visualization
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud_neg, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            except Exception as e:
                st.info("Word cloud generation requires additional setup. Showing word list instead.")
                st.write(", ".join(negative_words[:20]))
        
        with col2:
            st.markdown("#### 🟢 Positive Word Cloud")
            try:
                # Create word cloud for positive words
                wordcloud_pos = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    colormap='Greens',
                    max_words=100
                ).generate(' '.join(positive_words * 10))
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud_pos, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            except Exception as e:
                st.info("Word cloud generation requires additional setup. Showing word list instead.")
                st.write(", ".join(positive_words[:20]))
        # ---------------- HISTOGRAM (MODEL CONFIDENCE) ----------------

        st.markdown("---")
        st.markdown("## 📊 Prediction Confidence Histogram")

        try:
            confidence_scores = []

            # sample reviews to check model confidence
            sample_reviews = [
                "This product is amazing and works perfectly",
                "Worst purchase ever very disappointed",
                "Average quality nothing special",
                "I really loved this item highly recommend",
                "Not worth the money",
                "It is okay",
                "Very bad quality and broken",
                "Superb performance excellent",
                "Packaging was normal",
                "Terrible experience"
            ]

            for text in sample_reviews:
                cleaned = preprocessor.preprocess(text)
                features = vectorizer.transform([cleaned])
                probs = model.predict_proba(features)[0]

                confidence = max(probs) * 100
                confidence_scores.append(confidence)

            # Create histogram
            fig = px.histogram(
                x=confidence_scores,
                nbins=10,
                labels={"x": "Prediction Confidence (%)", "y": "Number of Reviews"},
                color_discrete_sequence=["#00c6ff"]
            )

            fig.update_layout(
                title="Model Confidence Score Distribution",
                xaxis_title="Confidence %",
                yaxis_title="Frequency",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                height=450
            )

            st.plotly_chart(fig, use_container_width=True)

            st.info("Histogram shows how confident the model is while predicting sentiment. High confidence near 100% means strong prediction reliability.")

        except Exception as e:
            st.warning("Histogram could not be generated.")


# PAGE 4: BATCH ANALYSIS
elif page == "📁 Batch Analysis":
    st.markdown('<p class="main-header">📁 BATCH REVIEW ANALYSIS</p>', unsafe_allow_html=True)
    
    model, vectorizer, preprocessor = load_model()
    
    if model is None:
        st.error("⚠️ Model files not found! Please run 'train_model.py' first.")
    else:
        st.markdown("### 📤 Upload CSV File")
        uploaded_file = st.file_uploader(
            "Choose a CSV file with reviews to analyze",
            type=['csv'],
            help="Upload any CSV file containing reviews. The system will automatically detect the review column, or you can select it manually."
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} rows")
                
                # Smart column detection - try multiple strategies
                review_column = None
                
                # Strategy 1: Check for exact matches (case-insensitive)
                possible_exact_names = ['review', 'reviews', 'text', 'texts', 'summary', 'summaries', 
                                      'comment', 'comments', 'feedback', 'feedbacks', 'opinion', 'opinions',
                                      'description', 'descriptions', 'content', 'contents', 'message', 'messages']
                
                df_columns_lower = [col.lower() for col in df.columns]
                for exact_name in possible_exact_names:
                    if exact_name in df_columns_lower:
                        idx = df_columns_lower.index(exact_name)
                        review_column = df.columns[idx]
                        break
                
                # Strategy 2: Check for columns containing "review" or "text" (case-insensitive)
                if review_column is None:
                    for col in df.columns:
                        col_lower = col.lower()
                        if 'review' in col_lower or 'text' in col_lower or 'comment' in col_lower:
                            # Check if it's a text-like column
                            if df[col].dtype == 'object' or df[col].dtype == 'string':
                                review_column = col
                                break
                
                # Strategy 3: Find the longest text column (most likely to be reviews)
                if review_column is None:
                    text_columns = []
                    for col in df.columns:
                        if df[col].dtype == 'object' or df[col].dtype == 'string':
                            # Calculate average text length
                            avg_length = df[col].astype(str).str.len().mean()
                            if avg_length > 20:  # Likely a review if average length > 20 chars
                                text_columns.append((col, avg_length))
                    
                    if text_columns:
                        # Sort by average length and pick the longest
                        text_columns.sort(key=lambda x: x[1], reverse=True)
                        review_column = text_columns[0][0]
                
                # Strategy 4: Let user manually select if auto-detection fails
                if review_column is None:
                    st.warning("⚠️ Could not automatically detect review column. Please select it manually.")
                    st.info("Available columns: " + ", ".join(df.columns.tolist()))
                    
                    # Get text-like columns for selection
                    text_like_columns = [col for col in df.columns 
                                       if df[col].dtype == 'object' or df[col].dtype == 'string']
                    
                    if text_like_columns:
                        review_column = st.selectbox(
                            "Select the column containing reviews:",
                            text_like_columns,
                            help="Choose the column that contains the review text"
                        )
                    else:
                        st.error("❌ No text-like columns found in the CSV file.")
                        review_column = None
                if review_column:
                    st.info(f"📋 Auto-detected review column: **{review_column}**")
                    
                    # Allow user to change it if needed
                    text_like_columns = [col for col in df.columns 
                                       if df[col].dtype == 'object' or df[col].dtype == 'string']
                    if len(text_like_columns) > 1:
                        with st.expander("🔧 Change Review Column", expanded=False):
                            selected_col = st.selectbox(
                                "Select a different column:",
                                text_like_columns,
                                index=text_like_columns.index(review_column) if review_column in text_like_columns else 0
                            )
                            review_column = selected_col
                
                if review_column:
                    # Show preview
                    with st.expander("👀 Data Preview", expanded=False):
                        st.dataframe(df.head(10))
                        st.write(f"**Total rows:** {len(df)}")
                        st.write(f"**Review column:** {review_column}")
                        st.write(f"**Sample review:** {str(df[review_column].iloc[0])[:200]}..." if len(str(df[review_column].iloc[0])) > 200 else f"**Sample review:** {str(df[review_column].iloc[0])}")
                    
                    if st.button("🚀 Analyze All Reviews", use_container_width=True):
                        with st.spinner("Analyzing reviews... This may take a while for large datasets."):
                            # Preprocess and predict
                            results = []
                            total = len(df)
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for idx, review in enumerate(df[review_column]):
                                try:
                                    cleaned = preprocessor.preprocess(str(review))

                                    if cleaned.strip():
                                        features = vectorizer.transform([cleaned])

                                        # get probabilities
                                        probabilities = model.predict_proba(features)[0]

                                        negative_prob = probabilities[0]
                                        positive_prob = probabilities[1]

                                        confidence = max(negative_prob, positive_prob)

                                        # ---- NEUTRAL LOGIC ----
                                        if confidence < 0.60:
                                            final_prediction = "Neutral"
                                        elif positive_prob > negative_prob:
                                            final_prediction = "Positive"
                                        else:
                                            final_prediction = "Negative"

                                        results.append({
                                            'Review': str(review)[:100] + "..." if len(str(review)) > 100 else str(review),
                                            'Sentiment': final_prediction,
                                            'Confidence': f"{confidence*100:.2f}%"
                                        })

                                    else:
                                        results.append({
                                            'Review': str(review)[:100] + "..." if len(str(review)) > 100 else str(review),
                                            'Sentiment': 'Unknown',
                                            'Confidence': 'N/A',
                                            'Positive_Prob': 0,
                                            'Negative_Prob': 0
                                        })
                                except Exception as e:
                                    results.append({
                                        'Review': str(review)[:100] + "..." if len(str(review)) > 100 else str(review),
                                        'Sentiment': 'Error',
                                        'Confidence': f'Error: {str(e)}',
                                        'Positive_Prob': 0,
                                        'Negative_Prob': 0
                                    })
                                
                                # Update progress
                                progress = (idx + 1) / total
                                progress_bar.progress(progress)
                                status_text.text(f"Processing {idx + 1} of {total} reviews...")
                            
                            results_df = pd.DataFrame(results)
                            
                            # Calculate summary statistics
                            positive_count = len(results_df[results_df['Sentiment'] == 'Positive'])
                            negative_count = len(results_df[results_df['Sentiment'] == 'Negative'])
                            neutral_count  = len(results_df[results_df['Sentiment'] == 'Neutral'])

                            avg_confidence = results_df[results_df['Sentiment'] != 'Unknown'][results_df['Sentiment'] != 'Error']['Confidence'].str.rstrip('%').astype(float).mean()
                            
                            st.markdown("### 📊 Analysis Results")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Reviews", len(results_df))
                            with col2:
                                st.metric("✅ Positive", f"{positive_count} ({positive_count/len(results_df)*100:.1f}%)")
                            with col3:
                                st.metric("❌ Negative", f"{negative_count} ({negative_count/len(results_df)*100:.1f}%)")
                            with col4:
                                st.metric("😐 Neutral", f"{neutral_count} ({neutral_count/len(results_df)*100:.1f}%)")

                            # Display results table
                            st.markdown("### 📋 Detailed Results Table")
                            st.dataframe(results_df[['Review', 'Sentiment', 'Confidence']], use_container_width=True, height=400)
                            
                            # Download button
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="💾 Download Results CSV",
                                data=csv,
                                file_name="sentiment_analysis_results.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            
                            # Visualization
                            st.markdown("### 📈 Sentiment Distribution")
                            # --------- Attractive Donut Chart ---------

                            total_reviews = positive_count + negative_count + neutral_count

                            fig = go.Figure(data=[go.Pie(
                                labels=['Positive 😊', 'Negative 😡', 'Neutral 😐'],
                                values=[positive_count, negative_count, neutral_count],
                                hole=0.55,   # makes donut chart
                                textinfo='percent+label',
                                textfont=dict(size=15, color='white'),
                                marker=dict(
                                    colors=['#2ecc71', '#e74c3c', '#f1c40f'],
                                    line=dict(color='#0f2027', width=3)
                                ),
                                hovertemplate=
                                "<b>%{label}</b><br>" +
                                "Reviews: %{value}<br>" +
                                "Percentage: %{percent}<extra></extra>"
                            )])

                            # Center Text
                            fig.add_annotation(
                                text=f"<b>{total_reviews}</b><br>Total Reviews",
                                x=0.5, y=0.5,
                                font_size=22,
                                showarrow=False,
                                font_color="white"
                            )

                            # Layout styling
                            fig.update_layout(
                                title={
                                    'text': "📊 Overall Sentiment Distribution",
                                    'y':0.95,
                                    'x':0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top',
                                    'font': dict(size=22, color='white')
                                },
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=-0.15,
                                    xanchor="center",
                                    x=0.5,
                                    font=dict(size=13, color="white")
                                ),
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                height=500
                            )

                            st.plotly_chart(fig, use_container_width=True)

                            
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                st.info("Please ensure your file is a valid CSV format.")

# PAGE 5: ABOUT
elif page == "ℹ️ About":
    st.markdown('<p class="main-header">ℹ️ ABOUT THIS PROJECT</p>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Project Overview
    
    Advanced NLP-powered sentiment analysis system for Amazon food product reviews. 
    Achieves **84.8% accuracy** using Naive Bayes classification and TF-IDF features.
    
    This dashboard provides an interactive interface to:
    - Analyze individual reviews in real-time
    - Process bulk reviews from CSV files
    - Explore model insights and word importance
    - View comprehensive performance metrics
    """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Dataset Information")
    st.markdown("""
    - **Source:** Amazon Fine Food Reviews
    - **Total Reviews:** 568,453
    - **Time Span:** Oct 2020 - Oct 2024
    - **Positive Reviews:** 78.07%
    - **Negative Reviews:** 21.93%
    """)
    
    st.markdown("---")
    
    st.markdown("### 🛠️ Methodology")
    
    st.markdown("""
    **1️⃣ Data Preprocessing**
    - Text cleaning & normalization
    - Stopword removal
    - Lowercasing & punctuation removal
    - URL and email removal
    
    **2️⃣ Feature Engineering**
    - TF-IDF vectorization (5000 features)
    - Unigrams & bigrams
    - Minimum document frequency filtering
    
    **3️⃣ Model Training**
    - Multinomial Naive Bayes classifier
    - 80/20 train-test split
    - Stratified sampling for balanced distribution
    
    **4️⃣ Evaluation**
    - Accuracy: 84.80%
    - Precision, Recall, F1-Score metrics
    - Confusion matrix analysis
    """)
    
    st.markdown("---")
    
    st.markdown("### 📈 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔴 Negative Class:**")
        st.markdown("""
        - Precision: 87.27%
        - Recall: 35.93%
        - F1-Score: 50.91%
        """)
    
    with col2:
        st.markdown("**🟢 Positive Class:**")
        st.markdown("""
        - Precision: 84.55%
        - Recall: 98.53%
        - F1-Score: 91.01%
        """)
    
    st.markdown("---")
    
    st.markdown("### 🚀 Getting Started")
    st.markdown("""
    1. **Train the Model:** Run `python train_model.py` to train and save the model
    2. **Launch Dashboard:** Run `streamlit run app.py` to start the interactive dashboard
    3. **Analyze Reviews:** Use the Live Predictor for single reviews or Batch Analysis for CSV files
    """)
    
    st.markdown("---")
    
    st.markdown("""
    <div class='about-card' style='text-align: center; padding: 2rem; border-radius: 10px;'>

        <p style='font-size: 1.2rem; margin: 0.5rem;'>Made with ❤️ using Streamlit</p>
        <p style='font-size: 1rem; margin: 0.5rem;'>Product Review Sentiment Analysis Dashboard</p>
    </div>
    """, unsafe_allow_html=True)