# 🎯 Product Review Sentiment Analyzer - Streamlit Dashboard

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python train_model.py
```
This will:
- Load the Reviews.csv dataset
- Train the sentiment analysis model
- Save model files (sentiment_model.pkl, tfidf_vectorizer.pkl, etc.)

### Step 3: Launch the Dashboard
```bash
streamlit run app.py
```

The dashboard will open in your browser automatically!

## 📱 Dashboard Features

### 🏠 Home
- Overview metrics (total reviews, accuracy, positive/negative counts)
- Confusion matrix visualization
- Class performance metrics

### 🔮 Live Predictor
- Real-time sentiment analysis for individual reviews
- Confidence scores and probability breakdowns
- Interactive gauge charts

### 📊 Model Insights
- Top positive and negative words
- Word importance visualizations
- Word clouds (if wordcloud library is properly configured)

### 📁 Batch Analysis
- Upload CSV files with reviews
- Bulk sentiment analysis
- Download results as CSV
- Sentiment distribution charts

### ℹ️ About
- Project overview and methodology
- Dataset information
- Model performance details

## 📋 CSV Upload Format

For Batch Analysis, your CSV file should have one of these column names:
- `review` or `Review`
- `text` or `Text`
- `Summary` or `summary`

## 🔧 Requirements

All dependencies are listed in `requirements.txt`. Key libraries:
- Streamlit (dashboard framework)
- Scikit-learn (ML models)
- NLTK (NLP preprocessing)
- Plotly (interactive charts)
- Pandas (data handling)

## 📝 Notes

- The model must be trained first using `train_model.py` before using the dashboard
- The training script uses the `Reviews.csv` file in the project directory
- Model files are saved as pickle files for fast loading
