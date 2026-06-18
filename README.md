# 🔍 Fake News Detector

An end-to-end fake news detection system built using NLP, 
Machine Learning, and Generative AI.

> Live Demo: [Add Streamlit link after deployment]

---

## Overview

This project detects whether a news article is fake or real using 
a TF-IDF + Logistic Regression pipeline trained on the WELFake dataset 
(72,134 articles). Google Gemini explains each prediction in plain language.

The project also includes an educational article generator that produces 
fake and credible-style articles on any topic for media literacy purposes.

---

## Screenshots

### Prediction Result
![Prediction Result](screenshots/prediction_result.png)

### AI Explanation
![AI Explanation](screenshots/ai_explanation.png)

### Article Generation
![Article Generation](screenshots/article_generation.png)

---

## Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 95.25% |
| Precision | 94.59% |
| Recall    | 96.26% |
| F1 Score  | 95.41% |

Trained on WELFake dataset — 72,134 real news articles from 
multiple verified sources.

---

## Project Structure

Fake_News_Detector/

├── dataset/

│ ├── WELFake_Dataset.csv

│ └── WELFake_Cleaned.csv

├── notebooks/

│ ├── 01_dataset_understanding.ipynb

│ ├── 02_preprocessing.ipynb

│ ├── 03_feature_engineering.ipynb

│ └── 04_model_training.ipynb

├── models/

│ ├── tfidf_vectorizer.pkl

│ └── logistic_regression_model.pkl

├── screenshots/

├── utils.py

├── predict.py

├── genai.py

├── train.py

├── app.py

├── requirements.txt

├── README.md

└── .gitignore

---

## How It Works

User Input (News Article)
            ↓
Text Cleaning & Preprocessing
            ↓
TF-IDF Vectorization
            ↓
Logistic Regression Model
            ↓
Prediction + Confidence Score
            ↓
Google Gemini Explanation
            ↓
Displayed in Streamlit Interface

---

## Tech Stack

| Component        | Technology                    |
|-----------------|-------------------------------|
| Language         | Python 3.10+                  |
| ML Framework     | scikit-learn                  |
| NLP              | NLTK, TF-IDF                  |
| Generative AI    | Google Gemini 2.5 Flash       |
| Web Interface    | Streamlit                     |
| Data Processing  | Pandas, NumPy                 |

---

## Dataset

**WELFake Dataset**  
- 72,134 news articles (fake + real)  
- Sources: Kaggle, McIntire, Reuters, BuzzFeed Political  
- Nearly balanced: 51% fake, 49% real  
- Selected over synthetic datasets to ensure real-world applicability

Due to repository size limitations, dataset files are not included.

Download:
https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
---

## Setup

### Prerequisites
- Python 3.10+
- Google Gemini API key ([Get one here](https://aistudio.google.com/app/apikey))

### Installation

```bash
# Clone the repository
git clone https://github.com/vansharora200-max/fake-news-detector.git
cd fake-news-detector

# Install dependencies
pip install -r requirements.txt

# Add your Gemini API key
echo "GEMINI_API_KEY=your_key_here" > .env

# Run the application (models already included)
streamlit run app.py
```

### Retrain from Scratch (Optional)

```bash
python train.py
```

---

## Design Decisions

**Why WELFake over other datasets?**  
An initial synthetic dataset was rejected after manual inspection revealed 
artificial word sequences rather than real news language. WELFake contains 
genuine articles and better represents real-world user input.

**Why TF-IDF over word embeddings?**  
TF-IDF is interpretable, fast, and appropriate for a Bag-of-Words 
classification task. Bigrams (ngram_range=(1,2)) partially compensate 
for the lack of word order.

**Why Logistic Regression?**  
Produces calibrated probability scores, trains quickly, and performs 
strongly on high-dimensional sparse text features.

**Known limitation:**  
Stopword removal eliminates negation words like "not", which can alter 
semantic meaning. This is an accepted trade-off for Bag-of-Words models. 
Context-aware models like BERT would handle this differently.

---

## Acknowledgements

- Fake News Classification WELFake Dataset
- Google Gemini API
- Streamlit