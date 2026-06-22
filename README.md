# 🔍 Fake News Detector

An end-to-end fake news detection system combining NLP, Machine Learning,
and Generative AI — trained on 72,134 real-world news articles.

🚀 **[Live Demo](https://fake-news-detector-lx6xras373wxp86nkram4p.streamlit.app/)**

---

## Overview

This project classifies news articles as Fake or Real using a TF-IDF +
Logistic Regression pipeline, then uses Google Gemini to generate a plain-
language explanation of the prediction. An educational article generator
demonstrates the linguistic differences between fake and credible journalism.

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

Trained and evaluated on the WELFake dataset (72,134 articles, 80/20 split).

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 95.25% |
| Precision | 94.59% |
| Recall    | 96.26% |
| F1 Score  | 95.41% |

---

## How It Works

```
User Input → Text Cleaning → TF-IDF Vectorization → Logistic Regression
→ Prediction + Confidence Score → Gemini Explanation → Streamlit UI
```

1. Raw article text is cleaned: lowercased, URLs/mentions removed,
   stopwords filtered, non-alphabetic characters stripped.
2. TF-IDF converts cleaned text into a 5,000-feature sparse matrix
   using unigrams and bigrams.
3. Logistic Regression predicts Fake or Real with a calibrated
   probability score.
4. Google Gemini generates a 3–4 paragraph educational explanation
   of the prediction (optional, togglable to conserve API quota).

---

## Tech Stack

| Component      | Technology                  |
|----------------|-----------------------------|
| Language       | Python 3.10+                |
| ML             | scikit-learn (LR + TF-IDF)  |
| NLP            | NLTK                        |
| Generative AI  | Google Gemini 2.5 Flash     |
| Interface      | Streamlit                   |
| Data           | Pandas                      |

---

## Dataset

**WELFake Dataset** — 72,134 articles (51% fake, 49% real) sourced from
Kaggle, Reuters, BuzzFeed Political, and McIntire.

Not included in this repository due to size.
[Download from Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env
streamlit run app.py
```

To retrain the model from scratch:
```bash
python train.py
```

---

## Design Decisions

**Why WELFake?**
An initial synthetic dataset was rejected after manual row inspection
revealed artificial word sequences. WELFake contains genuine scraped
articles and better represents real-world inputs.

**Why TF-IDF over embeddings?**
TF-IDF is interpretable, fast to train, and well-suited to high-dimensional
sparse text classification. Bigrams partially compensate for lack of word
order. Logistic regression coefficients remain directly inspectable.

**Why Logistic Regression?**
Produces calibrated probability scores, trains in seconds on 72K samples,
and performs strongly on sparse TF-IDF features. Interpretability was
prioritized for an educational use case.

---

## Limitations & Key Findings

**Shortcut learning (source leakage)**
Coefficient analysis shows the model's strongest real-news indicators are
source attribution tokens: `reuters`, `washington reuters`, `york times`.
The model learned to associate wire service bylines with credibility rather
than linguistic or factual features of the article itself.

Adversarial probe confirming this:
> *"Reuters reported that the moon is made of cheese. Washington Reuters
> confirms that drinking bleach cures cancer."*
> **Classified: 100% Real**

**Distribution mismatch**
LLM-generated "real-style" articles are consistently classified as Fake
(65–78% fake probability). The model was not trained on LLM-generated text,
and Gemini's vocabulary distribution differs from the WELFake corpus.
Genuine Reuters articles classify correctly.

**Negation loss**
Stopword removal eliminates words like "not", altering semantic meaning
in edge cases. Context-aware models such as BERT would handle this
differently at the cost of interpretability and training speed.

---

## Project Structure

```
├── app.py              # Streamlit application
├── predict.py          # ML prediction pipeline
├── genai.py            # Gemini explanation and article generation
├── train.py            # Model training pipeline
├── utils.py            # Shared text cleaning
├── models/             # Saved vectorizer and classifier (.pkl)
├── notebooks/          # EDA, preprocessing, feature engineering, training
├── screenshots/
├── requirements.txt
└── README.md
```