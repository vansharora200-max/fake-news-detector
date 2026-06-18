import joblib
from utils import clean

vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
model = joblib.load('models/logistic_regression_model.pkl')
labels = {0:'Real', 1:'Fake'}

def predict(raw_text):
    '''
    This functions runs the full prediction pipeline on a raw news article.

    Steps:
    1. clean raw text
    2. transform using the tf-idf vectorizer
    3. predict label and probability using logistic regression
    4. return the structured result dict

    '''
    cleaned = clean(raw_text)
    features = vectorizer.transform([cleaned])
    label_index = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    fake_prob = probabilities[1]
    real_prob = probabilities[0]

    return {
        'label': labels[label_index],
        'confidence': round(float(max(probabilities)), 4),
        'fake_probability': round(float(fake_prob), 4),
        'real_probability': round(float(real_prob), 4),
        'cleaned_text': cleaned
    }

if __name__ == '__main__':

    test_article = '''Scientists have confirmed that the new vaccine is safe and effective
    after conducting trials across multiple countries. The results were
    published in the New England Journal of Medicine and peer reviewed
    by independent researchers.'''

    result = predict(test_article)

    print(f'Prediction Result:')
    print(f"label:              {result['label']}")
    print(f"confidence:         {result['confidence'] * 100: .1f}%")
    print(f"fake probability:   {result['fake_probability'] * 100: .1f}%")
    print(f"real probability:   {result['real_probability'] * 100: .1f}%")
    print(f"\nCleaned text:")
    print(f"{result['cleaned_text']}")