import joblib
from utils import clean

vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
lr = joblib.load('models/logistic_regression_model.pkl')
mnb = joblib.load('models/Multinomial_Naive_Bayes_model.pkl')
svm = joblib.load('models/Linear_SVM_model.pkl')

labels = {0:'Real', 1:'Fake'}

def predict_with_model(model, features):
    '''
    Takes one model at a time and gives it's results.
    '''
    label_index = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    fake_prob = probabilities[1]
    real_prob = probabilities[0]

    return {
        'label': labels[label_index],
        'confidence': round(float(max(probabilities)), 4),
        'fake_probability': round(float(fake_prob), 4),
        'real_probability': round(float(real_prob), 4),
    }

def predict_all(raw_text):
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

    return{
        'cleaned_text': cleaned,
        'lr': predict_with_model(lr, features),
        'mnb': predict_with_model(mnb, features),
        'svm': predict_with_model(svm, features)
    }

if __name__ == '__main__':

    test_article = '''Scientists have confirmed that the new vaccine is safe and effective
    after conducting trials across multiple countries. The results were
    published in the New England Journal of Medicine and peer reviewed
    by independent researchers.'''

    result = predict_all(test_article)

    print(f'Logistic Regression Prediction Result:')
    print(f"label:              {result['lr']['label']}")
    print(f"confidence:         {result['lr']['confidence'] * 100: .1f}%")
    print(f"fake probability:   {result['lr']['fake_probability'] * 100: .1f}%")
    print(f"real probability:   {result['lr']['real_probability'] * 100: .1f}%")

    print(f'Multinomial Naive Bayes Prediction Result:')
    print(f"label:              {result['mnb']['label']}")
    print(f"confidence:         {result['mnb']['confidence'] * 100: .1f}%")
    print(f"fake probability:   {result['mnb']['fake_probability'] * 100: .1f}%")
    print(f"real probability:   {result['mnb']['real_probability'] * 100: .1f}%")

    print(f'Linear SVM Prediction Result:')
    print(f"label:              {result['svm']['label']}")
    print(f"confidence:         {result['svm']['confidence'] * 100: .1f}%")
    print(f"fake probability:   {result['svm']['fake_probability'] * 100: .1f}%")
    print(f"real probability:   {result['svm']['real_probability'] * 100: .1f}%")

    print(f"\nCleaned text:")
    print(f"{result['cleaned_text']}")