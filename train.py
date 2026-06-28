import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
from utils import clean
import nltk
nltk.download('stopwords', quiet=True)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def load_data(filepath):
    '''
    Load and validate the cleaned dataset.
    '''
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)

    required = ['Cleaned', 'label']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f'Dataset missing required columns: {missing}')
    
    df = df[df['Cleaned'].str.strip() != '']

    print(f'Dataset loaded: {df.shape[0]} rows')
    print(f'Label Distribution:\n{df['label'].value_counts()}\n')
    return df

def build_features(df):
    '''
    Split data and fit TF-IDF vectorizer.
    '''
    X = df['Cleaned']
    Y = df['label']

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    print(f'Train Size: {len(X_train)}  |  Test Size: {len(X_test)}')

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2),
        min_df=2,
        max_df=0.95
    )

    print(f'Fitting TF-IDF vectorizer...')

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f'Feature matrix shape: {X_train_tfidf.shape}\n')
    return X_train_tfidf, X_test_tfidf, Y_train, Y_test, vectorizer

def train_models(X_train_tfidf, Y_train):
    '''
    To Train logistic regression, Multinomial Naive Bayes and Linear SVM classifier.
    '''
    print(f'Training Logistic Regression Model...')
    log_reg = LogisticRegression(max_iter=1000,random_state=42)
    log_reg.fit(X_train_tfidf, Y_train)
    print('Done')
    print(f'Training Multinomial Naive Bayes Model...')
    multi_nb = MultinomialNB()
    multi_nb.fit(X_train_tfidf, Y_train)
    print('Done')
    print(f'Training Linear SVM Model...')
    lin_svm_base = LinearSVC(max_iter=2000,random_state=42)
    lin_svm = CalibratedClassifierCV(lin_svm_base, cv=5)
    lin_svm.fit(X_train_tfidf, Y_train)
    print('Done\n')
    return log_reg, multi_nb, lin_svm

def evaluate_models(lr, mnb, l_svm, X_test_tfidf, Y_test):
    '''
    Print evaluation metrics.
    '''
    Y_pred_lr = lr.predict(X_test_tfidf)
    print('=' * 50)
    print('LOGISTIC REGRESSION EVALUATION RESULTS:')
    print('=' * 50)
    print(f'Accuracy  : {accuracy_score(Y_test,Y_pred_lr):.4f}')
    print(f'Precision : {precision_score(Y_test,Y_pred_lr):.4f}')
    print(f'Recall    : {recall_score(Y_test,Y_pred_lr):.4f}')
    print(f'F1 Score  : {f1_score(Y_test,Y_pred_lr):.4f}')
    print('=' * 50)
    print('\nLinear Regression Classification Report:')
    print(classification_report(Y_test,Y_pred_lr,target_names=['Real','Fake']))

    Y_pred_mnb = mnb.predict(X_test_tfidf)
    print('=' * 50)
    print('MULTINOMIAL NAIVE BAYES EVALUATION RESULTS:')
    print('=' * 50)
    print(f'Accuracy  : {accuracy_score(Y_test,Y_pred_mnb):.4f}')
    print(f'Precision : {precision_score(Y_test,Y_pred_mnb):.4f}')
    print(f'Recall    : {recall_score(Y_test,Y_pred_mnb):.4f}')
    print(f'F1 Score  : {f1_score(Y_test,Y_pred_mnb):.4f}')
    print('=' * 50)
    print('\nMultinomial Naive Bayes Classification Report:')
    print(classification_report(Y_test,Y_pred_mnb,target_names=['Real','Fake']))

    Y_pred_svm = l_svm.predict(X_test_tfidf)
    print('=' * 50)
    print('LINEAR SVM EVALUATION RESULTS:')
    print('=' * 50)
    print(f'Accuracy  : {accuracy_score(Y_test,Y_pred_svm):.4f}')
    print(f'Precision : {precision_score(Y_test,Y_pred_svm):.4f}')
    print(f'Recall    : {recall_score(Y_test,Y_pred_svm):.4f}')
    print(f'F1 Score  : {f1_score(Y_test,Y_pred_svm):.4f}')
    print('=' * 50)
    print('\nLinear SVM Classification Report:')
    print(classification_report(Y_test,Y_pred_svm,target_names=['Real','Fake']))

def save_artifacts(lr, mnb, l_svm, vectorizer, output_dir = 'models'):
    '''
    Save trained models and vectorizer to disk.
    '''
    os.makedirs(output_dir, exist_ok=True)
    lr_path = os.path.join(output_dir, 'logistic_regression_model.pkl')
    mnb_path = os.path.join(output_dir, 'Multinomial_Naive_Bayes_model.pkl')
    l_svm_path = os.path.join(output_dir, 'Linear_SVM_model.pkl')
    vectorizer_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')

    joblib.dump(lr, lr_path)
    joblib.dump(mnb, mnb_path)
    joblib.dump(l_svm, l_svm_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f'\n Models saved to {lr_path}, {mnb_path} and {l_svm_path}')
    print(f'\n Vectorizer saved to {vectorizer_path}')


if __name__ == '__main__':
    # FULL TRAINING PIPELINE
    dataset_path = 'dataset/WELFake_Cleaned.csv'

    df = load_data(dataset_path)
    X_train_tfidf, X_test_tfidf, Y_train, Y_test, vectorizer = build_features(df)
    lr, mnb, l_svm = train_models(X_train_tfidf, Y_train)    
    evaluate_models(lr, mnb, l_svm, X_test_tfidf, Y_test)
    save_artifacts(lr, mnb, l_svm, vectorizer)

    print('\nTraining pipeline complete.')
    print("Run 'streamlit run app.py' to launch the application.")