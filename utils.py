import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet = True)

stop_words = set(stopwords.words('english'))

def clean(text):
    '''
    This function cleans a raw news article given by user into a cleaned string for NLP processing.

    Steps:
    1. lowercase
    2. remove urls
    3. remove mentions and hashtags
    4. remove content inside square brackets
    5. remove non-alphabetic characters
    6. remove stopwords and single character tokens
    7. remove extra whitespace

    '''
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+ | www\S+', '', text)
    text = re.sub(r'@\w+ | #\w+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([i for i in text.split() if i not in stop_words and len(i) > 1])
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == '__main__':
    # Just to check if the function is working properly
    sample = 'PRESIDENT UNDER ATTACK!!!! https://newsx.com # Breaking News #9199'
    print(f'Input Sample: {sample}')
    print(f'\nCleaned Output after clean() function: {clean(sample)}')