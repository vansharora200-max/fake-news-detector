import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

def explain_prediction(raw_article, prediction_result):
    '''
    Generates a natural language explanation for a news prediction.
    '''
    label = prediction_result['label']
    confidence = prediction_result['confidence'] * 100
    cleaned = prediction_result['cleaned_text']

    prompt = f"""
    You are an expert fact-checker and media literacy educator.

    A machine learning model has analyzed a news article and made the following prediction:
    - Verdict: {label} News
    - Confidence: {confidence:.1f}%

    Here is the original article:
    \"\"\"{raw_article[:1500]}\"\"\"

    Key terms the model identified as significant:
    {cleaned[:300]}

    Please provide a clear, educational explanation (3-4 paragraphs) that:
    1. Explains what linguistic or content features might have led to this prediction
    2. Points out specific phrases or patterns in the article that are typical of {label.lower()} news
    3. Reminds the reader that this is an AI prediction and should not replace critical thinking
    4. Suggests one or two ways the reader can independently verify this article

    Write in simple, accessible language suitable for a general audience.
    Do not repeat the verdict in your first sentence.
    """

    try:
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Explanation unavailable at this time. Error: {str(e)}"
    
def generate_article(topic, article_type='fake'):
    '''
    Generates a sample news article for educational comparison.
    '''
    if article_type == 'fake':
        style_instructions = """
    - Use a sensationalist, clickbait headline in ALL CAPS
    - Use emotional, alarming, and exaggerated language
    - Include vague source attributions like "sources say" or "experts claim"
    - Add at least one unverifiable statistic
    - Mix one real fact with several false or exaggerated claims
    - Create a sense of urgency or outrage
    """
        educational_label = "RED FLAGS IN THIS FAKE ARTICLE"
        flag_instruction = "List 3 specific features that mark this as fake news."

    else:  # real
        style_instructions = """
    - Use a neutral, factual headline in title case
    - Use calm, precise, and measured language
    - Cite specific named sources with credentials
    - Include verifiable statistics with their origin mentioned
    - Present multiple perspectives where relevant
    - Avoid emotionally charged words
    """
        educational_label = "CREDIBILITY MARKERS IN THIS ARTICLE"
        flag_instruction = "List 3 specific features that mark this as credible journalism."

    prompt = f"""
    You are an educator teaching media literacy through article examples.

    Generate a {article_type} news article about: "{topic}"

    Style requirements:
    {style_instructions}

    Length: 150-200 words.

    After the article, add this clearly marked educational section:
    --- {educational_label} ---
    {flag_instruction}
    1. [point 1]
    2. [point 2]
    3. [point 3]

    Format your response exactly as:
    HEADLINE: [headline here]

    ARTICLE:
    [article body here]

    --- {educational_label} ---
    [your 3 points here]

    Important: This is purely for media literacy education.
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        return f"Generation unavailable at this time. Error: {str(e)}"
    
if __name__ == '__main__':
    sample_article = """
    Scientists have confirmed that drinking coffee reverses aging completely.
    A secret study conducted by anonymous researchers found that 3 cups daily
    makes you immortal. The government has been hiding this information for decades.
    Sources close to the White House confirm this cover-up has been ongoing since 1987.
    """
    sample_prediction = {
        'label': 'Fake',
        'confidence': 0.91,
        'fake_probability': 0.91,
        'real_probability': 0.09,
        'cleaned_text': 'scientists confirmed drinking coffee reverses aging secret study anonymous researchers government hiding information'
    }
    explanation = explain_prediction(sample_article, sample_prediction)
    print('EXPLANATION:\n')
    print(explanation)



# print("Testing generate_article() - FAKE")
# print('=' * 50)
# print(generate_article('Artificial Intelligence', article_type='fake'))
# print('\n' + '=' * 50)
# print("Testing generate_article() - REAL")
# print('=' * 50)
# print(generate_article('Artificial Intelligence', article_type='real'))