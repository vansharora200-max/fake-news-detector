import streamlit as st
import joblib
from predict import predict_all
from genai import explain_prediction, generate_article
from utils import clean
def display_model_result(model_name, model_result):
    st.subheader(model_name)
    if model_result['label'] == 'Fake':
        st.error(f"🚨 Verdict: FAKE NEWS - {model_result['confidence']*100:.1f}% confidence")
    else:
        st.success(f" Verdict: REAL NEWS - {model_result['confidence']*100:.1f}% confidence")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Fake Probability", f"{model_result['fake_probability']*100:.1f}%")
    with col2:
        st.metric("Real Probability", f"{model_result['real_probability']*100:.1f}%")

    st.progress(model_result['fake_probability'], text="Fake likelihood")
    st.progress(model_result['real_probability'], text="Real likelihood")

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Fake News Detector")
st.markdown(
    "Analyze news articles using Machine Learning and "
    "get AI-powered explanations of the prediction."
)
st.divider()

tab1, tab2 = st.tabs(["📰 Analyze Article", "✍️ Generate Sample Article"])

with tab1:
    st.subheader("Paste a news article to analyze")

    user_input = st.text_area(
        label="Article Text",
        placeholder="Paste the full article or headline here...",
        height=200,
        label_visibility="collapsed"
    )
    
    enable_explanation = st.checkbox("Generate AI Explanation (uses API limits)", value=True)
    
    analyze_button = st.button("Analyze Article", type="primary")

    if analyze_button:
        if not user_input.strip():
            st.warning("Please paste an article before analyzing.")
        
        else:
            with st.spinner("Analyzing Article..."):
                result = predict_all(user_input)
            st.session_state['result'] = result
            st.session_state['user_input'] = user_input
            st.session_state['explanation'] = None

    if 'result' in st.session_state:
            result = st.session_state['result']
            
            st.divider()
            display_model_result('Logistic Regression', result['lr'])
            st.divider()
            display_model_result('Multinomial Naive Bayes', result['mnb'])
            st.divider()
            display_model_result('Linear SVM', result['svm'])
            st.divider()
            st.subheader("AI Explanation")
            
            if enable_explanation:
                if st.session_state.get('explanation') is None:
                    with st.spinner("Generating Explanation..."):
                        st.session_state['explanation'] = explain_prediction(st.session_state['user_input'], result['lr'], result['cleaned_text'])   
                st.markdown(st.session_state['explanation'])
            else:
                st.info("AI Explanation disabled.")

            with st.expander("View Preprocessed Text"):
                st.caption(
                    'This is what the model analyzed after'
                    'removing stopwords, punctuations and URLs.'
                )
                st.text(result['cleaned_text'])

with tab2:
    st.subheader("Generate a sample article for educational comparison")
    st.caption(
        "⚠️ All content generated here is synthetic and created purely "
        "for media literacy education. It does not represent real events."
    )

    topic_input = st.text_input(
        label="Topic",
        placeholder="e.g. Weather forecast, Election Results, Artificial Intelligence, Sports Highlights"
    )

    article_type = st.radio(
        label="Article Style",
        options=['Fake','Real'],
        horizontal=True,
        help="Generate a fake article to see misinformation patterns, "
             "or a credible-style article to compare writing differences."
    )

    generate_button = st.button("Generate Article", type="primary")

    if generate_button:
        if not topic_input.strip():
            st.warning("Please enter a topic before generating.")

        else:
            with st.spinner(f"Generating {article_type.lower()} article..."):
                generated = generate_article(topic_input, article_type.lower())

            st.divider()

            if article_type == 'Fake':
                st.error(
                    "🚨 GENERATED FAKE ARTICLE — This is AI-created synthetic "
                    "content for educational purposes only. Not real news."
                )
            else:
                st.info(
                    "📰 CREDIBLE STYLE ARTICLE — This is AI-generated content "
                    "written in the style of credible journalism. "
                    "Not a real news report."
                )
            
            st.markdown(generated)