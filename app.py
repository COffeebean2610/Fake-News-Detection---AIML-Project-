import streamlit as st
import joblib

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

st.title("🔍 Fake News Detection System")
st.markdown("### A machine learning-powered web application that analyzes news articles using NLP and logistic regression")

with st.expander("📖 How It Works"):
    st.markdown("""
    1. **Text Preprocessing**: Input text is vectorized using TF-IDF
    2. **Model Prediction**: Pre-trained logistic regression analyzes text patterns
    3. **Classification**: Returns binary prediction (Fake/Real)
    """)

with st.expander("🛠️ Libraries Used"):
    st.markdown("""
    - **Streamlit**: Web application framework
    - **Joblib**: Model serialization and loading
    - **Scikit-learn**: TF-IDF Vectorizer, Logistic Regression
    """)

with st.expander("🎯 Applications"):
    st.markdown("""
    - **Media Verification**: Journalists verify news authenticity
    - **Social Media Monitoring**: Detect misinformation on platforms
    - **Educational Tool**: Teach critical thinking about news
    - **Content Moderation**: Automated fake news filtering
    - **Research**: Study misinformation patterns
    - **Fact-Checking**: Support preliminary analysis
    """)

st.markdown("---")
st.markdown("### 📰 Test the Model")
st.write("Enter a news article below to check whether it is Fake or Real:")

inputn = st.text_area("News Article:", placeholder="Paste your news article here...")

if st.button("🔍 Check News", type="primary"):
    if inputn.strip():
        transform_input = vectorizer.transform([inputn])
        prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("✅ The News is Real!")
        else:
            st.error("❌ The News is Fake!")
    else:
        st.warning("⚠️ Please enter some text to analyze.")

st.markdown("---")
st.markdown("### 📝 Sample News Articles for Testing")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**🔴 Fake News Sample:**")
    fake_sample = "Scientists at MIT have discovered that drinking coffee while standing on one foot for 30 seconds daily can increase IQ by 15 points. The study, conducted over 6 months with 10,000 participants, showed remarkable cognitive improvements. Dr. Sarah Johnson, lead researcher, claims this breakthrough could revolutionize education systems worldwide."
    if st.button("📋 Use Fake Sample"):
        st.text_area("Selected Sample:", fake_sample, height=150, key="fake_display")
        
with col2:
    st.markdown("**🟢 Real News Sample:**")
    real_sample = "NASA's James Webb Space Telescope has captured detailed images of the Crab Nebula, located about 6,500 light-years from Earth. The infrared observations reveal intricate structures within the nebula's gas and dust clouds, providing new insights into stellar evolution. The images were released by NASA on January 15, 2024, as part of ongoing deep space observations."
    if st.button("📋 Use Real Sample"):
        st.text_area("Selected Sample:", real_sample, height=150, key="real_display") 