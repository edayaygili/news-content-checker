import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Modeli yükle
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L12-v2")

model = load_model()

# Kategoriler
content_options = [
    "black voices", "business", "comedy", "crime", "culture&arts",
    "education", "entertainment", "environment", "food&drink", "home&living",
    "media", "parenting", "politics", "sports", "style&beauty",
    "tech", "u.s news", "weird news", "world news"
]

# Arayüz
st.title("🗞️ News Relevance Checker")
st.write("This app checks if a short news description is relevant to a selected content category using BERT.")

text = st.text_area("News Description", height=200)
category = st.selectbox("Content Category", content_options)

if st.button("Check Relevance"):
    if text and category:
        text_embed = model.encode([text])
        category_embed = model.encode([category])
        similarity = cosine_similarity(text_embed, category_embed)[0][0]
        st.markdown(f"**Similarity Score:** {similarity:.3f}")
        if similarity >= 0.10:
            st.success("✅ Relevant")
        else:
            st.error("❌ Non-Relevant")
    else:
        st.warning("Please fill in both fields.")

