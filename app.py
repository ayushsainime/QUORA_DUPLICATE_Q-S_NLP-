# app.py
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import numpy as np

from utils import load_model, load_vectorizer, query_point_creator

st.set_page_config(page_title="Quora Duplicate Detector", layout="centered")

# 1) Load model & vectorizer once at startup
@st.cache_resource
def load_artifacts():
    rf_model = load_model("model.pkl")         # your trained RF or XGB models/model.pkl
    tfidf_vec = load_vectorizer("tfidf.pkl")   # the fitted TfidfVectorizer models/tfidf.pkl
    return rf_model, tfidf_vec

rf_model, tfidf_vec = load_artifacts()

# 2) Page title + description
st.title("  🔍 QUORA  DUPLICATE  QUESTION  DETECTOR") 
st.write(
    """
    Enter two questions below and click **Predict** to see if they are likely duplicates.
    This model was trained on a sample of Quora’s duplicated‐questions dataset, 
    using a combination of handcrafted features (token‐overlap, fuzzy scores, etc.) and TF-IDF vectors.
    """
)

# 3) User inputs
q1 = st.text_area("Question 1", height=100, placeholder="Type the first question here...")
q2 = st.text_area("Question 2", height=100, placeholder="Type the second question here...")

# 4) When “Predict” is clicked:
if st.button("Predict"):
    if not q1.strip() or not q2.strip():
        st.error("Please enter both Question 1 and Question 2.")
    else:
        # Build features array (1 × D) using utils.query_point_creator
        features = query_point_creator(q1, q2, tfidf_vec)  # shape = (1, 22 + 2*TFIDF_DIM)

        if features.ndim == 1:
            features = features.reshape(1, -1)

        
        # Run model prediction
        pred_label = rf_model.predict(features)[0]         # 0 or 1
        pred_prob  = rf_model.predict_proba(features)[0][1]  # probability of “duplicate”

        # Display results
        if pred_label > 0.7:    
            st.success(f"✅ These questions are predicted to be *duplicates* (p = {pred_prob:.2f}).")
        else:
            st.warning(f"❌ These questions are predicted to be *not duplicates* (p = {pred_prob:.2f}).")
        
        # (Optional) Show feature breakdown, etc.
        with st.expander("Show raw feature vector (length = {})".format(features.shape[1])):
            st.write(features)
#%%

#import sklearn
#print(sklearn.__version__)




# %%
