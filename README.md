# ❓ Duplicate Question Detection using ML

A Machine Learning-powered web app to detect if two given questions are duplicates, inspired by Quora's question pair challenge.

---

## 🚀 Demo

Try it live: *   *


![Screenshot 2025-06-07 001746](https://github.com/user-attachments/assets/1cc882aa-ff00-4c99-bc2d-5c498a8c2159)


![Screenshot 2025-06-07 001700](https://github.com/user-attachments/assets/1910bc41-1319-4533-af07-aff2024b7c34)


---

## 🧠 Problem Statement

Given two questions (e.g., from a forum like Quora), determine whether they are semantically similar or duplicates of each other.

This helps reduce redundancy in platforms by merging identical questions and improving user experience.

---

## 📊 Accuracy

**Model Accuracy:** `79.98%`

Evaluated on a held-out test set using a Random Forest Classifier trained on rich handcrafted features and TF-IDF representations.

---

## 🧰 Tech Stack Used

- **Frontend / Interface**: `Streamlit`
- **Text Preprocessing**: `spaCy`, `BeautifulSoup`, `contractions`, `NLTK`
- **Feature Engineering**:
  - Token-based, length-based, fuzzy features
  - TF-IDF Vectorization
- **Machine Learning**: `RandomForestClassifier` / `XGBoost`
- **Vectorization**: `TfidfVectorizer (scikit-learn)`
- **Others**: `fuzzywuzzy`, `distance`, `pickle`, `numpy`, `scipy`

---

## 📂 Dataset

- **Name**: Quora Question Pairs Dataset  
- **Download Link**: [Quora Duplicate Questions Dataset](https://www.kaggle.com/c/quora-question-pairs/data)


## 🏗️ How It Works

1. **Input**: Two user-provided questions
2. **Preprocessing**:
   - HTML stripping, lowercasing, contraction expansion
   - Lemmatization using spaCy
   - TF-IDF vectorization
3. **Feature Engineering**:
   - Token overlap, fuzzy ratios, length difference, etc.
4. **Prediction**:
   - Model predicts 1 if the questions are duplicates, 0 otherwise

---


