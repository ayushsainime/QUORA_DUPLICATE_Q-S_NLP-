# ❓ Duplicate Question Detection using ML

A Machine Learning-powered web app to detect if two given questions are duplicates, inspired by Quora's question pair challenge.


Our model uses a comprehensive set of features including basic text statistics, token overlap features, length-based metrics, fuzzy string matching scores, and TF-IDF vector representations to accurately capture the similarity between question pairs. . 


---

## 🚀 Demo

Try it live: *   https://gscvlubtbxkbetufv4frfy.streamlit.app/ *


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

## Features Used

The model leverages a comprehensive set of features engineered to capture various aspects of similarity between question pairs. These features include:

### 1. Basic Text Features
- Character length of each question
- Word count of each question
- Number of common words between the two questions
- Total number of words in both questions combined
- Word share ratio (common words divided by total words)

### 2. Token-based Features
- Ratio of common non-stopwords (minimum and maximum denominators)
- Ratio of common stopwords (minimum and maximum denominators)
- Ratio of common tokens overall (minimum and maximum denominators)
- Binary indicators if the first words of both questions are the same
- Binary indicators if the last words of both questions are the same

### 3. Length-based Features
- Absolute difference in the number of tokens between the two questions
- Average length of both questions (in tokens)
- Longest common substring ratio relative to the smaller question length

### 4. Fuzzy Matching Features
- Fuzzy ratio (QRatio) between questions
- Partial fuzzy ratio
- Token sort ratio
- Token set ratio

### 5. TF-IDF Features
- TF-IDF vector representations of questions to capture weighted word importance based on corpus statistics

---

These features help the model effectively identify subtle semantic and lexical similarities, contributing to its overall accuracy in detecting duplicate questions.


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


