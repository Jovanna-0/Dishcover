# 🍳 Dishcover: Machine Learning Recipe Recommender

Dishcover is an intelligent, ingredient-based recipe recommendation system designed to help users minimize food waste by finding recipes that maximize the use of their available pantry ingredients.

## 🚀 The Architecture

This project utilizes an **Unsupervised Machine Learning** approach to map and retrieve recipes based on ingredient overlap and statistical relevance, rather than simple keyword matching.

### 1. Feature Extraction (TF-IDF)
The text corpus of ~13,500 recipes was processed using a `TfidfVectorizer`. 
* Applied **Inverse Document Frequency (IDF)** to penalize ubiquitous ingredients (like salt and water) and boost the weight of unique, defining ingredients.
* Tuned hyperparameters (`max_df=0.85`, `max_features=5000`, `ngram_range=(1,2)`) to eliminate noise and capture compound ingredients (e.g., "soy_sauce").

### 2. Machine Learning Model (K-Nearest Neighbors)
The resulting TF-IDF sparse matrix was passed into a **K-Nearest Neighbors (KNN)** model. 
* All recipes are plotted in a high-dimensional vector space.
* User inputs are vectorized and plotted as a new coordinate.
* The model utilizes a **Cosine distance metric** to expand a search radius and return the $K$ mathematically closest recipes.

### 3. The Smart Sorter
To ensure highly practical recommendations, the KNN output is re-ranked using a custom Smart Sort function. It prioritizes the absolute **Match Count** (how many of the user's requested ingredients are physically present in the recipe) and uses the **Cosine Similarity Score** as the secondary tie-breaker.

## 📊 Model Evaluation (Precision@K)

To evaluate the mathematical validity of the ML architecture, the KNN model was tested against a standard Jaccard Similarity Baseline using 10 diverse ground-truth test cases. 

**Final Performance Averages:**
* **Jaccard Baseline:** Precision@3: 30.00% | Precision@5: 34.00%
* **TF-IDF + KNN Model:** Precision@3: 56.67% | Precision@5: 50.00%

The KNN model demonstrated a significant **88.9% improvement** over the baseline in top-3 precision, proving the effectiveness of the weighted vector space approach.

## 📁 Project Structure
```text
Dishcover/
├── data/
│   ├── recipes.csv               # Raw dataset
│   └── cleaned_recipes.csv       # Preprocessed dataset
├── models/
│   ├── tfidf_vectorizer.pkl      # Saved TF-IDF state
│   └── knn_model.pkl             # Saved KNN model
├── notebooks/
│   └── 01_Data_Cleaning_and_EDA.ipynb # Full pipeline & evaluation
├── app.py                        # Streamlit web application (WIP)
└── README.md