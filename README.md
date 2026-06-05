# 🍽️ Dishcover — Recipe Recommendation System
**Group 8 | COMP6577001 Machine Learning | BINUS University 2025/2026**

> Recommending recipes based on available ingredients using TF-IDF and Cosine Similarity.

---

## Project Structure

```
dishcover/
├── app.py            ← Streamlit frontend + UI
├── model.py          ← TF-IDF + Cosine Similarity ML pipeline
├── requirements.txt  ← Python dependencies
├── README.md         ← This file
└── data/
    └── recipes.csv   ← Put your Kaggle dataset here
```

---

## Setup Instructions (Step by Step)

### Step 1 — Install Python dependencies

Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
```

### Step 2 — Prepare your dataset

1. Download the dataset from Kaggle:  
   https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images

2. Create a `data/` folder inside `dishcover/`:
   ```bash
   mkdir data
   ```

3. Put the CSV file (e.g. `epicurious-recipes-with-rating-and-nutrition.csv` or similar) inside `data/` and rename it `recipes.csv`:
   ```
   data/recipes.csv
   ```

### Step 3 — Build the model (run ONCE)

This step preprocesses the dataset and saves the TF-IDF model to disk.  
You only need to do this once (or when you change the dataset).

```bash
python model.py data/recipes.csv
```

You should see output like:
```
[DishcoverModel] Loading dataset from: data/recipes.csv
[DishcoverModel] Recipes after cleaning: 13501
[DishcoverModel] TF-IDF matrix shape: (13501, 4823)
[DishcoverModel] Model saved to: dishcover_model.pkl
```

### Step 4 — Launch the app

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

---

## How to Use the App

1. Use the **left sidebar** to select ingredients you have at home
2. You can also **type custom ingredients** in the text box
3. Adjust **number of results** and **minimum similarity score** with the sliders
4. Results are shown as **recipe cards**, ranked by similarity score

---

## How the Model Works

### TF-IDF Vectorization
- Each recipe's ingredient list is converted into a numerical vector
- **TF (Term Frequency):** How often an ingredient appears in a recipe
- **IDF (Inverse Document Frequency):** Reduces weight of common ingredients (salt, water) and increases weight of rare ones (saffron, tahini)
- User input is vectorized the same way

### Cosine Similarity
- Measures the angle between the user's ingredient vector and each recipe vector
- Score of **1.0** = perfect match, **0.0** = no match
- Top-K recipes with highest scores are returned

### Evaluation: Precision@K
- Measures how many of the top-K results are actually relevant
- A recipe is "relevant" if its similarity score ≥ 0.05
- Displayed in the app after each search

### Baseline Comparison
- Baseline: Jaccard Similarity (simple ingredient overlap counting)
- Our TF-IDF approach outperforms the baseline by weighing rare ingredients more heavily

---

## Git Setup

```bash
git init
git add .
git commit -m "Initial commit: Dishcover recipe recommendation system"
git remote add origin https://github.com/YOUR_USERNAME/dishcover.git
git push -u origin main
```

> **Note:** Add `dishcover_model.pkl` to `.gitignore` if the file is large.

```bash
echo "dishcover_model.pkl" >> .gitignore
echo "__pycache__/" >> .gitignore
echo ".streamlit/" >> .gitignore
```

---

## Team

| Name | NIM |
|---|---|
| Jovanna | 2802499030 |
| Khalisha Ramadhany | 2802560990 |
| Thania Calista | 2802451453 |

---

## References

- Central Insight. (2025). *The shocking impact of food waste in Indonesia.*
- Goel, S., Desai, A., & Tanvi. (2021). *Food Ingredients and Recipes Dataset with Images.* Kaggle.
- Scikit-learn: Machine Learning in Python. https://scikit-learn.org
