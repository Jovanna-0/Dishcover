"""
model.py — Dishcover Recommendation Engine
TF-IDF + Cosine Similarity pipeline for recipe recommendation.
"""

import pickle
import re
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────
# Text cleaning helpers
# ─────────────────────────────────────────────

def clean_ingredient_text(raw: str) -> str:
    """
    Normalize an ingredient string:
    - lowercase
    - remove quantities and units  (e.g. "2 cups", "½ tsp")
    - remove punctuation / special chars
    - collapse whitespace
    """
    if not isinstance(raw, str):
        return ""
    text = raw.lower()
    # Remove fraction characters and numbers with units
    text = re.sub(r"[\u00bc-\u00be\u2150-\u215e]", " ", text)   # unicode fractions
    text = re.sub(r"\d+[\./]?\d*", " ", text)                     # numeric quantities
    # Remove common unit words
    units = (r"\b(cup|cups|tbsp|tsp|tablespoon|tablespoons|teaspoon|teaspoons|"
             r"oz|ounce|ounces|lb|pound|pounds|gram|grams|g|kg|ml|liter|liters|"
             r"pinch|dash|handful|slice|slices|clove|cloves|can|cans|package|"
             r"pkg|bunch|head|large|medium|small|fresh|dried|chopped|minced|"
             r"diced|sliced|grated|shredded|cooked|raw|optional|to taste)\b")
    text = re.sub(units, " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
# Main model class
# ─────────────────────────────────────────────

class DishcoverModel:
    """
    Offline preprocessing + online inference for recipe recommendation.

    Usage
    -----
    # First time (or when dataset changes):
    model = DishcoverModel()
    model.fit("path/to/recipes.csv")
    model.save("dishcover_model.pkl")

    # Every subsequent run:
    model = DishcoverModel.load("dishcover_model.pkl")
    results = model.recommend(["chicken", "garlic", "lemon"], top_k=5)
    """

    def __init__(self):
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None          # shape: (n_recipes, n_features)
        self.df: pd.DataFrame | None = None
        self.all_ingredients: list[str] = []

    # ── Fitting ────────────────────────────────

    def fit(self, csv_path: str) -> None:
        """Load CSV, clean data, build TF-IDF matrix."""
        print(f"[DishcoverModel] Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)

        # ── Column normalisation ──
        # The Kaggle dataset uses 'Title', 'Ingredients', 'Instructions'
        # Accept minor variations in column naming.
        col_map = {}
        for col in df.columns:
            lc = col.lower().strip()
            if lc in ("title", "recipe_title", "name", "recipe name"):
                col_map[col] = "title"
            elif lc in ("ingredients", "ingredient", "ingredient_list"):
                col_map[col] = "ingredients"
            elif lc in ("instructions", "instruction", "directions", "steps"):
                col_map[col] = "instructions"
        df.rename(columns=col_map, inplace=True)

        required = {"title", "ingredients"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Dataset is missing required columns: {missing}. "
                f"Found columns: {list(df.columns)}"
            )

        # Drop rows without ingredients or title
        df.dropna(subset=["title", "ingredients"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Clean ingredient text
        df["clean_ingredients"] = df["ingredients"].apply(clean_ingredient_text)

        # Drop rows that became empty after cleaning
        df = df[df["clean_ingredients"].str.strip() != ""].reset_index(drop=True)

        print(f"[DishcoverModel] Recipes after cleaning: {len(df)}")

        # ── TF-IDF vectorisation ──
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams (e.g. "olive oil")
            min_df=2,             # ignore ingredients appearing in < 2 recipes
            max_df=0.95,          # ignore very common ingredients (appear in 95 %+ recipes)
            sublinear_tf=True,    # apply log normalization on TF
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(df["clean_ingredients"])
        print(f"[DishcoverModel] TF-IDF matrix shape: {self.tfidf_matrix.shape}")

        self.df = df

        # Build sorted list of all recognised ingredients for the UI dropdown
        feature_names = self.vectorizer.get_feature_names_out()
        # Keep only single-word features for the dropdown (cleaner UX)
        self.all_ingredients = sorted(
            [f for f in feature_names if " " not in f and len(f) > 2]
        )
        print(f"[DishcoverModel] Unique ingredient tokens: {len(self.all_ingredients)}")

    # ── Persistence ────────────────────────────

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[DishcoverModel] Model saved to: {path}")

    @staticmethod
    def load(path: str) -> "DishcoverModel":
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"[DishcoverModel] Model loaded from: {path}")
        return model

    # ── Inference ──────────────────────────────

    def recommend(
        self,
        user_ingredients: list[str],
        top_k: int = 5,
    ) -> tuple[pd.DataFrame, float]:
        """
        Return top-k recipe recommendations for the given ingredients.

        Parameters
        ----------
        user_ingredients : list of ingredient name strings
        top_k            : number of recipes to return

        Returns
        -------
        results_df : DataFrame with columns [title, ingredients, instructions,
                                             similarity_score, matched_count,
                                             total_ingredients]
        elapsed_ms : query execution time in milliseconds
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        if not user_ingredients:
            raise ValueError("Please provide at least one ingredient.")

        # Clean user input the same way as training data
        query_text = clean_ingredient_text(" ".join(user_ingredients))

        t0 = time.perf_counter()
        query_vec = self.vectorizer.transform([query_text])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Get top-k indices (sorted descending)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            score = float(scores[idx])
            if score == 0:
                continue

            # Count how many user ingredients actually appear in this recipe
            recipe_text = row["clean_ingredients"].lower()
            matched = sum(
                1 for ing in user_ingredients
                if clean_ingredient_text(ing) in recipe_text
            )
            # Estimate total unique ingredients in this recipe
            total_ings = len(set(row["clean_ingredients"].split()))

            results.append({
                "title": row["title"],
                "ingredients": row.get("ingredients", ""),
                "instructions": row.get("instructions", "No instructions available."),
                "similarity_score": round(score, 4),
                "matched_count": matched,
                "total_ingredients": total_ings,
            })

        results_df = pd.DataFrame(results)
        return results_df, round(elapsed_ms, 2)

    # ── Evaluation (Precision@K) ────────────────

    @staticmethod
    def precision_at_k(
        results_df: pd.DataFrame,
        user_ingredients: list[str],
        k: int = 5,
        relevance_threshold: float = 0.05,
    ) -> float:
        """
        Compute Precision@K: proportion of top-K recipes with
        similarity score >= relevance_threshold.
        """
        if results_df.empty:
            return 0.0
        top_k_results = results_df.head(k)
        relevant = (top_k_results["similarity_score"] >= relevance_threshold).sum()
        return round(relevant / min(k, len(top_k_results)), 4)


# ─────────────────────────────────────────────
# CLI helper — run once to build the model file
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python model.py <path_to_csv>")
        print("Example: python model.py data/recipes.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    model = DishcoverModel()
    model.fit(csv_path)
    model.save("dishcover_model.pkl")
    print("\nDone! Run the app with:  streamlit run app.py")
