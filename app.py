"""
app.py — Dishcover
Recipe Recommendation App using TF-IDF + Cosine Similarity
COMP6577001 Machine Learning — Group 8, BINUS University
"""

import os
import streamlit as st
import pandas as pd
from model import DishcoverModel

# ─────────────────────────────────────────────
# Page config  (must be the FIRST st call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Dishcover",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Warm cream background ── */
.stApp {
    background-color: #FAF7F2;
}

/* ── Hero title ── */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 700;
    color: #2C1A0E;
    line-height: 1.15;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1.05rem;
    color: #7A5C44;
    margin-bottom: 1.5rem;
}

/* ── Section headers ── */
h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #2C1A0E !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #2C1A0E;
}
section[data-testid="stSidebar"] * {
    color: #F5EFE6 !important;
}
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color: #E8D9C8 !important;
    font-weight: 500;
}

/* ── Recipe cards ── */
.recipe-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    border-left: 5px solid #C9541A;
    box-shadow: 0 2px 12px rgba(44,26,14,0.07);
}
.recipe-card h3 {
    font-family: 'Playfair Display', serif;
    color: #2C1A0E !important;
    font-size: 1.25rem;
    margin-bottom: 0.3rem;
}
.recipe-card .score-badge {
    display: inline-block;
    background: #C9541A;
    color: #fff;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 30px;
    margin-bottom: 0.7rem;
}
.recipe-card .match-badge {
    display: inline-block;
    background: #EEF7EE;
    color: #2E6B30;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 2px 10px;
    border-radius: 30px;
    margin-left: 6px;
    margin-bottom: 0.7rem;
}
.recipe-card .ingredient-list {
    font-size: 0.88rem;
    color: #5C3D2E;
    margin-bottom: 0.6rem;
}
.recipe-card .instructions {
    font-size: 0.9rem;
    color: #4A3728;
    line-height: 1.65;
    border-top: 1px solid #EDE5DB;
    padding-top: 0.7rem;
    margin-top: 0.4rem;
}

/* ── Metrics ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0 1.5rem;
}
.metric-box {
    background: #fff;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    flex: 1;
    box-shadow: 0 1px 8px rgba(44,26,14,0.06);
    text-align: center;
}
.metric-box .value {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: #C9541A;
    font-weight: 700;
}
.metric-box .label {
    font-size: 0.78rem;
    color: #7A5C44;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ── Buttons ── */
.stButton > button {
    background-color: #C9541A !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.55rem 2.2rem !important;
    transition: background 0.2s ease !important;
}
.stButton > button:hover {
    background-color: #A8431477 !important;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #9B7B63;
}
.empty-state .icon {
    font-size: 4rem;
    margin-bottom: 0.8rem;
}

/* ── Tag pills ── */
.tag {
    display: inline-block;
    background: #F5EDE3;
    color: #7A4E2D;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8rem;
    margin: 2px 3px 2px 0;
}

/* ── About section ── */
.about-box {
    background: #2C1A0E;
    color: #F5EFE6;
    border-radius: 14px;
    padding: 1.5rem;
    margin-top: 2rem;
    font-size: 0.9rem;
    line-height: 1.7;
}
.about-box b { color: #F5C18A; }

/* ── How-it-works ── */
.step-row {
    display: flex;
    gap: 0.8rem;
    align-items: flex-start;
    margin-bottom: 0.7rem;
}
.step-num {
    background: #C9541A;
    color: #fff;
    border-radius: 50%;
    width: 26px;
    height: 26px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.78rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 2px;
}
.step-text { font-size: 0.87rem; color: #F5EFE6; }

/* ── Precision bar ── */
.prec-bar-bg {
    background: #EDE5DB;
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
    margin-top: 4px;
}
.prec-bar-fill {
    background: #C9541A;
    height: 10px;
    border-radius: 6px;
    transition: width 0.4s ease;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Model loading (cached so it only runs once)
# ─────────────────────────────────────────────

MODEL_PATH = "dishcover_model.pkl"

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return DishcoverModel.load(MODEL_PATH)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

def render_sidebar(model: DishcoverModel | None):
    with st.sidebar:
        st.markdown("## 🍽️ Dishcover")
        st.markdown("---")

        # ── Ingredient selector ──
        st.markdown("### Select your ingredients")
        st.caption("Pick what you have at home.")

        selected = []
        if model is not None:
            selected = st.multiselect(
                label="Ingredients",
                options=model.all_ingredients,
                default=[],
                label_visibility="collapsed",
                placeholder="Type to search ingredients…",
            )

            # Custom ingredient input
            custom_raw = st.text_input(
                "Add unlisted ingredient",
                placeholder="e.g. dragon fruit",
                help="Type any ingredient not in the list above.",
            )
            custom_list = [
                i.strip().lower()
                for i in custom_raw.split(",")
                if i.strip()
            ]
            selected = list(dict.fromkeys(selected + custom_list))  # deduplicate

            st.markdown("---")

            # ── Settings ──
            st.markdown("### Settings")
            top_k = st.slider("Number of recipes to show", 3, 15, 5, 1)
            min_score = st.slider(
                "Minimum similarity score",
                0.0, 0.5, 0.01, 0.01,
                help="0 = show everything, 0.1 = only decent matches",
            )
        else:
            top_k = 5
            min_score = 0.01

        return selected, top_k, min_score


# ─────────────────────────────────────────────
# Recipe card renderer
# ─────────────────────────────────────────────

def render_recipe_card(row: pd.Series, rank: int, user_ingredients: list[str]):
    score_pct = int(row["similarity_score"] * 100)
    matched = row["matched_count"]

    # Highlight user ingredients in the ingredient text
    ingredients_html = row["ingredients"]
    for ing in user_ingredients:
        if ing.lower() in ingredients_html.lower():
            ingredients_html = ingredients_html.replace(
                ing, f'<span style="background:#FDE8D5;padding:1px 3px;border-radius:3px;font-weight:600">{ing}</span>'
            )

    # Truncate very long instructions
    instructions = str(row["instructions"])
    show_full = len(instructions) <= 500
    short_instructions = instructions[:500] + ("…" if len(instructions) > 500 else "")

    card_html = f"""
    <div class="recipe-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start">
            <h3>#{rank} &nbsp; {row['title']}</h3>
            <span style="font-size:0.85rem;color:#9B7B63;margin-top:4px">Match</span>
        </div>
        <span class="score-badge">Similarity: {score_pct}%</span>
        <span class="match-badge">✓ {matched} of your ingredients used</span>
        <div class="ingredient-list">
            <b style="color:#2C1A0E">Ingredients:</b><br>
            {ingredients_html}
        </div>
        <div class="instructions">
            <b style="color:#2C1A0E">Instructions:</b><br>
            {short_instructions if not show_full else instructions}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    model = load_model()
    selected_ingredients, top_k, min_score = render_sidebar(model)

    # ── Hero ──
    st.markdown(
        '<div class="hero-title">Dishcover 🍽️</div>'
        '<div class="hero-sub">Find recipes based on what\'s already in your kitchen — reduce waste, cook smarter.</div>',
        unsafe_allow_html=True,
    )

    # ── Model not ready ──
    if model is None:
        st.warning(
            "**Model not found.** Please build it first by running:\n\n"
            "```bash\npython model.py data/recipes.csv\n```\n\n"
            "Then restart the app with:\n\n"
            "```bash\nstreamlit run app.py\n```",
        )
        return

    # ── No ingredients selected ──
    if not selected_ingredients:
        st.markdown(
            '<div class="empty-state">'
            '<div class="icon">🥕</div>'
            '<p><b>Select at least one ingredient</b> from the sidebar to get started.</p>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Show dataset stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Recipes in database", f"{len(model.df):,}")
        with col2:
            st.metric("Unique ingredients", f"{len(model.all_ingredients):,}")
        with col3:
            st.metric("Model", "TF-IDF + Cosine Similarity")
        return

    # ── Run recommendation ──
    with st.spinner("Finding recipes…"):
        try:
            results_df, elapsed_ms = model.recommend(
                user_ingredients=selected_ingredients,
                top_k=top_k,
            )
        except Exception as e:
            st.error(f"Error during recommendation: {e}")
            return

    # Filter by minimum score
    results_df = results_df[results_df["similarity_score"] >= min_score]

    if results_df.empty:
        st.info("No recipes matched your ingredients above the similarity threshold. Try lowering the threshold or adding more ingredients.")
        return

    # ── Precision@K metric ──
    precision = DishcoverModel.precision_at_k(
        results_df,
        selected_ingredients,
        k=top_k,
        relevance_threshold=0.05,
    )

    # ── Metrics row ──
    st.markdown("### Results")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Recipes found", len(results_df))
    with col2:
        st.metric("Query time", f"{elapsed_ms:.1f} ms")
    with col3:
        st.metric(f"Precision@{top_k}", f"{precision:.0%}")
    with col4:
        best_score = results_df["similarity_score"].max()
        st.metric("Best match", f"{best_score:.0%}")

    # Precision bar
    st.markdown(
        f'<div style="font-size:0.8rem;color:#7A5C44;margin-bottom:0.3rem">Showing {len(results_df)} recipes</div>'
        f'<div class="prec-bar-bg"><div class="prec-bar-fill" style="width:{precision*100:.0f}%"></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Your ingredients ──
    tags_html = "".join(
        f'<span class="tag">• {ing}</span>' for ing in selected_ingredients
    )
    st.markdown(
        f'<div style="margin-bottom:1rem"><b style="color:#2C1A0E">Your ingredients:</b><br>{tags_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Recipe cards ──
    for rank, (_, row) in enumerate(results_df.iterrows(), start=1):
        render_recipe_card(row, rank, selected_ingredients)




if __name__ == "__main__":
    main()