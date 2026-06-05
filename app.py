import streamlit as st
import pandas as pd
import joblib
import re
import ast
from collections import Counter
import json
import os

# --- 1. Page Configuration & CSS ---
st.set_page_config(page_title="Dishcover", page_icon="🍳", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #FAF7F2; }

/* ── Typography ── */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 700;
    color: #2C1A0E;
    line-height: 1.15;
    margin-bottom: 0.2rem;
}
.hero-sub { font-size: 1.05rem; color: #7A5C44; margin-bottom: 1.5rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background-color: #2C1A0E; }
section[data-testid="stSidebar"] * { color: #F5EFE6 !important; }

/* ── Recipe Cards ── */
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
    display: inline-block;
}
.recipe-card .match-badge {
    display: inline-block; background: #EEF7EE; color: #2E6B30;
    font-size: 0.75rem; font-weight: 500; padding: 2px 10px;
    border-radius: 30px; margin-bottom: 0.5rem; margin-top: 0.3rem;
}
.ingredient-list { font-size: 0.88rem; color: #5C3D2E; margin-bottom: 0.6rem; margin-top: 1rem;}
.instructions {
    font-size: 0.9rem; color: #4A3728; line-height: 1.65;
    border-top: 1px solid #EDE5DB; padding-top: 0.7rem; margin-top: 0.4rem;
}

/* ── Accordion Styles ── */
details summary {
    cursor: pointer;
    list-style: none; /* Removes the default arrow in some browsers */
    outline: none;
}
details summary::-webkit-details-marker {
    display: none; /* Removes the default arrow in Safari/Chrome */
}
details summary:hover h3 {
    color: #C9541A !important; /* Changes title color on hover to indicate clickability */
    transition: color 0.2s ease;
}

/* ── Empty State & Buttons ── */
.empty-state { text-align: center; padding: 3rem 1rem; color: #9B7B63; }
.empty-state .icon { font-size: 4rem; margin-bottom: 0.8rem; }
.stButton > button {
    background-color: #C9541A !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; width: 100%; padding: 0.6rem !important;
}
/* Bookmark & action icon buttons in narrow columns */
div[data-testid="column"]:last-child .stButton > button {
    background: transparent !important;
    border: 1px solid #E8DDD3 !important;
    box-shadow: none !important;
    font-size: 1.1rem !important;
    width: 2.4rem !important;
    height: 2.4rem !important;
    padding: 0 !important;
    min-width: unset !important;
    border-radius: 8px !important;
    margin-top: 0.6rem;
    color: #2C1A0E !important;
}
div[data-testid="column"]:last-child .stButton > button:hover {
    background: #F5EFE6 !important;
    border-color: #C9541A !important;
}
</style>
""", unsafe_allow_html=True)

# --- 2. Load KNN Models ---
@st.cache_resource(show_spinner=False)
def load_models():
    
    vectorizer = joblib.load('data/tfidf_vectorizer.pkl') 
    knn_model = joblib.load('data/knn_model.pkl') 
    df = pd.read_csv('data/cleaned_recipes.csv') 
    return vectorizer, knn_model, df

try:
    vectorizer, knn_model, df = load_models()
except Exception as e:
    st.error(f"Error loading files. Details: {e}")
    st.stop()

SAVE_FILE = "saved_recipes.json"

def load_saved_recipes():
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_saved_recipes(data):
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# --- 3. Early Session State Init (must be before sidebar reads it) ---
if 'saved_recipes' not in st.session_state:
    st.session_state.saved_recipes = load_saved_recipes()
if 'submitted_ingredients' not in st.session_state:
    st.session_state.submitted_ingredients = []

# --- Preprocessing ---
all_ingredients_text = " ".join(df['Cleaned_Ingredients'].dropna().astype(str).tolist())
word_counts = Counter(all_ingredients_text.split())
unique_ingredients = sorted([word for word, count in word_counts.items() if count >= 5 and len(word) > 1])

# --- 4. Sidebar UI (Settings) ---
with st.sidebar:

    st.markdown("## 🍽️ Dishcover")

    page = st.radio(
        "Navigation",
        [
            "Discover Recipes",
            "Saved Recipes"
        ]
    )

    st.markdown("---")
    st.markdown("## ⚙️ Settings")
    top_k = st.slider("Number of recipes to show", 3, 15, 5, 1)
    st.markdown("---")
    st.markdown(f"❤️ Saved Recipes: {len(st.session_state.saved_recipes)}")

# --- 5. Main Page UI ---
if page == "Discover Recipes":
    st.markdown(
        '<div class="hero-title">Dishcover 🍽️</div>'
        '<div class="hero-sub">Find recipes based on what\'s already in your kitchen — reduce waste, cook smarter.</div>',
        unsafe_allow_html=True
    )

    # Centered Search Box
    selected_ingredients = st.multiselect(
        "What's in your kitchen?", 
        options=unique_ingredients, 
        placeholder="Type to search ingredients...",
        format_func=lambda x: x.replace('_', ' ').title() 
    )

    # The Button
    search_button = st.button("Find Recipes 🔍")

    # If the button is clicked, officially lock in the ingredients to memory!
    if search_button:
        st.session_state.submitted_ingredients = selected_ingredients.copy()

    st.markdown("---")

    # --- 6. ML Logic & Rendering ---

    # Check the memory instead of checking the live search box
    if not st.session_state.submitted_ingredients:
        if not selected_ingredients:
            st.markdown(
                '<div class="empty-state">'
                '<div class="icon">🥕</div>'
                '<p style="color: #9B7B63; font-size: 1.05rem;"><b>Select at least one ingredient</b> from the search bar to get started.</p>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="empty-state">'
                '<div class="icon">🍽️</div>'
                '<p style="color: #9B7B63; font-size: 1.05rem;">Great! Now click <b>Find Recipes</b> to see what you can cook.</p>'
                '</div>',
                unsafe_allow_html=True
            )
            
    else:
        with st.spinner('Powering up the Machine Learning Radar...'):
            
            # Load the locked-in ingredients from memory
            active_ingredients = st.session_state.submitted_ingredients
            
            user_search_string = " ".join(active_ingredients)
            user_vector = vectorizer.transform([user_search_string])
            distances, indices = knn_model.kneighbors(user_vector, n_neighbors=50)
            similarity_scores = 1 - distances.flatten()
            
            top_results = df.iloc[indices.flatten()].copy()
            top_results['Cosine_Score'] = similarity_scores
            
            def count_matches(recipe_ingredients_string):
                count = 0
                for item in active_ingredients:
                    if re.search(rf'\b{item}\b', str(recipe_ingredients_string)):
                        count += 1
                return count
                
            top_results['Match_Count'] = top_results['Cleaned_Ingredients'].apply(count_matches)
            
            final_results = top_results.sort_values(by=['Match_Count', 'Cosine_Score'], ascending=[False, False]).head(top_k)
            
            if final_results.empty:
                st.info("No recipes found. Try adding different ingredients.")
            else:
                st.markdown(f"### Top {len(final_results)} Matches")
                for rank, (_, row) in enumerate(final_results.iterrows(), start=1):
                    recipe_name = str(row["Title"])
                    is_saved = recipe_name in st.session_state.saved_recipes
                    btn_key = f"toggle_save_{recipe_name}_{rank}"
                    
                    matched = row["Match_Count"]
                    total_selected = len(active_ingredients)
                    
                    found_in_recipe = []
                    recipe_text_lower = str(row['Cleaned_Ingredients']).lower()
                    for item in active_ingredients:
                        if re.search(rf'\b{item}\b', recipe_text_lower):
                            found_in_recipe.append(item.replace('_', ' ').title())
                    
                    if found_in_recipe:
                        used_preview_string = ", ".join(found_in_recipe)
                    else:
                        used_preview_string = "None perfectly matched (Similar profile)"

                    try:
                        ing_list = ast.literal_eval(row['Ingredients'])
                        formatted_ings = []
                        for ing in ing_list:
                            highlighted_ing = ing
                            for search_term in sorted(active_ingredients, key=len, reverse=True):
                                display_term = search_term.replace('_', ' ')
                                if search_term == 'sugar':
                                    pattern = re.compile(r'(?<!brown )\b' + re.escape(display_term) + r'\b', re.IGNORECASE)
                                else:
                                    pattern = re.compile(r'\b' + re.escape(display_term) + r'\b', re.IGNORECASE)
                                highlighted_ing = pattern.sub(r'<b style="color: #C9541A;">\g<0></b>', highlighted_ing)
                            formatted_ings.append(f"• {highlighted_ing}")
                        ingredients_html = "<br>".join(formatted_ings)
                    except:
                        highlighted_ing = str(row['Cleaned_Ingredients']).replace('_', ' ').title()
                        for search_term in sorted(active_ingredients, key=len, reverse=True):
                            display_term = search_term.replace('_', ' ')
                            if search_term == 'sugar':
                                pattern = re.compile(r'(?<!brown )\b' + re.escape(display_term) + r'\b', re.IGNORECASE)
                            else:
                                pattern = re.compile(r'\b' + re.escape(display_term) + r'\b', re.IGNORECASE)
                            highlighted_ing = pattern.sub(r'<b style="color: #C9541A;">\g<0></b>', highlighted_ing)
                        ingredients_html = highlighted_ing
                    
                    instructions = str(row.get('Instructions', 'No instructions provided.'))
                    
                    bookmark_color = "#F5C518" if is_saved else "none"
                    bookmark_stroke = "#C9541A"

                    card_html = f"""
<div class="recipe-card" style="position:relative; padding-right: 3rem;">
<details>
<summary>
<div style="display:flex; justify-content:space-between; align-items:center; padding-right: 0.5rem;">
<h3 style="margin-top: 0; margin-bottom: 0;">#{rank} &nbsp; {str(row['Title']).title()} <span style="font-size: 0.9rem; color: #C9541A;">▼</span></h3>
</div>
<div style="margin-top: 0.3rem;">
<span class="match-badge">✓ {matched} out of {total_selected} selected ingredients used</span>
</div>
<div style="font-size: 0.9rem; color: #7A5C44; margin-top: 0.2rem;">
<b style="color: #2C1A0E;">Used from kitchen:</b> {used_preview_string}
</div>
</summary>
<div style="margin-top: 0.5rem;">
<div class="ingredient-list">
<b style="color:#2C1A0E">All Ingredients Required:</b><br>
{ingredients_html}
</div>
<div class="instructions">
<b style="color:#2C1A0E">Instructions:</b><br>
{instructions}
</div>
</div>
</details>
</div>
"""
                    card_col, bm_col = st.columns([20, 1])
                    with card_col:
                        st.markdown(card_html, unsafe_allow_html=True)
                    with bm_col:
                        bm_label = "❤️" if is_saved else "🤍"
                        if st.button(bm_label, key=btn_key):
                            if is_saved:
                                del st.session_state.saved_recipes[recipe_name]
                            else:
                                st.session_state.saved_recipes[recipe_name] = {
                                    "Title": recipe_name,
                                    "Ingredients": str(row.get("Ingredients", "")),
                                    "Instructions": str(row.get("Instructions", "")),
                                    "matched_ingredients": found_in_recipe,
                                    "matched_count": matched,
                                    "total_selected": total_selected,
                                }
                                save_saved_recipes(st.session_state.saved_recipes)
                            save_saved_recipes(st.session_state.saved_recipes)
                            st.rerun()

elif page == "Saved Recipes":

    st.markdown(
        """
        <div class="hero-title">Saved Recipes ❤️</div>
        <div class="hero-sub">Click on a recipe to see the full ingredients and instructions.</div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.saved_recipes:
        st.markdown(
            """
            <div class="empty-state">
                <div class="icon">❤️</div>
                <p>No saved recipes yet. Bookmark a recipe from <b>Discover Recipes</b> to see it here.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(f"**{len(st.session_state.saved_recipes)} saved recipe{'s' if len(st.session_state.saved_recipes) != 1 else ''}**")
        st.markdown("")

        for i, (recipe_name, recipe) in enumerate(st.session_state.saved_recipes.items(), start=1):

            # Parse ingredients for count and preview
            try:
                ing_list = ast.literal_eval(recipe['Ingredients'])
                total_ing_count = len(ing_list)
                ing_preview = ", ".join(ing_list[:5])
                if total_ing_count > 5:
                    ing_preview += f", <i>+{total_ing_count - 5} more</i>"
            except:
                ing_list = []
                total_ing_count = "?"
                ing_preview = recipe['Ingredients'][:120] + "..." if len(recipe['Ingredients']) > 120 else recipe['Ingredients']

            # Matched ingredients info (may not exist for older saved recipes)
            matched_ings = recipe.get("matched_ingredients", [])
            matched_count = recipe.get("matched_count", None)
            total_selected = recipe.get("total_selected", None)

            if matched_ings:
                matched_html = f"""
<div style="font-size:0.85rem; color:#7A5C44; margin-top:0.3rem;">
  <b style="color:#2C1A0E;">Your ingredients used:</b>
  <span style="color:#C9541A; font-weight:500;"> {matched_count} of {total_selected} matched</span>
  &nbsp;·&nbsp; {", ".join(matched_ings)}
</div>"""
            else:
                matched_html = ""

            # Format full ingredients for expanded view
            try:
                full_ing_html = "<br>".join([f"• {ing}" for ing in ing_list])
            except:
                full_ing_html = recipe['Ingredients']

            instructions = recipe.get('Instructions', 'No instructions available.')

            card_html = f"""
<div class="recipe-card">
<details>
<summary>
<div style="display:flex; justify-content:space-between; align-items:center;">
  <h3 style="margin-top:0; margin-bottom:0;">{recipe_name.title()} <span style="font-size:0.9rem; color:#C9541A;">▼</span></h3>
  <span style="font-size:0.8rem; color:#9B7B63; font-weight:500; white-space:nowrap; margin-left:0.8rem;">{total_ing_count} ingredients</span>
</div>
<div style="font-size:0.85rem; color:#7A5C44; margin-top:0.4rem;">

</div>
{matched_html}
</summary>
<div style="margin-top:0.8rem;">
  <div class="ingredient-list">
    <b style="color:#2C1A0E;">All Ingredients Required:</b><br>
    {full_ing_html}
  </div>
  <div class="instructions">
    <b style="color:#2C1A0E;">Instructions:</b><br>
    {instructions}
  </div>
</div>
</details>
</div>
"""
            card_col, bm_col = st.columns([20, 1])
            with card_col:
                st.markdown(card_html, unsafe_allow_html=True)
            with bm_col:
                if st.button("🗑️", key=f"remove_{recipe_name}_{i}"):
                    del st.session_state.saved_recipes[recipe_name]
                    save_saved_recipes(st.session_state.saved_recipes)
                    st.rerun()