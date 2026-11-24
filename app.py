import streamlit as st
import pandas as pd
import joblib
import re
import ast
from collections import Counter
import json
import os
import inflect
from streamlit_local_storage import LocalStorage
import uuid
import datetime
import math
import base64

# Initialize the bridge to the user's browser
localS = LocalStorage()
p = inflect.engine()

# --- 1. Page Configuration & CSS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "assets", "DishCover Logo.png")
LOGO_PATH2 = os.path.join(BASE_DIR, "assets", "DishCover Logo2.png")

with open(LOGO_PATH, "rb") as f:
    logo_b64 = base64.b64encode(f.read()).decode()

with open(LOGO_PATH2, "rb") as f:
    logo2 = base64.b64encode(f.read()).decode()

st.set_page_config(
    page_title="Dishcover", 
    page_icon=LOGO_PATH, 
    layout="wide",
    initial_sidebar_state="auto",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@400;500&display=swap');
label[data-testid="stWidgetLabel"] p { color: #2C1A0E !important; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #FAF7F2; }
            
/* Reduce excessive bottom whitespace */
.block-container,
[data-testid="stMainBlockContainer"] {
    padding-bottom: 2rem !important;
}

/* Makes st.logo() rounded */
[data-testid="stLogo"] {
    border-radius: 50%;      /* Use 12px instead for rounded corners */
    object-fit: cover;
}
            
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
section[data-testid="stSidebar"] *:not([data-testid="stIconMaterial"]) { 
    color: #F5EFE6 !important; 
    font-family: 'DM Sans', sans-serif !important;
}
section[data-testid="stSidebar"] [data-testid="stIconMaterial"] {
    color: #F5EFE6 !important;
}
section[data-testid="stSidebar"] .stButton > button > div {
    width: 100%;
    text-align: left !important;
}
            
.sidebar-brand {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    color: #F5EFE6 !important;
    margin-bottom: 1.1rem;
    letter-spacing: 0.01em;
}
.sidebar-section-label {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: rgba(245, 239, 230, 0.55) !important;
    margin-top: 0.2rem;
    margin-bottom: 0.5rem;
}
section[data-testid="stSidebar"] hr {
    margin: 1rem 0 !important;
    border-color: rgba(245, 239, 230, 0.12) !important;
}
section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p,
section[data-testid="stSidebar"] [data-testid="stCheckbox"] p {
    color: #F5EFE6 !important;
    font-size: 0.9rem !important;
}

/* Sidebar Expander (Pantry Staples) */
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    background-color: rgba(0, 0, 0, 0.22) !important;
    border: 1px solid rgba(245, 239, 230, 0.14) !important;
    border-radius: 8px !important;
    margin-top: 0.4rem;
}

/* ── Sidebar Navigation, Daily Picks & Recent Search Buttons ── */
section[data-testid="stSidebar"] .stButton button {
    background-color: transparent !important;
    color: #F5EFE6 !important;
    border: none !important;
    border-radius: 8px !important;
    display: flex !important;
    justify-content: flex-start !important;
    align-items: center !important;
    text-align: left !important;
    padding: 0.55rem 0.85rem !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    transition: background-color 0.2s ease !important;
}
section[data-testid="stSidebar"] .stButton button div,
section[data-testid="stSidebar"] .stButton button p {
    width: 100% !important;
    text-align: left !important;
    justify-content: flex-start !important;
    margin: 0 !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background-color: rgba(0, 0, 0, 0.28) !important;
}
section[data-testid="stSidebar"] .stButton button[kind="primary"],
section[data-testid="stSidebar"] .stButton button[data-testid*="primary"] {
    background-color: rgba(0, 0, 0, 0.42) !important;
    border-left: 3px solid #C9541A !important;
    font-weight: 600 !important;
}

/* Unify multiselect tags with Dishcover burnt orange */
span[data-baseweb="tag"] {
    background-color: #C9541A !important;
    color: #FFFFFF !important;
    border-radius: 6px !important;
}

/* Darken section on hover */
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: rgba(0, 0, 0, 0.28) !important;
}

/* Active section highlight */
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
    background-color: rgba(0, 0, 0, 0.42) !important;
    border-left: 4px solid #C9541A !important;
    font-weight: 600 !important;
}

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
div[data-testid="column"]:last-child .stButton > button,
div[data-testid="stColumn"]:last-child .stButton > button {
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
div[data-testid="column"]:last-child .stButton > button:hover,
div[data-testid="stColumn"]:last-child .stButton > button:hover {
    background: #F5EFE6 !important;
    border-color: #C9541A !important;
}

/* ── Bottom Pagination Arrow Buttons ── */
.st-key-pg_first button,
.st-key-pg_prev button,
.st-key-pg_next button,
.st-key-pg_last button {
    background-color: transparent !important;
    color: #2C1A0E !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0.3rem 0.2rem !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    min-height: 2.2rem !important;
}
.st-key-pg_first button:hover:not(:disabled),
.st-key-pg_prev button:hover:not(:disabled),
.st-key-pg_next button:hover:not(:disabled),
.st-key-pg_last button:hover:not(:disabled) {
    background-color: #EDE5DB !important;
    color: #C9541A !important;
}
.st-key-pg_first button:disabled,
.st-key-pg_prev button:disabled,
.st-key-pg_next button:disabled,
.st-key-pg_last button:disabled {
    opacity: 0.35 !important;
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
    # 1. Ask the user's browser if they have any saved data
    browser_data = localS.getItem("dishcover_recipes")
    
    # 2. If data exists, decode it and load it in!
    if browser_data:
        try:
            return json.loads(browser_data)
        except:
            return {}
    return {}

def save_saved_recipes(data, unique_widget_key):
    # --- NEW: Master Save Logic ---
    if 'trigger_save' not in st.session_state:
        st.session_state.trigger_save = False

    # If a button flipped the flag to True, generate a fresh widget to save the data
    if st.session_state.trigger_save:
        localS.setItem("dishcover_recipes", json.dumps(st.session_state.saved_recipes), key=f"save_{uuid.uuid4()}")
        st.session_state.trigger_save = False

# --- 3. Early Session State Init (must be before sidebar reads it) ---
if 'saved_recipes' not in st.session_state:
    st.session_state.saved_recipes = load_saved_recipes()
if 'submitted_ingredients' not in st.session_state:
    st.session_state.submitted_ingredients = []
if 'page' not in st.session_state:
    st.session_state.page = "Discover Recipes"
if 'recent_searches' not in st.session_state:
    st.session_state.recent_searches = []
if 'per_page' not in st.session_state:
    st.session_state.per_page = 5
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1

# --- Preprocessing ---
all_ingredients_text = " ".join(df['Cleaned_Ingredients'].dropna().astype(str).tolist())
word_counts = Counter(all_ingredients_text.split())
unique_ingredients = sorted([word for word, count in word_counts.items() if count >= 5 and len(word) > 1])

BASIC_STAPLES = [
    "salt", "pepper", "water", "oil", "sugar", "butter", "onion",
    "garlic", "flour", "vinegar", "soy sauce"
]

# -- Top Recipes -- 
@st.dialog("🍽️ Recipe of the Day")
def show_daily_recipe(row):
    recipe_name = str(row["Title"])
    st.markdown(f"<h3 style='font-family: Playfair Display, serif; color: #2C1A0E; margin-top: 0;'>{recipe_name.title()}</h3>", unsafe_allow_html=True)

    # Format ingredients
    try:
        ing_list = ast.literal_eval(row["Ingredients"])
        ing_html = "<br>".join([f"• {ing}" for ing in ing_list])
    except Exception:
        ing_html = str(row.get("Cleaned_Ingredients", "")).replace("_", " ").title()

    st.markdown(f"<div style='font-size: 0.9rem; color: #5C3D2E; margin-bottom: 1rem;'><b>Ingredients:</b><br>{ing_html}</div>", unsafe_allow_html=True)

    # Format numbered instructions
    raw_instructions = str(row.get("Instructions", "No instructions provided."))
    if raw_instructions == "No instructions provided." or raw_instructions.lower() == "nan":
        formatted_instructions = "No instructions provided."
    else:
        steps = [step.strip() for step in raw_instructions.split(". ") if len(step.strip()) > 0]
        formatted_instructions = "<br>".join([f"<b>{i+1}.</b> {step}." if not step.endswith(".") else f"<b>{i+1}.</b> {step}" for i, step in enumerate(steps)])

    st.markdown(f"<div style='font-size: 0.9rem; color: #4A3728; line-height: 1.6; border-top: 1px solid #EDE5DB; padding-top: 0.7rem;'><b>Instructions:</b><br>{formatted_instructions}</div>", unsafe_allow_html=True)

    st.markdown("")
    is_saved = recipe_name in st.session_state.saved_recipes
    if st.button("❤️ Saved" if is_saved else "🤍 Save Recipe", use_container_width=True):
        if is_saved:
            del st.session_state.saved_recipes[recipe_name]
        else:
            st.session_state.saved_recipes[recipe_name] = {
                "Title": recipe_name,
                "Ingredients": str(row.get("Ingredients", "")),
                "Instructions": str(row.get("Instructions", "")),
            }
        st.session_state.trigger_save = True
        st.rerun()


# --- 4. Sidebar UI (Settings) ---
with st.sidebar:

    st.markdown(
        f'<div class="sidebar-brand" style="display:flex; align-items:center; gap:0.65rem;">'
        f'<img src="data:image/png;base64,{logo2}" style="width:48px; height:48px; flex-shrink:0; border-radius:50%; object-fit:cover;">'
        f'<span>Dishcover</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── TOP: Navigation ──
    st.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)

    saved_count = len(st.session_state.saved_recipes)
    nav_items = [
        ("🔍  Discover Recipes", "Discover Recipes"),
        (f"❤️  Saved Recipes ({saved_count})", "Saved Recipes"),
    ]

    for label, page_key in nav_items:
        is_active = (st.session_state.page == page_key)
        if st.button(
            label,
            key=f"nav_{page_key}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.page = page_key
            st.rerun()

    page = st.session_state.page

    st.markdown("---")

    # ── MIDDLE 1: Top Recipes of the Day ──
    st.markdown('<div class="sidebar-section-label">Top Recipes of the Day</div>', unsafe_allow_html=True)

    today_seed = int(datetime.date.today().strftime("%Y%m%d"))
    daily_picks = df.sample(n=min(3, len(df)), random_state=today_seed)

    for idx, (_, pick_row) in enumerate(daily_picks.iterrows()):
        pick_title = str(pick_row["Title"]).title()
        short_title = pick_title[:24] + "..." if len(pick_title) > 27 else pick_title
        if st.button(f"#{idx + 1}  {short_title}", key=f"daily_pick_{idx}", use_container_width=True):
            show_daily_recipe(pick_row)

    st.markdown("---")

    # ── MIDDLE 2: Recent Searches ──
    st.markdown('<div class="sidebar-section-label">Recent Searches</div>', unsafe_allow_html=True)

    if not st.session_state.recent_searches:
        st.caption("Your last 3 searches will appear here.")
    else:
        for idx, combo in enumerate(st.session_state.recent_searches):
            combo_label = ", ".join(combo)
            if len(combo_label) > 26:
                combo_label = combo_label[:23] + "..."
            if st.button(f"↩  {combo_label}", key=f"recent_{idx}", use_container_width=True):
                st.session_state.submitted_ingredients = combo.copy()
                st.session_state.kitchen_multiselect = combo.copy()
                st.session_state.current_page = 1
                st.session_state.page = "Discover Recipes"
                st.rerun()

    st.markdown("---")

    # ── BOTTOM: Settings ──
    st.markdown('<div class="sidebar-section-label">Settings</div>', unsafe_allow_html=True)
    
    strict_mode = st.toggle(
        "Strict Mode", 
        value=False,
        help="Prioritizes 100% ready-to-cook recipes with zero missing ingredients."
    )

    assume_staples = st.toggle(
        "Ignore basic staples", 
        value=True,
        help="Ignores the items selected in Pantry Staples below when counting missing ingredients."
    )

    with st.expander("🧂 Pantry Staples"):
        active_staples = st.multiselect(
            "Customize ignored basics:",
            options=BASIC_STAPLES,
            default=BASIC_STAPLES,
            key="custom_staples"
        )


# --- 5. Main Page UI ---
if page == "Discover Recipes":
    st.markdown(
        f'<div class="hero-title" style="display:flex; align-items:center; gap:0.1rem;">'
        f'<span>Dishcover</span>'
        f'<img src="data:image/png;base64,{logo_b64}" style="width:80px; height:80px; flex-shrink:0; border-radius:50%; object-fit:cover; transform:translateY(6px);">'
        f'</div>'
        f'<div class="hero-sub">Find recipes based on what\'s already in your kitchen — reduce waste, cook smarter.</div>',
        unsafe_allow_html=True
    )

    # --- 1. Define words that describe preparation ---
    fluff_words = [
        ' can ', ' cans ', ' canned ', ' jar ', ' jars ', ' package ', ' packages ', ' of ', 
        ' cut ', ' inch ', ' halved ', ' pitted ', ' thinly ', ' sliced ', 'adjustableblade',
        ' chopped ', ' diced ', ' peeled ', ' fresh ', ' large ', ' small ', 
        ' medium ', ' pieces ', ' whole ', ' halves ', 'more', 'slice', 'stock', 'broth', 'ground', 'grind', 'crumble', 
        'crumbled', 'zest', 'juice', 'squeeze', 'drizzle', 'boneless', 'chuck', 'roast', 'lowsalt', 'serving', 'rotisserie', 'lowsodium',
        ' yolk ', ' yolks ', ' white ', ' whites ', ' beaten ', ' whisked ', 
        ' divided ', ' separated ', 'blend', 'puree', 'mashed', 'grated', 'shredded', 'minced', 'crushed',
        'lightly ', 'finely ', 'coarsely ', 'roughly ', 'thinly ', 'thickly ', 'julienned ', 'ribboned ',
        'cubed ', 'quartered ', 'sliced ', 'diced ', 'chopped ', 'minced ', 'zested ', 'peeled ',
        'seeded ', 'pitted ', 'deveined ', 'trimmed ', 'rinsed ', 'drained ', 'patted dry ',
        'room temperature ', 'cold ', 'frozen ', 'freshly ', 'softened ', 'melted ', 'grated ', 'shredded ', 'sliced thinly ',
        'from the can ', 'from the jar ', 'from the package ', 'canned ', 'jarred ', 'packaged ',
        'from', 'glaze', 'minute', 'plus', 'cooked', 'uncooked', 'raw', 'ripe', 'overripe', 'underripe', 'dried', 'fresh ', 'frozen ', 'cube'
    ]
    # --- 2. Clean your original list ---
    display_ingredients = set()

    for item in unique_ingredients: 
        clean_item = f" {item.lower().replace('_', ' ')} " 
        
        for word in fluff_words:
            clean_item = clean_item.replace(word, ' ')
                
        words = clean_item.split()
        final_words = []
        
        for w in words:
            if len(w) <= 1:
                continue
                
            sing = p.singular_noun(w)
            final_w = sing if sing else w 
            final_words.append(final_w)
            
        unique_words = list(dict.fromkeys(final_words))
        final_item = " ".join(unique_words).title()
        
        if final_item:
            display_ingredients.add(final_item)

    # --- 3. Sort alphabetically ---
    clean_dropdown_options = sorted(list(display_ingredients))

    # --- 4. The actual Streamlit UI component ---
    selected_ingredients = st.multiselect(
        "What's in your kitchen?", 
        options=clean_dropdown_options,
        placeholder="Type to search ingredients...",
        format_func=lambda x: x.replace('_', ' ').title(),
        key="kitchen_multiselect",
        select_all=False
    )

    # The Button
    search_button = st.button("Find Recipes 🔍", use_container_width=True)

    # If the button is clicked, lock in the ingredients and record in Recent Searches
    if search_button:
        st.session_state.submitted_ingredients = selected_ingredients.copy()
        st.session_state.current_page = 1
        if selected_ingredients:
            if selected_ingredients in st.session_state.recent_searches:
                st.session_state.recent_searches.remove(selected_ingredients)
            st.session_state.recent_searches.insert(0, selected_ingredients.copy())
            st.session_state.recent_searches = st.session_state.recent_searches[:3]
            st.rerun()


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

            exclusion_rules = {
                'sugar': ['brown', 'powdered', 'confectioners', 'icing', 'coconut'],
                'onion': ['red', 'green', 'spring', 'pearl'],
                'cheese': ['cream', 'blue', 'cottage', 'goat'],
                'flour': ['almond', 'coconut', 'bread', 'cake'],
                'milk': ['coconut', 'almond', 'soy', 'oat', 'condensed', 'evaporated'],
                'rice': ['wild', 'brown', 'cauliflower'],
                'oil': ['sesame', 'chili', 'coconut', 'truffle']
            }

            # Load the locked-in ingredients from memory
            active_ingredients = st.session_state.submitted_ingredients
            
            # Include both spaced and underscored versions so multi-word ingredients match TF-IDF
            search_tokens = []
            for ing in active_ingredients:
                ing_lower = ing.lower()
                search_tokens.append(ing_lower)
                search_tokens.append(ing_lower.replace(' ', '_'))
            user_search_string = " ".join(search_tokens)
            user_vector = vectorizer.transform([user_search_string])
            distances, indices = knn_model.kneighbors(user_vector, n_neighbors=100)
            similarity_scores = 1 - distances.flatten()
            
            top_results = df.iloc[indices.flatten()].copy()
            top_results['Cosine_Score'] = similarity_scores

            def build_pattern(term):
                term = term.lower().replace('_', ' ')
                if term in exclusion_rules:
                    lookbehinds = "".join([f"(?<!{adj} )" for adj in exclusion_rules[term]])
                    return re.compile(rf'{lookbehinds}\b{re.escape(term)}\b', re.IGNORECASE)
                return re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)

            active_patterns = [(item, build_pattern(item)) for item in active_ingredients]
            staple_patterns = [re.compile(rf'\b{re.escape(s)}\b', re.IGNORECASE) for s in active_staples]

            UNITS_PATTERN = re.compile(
                r'^(?:cups?|tablespoons?|tbsp\.?|teaspoons?|tsp\.?|ounces?|oz\.?|pounds?|lbs?\.?|'
                r'grams?|g\.?|kg\.?|ml\.?|liters?|l\.?|cans?|jars?|packages?|pkgs?\.?|bottles?|'
                r'cloves?|bunches?|pinch(?:es)?|dashes?|slices?|pieces?|sticks?|sprigs?|heads?|'
                r'bulbs?|stalks?|handfuls?|large|medium|small|fresh|dried|frozen|peeled|chopped|'
                r'diced|minced|sliced|grated|shredded|ground|whole|halved|of)\b\s*',
                re.IGNORECASE
            )

            def clean_ingredient_name(raw_ing):
                text = str(raw_ing).strip()
                text = re.sub(r'\([^)]*\)', '', text)
                text = text.split(',')[0]
                text = re.sub(r'^[\d\s/.,–\-¼½¾⅓⅔⅛⅜⅝⅞]+', '', text)
                while True:
                    new_text = UNITS_PATTERN.sub('', text).strip()
                    if new_text == text:
                        break
                    text = new_text
                text = re.sub(r'\s+', ' ', text).strip(' -–—')
                return text.title() if text else str(raw_ing).strip()

            def evaluate_recipe(row):
                try:
                    ing_lines = ast.literal_eval(row['Ingredients'])
                except Exception:
                    ing_lines = [str(row['Cleaned_Ingredients'])]

                matched_user_items = set()
                missing_items = []

                for line in ing_lines:
                    line_str = str(line).strip()
                    line_matched = False

                    # 1. Check if this recipe line uses any of the user's ingredients
                    for item, pat in active_patterns:
                        if pat.search(line_str):
                            matched_user_items.add(item)
                            line_matched = True

                    # 2. If not matched, check if it's a basic pantry staple
                    if not line_matched and assume_staples:
                        if any(sp.search(line_str) for sp in staple_patterns):
                            line_matched = True

                    # 3. Otherwise, it's a missing ingredient
                    if not line_matched:
                        missing_items.append(clean_ingredient_name(line_str))

                # Remove duplicates while preserving order
                unique_missing = list(dict.fromkeys(missing_items))
                matched_list = [item.replace('_', ' ').title() for item in matched_user_items]
                return pd.Series([len(matched_list), len(unique_missing), unique_missing, matched_list])

            top_results[['Match_Count', 'Missing_Count', 'Missing_Ingredients', 'Matched_Ingredients']] = top_results.apply(evaluate_recipe, axis=1)
            
            # Only keep recipes that actually use at least 1 of the user's ingredients
            matched_results = top_results[top_results['Match_Count'] > 0]

            # Sort by: fewest missing ingredients first -> most user matches -> highest cosine score
            ranked_results = matched_results.sort_values(
                by=['Missing_Count', 'Match_Count', 'Cosine_Score'],
                ascending=[True, False, False]
            )

            # Apply Strict Mode with smart fallback
            if strict_mode:
                exact_matches = ranked_results[ranked_results['Missing_Count'] == 0]
                if not exact_matches.empty:
                    all_results = exact_matches
                else:
                    st.info(
                        "No 100% exact matches found with just those ingredients. "
                        "Showing the closest recipes with the fewest missing ingredients instead!"
                    )
                    all_results = ranked_results
            else:
                all_results = ranked_results

            if all_results.empty:
                st.info("No recipes found. Try adding different ingredients.")
            else:
                max_available = min(len(all_results), 25)
                per_page = st.session_state.per_page
                final_results = all_results.head(min(per_page, max_available))
                total_shown = len(final_results)

                st.markdown("<h3 style='color: #2C1A0E; margin-top: -0.5rem; margin-bottom: 1rem;'>Top Matches</h3>", unsafe_allow_html=True)
                for rank, (_, row) in enumerate(final_results.iterrows(), start=1):
                    recipe_name = str(row["Title"])
                    is_saved = recipe_name in st.session_state.saved_recipes
                    btn_key = f"toggle_save_{recipe_name}_{rank}"
                    
                    matched = row["Match_Count"]
                    total_selected = len(active_ingredients)
                    missing = int(row["Missing_Count"])
                    if missing == 0:
                        missing_badge = '<span class="match-badge" style="background:#E8F5E9; color:#2E7D32; margin-left:6px;">✨ 100% Ready to Cook</span>'
                    else:
                        missing_badge = f'<span class="match-badge" style="background:#FFF3E0; color:#C9541A; margin-left:6px;">🛒 Missing {missing} ingredient{"s" if missing != 1 else ""}</span>'


                    # --- UNIFIED PREVIEW TEXT LOGIC ---
                    found_in_recipe = row["Matched_Ingredients"]
                    if found_in_recipe:
                        used_preview_string = ", ".join(found_in_recipe)
                    else:
                        used_preview_string = "None perfectly matched (Similar profile)"

                    missing_items = row["Missing_Ingredients"]
                    if missing_items:
                        missing_preview_html = f'<div style="font-size: 0.9rem; color: #7A5C44; margin-top: 0.2rem;"><b style="color: #C9541A;">Missing ingredients:</b> {", ".join(missing_items)}</div>'
                    else:
                        missing_preview_html = ""

                    # 3. THE ORANGE HIGHLIGHTER
                    try:
                        ing_list = ast.literal_eval(row['Ingredients'])
                        formatted_ings = []
                        for ing in ing_list:
                            highlighted_ing = ing
                            for search_term in sorted(active_ingredients, key=len, reverse=True):
                                display_term = search_term.lower().replace('_', ' ')
                                
                                if display_term in exclusion_rules:
                                    lookbehinds = "".join([f"(?<!{adj} )" for adj in exclusion_rules[display_term]])
                                    pattern = re.compile(rf'{lookbehinds}\b{re.escape(display_term)}\b', re.IGNORECASE)
                                else:
                                    pattern = re.compile(rf'\b{re.escape(display_term)}\b', re.IGNORECASE)
                                    
                                highlighted_ing = pattern.sub(r'<b style="color: #C9541A;">\g<0></b>', highlighted_ing)
                            formatted_ings.append(f"• {highlighted_ing}")
                        ingredients_html = "<br>".join(formatted_ings)
                    except:
                        highlighted_ing = str(row['Cleaned_Ingredients']).replace('_', ' ').title()
                        for search_term in sorted(active_ingredients, key=len, reverse=True):
                            display_term = search_term.lower().replace('_', ' ')
                            if display_term in exclusion_rules:
                                lookbehinds = "".join([f"(?<!{adj} )" for adj in exclusion_rules[display_term]])
                                pattern = re.compile(rf'{lookbehinds}\b{re.escape(display_term)}\b', re.IGNORECASE)
                            else:
                                pattern = re.compile(rf'\b{re.escape(display_term)}\b', re.IGNORECASE)
                            highlighted_ing = pattern.sub(r'<b style="color: #C9541A;">\g<0></b>', highlighted_ing)
                        ingredients_html = highlighted_ing
                    
                    # --- NEW: Numbered Instructions Logic ---
                    raw_instructions = str(row.get('Instructions', 'No instructions provided.'))
                    
                    if raw_instructions == 'No instructions provided.' or raw_instructions.lower() == 'nan':
                        formatted_instructions = "No instructions provided."
                    else:
                        steps = [step.strip() for step in raw_instructions.split('. ') if len(step.strip()) > 0]
                        formatted_instructions = "<br>".join([f"<b>{i+1}.</b> {step}." if not step.endswith('.') else f"<b>{i+1}.</b> {step}" for i, step in enumerate(steps)])
                    
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
{missing_badge}
</div>
<div style="font-size: 0.9rem; color: #7A5C44; margin-top: 0.2rem;">
<b style="color: #2C1A0E;">Used from kitchen:</b> {used_preview_string}
</div>
{missing_preview_html}
</summary>
<div style="margin-top: 0.5rem;">
<div class="ingredient-list">
<b style="color:#2C1A0E">All Ingredients Required:</b><br>
{ingredients_html}
</div>
<div class="instructions">
<b style="color:#2C1A0E">Instructions:</b><br>
{formatted_instructions}
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
                            
                            st.session_state.trigger_save = True
                            st.rerun()

                # --- Bottom Pagination Bar ---
                st.markdown("<hr style='margin: 1.5rem 0 0.8rem 0; border-color: #E8DDD3;'>", unsafe_allow_html=True)

                spacer, c_res, c_show_lbl, c_show = st.columns(
                    [7.5, 2.0, 0.9, 1.4],
                    vertical_alignment="center"
                )

                with c_res:
                    st.markdown(
                        f"<div style='text-align:right; color:#5C3D2E; font-size:0.9rem; white-space:nowrap;'>"
                        f"Showing {total_shown} of {max_available} Result(s)</div>",
                        unsafe_allow_html=True
                    )

                with c_show_lbl:
                    st.markdown("<div style='text-align:right; color:#5C3D2E; font-size:0.9rem;'>Show:</div>", unsafe_allow_html=True)

                with c_show:
                    page_sizes = [5, 10, 15, 25]
                    new_per_page = st.selectbox(
                        "Show",
                        options=page_sizes,
                        index=page_sizes.index(per_page) if per_page in page_sizes else 0,
                        label_visibility="collapsed",
                        key="pagination_show_select"
                    )
                    if new_per_page != st.session_state.per_page:
                        st.session_state.per_page = new_per_page
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
        st.markdown(f"<p style='color: #7A5C44; font-weight: 600;'>{len(st.session_state.saved_recipes)} saved recipe{'s' if len(st.session_state.saved_recipes) != 1 else ''}</p>", unsafe_allow_html=True)
        st.markdown("")

        for i, (recipe_name, recipe) in enumerate(st.session_state.saved_recipes.items(), start=1):

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

            matched_ings = recipe.get("matched_ingredients", [])
            matched_count = recipe.get("matched_count", None)
            total_selected = recipe.get("total_selected", None)

            if matched_count is not None:
                display_ings = ", ".join(matched_ings) if matched_ings else "None perfectly matched"
                
                matched_html = f"""
<div style="font-size:0.85rem; color:#7A5C44; margin-top:0.3rem;">
  <b style="color:#2C1A0E;">Your ingredients used:</b>
  <span style="color:#C9541A; font-weight:500;"> {matched_count} of {total_selected} matched</span>
  &nbsp;·&nbsp; {display_ings}
</div>"""
            else:
                matched_html = ""

            try:
                full_ing_html = "<br>".join([f"• {ing}" for ing in ing_list])
            except:
                full_ing_html = recipe['Ingredients']

            raw_instructions = str(recipe.get('Instructions', 'No instructions provided.'))
            
            if raw_instructions == 'No instructions provided.' or raw_instructions.lower() == 'nan':
                formatted_instructions = "No instructions provided."
            else:
                steps = [step.strip() for step in raw_instructions.split('. ') if len(step.strip()) > 0]
                formatted_instructions = "<br>".join([f"<b>{i+1}.</b> {step}." if not step.endswith('.') else f"<b>{i+1}.</b> {step}" for i, step in enumerate(steps)])

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
    {formatted_instructions}
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
                    st.session_state.trigger_save = True
                    st.rerun()